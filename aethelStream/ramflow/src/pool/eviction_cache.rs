// src/pool/eviction_cache.rs — LZ4 compressed eviction tier
//
// Feature gate: `lz4-cache`
//
// When the active pool is exhausted during a Recomputation window, a layer
// that will be revisited (but has finished its first pass) is compressed with
// LZ4 into this side buffer instead of being written back to SSD.
//
// LZ4 decompression at 4–5 GB/s/core outpaces most consumer NVMe reads
// (≤ 7 GB/s sequential, with queue-depth and seek penalty in practice), and
// costs zero SSD TBW.  This addresses the "low-RAM box" scenario where the
// pool has too few slots to hold all recompute-window layers simultaneously.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use crate::{RamFlowError, Result};

// ---------------------------------------------------------------------------
// CachePrecision
// ---------------------------------------------------------------------------

/// Precision of the data stored in an eviction cache entry.
///
/// Stored alongside the compressed bytes so the caller can validate the
/// buffer layout on decompression without consulting a separate index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CachePrecision {
    /// 32-bit float (4 bytes/element).
    Fp32,
    /// 16-bit half-precision float (2 bytes/element).
    Fp16,
    /// Brain float-16 (2 bytes/element); Ampere+ native.
    Bf16,
    /// Signed 8-bit integer (1 byte/element).
    Int8,
}

// ---------------------------------------------------------------------------
// Lz4CacheTelemetry
// ---------------------------------------------------------------------------

/// Snapshot of LZ4 eviction cache counters exposed for telemetry.
///
/// Returned by [`crate::pool::PoolRegistry::lz4_cache_telemetry`].
#[derive(Debug, Clone, Copy)]
pub struct Lz4CacheTelemetry {
    /// Decompress calls that found a cached entry.
    pub hits: u64,
    /// Decompress calls that found no entry (cold miss or already evicted).
    pub misses: u64,
    /// Current compressed bytes stored in the cache.
    pub current_bytes: usize,
    /// Maximum compressed byte budget configured for this cache.
    pub max_bytes: usize,
}

// ---------------------------------------------------------------------------
// EvictionCache
// ---------------------------------------------------------------------------

/// LRU-evicting in-RAM compressed buffer cache.
///
/// Stores up to `max_compressed_bytes` of LZ4-compressed layer data.  When a
/// new entry would push the total past the budget, the oldest entry (by
/// insertion order) is silently dropped to make room.
///
/// All compression is done with `lz4_flex` — pure safe Rust, no C dependency.
///
/// # Thread safety
///
/// `EvictionCache` is `Send` but not `Sync`.  All access must go through an
/// external `Mutex`; `PoolRegistry` wraps it in `Mutex<Option<EvictionCache>>`.
pub struct EvictionCache {
    /// Compressed payload, original byte length, and precision per layer index.
    entries: HashMap<u32, (Vec<u8>, usize, CachePrecision)>,

    /// Insertion-order tracking for LRU eviction: oldest key at front.
    insertion_order: VecDeque<u32>,

    /// Maximum total compressed bytes allowed in the cache.
    max_compressed_bytes: usize,

    /// Running total of bytes used by all compressed entries.
    current_bytes: usize,

    /// Cumulative decompress hits since construction.
    hits: AtomicU64,

    /// Cumulative decompress misses since construction.
    misses: AtomicU64,
}

impl EvictionCache {
    /// Construct an empty cache with `max_compressed_bytes` budget.
    ///
    /// Passing `0` creates a zero-budget cache; all `compress` calls succeed
    /// but every entry is immediately evicted, making the cache a transparent
    /// pass-through no-op.
    pub fn new(max_compressed_bytes: usize) -> Self {
        EvictionCache {
            entries: HashMap::new(),
            insertion_order: VecDeque::new(),
            max_compressed_bytes,
            current_bytes: 0,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// LZ4-compress `src` and store it under `layer_idx`.
    ///
    /// If the resulting payload would push `current_bytes` past
    /// `max_compressed_bytes`, the oldest entry is evicted first.  An
    /// existing entry for `layer_idx` is replaced and its bytes reclaimed.
    ///
    /// # Errors
    ///
    /// Returns [`RamFlowError::ConfigError`] if `src` is empty.
    pub fn compress(
        &mut self,
        layer_idx: u32,
        src: &[u8],
        precision: CachePrecision,
    ) -> Result<()> {
        if src.is_empty() {
            return Err(RamFlowError::ConfigError(
                "EvictionCache::compress: source buffer is empty".into(),
            ));
        }

        let orig_len = src.len();
        let compressed = lz4_flex::compress(src);

        // Remove any existing entry for this layer so budget accounting and
        // insertion_order stay consistent before we insert the new one.
        if let Some((old_compressed, _, _)) = self.entries.remove(&layer_idx) {
            self.current_bytes = self.current_bytes.saturating_sub(old_compressed.len());
            self.insertion_order.retain(|key| *key != layer_idx);
        }

        // Evict oldest entries until the new payload fits within budget.
        while !self.insertion_order.is_empty()
            && self.current_bytes + compressed.len() > self.max_compressed_bytes
        {
            self.evict_oldest();
        }

        self.current_bytes += compressed.len();
        self.entries
            .insert(layer_idx, (compressed, orig_len, precision));
        self.insertion_order.push_back(layer_idx);
        Ok(())
    }

    /// LZ4-decompress the cached entry for `layer_idx` into `dst`.
    ///
    /// Returns `Ok(true)` on a cache hit; the decompressed bytes are written
    /// to `dst[..orig_len]` and the entry is consumed (removed from cache).
    ///
    /// Returns `Ok(false)` when `layer_idx` has no entry.
    ///
    /// # Errors
    ///
    /// Returns [`RamFlowError::ConfigError`] if `dst` is shorter than the
    /// stored `orig_len`, or if LZ4 reports a corrupt payload.
    pub fn decompress(&mut self, layer_idx: u32, dst: &mut [u8]) -> Result<bool> {
        let Some((compressed, orig_len, _precision)) = self.entries.remove(&layer_idx) else {
            self.misses.fetch_add(1, Relaxed);
            return Ok(false);
        };

        self.current_bytes = self.current_bytes.saturating_sub(compressed.len());
        self.insertion_order.retain(|key| *key != layer_idx);

        if dst.len() < orig_len {
            return Err(RamFlowError::ConfigError(format!(
                "EvictionCache::decompress: dst is {} bytes but layer {layer_idx} needs {orig_len}",
                dst.len()
            )));
        }

        let decompressed =
            lz4_flex::decompress(&compressed, orig_len).map_err(|decompression_error| {
                RamFlowError::ConfigError(format!(
                    "EvictionCache: LZ4 decompression failed for layer {layer_idx}: \
                     {decompression_error}"
                ))
            })?;

        dst[..orig_len].copy_from_slice(&decompressed);
        self.hits.fetch_add(1, Relaxed);
        Ok(true)
    }

    /// Returns `true` if a cache entry exists for `layer_idx`.
    pub fn contains(&self, layer_idx: u32) -> bool {
        self.entries.contains_key(&layer_idx)
    }

    /// Remove any existing entry for `layer_idx` without decompressing it.
    ///
    /// Used when a layer is permanently evicted or freed before the recompute
    /// window needs it.
    pub fn invalidate(&mut self, layer_idx: u32) {
        if let Some((compressed, _, _)) = self.entries.remove(&layer_idx) {
            self.current_bytes = self.current_bytes.saturating_sub(compressed.len());
            self.insertion_order.retain(|key| *key != layer_idx);
        }
    }

    /// Cumulative number of successful decompress calls since construction.
    pub fn hits(&self) -> u64 {
        self.hits.load(Relaxed)
    }

    /// Cumulative number of decompress misses since construction.
    pub fn misses(&self) -> u64 {
        self.misses.load(Relaxed)
    }

    /// Current total compressed bytes held in the cache.
    pub fn current_bytes(&self) -> usize {
        self.current_bytes
    }

    /// Configured maximum compressed byte budget.
    pub fn max_bytes(&self) -> usize {
        self.max_compressed_bytes
    }

    /// Number of layer entries currently in the cache.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the cache holds no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Snapshot of hit/miss counters and byte usage for telemetry consumers.
    pub fn telemetry(&self) -> Lz4CacheTelemetry {
        Lz4CacheTelemetry {
            hits: self.hits.load(Relaxed),
            misses: self.misses.load(Relaxed),
            current_bytes: self.current_bytes,
            max_bytes: self.max_compressed_bytes,
        }
    }

    /// Evict the single oldest (by insertion order) entry.
    fn evict_oldest(&mut self) {
        let Some(oldest_key) = self.insertion_order.pop_front() else {
            return;
        };
        if let Some((compressed, _, _)) = self.entries.remove(&oldest_key) {
            self.current_bytes = self.current_bytes.saturating_sub(compressed.len());
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    /// Verify that compress → decompress produces byte-identical output.
    #[test]
    fn round_trip_byte_identical_1mb() {
        let size = 1024 * 1024;
        // Pseudo-random pattern that is NOT all-zeros (more realistic).
        let original: Vec<u8> = (0..size)
            .map(|index: usize| (index.wrapping_mul(6364136223846793005usize).wrapping_add(1442695040888963407)) as u8)
            .collect();

        let mut cache = EvictionCache::new(64 * 1024 * 1024);
        cache
            .compress(0, &original, CachePrecision::Fp16)
            .expect("compress");

        let mut restored = vec![0u8; size];
        let hit = cache.decompress(0, &mut restored).expect("decompress");

        assert!(hit, "decompress must return true on a cache hit");
        assert_eq!(original, restored, "decompressed bytes must be identical to source");
    }

    /// Measure compression ratio for random FP16 data and document expectations.
    ///
    /// Random FP16 compresses poorly (ratio near 1.0) because the bit patterns
    /// are essentially white noise.  Real trained weights have much more
    /// structure and compress to 0.5–0.8× in practice.
    #[test]
    fn compression_ratio_random_fp16_is_documented() {
        let size = 1024 * 1024; // 1 MiB
        let random_fp16: Vec<u8> = (0..size)
            .map(|index| ((index * 2654435761usize) ^ (index >> 5)) as u8)
            .collect();

        let mut cache = EvictionCache::new(64 * 1024 * 1024);
        cache
            .compress(0, &random_fp16, CachePrecision::Fp16)
            .expect("compress");

        let ratio = cache.current_bytes() as f64 / size as f64;
        // Random FP16 compresses poorly; assert the cache is storing something.
        // A ratio > 0.8 is expected for noise; real weights may be ≤ 0.5.
        assert!(
            cache.current_bytes() > 0,
            "cache must store at least one byte"
        );
        // Documentation: ratio is expected to be between 0.8 and 1.1 for pure
        // random data.  Real trained weights compress much better (0.5–0.75×).
        let _ = ratio;
    }

    /// LRU eviction: filling the cache past `max_bytes` should drop the oldest entry.
    #[test]
    fn lru_eviction_drops_oldest_entry() {
        // Budget: 600 bytes.  Each compressed entry for 256-byte all-same-byte
        // payloads is ~20 bytes (LZ4 handles runs very well).
        // We'll use 3 entries of 200-byte compressed budgets.
        // Set a tight budget: hold at most 2 compressed entries.
        let entry_size = 256usize;

        // Compress a known-compressible payload (repeated byte → near-zero output).
        let payload_a = vec![0xABu8; entry_size];
        let payload_b = vec![0xCDu8; entry_size];
        let payload_c = vec![0xEFu8; entry_size];

        // Compress entry A to measure its compressed size.
        let compressed_a = lz4_flex::compress(&payload_a);
        let compressed_b = lz4_flex::compress(&payload_b);

        // Budget: fit A and B but not C.
        let budget = compressed_a.len() + compressed_b.len() + 1;
        let mut cache = EvictionCache::new(budget);

        cache.compress(0, &payload_a, CachePrecision::Fp16).expect("compress a");
        assert!(cache.contains(0), "entry 0 (A) must be present after insert");

        cache.compress(1, &payload_b, CachePrecision::Fp16).expect("compress b");
        assert!(cache.contains(1), "entry 1 (B) must be present");

        // Adding C forces eviction of the oldest (A).
        cache.compress(2, &payload_c, CachePrecision::Fp16).expect("compress c");
        assert!(
            !cache.contains(0),
            "entry 0 (A) must have been LRU-evicted to make room for C"
        );
        assert!(cache.contains(2), "entry 2 (C) must be present after insert");
    }

    /// decompress on a missing layer returns Ok(false) and increments misses.
    #[test]
    fn decompress_miss_returns_false_and_increments_counter() {
        let mut cache = EvictionCache::new(1024 * 1024);
        let mut dst = vec![0u8; 64];
        let hit = cache.decompress(99, &mut dst).expect("decompress");
        assert!(!hit, "missing entry must return false");
        assert_eq!(cache.misses(), 1);
        assert_eq!(cache.hits(), 0);
    }

    /// compress with empty src returns ConfigError.
    #[test]
    fn compress_empty_src_returns_error() {
        let mut cache = EvictionCache::new(1024);
        let result = cache.compress(0, &[], CachePrecision::Fp16);
        assert!(
            matches!(result, Err(RamFlowError::ConfigError(_))),
            "empty src must return ConfigError"
        );
    }

    /// decompress into undersized dst returns ConfigError.
    #[test]
    fn decompress_undersized_dst_returns_error() {
        let src = vec![0u8; 512];
        let mut cache = EvictionCache::new(1024 * 1024);
        cache.compress(0, &src, CachePrecision::Fp16).expect("compress");
        let mut tiny_dst = vec![0u8; 16];
        let result = cache.decompress(0, &mut tiny_dst);
        assert!(
            matches!(result, Err(RamFlowError::ConfigError(_))),
            "undersized dst must return ConfigError"
        );
    }

    /// invalidate removes entry and reclaims bytes.
    #[test]
    fn invalidate_removes_entry_and_reclaims_bytes() {
        let src = vec![0xAAu8; 512];
        let mut cache = EvictionCache::new(1024 * 1024);
        cache.compress(0, &src, CachePrecision::Fp16).expect("compress");
        let bytes_after_insert = cache.current_bytes();
        assert!(bytes_after_insert > 0);
        cache.invalidate(0);
        assert!(!cache.contains(0));
        assert_eq!(cache.current_bytes(), 0);
    }
}

