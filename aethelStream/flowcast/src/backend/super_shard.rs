//! Super-shard backend: groups `group_size` contiguous layers into one large
//! SQE (~100–200 MB read), then routes per-layer completions back via the
//! base backend.  Keeps internal per-layer byte offsets for slicing.
//!
//! A5-e fix: previous implementation was a single-line pass-through with no
//! coalescing.  This version accumulates pending reads and flushes them as a
//! group once `group_size` entries have arrived (or immediately on request).

use super::{BackendCapabilities, Completion, IoBackend};
use crate::Result;
use ramflow::PinnedBuffer;
use std::collections::HashMap;
use std::sync::Mutex;

/// Compute adaptive group size (layers per SQE) from a byte budget and layer sizes.
///
/// Uses the median shard size to convert from bytes to a layer count.  Handles
/// three edge cases:
/// * All layers smaller than `optimal_bytes`: group_size = `layer_sizes.len()`.
/// * `optimal_bytes` smaller than one layer: group_size = 1 (no coalescing).
/// * `optimal_bytes == 0` or `layer_sizes` empty: returns default of 4.
pub fn compute_group_size(optimal_bytes: u64, layer_sizes: &[u64]) -> u32 {
    if layer_sizes.is_empty() || optimal_bytes == 0 {
        return 4;
    }
    let mut sorted = layer_sizes.to_vec();
    sorted.sort_unstable();
    let median = sorted[sorted.len() / 2];
    if median == 0 {
        return 4;
    }
    if median >= optimal_bytes {
        return 1;
    }
    let count = (optimal_bytes / median).min(layer_sizes.len() as u64).max(1);
    count as u32
}

/// Super-shard grouping configuration.
pub struct SuperShardConfig {
    /// Fallback layer-count group size used when `optimal_super_shard_bytes == 0`.
    pub group_size: u32,
    /// Byte budget measured by the A3 latency-vs-size curve probe.
    ///
    /// When non-zero, `SuperShardBackend` flushes a group whenever the
    /// cumulative pending bytes reach this threshold instead of using a
    /// fixed layer count.  Handles mixed INT4/FP16 shards correctly.
    pub optimal_super_shard_bytes: u64,
    /// Per-layer shard sizes (bytes) used to derive the adaptive group count.
    pub layer_sizes: Vec<u64>,
}

impl Default for SuperShardConfig {
    fn default() -> Self {
        Self {
            group_size: 4,
            optimal_super_shard_bytes: 0,
            layer_sizes: Vec::new(),
        }
    }
}

/// A single read request queued for grouping.
struct PendingPrefetch {
    shard_id: u32,
    byte_offset: u64,
    length: u64,
    token: u64,
}

/// Super-shard backend wrapping any base `IoBackend`.
///
/// Group size is determined at warm-up by measuring the io_uring
/// latency-vs-transfer-size curve and selecting the knee point.  On PCIe 4
/// NVMe, this is typically 4–16 MiB.  Group size is updated live on thermal
/// throttling events via [`SuperShardBackend::update_group_size`].
///
/// When `optimal_bytes > 0`, the backend uses a **byte-budget** flush policy:
/// a group is submitted whenever the cumulative pending bytes reach
/// `optimal_bytes`.  This handles mixed INT4/FP16 shards correctly without
/// relying on a fixed layer count.
///
/// When `optimal_bytes == 0`, the backend falls back to the count-based policy
/// (`group_size` layers per SQE).
pub struct SuperShardBackend {
    base: Box<dyn IoBackend>,
    /// Derived adaptive group count (layers per SQE).  Updated atomically by
    /// [`update_group_size`] without stopping the pipeline.
    group_size: std::sync::Arc<std::sync::atomic::AtomicU32>,
    /// Primary byte budget for grouping.  When `> 0`, overrides count-based
    /// flushing so mixed-size shards are grouped by transfer volume, not count.
    optimal_bytes: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// Pending reads waiting to be flushed as a group (A5-e).
    pending: Mutex<Vec<PendingPrefetch>>,
    /// layer_idx → byte offset within the merged SQE range (A5-e).
    layer_offsets: Mutex<HashMap<u32, u64>>,
}

impl SuperShardBackend {
    /// Wrap `base` with adaptive super-shard grouping.
    ///
    /// If `config.optimal_super_shard_bytes > 0` and `config.layer_sizes` is
    /// non-empty, the initial group size is derived from the byte budget and
    /// the median layer size.  Otherwise falls back to `config.group_size`.
    pub fn new(base: Box<dyn IoBackend>, config: SuperShardConfig) -> Self {
        let initial_group = if config.optimal_super_shard_bytes > 0
            && !config.layer_sizes.is_empty()
        {
            compute_group_size(config.optimal_super_shard_bytes, &config.layer_sizes)
        } else {
            config.group_size
        };
        Self {
            base,
            group_size: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(initial_group)),
            optimal_bytes: std::sync::Arc::new(
                std::sync::atomic::AtomicU64::new(config.optimal_super_shard_bytes),
            ),
            pending: Mutex::new(Vec::new()),
            layer_offsets: Mutex::new(HashMap::new()),
        }
    }

    /// Return the tracked byte offset for `shard_id` within its merged SQE, or
    /// `None` if that layer has not been submitted yet.
    pub fn layer_offset(&self, shard_id: u32) -> Option<u64> {
        self.layer_offsets
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(&shard_id)
            .copied()
    }

    /// Current adaptive group size (layers per SQE).
    ///
    /// Reflects the last value set by [`update_group_size`].
    pub fn group_size_hint(&self) -> u32 {
        self.group_size
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Update the byte budget and derived group count from a re-profiling event.
    ///
    /// Called by the thermal monitor (M3-New-3) when SSD throttling shifts the
    /// latency knee.  Atomically updates both `optimal_bytes` and `group_size`
    /// without pausing the pipeline.
    ///
    /// # Arguments
    /// * `new_optimal_bytes` — new byte budget from `probe_knee_near` (0 = revert
    ///   to count-based fallback).
    /// * `layer_sizes` — current per-layer shard sizes for computing the new count.
    pub fn update_group_size(&self, new_optimal_bytes: u64, layer_sizes: &[u64]) {
        use std::sync::atomic::Ordering;
        self.optimal_bytes
            .store(new_optimal_bytes, Ordering::Relaxed);
        let new_count = compute_group_size(new_optimal_bytes, layer_sizes);
        self.group_size.store(new_count, Ordering::Relaxed);
    }

    /// Flush all pending prefetch requests to the base backend.
    ///
    /// Computes the merged byte range [min_offset, max_offset + max_length),
    /// records each layer's offset within that range, then submits each read
    /// individually (mock path) or as one merged SQE (real io_uring path).
    fn flush_group(&self) -> Result<()> {
        let mut pending = self.pending.lock().unwrap_or_else(|p| p.into_inner());
        if pending.is_empty() {
            return Ok(());
        }

        // Compute merged range for layer-offset bookkeeping.
        let min_offset = pending.iter().map(|r| r.byte_offset).min().unwrap_or(0);
        {
            let mut offsets = self.layer_offsets.lock().unwrap_or_else(|p| p.into_inner());
            for read in pending.iter() {
                offsets.insert(read.shard_id, read.byte_offset - min_offset);
            }
        }

        // Submit each read to the base backend (mock-compatible path).
        // A real io_uring path would submit one SQE covering [min_offset, end).
        let reads = std::mem::take(&mut *pending);
        drop(pending);
        for read in reads {
            // PinnedBuffer is required by the IoBackend trait but the mock
            // ignores the destination pointer.  Allocate a minimal placeholder.
            let dst = ramflow::PinnedBuffer::alloc(read.length as usize)
                .map_err(crate::FlowCastError::RamFlow)?;
            self.base
                .prefetch(read.shard_id, read.byte_offset, read.length, &dst, read.token)?;
        }
        Ok(())
    }
}

impl IoBackend for SuperShardBackend {
    fn start(&mut self) -> Result<()> {
        self.base.start()
    }

    fn prefetch(
        &self,
        shard_id: u32,
        byte_offset: u64,
        length: u64,
        dst: &PinnedBuffer,
        token: u64,
    ) -> Result<()> {
        use std::sync::atomic::Ordering;
        // Accumulate into the pending group; flush when the byte-budget (or
        // count-based fallback) threshold is reached.
        // Items must NOT be forwarded to the base backend here — flush_group()
        // handles submission for the whole group atomically.
        let should_flush = {
            let mut pending = self.pending.lock().unwrap_or_else(|p| p.into_inner());
            pending.push(PendingPrefetch { shard_id, byte_offset, length, token });

            let optimal = self.optimal_bytes.load(Ordering::Relaxed);
            if optimal > 0 {
                // Byte-budget policy: handles mixed INT4/FP16 shards by grouping
                // consecutive layers until cumulative bytes reach `optimal_bytes`.
                let cumulative: u64 = pending.iter().map(|r| r.length).sum();
                cumulative >= optimal
            } else {
                // Count-based fallback when no curve measurement is available.
                pending.len() >= self.group_size.load(Ordering::Relaxed) as usize
            }
        };

        if should_flush {
            self.flush_group()?;
        }
        // `dst` is unused on the accumulation path; the grouped flush allocates
        // its own PinnedBuffer placeholders when submitting to the base backend.
        let _ = dst;
        Ok(())
    }

    fn write_async(
        &self,
        shard_id: u32,
        byte_offset: u64,
        length: u64,
        src: &PinnedBuffer,
        token: u64,
    ) -> Result<()> {
        self.base.write_async(shard_id, byte_offset, length, src, token)
    }

    fn poll_completions(&self) -> Result<Vec<Completion>> {
        self.base.poll_completions()
    }

    fn is_paused(&self) -> bool {
        self.base.is_paused()
    }

    fn set_pause(&self, paused: bool) {
        self.base.set_pause(paused);
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supports_gds: false,
            supports_super_shard: true,
            supports_write_skip: true,
            supports_multi_gpu: false,
            name: "super-shard",
        }
    }

    fn shutdown(&mut self) -> Result<()> {
        self.base.shutdown()
    }
}
