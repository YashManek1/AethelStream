// src/pool/subpools.rs — PoolRegistry: central registry of all pool shards
//
// Sprint 3A: full implementation replacing Sprint 0 stub.
//
// lz4-cache feature: when the pool is exhausted during a Recomputation window
// and at least one eviction candidate has been registered with
// offer_for_lz4_eviction, the slow-path checks the LZ4 tier before blocking.
// This is transparent to callers of the frozen claim() API.

use std::collections::HashMap;
#[cfg(feature = "lz4-cache")]
use std::sync::atomic::AtomicBool;
#[cfg(feature = "lz4-cache")]
use std::sync::atomic::AtomicU32;
#[cfg(feature = "lz4-cache")]
use std::sync::atomic::Ordering::{Acquire, Release};
use std::sync::{Arc, Mutex};

use crate::cuda_bridge::zero_copy::ZeroCopyRouter;
use crate::pool::slow_path::SlowPathAllocator;
use crate::pool::{LayerKind, PoolSlot, RingBuffer, TensorLocationDict, TensorSlab};
use crate::{RamFlowError, Result};

/// Low-RAM fallback mode for the embedding pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingFallbackMode {
    /// Normal: embedding table kept in pinned RAM (default).
    PinnedRam,
    /// SSD fallback: embedding loaded from NVMe only during lookup.
    SsdLookup,
}

/// Slab allocation strategy for low-RAM hosts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlabInitMode {
    /// Build all layer slabs during pool registry startup.
    Eager,
    /// Allocate a layer slab only on first use and allow explicit release.
    Lazy,
}

const DEFAULT_ATTENTION_SLOT_COUNT: usize = 4;
const DEFAULT_MLP_SLOT_COUNT: usize = 4;
const DEFAULT_NORM_SLOT_COUNT: usize = 4;
const DEFAULT_EMBEDDING_SLOT_COUNT: usize = 2;
const DEFAULT_LARGE_SLOT_BYTES: usize = 64 * 1024 * 1024;
const DEFAULT_NORM_SLOT_BYTES: usize = 1024 * 1024;
const DEFAULT_EMBEDDING_SLOT_BYTES: usize = 32 * 1024 * 1024;
const MINIMUM_SLOT_BYTES: usize = 512;
const DEFAULT_MAX_PINNED_RAM_FRACTION: f64 = 0.9;

/// Central registry of all pool shards.
///
/// Owns four [`RingBuffer`]s (attention, mlp, norm, embedding) and routes
/// `claim(LayerKind)` to the correct ring.  Falls back to the LZ4 eviction
/// tier (feature = "lz4-cache") during Recomputation windows before
/// blocking on the slow path.
///
/// # Thread safety
///
/// `PoolRegistry` is `Send + Sync`.
pub struct PoolRegistry {
    attention_ring: Arc<RingBuffer>,
    mlp_ring: Arc<RingBuffer>,
    norm_ring: Arc<RingBuffer>,
    embedding_ring: Arc<RingBuffer>,
    slow_path: SlowPathAllocator,
    embedding_fallback: EmbeddingFallbackMode,
    slabs: Vec<TensorSlab>,
    lazy_slabs: Mutex<HashMap<u32, TensorSlab>>,
    slab_init_mode: SlabInitMode,
    slab_threshold: usize,
    /// NUMA topology detected at startup.
    numa_config: crate::allocator::NumaConfig,

    /// True when pool rings were allocated via mmap fallback (feature = "mmap-fallback").
    #[cfg(feature = "mmap-fallback")]
    mmap_fallback_active: bool,

    /// LZ4 compressed eviction cache (None until enable_lz4_cache is called).
    #[cfg(feature = "lz4-cache")]
    eviction_cache: Mutex<Option<crate::pool::eviction_cache::EvictionCache>>,

    /// Layers offered for LZ4 eviction: (layer_idx, slot, kind, precision).
    /// Each slot remains claimed (counted in ring active_claims) until evicted.
    #[cfg(feature = "lz4-cache")]
    eviction_candidates: Mutex<
        std::collections::VecDeque<(
            u32,
            PoolSlot,
            LayerKind,
            crate::pool::eviction_cache::CachePrecision,
        )>,
    >,

    /// True while the training loop is inside a Recomputation window.
    /// AcqRel so set_recompute_window / claim observe consistent bounds.
    #[cfg(feature = "lz4-cache")]
    recompute_window_active: AtomicBool,

    /// Inclusive layer-index start of the current Recomputation window.
    #[cfg(feature = "lz4-cache")]
    recompute_window_start: AtomicU32,

    /// Inclusive layer-index end of the current Recomputation window.
    #[cfg(feature = "lz4-cache")]
    recompute_window_end: AtomicU32,
}

impl PoolRegistry {
    /// Construct from a hardware profile and tensor location dict.
    ///
    /// Derives slot counts from profile slots_needed fields (minimum 1).
    /// Derives slot sizes from zero_copy_threshold.
    ///
    /// # Errors
    ///
    /// Returns [`RamFlowError::AllocationFailed`] if PinnedBuffer allocations fail.
    /// Returns [`RamFlowError::ConfigError`] if zero_copy_threshold is zero.
    pub fn new(
        profile: &crate::phase::classifier::PhaseMemoryProfile,
        dict: &TensorLocationDict,
        zero_copy_threshold: usize,
    ) -> Result<Self> {
        if zero_copy_threshold == 0 {
            return Err(RamFlowError::ConfigError(
                "zero_copy_threshold must be non-zero".into(),
            ));
        }
        ZeroCopyRouter::set_threshold(zero_copy_threshold);

        let large_slot_bytes = (zero_copy_threshold / 2).max(MINIMUM_SLOT_BYTES);
        let norm_slot_bytes = DEFAULT_NORM_SLOT_BYTES;
        let embedding_slot_bytes = (zero_copy_threshold / 4).max(MINIMUM_SLOT_BYTES);
        let attention_slot_count = (profile.attention_slots_needed as usize).max(1);
        let mlp_slot_count = (profile.mlp_slots_needed as usize).max(1);
        let norm_slot_count = (profile.norm_slots_needed as usize).max(1);
        let embedding_slot_count = (profile.optimizer_slots_needed as usize).max(1);

        let budget_plan = PoolBudgetPlan {
            large_slot_bytes,
            norm_slot_bytes,
            embedding_slot_bytes,
            attention_slot_count,
            mlp_slot_count,
            norm_slot_count,
            embedding_slot_count,
        };
        let preflight_result = preflight_profile_memory_budget(
            profile,
            &budget_plan,
            available_ram_bytes(),
            configured_pool_ram_fraction(),
        );

        #[cfg(feature = "mmap-fallback")]
        let use_mmap_fallback = matches!(
            &preflight_result,
            Err(RamFlowError::ConfigError(message)) if message.contains("pre-flight budget")
        );
        #[cfg(not(feature = "mmap-fallback"))]
        let use_mmap_fallback = false;

        if !use_mmap_fallback {
            preflight_result?;
        } else {
            // Insufficient pinned RAM — fall back to mmap-backed pool slots.
            // mmap decompression at ~4 GB/s is slower than pinned DMA (~14 GB/s),
            // but allows training on 16–32 GB machines without OOM.
            eprintln!(
                "[ramflow] WARNING: Insufficient pinned RAM for optimal streaming. \
                 Falling back to mmap-backed buffers. \
                 Expect ~2–4× lower throughput."
            );
        }

        #[cfg(feature = "mmap-fallback")]
        let (attention_ring, mlp_ring, norm_ring, embedding_ring) = if use_mmap_fallback {
            (
                Arc::new(RingBuffer::new_mmap(large_slot_bytes, attention_slot_count)?),
                Arc::new(RingBuffer::new_mmap(large_slot_bytes, mlp_slot_count)?),
                Arc::new(RingBuffer::new_mmap(norm_slot_bytes, norm_slot_count)?),
                Arc::new(RingBuffer::new_mmap(embedding_slot_bytes, embedding_slot_count)?),
            )
        } else {
            (
                Arc::new(RingBuffer::new(large_slot_bytes, attention_slot_count)?),
                Arc::new(RingBuffer::new(large_slot_bytes, mlp_slot_count)?),
                Arc::new(RingBuffer::new(norm_slot_bytes, norm_slot_count)?),
                Arc::new(RingBuffer::new(embedding_slot_bytes, embedding_slot_count)?),
            )
        };
        #[cfg(not(feature = "mmap-fallback"))]
        let attention_ring = Arc::new(RingBuffer::new(large_slot_bytes, attention_slot_count)?);
        #[cfg(not(feature = "mmap-fallback"))]
        let mlp_ring = Arc::new(RingBuffer::new(large_slot_bytes, mlp_slot_count)?);
        #[cfg(not(feature = "mmap-fallback"))]
        let norm_ring = Arc::new(RingBuffer::new(norm_slot_bytes, norm_slot_count)?);
        #[cfg(not(feature = "mmap-fallback"))]
        let embedding_ring =
            Arc::new(RingBuffer::new(embedding_slot_bytes, embedding_slot_count)?);
        let slabs = build_eager_slabs(dict, zero_copy_threshold)?;

        let numa_config = crate::allocator::numa::detect(None);
        #[cfg(all(feature = "numa", target_os = "linux"))]
        if let Some(node) = numa_config.gpu_node {
            for ring in [&attention_ring, &mlp_ring, &norm_ring, &embedding_ring] {
                ring.apply_numa_binding(node);
            }
        }

        Ok(PoolRegistry {
            attention_ring,
            mlp_ring,
            norm_ring,
            embedding_ring,
            slow_path: SlowPathAllocator::new(),
            embedding_fallback: EmbeddingFallbackMode::PinnedRam,
            slabs,
            lazy_slabs: Mutex::new(HashMap::new()),
            slab_init_mode: SlabInitMode::Eager,
            slab_threshold: zero_copy_threshold,
            numa_config,
            #[cfg(feature = "mmap-fallback")]
            mmap_fallback_active: use_mmap_fallback,
            #[cfg(feature = "lz4-cache")]
            eviction_cache: Mutex::new(None),
            #[cfg(feature = "lz4-cache")]
            eviction_candidates: Mutex::new(std::collections::VecDeque::new()),
            #[cfg(feature = "lz4-cache")]
            recompute_window_active: AtomicBool::new(false),
            #[cfg(feature = "lz4-cache")]
            recompute_window_start: AtomicU32::new(0),
            #[cfg(feature = "lz4-cache")]
            recompute_window_end: AtomicU32::new(0),
        })
    }

    /// Construct with lazy slab allocation for low-RAM systems.
    pub fn new_lazy(
        profile: &crate::phase::classifier::PhaseMemoryProfile,
        _dict: &TensorLocationDict,
        zero_copy_threshold: usize,
    ) -> Result<Self> {
        let mut registry = Self::new(profile, &TensorLocationDict::empty(), zero_copy_threshold)?;
        registry.slabs.clear();
        registry.slab_init_mode = SlabInitMode::Lazy;
        registry.slab_threshold = zero_copy_threshold;
        Ok(registry)
    }

    /// Convenience constructor with small built-in defaults for unit tests.
    ///
    /// # Errors
    ///
    /// Returns [`RamFlowError::AllocationFailed`] if any ring PinnedBuffer allocations fail.
    pub fn with_defaults() -> Result<Self> {
        let attention_ring = Arc::new(RingBuffer::new(
            DEFAULT_LARGE_SLOT_BYTES,
            DEFAULT_ATTENTION_SLOT_COUNT,
        )?);
        let mlp_ring = Arc::new(RingBuffer::new(
            DEFAULT_LARGE_SLOT_BYTES,
            DEFAULT_MLP_SLOT_COUNT,
        )?);
        let norm_ring =
            Arc::new(RingBuffer::new(DEFAULT_NORM_SLOT_BYTES, DEFAULT_NORM_SLOT_COUNT)?);
        let embedding_ring = Arc::new(RingBuffer::new(
            DEFAULT_EMBEDDING_SLOT_BYTES,
            DEFAULT_EMBEDDING_SLOT_COUNT,
        )?);
        Ok(PoolRegistry {
            attention_ring,
            mlp_ring,
            norm_ring,
            embedding_ring,
            slow_path: SlowPathAllocator::new(),
            embedding_fallback: EmbeddingFallbackMode::PinnedRam,
            slabs: Vec::new(),
            lazy_slabs: Mutex::new(HashMap::new()),
            slab_init_mode: SlabInitMode::Lazy,
            slab_threshold: 4 * 1024 * 1024,
            numa_config: crate::allocator::NumaConfig::disabled(),
            #[cfg(feature = "mmap-fallback")]
            mmap_fallback_active: false,
            #[cfg(feature = "lz4-cache")]
            eviction_cache: Mutex::new(None),
            #[cfg(feature = "lz4-cache")]
            eviction_candidates: Mutex::new(std::collections::VecDeque::new()),
            #[cfg(feature = "lz4-cache")]
            recompute_window_active: AtomicBool::new(false),
            #[cfg(feature = "lz4-cache")]
            recompute_window_start: AtomicU32::new(0),
            #[cfg(feature = "lz4-cache")]
            recompute_window_end: AtomicU32::new(0),
        })
    }

    /// Claim one slot of the pool appropriate for `kind`.
    ///
    /// Three-tier dispatch:
    ///   1. Fast path: `try_claim()` on the matching ring.
    ///   2. LZ4 path (feature="lz4-cache", Recomputation phase only): compress
    ///      oldest eligible eviction candidate, free its slot, reclaim it.
    ///   3. Slow path: signal gauge stall + blocking wait for a returned slot.
    ///
    /// # Errors
    ///
    /// Infallible for Sprint 3A; propagates OS errors from overflow allocation.
    pub fn claim(&self, kind: LayerKind) -> Result<PoolSlot> {
        let ring = match kind {
            LayerKind::Attention => &self.attention_ring,
            LayerKind::Mlp => &self.mlp_ring,
            LayerKind::Norm => &self.norm_ring,
            LayerKind::Embedding => &self.embedding_ring,
        };

        // Sprint 4 seam: SsdLookup falls through to the normal ring claim.
        // Sprint 4 replaces this with an SSD-backed handle return.
        let _ssd_embedding_seam =
            kind == LayerKind::Embedding && self.embedding_fallback == EmbeddingFallbackMode::SsdLookup;

        if let Some(pool_slot) = ring.try_claim() {
            return Ok(pool_slot);
        }

        // LZ4 eviction path: compress oldest eligible candidate before blocking.
        #[cfg(feature = "lz4-cache")]
        if let Some(slot) = self.try_lz4_evict_and_reclaim(ring, kind) {
            return Ok(slot);
        }

        self.slow_path.handle_exhaustion(ring, kind)
    }

    /// Claim a slot for `layer_idx`, automatically decompressing from the LZ4
    /// cache if a prior eviction stored data for that layer.
    ///
    /// Identical to `claim` when the `lz4-cache` feature is disabled.
    ///
    /// # Errors
    ///
    /// Returns the same errors as `claim` plus [`RamFlowError::ConfigError`]
    /// if a cached payload is corrupt.
    pub fn claim_for_layer(&self, kind: LayerKind, layer_idx: u32) -> Result<PoolSlot> {
        #[cfg(feature = "lz4-cache")]
        let mut slot = self.claim(kind)?;
        #[cfg(not(feature = "lz4-cache"))]
        let slot = {
            let _ = layer_idx;
            self.claim(kind)?
        };
        #[cfg(feature = "lz4-cache")]
        self.maybe_decompress_into_slot(layer_idx, &mut slot)?;
        Ok(slot)
    }

    // -----------------------------------------------------------------------
    // lz4-cache public API
    // -----------------------------------------------------------------------

    /// Enable the LZ4 eviction cache with the given byte budget.
    ///
    /// Replaces any previously installed cache.  `0` installs a zero-budget
    /// pass-through (every entry is immediately evicted — effectively a no-op).
    ///
    /// Recommended budget: `min(512 * 1024 * 1024, available_ram * 5 / 100)`.
    #[cfg(feature = "lz4-cache")]
    pub fn enable_lz4_cache(&self, max_compressed_bytes: usize) {
        let cache = crate::pool::eviction_cache::EvictionCache::new(max_compressed_bytes);
        *self
            .eviction_cache
            .lock()
            .unwrap_or_else(|p| p.into_inner()) = Some(cache);
    }

    /// Declare that the training loop is entering a Recomputation window.
    ///
    /// While active, `claim()` will compress eviction candidates whose layer
    /// index falls within `window_start..=window_end` instead of blocking.
    ///
    /// Stores `window_start` and `window_end` before the `active` flag so
    /// callers that observe `active=true` always see valid bounds.
    #[cfg(feature = "lz4-cache")]
    pub fn set_recompute_window(&self, window_start: u32, window_end: u32) {
        self.recompute_window_start.store(window_start, Release);
        self.recompute_window_end.store(window_end, Release);
        self.recompute_window_active.store(true, Release);
    }

    /// Clear the Recomputation window; LZ4 eviction will no longer trigger.
    #[cfg(feature = "lz4-cache")]
    pub fn clear_recompute_window(&self) {
        self.recompute_window_active.store(false, Release);
    }

    /// Register `slot` as a candidate for LZ4 eviction.
    ///
    /// The slot's data are compressed and the slot freed by the next `claim()`
    /// call that hits the slow path during a Recomputation window, provided the
    /// layer index falls within `[window_start, window_end]`.
    ///
    /// If the LZ4 cache has not been enabled via `enable_lz4_cache`, the slot
    /// is immediately dropped (returned to its ring) — the call is a no-op.
    ///
    /// # Errors
    ///
    /// Currently infallible; returns `Result` for forward-compatibility.
    #[cfg(feature = "lz4-cache")]
    pub fn offer_for_lz4_eviction(
        &self,
        layer_idx: u32,
        kind: LayerKind,
        slot: PoolSlot,
        precision: crate::pool::eviction_cache::CachePrecision,
    ) -> Result<()> {
        let cache_enabled = self
            .eviction_cache
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .is_some();
        if !cache_enabled {
            drop(slot);
            return Ok(());
        }
        self.eviction_candidates
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push_back((layer_idx, slot, kind, precision));
        Ok(())
    }

    /// Snapshot of LZ4 cache counters for telemetry consumers.
    ///
    /// Returns `None` when the cache has not been enabled.
    #[cfg(feature = "lz4-cache")]
    pub fn lz4_cache_telemetry(
        &self,
    ) -> Option<crate::pool::eviction_cache::Lz4CacheTelemetry> {
        self.eviction_cache
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .map(|cache| cache.telemetry())
    }

    // -----------------------------------------------------------------------
    // lz4-cache private helpers
    // -----------------------------------------------------------------------

    /// Try to free a ring slot by compressing the oldest eligible eviction
    /// candidate, then reclaim that freed slot for the caller.
    ///
    /// Invariants:
    ///   - The candidates Mutex is held only during candidate lookup; the cache
    ///     Mutex is held only during compression; neither is held during drop.
    ///   - Slot drop happens outside both Mutexes to prevent any ordering issue.
    #[cfg(feature = "lz4-cache")]
    fn try_lz4_evict_and_reclaim(
        &self,
        ring: &Arc<RingBuffer>,
        kind: LayerKind,
    ) -> Option<PoolSlot> {
        if !self.recompute_window_active.load(Acquire) {
            return None;
        }
        let window_start = self.recompute_window_start.load(Acquire);
        let window_end = self.recompute_window_end.load(Acquire);

        // Step 1: pop oldest matching candidate (candidates lock, then release).
        let candidate = {
            let mut candidates = self
                .eviction_candidates
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            let position =
                candidates
                    .iter()
                    .position(|(layer_idx, _, candidate_kind, _)| {
                        *candidate_kind == kind
                            && *layer_idx >= window_start
                            && *layer_idx <= window_end
                    });
            match position {
                Some(index) => candidates.remove(index),
                None => return None,
            }
        };

        let (layer_idx, slot, _kind, precision) = candidate?;

        // Step 2: compress under cache lock only; capture success flag.
        let compressed_ok = {
            let mut cache_guard = self
                .eviction_cache
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            match cache_guard.as_mut() {
                None => {
                    // Cache was disabled after the offer; drop slot and bail.
                    drop(slot);
                    return ring.try_claim();
                }
                Some(cache) => cache
                    .compress(layer_idx, slot.buffer().as_slice(), precision)
                    .is_ok(),
            }
        };

        // Step 3: drop slot outside all locks — returns buffer to ring.
        drop(slot);

        if compressed_ok {
            ring.try_claim()
        } else {
            None
        }
    }

    /// Decompress a cached entry for `layer_idx` into `slot`'s buffer if one exists.
    ///
    /// No-op when the cache has no entry for that layer.
    #[cfg(feature = "lz4-cache")]
    fn maybe_decompress_into_slot(&self, layer_idx: u32, slot: &mut PoolSlot) -> Result<()> {
        let mut cache_guard = self
            .eviction_cache
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let Some(cache) = cache_guard.as_mut() else {
            return Ok(());
        };
        if !cache.contains(layer_idx) {
            return Ok(());
        }
        cache.decompress(layer_idx, slot.buffer_mut().as_mut_slice())?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Existing public API (frozen)
    // -----------------------------------------------------------------------

    /// Select the embedding pool residency strategy.
    pub fn set_embedding_fallback(&mut self, fallback_mode: EmbeddingFallbackMode) {
        self.embedding_fallback = fallback_mode;
    }

    /// Current embedding residency strategy.
    pub fn embedding_fallback(&self) -> EmbeddingFallbackMode {
        self.embedding_fallback
    }

    /// Returns `true` when pool slots are backed by mmap instead of pinned memory.
    ///
    /// When `true`, DMA transfers use a pinned staging copy and throughput is
    /// approximately 2–4× lower than pinned DMA.
    #[cfg(feature = "mmap-fallback")]
    pub fn is_mmap_fallback(&self) -> bool {
        self.mmap_fallback_active
    }

    /// Number of currently-claimed slots for one pool ring.
    pub fn claimed_slots_for(&self, kind: LayerKind) -> usize {
        self.ring_for_kind(kind).claimed_slots()
    }

    /// Total slot capacity for one pool ring.
    pub fn capacity_for(&self, kind: LayerKind) -> usize {
        self.ring_for_kind(kind).total_slots()
    }

    /// Resize all pool rings to match a phase memory profile.
    ///
    /// # Errors
    ///
    /// Returns [`RamFlowError::PhaseTransitionError`] if any ring still has
    /// an in-flight slot. Returns [`RamFlowError::AllocationFailed`] if new
    /// pinned allocations fail.
    pub fn resize_to_profile(
        &self,
        profile: &crate::phase::classifier::PhaseMemoryProfile,
    ) -> Result<()> {
        self.attention_ring
            .resize((profile.attention_slots_needed as usize).max(1))?;
        self.mlp_ring
            .resize((profile.mlp_slots_needed as usize).max(1))?;
        self.norm_ring
            .resize((profile.norm_slots_needed as usize).max(1))?;
        self.embedding_ring
            .resize((profile.optimizer_slots_needed as usize).max(1))?;
        Ok(())
    }

    fn ring_for_kind(&self, kind: LayerKind) -> &Arc<RingBuffer> {
        match kind {
            LayerKind::Attention => &self.attention_ring,
            LayerKind::Mlp => &self.mlp_ring,
            LayerKind::Norm => &self.norm_ring,
            LayerKind::Embedding => &self.embedding_ring,
        }
    }

    /// Number of currently-claimed slots across all pools.
    pub fn total_claimed_slots(&self) -> usize {
        self.attention_ring.claimed_slots()
            + self.mlp_ring.claimed_slots()
            + self.norm_ring.claimed_slots()
            + self.embedding_ring.claimed_slots()
    }

    /// Total slot capacity across all pools (free + in-flight).
    pub fn total_capacity(&self) -> usize {
        self.attention_ring.total_slots()
            + self.mlp_ring.total_slots()
            + self.norm_ring.total_slots()
            + self.embedding_ring.total_slots()
    }

    /// Total bytes currently allocated across all rings.
    pub fn bytes_allocated(&self) -> usize {
        self.attention_ring.bytes_allocated()
            + self.mlp_ring.bytes_allocated()
            + self.norm_ring.bytes_allocated()
            + self.embedding_ring.bytes_allocated()
    }

    /// Return a reference to the pre-built eager slab for `layer_idx`.
    pub fn slab_for_layer(&self, layer_idx: u32) -> Option<&crate::pool::TensorSlab> {
        self.slabs.get(layer_idx as usize)
    }

    /// Build a lazy slab for `layer_idx` if it is not already present.
    ///
    /// # Errors
    ///
    /// Returns allocation or tensor-index errors from [`TensorSlab::build_for_layer`].
    pub fn ensure_slab_for_layer(&self, layer_idx: u32, dict: &TensorLocationDict) -> Result<()> {
        if self.slab_init_mode == SlabInitMode::Eager {
            return Ok(());
        }
        let mut lazy_slabs = self
            .lazy_slabs
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if let std::collections::hash_map::Entry::Vacant(entry) = lazy_slabs.entry(layer_idx) {
            let slab = TensorSlab::build_for_layer(layer_idx, dict, self.slab_threshold)?;
            entry.insert(slab);
        }
        Ok(())
    }

    /// Release a lazily-created slab once it will not be revisited this pass.
    pub fn release_lazy_slab(&self, layer_idx: u32) {
        let mut lazy_slabs = self
            .lazy_slabs
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        lazy_slabs.remove(&layer_idx);
    }

    /// Register the pressure gauge with the slow-path allocator.
    ///
    /// After this call, pool exhaustion events call `gauge.signal_stall()`
    /// immediately — the NVMe engine pauses so in-flight DMA transfers complete
    /// and return their buffers before the training loop blocks.
    pub fn set_pressure_gauge(&self, gauge: crate::scheduler::MemoryPressureGauge) {
        self.slow_path.set_gauge(gauge);
    }

    /// NUMA topology detected at pool startup.
    pub fn numa_config(&self) -> crate::allocator::NumaConfig {
        self.numa_config
    }
}

fn build_eager_slabs(dict: &TensorLocationDict, threshold: usize) -> Result<Vec<TensorSlab>> {
    let layer_count = dict.num_layers().max(80);
    let mut slabs = Vec::with_capacity(layer_count);
    for layer_idx in 0..layer_count {
        slabs.push(TensorSlab::build_for_layer(
            layer_idx as u32,
            dict,
            threshold,
        )?);
    }
    Ok(slabs)
}

struct PoolBudgetPlan {
    large_slot_bytes: usize,
    norm_slot_bytes: usize,
    embedding_slot_bytes: usize,
    attention_slot_count: usize,
    mlp_slot_count: usize,
    norm_slot_count: usize,
    embedding_slot_count: usize,
}

fn preflight_profile_memory_budget(
    profile: &crate::phase::classifier::PhaseMemoryProfile,
    plan: &PoolBudgetPlan,
    available_bytes: Option<u64>,
    max_fraction: f64,
) -> Result<()> {
    let Some(available_bytes) = available_bytes else {
        return Ok(());
    };
    let pool_bytes = checked_sum(&[
        checked_mul(plan.large_slot_bytes, plan.attention_slot_count)?,
        checked_mul(plan.large_slot_bytes, plan.mlp_slot_count)?,
        checked_mul(plan.norm_slot_bytes, plan.norm_slot_count)?,
        checked_mul(plan.embedding_slot_bytes, plan.embedding_slot_count)?,
    ])?;
    let checkpoint_budget = profile.expected_peak_bytes as u128;
    let optimizer_budget = checked_mul(plan.embedding_slot_bytes, plan.embedding_slot_count)?;
    let estimated_peak = checked_sum(&[pool_bytes, checkpoint_budget, optimizer_budget])?;
    let allowed_bytes = (available_bytes as f64 * max_fraction.clamp(0.0, 1.0)) as u128;
    if estimated_peak > allowed_bytes {
        return Err(RamFlowError::ConfigError(format!(
            "pool pre-flight budget estimate {estimated_peak} bytes exceeds {:.1}% of available RAM ({available_bytes} bytes); consider enabling the `mmap-fallback` feature for degraded-mode operation on low-RAM machines",
            max_fraction.clamp(0.0, 1.0) * 100.0
        )));
    }
    Ok(())
}

fn checked_mul(bytes: usize, count: usize) -> Result<u128> {
    (bytes as u128)
        .checked_mul(count as u128)
        .ok_or_else(|| RamFlowError::ConfigError("pool pre-flight budget overflow".into()))
}

fn checked_sum(values: &[u128]) -> Result<u128> {
    values.iter().try_fold(0_u128, |accumulator, value| {
        accumulator
            .checked_add(*value)
            .ok_or_else(|| RamFlowError::ConfigError("pool pre-flight budget overflow".into()))
    })
}

fn configured_pool_ram_fraction() -> f64 {
    std::env::var("RAMFLOW_POOL_RAM_FRACTION")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| *value > 0.0)
        .unwrap_or(DEFAULT_MAX_PINNED_RAM_FRACTION)
}

fn available_ram_bytes() -> Option<u64> {
    if let Some(override_bytes) = std::env::var("RAMFLOW_MEM_AVAILABLE_BYTES")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
    {
        return Some(override_bytes);
    }
    #[cfg(target_os = "linux")]
    {
        read_linux_mem_available_bytes()
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

#[cfg(target_os = "linux")]
fn read_linux_mem_available_bytes() -> Option<u64> {
    let contents = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in contents.lines() {
        let Some(rest) = line.strip_prefix("MemAvailable:") else {
            continue;
        };
        let kib = rest.split_whitespace().next()?.parse::<u64>().ok()?;
        return kib.checked_mul(1024);
    }
    None
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::phase::{PhaseMemoryProfile, TrainingPhase};

    #[test]
    fn absurd_profile_is_rejected_by_preflight_budget() {
        let profile = PhaseMemoryProfile {
            phase: TrainingPhase::Forward {
                layers_in_flight: 1,
            },
            expected_peak_bytes: 8 * 1024 * 1024,
            attention_slots_needed: 1024,
            mlp_slots_needed: 1024,
            norm_slots_needed: 1024,
            optimizer_slots_needed: 1024,
        };
        let budget_plan = PoolBudgetPlan {
            large_slot_bytes: 1024 * 1024,
            norm_slot_bytes: 1024 * 1024,
            embedding_slot_bytes: 1024 * 1024,
            attention_slot_count: profile.attention_slots_needed as usize,
            mlp_slot_count: profile.mlp_slots_needed as usize,
            norm_slot_count: profile.norm_slots_needed as usize,
            embedding_slot_count: profile.optimizer_slots_needed as usize,
        };
        let result = preflight_profile_memory_budget(
            &profile,
            &budget_plan,
            Some(64 * 1024 * 1024),
            DEFAULT_MAX_PINNED_RAM_FRACTION,
        );
        assert!(
            matches!(result, Err(RamFlowError::ConfigError(ref message)) if message.contains("pre-flight budget")),
            "absurd pool profile should fail before allocation, got {result:?}"
        );
    }
}


