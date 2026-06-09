// src/pool/subpools.rs — PoolRegistry: central registry of all pool shards
//
// Sprint 3A: full implementation replacing Sprint 0 stub.
//
// ─── DESIGN ───────────────────────────────────────────────────────────────────
//
// PoolRegistry owns four RingBuffers — one per LayerKind — and a
// SlowPathAllocator.  `claim(LayerKind)` dispatches to the matching ring
// via `try_claim` (fast path); on exhaustion the slow path blocks until a
// slot is returned.
//
// All slot sizes and capacities are fixed at construction time from either
// the `PhaseMemoryProfile` (production) or built-in defaults (testing / CI).
// Resize is a Sprint 4 operation (triggered by the Phase Rebalancer).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::pool::slow_path::SlowPathAllocator;
use crate::pool::{LayerKind, PoolSlot, RingBuffer, TensorLocationDict, TensorSlab};
use crate::{RamFlowError, Result};

// ---------------------------------------------------------------------------
// EmbeddingFallbackMode
// ---------------------------------------------------------------------------

/// Low-RAM fallback mode for the embedding pool.
///
/// On machines where a single embedding table does not fit in pinned RAM,
/// the embedding pool can defer to the SSD path: the embedding weights remain
/// on NVMe and are loaded fresh only during the lookup step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingFallbackMode {
    /// Normal: embedding table kept in pinned RAM (default).
    PinnedRam,

    /// SSD fallback: embedding loaded from NVMe only during lookup.
    ///
    /// The pool slot is allocated fresh per lookup call (not pre-pooled).
    /// Sprint 4 wires up the actual SSD-backed path; Sprint 3A falls through
    /// to the normal ring claim to maintain the calling convention.
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

// ---------------------------------------------------------------------------
// Default slot configuration constants (used by `with_defaults`)
// ---------------------------------------------------------------------------

/// Slot count for the attention pool in default / test mode.
const DEFAULT_ATTENTION_SLOT_COUNT: usize = 4;
/// Slot count for the MLP pool in default / test mode.
const DEFAULT_MLP_SLOT_COUNT: usize = 4;
/// Slot count for the norm pool in default / test mode.
const DEFAULT_NORM_SLOT_COUNT: usize = 4;
/// Slot count for the embedding pool in default / test mode.
const DEFAULT_EMBEDDING_SLOT_COUNT: usize = 2;

/// Attention / MLP slot size in default mode: 64 MiB.
const DEFAULT_LARGE_SLOT_BYTES: usize = 64 * 1024 * 1024;
/// Norm slot size in default mode: 1 MiB (LayerNorm weights are small).
const DEFAULT_NORM_SLOT_BYTES: usize = 1024 * 1024;
/// Embedding slot size in default mode: 32 MiB.
const DEFAULT_EMBEDDING_SLOT_BYTES: usize = 32 * 1024 * 1024;

/// Minimum slot size used when deriving sizes from zero_copy_threshold.
const MINIMUM_SLOT_BYTES: usize = 512;

// ---------------------------------------------------------------------------
// PoolRegistry
// ---------------------------------------------------------------------------

/// Central registry of all pool shards.
///
/// Owns four [`RingBuffer`]s (attention, mlp, norm, embedding) sized according
/// to the [`crate::phase::classifier::PhaseMemoryProfile`] supplied at
/// construction.  Routes `claim(LayerKind)` to the correct ring, falling back
/// to [`SlowPathAllocator`] on pool exhaustion.
///
/// # Thread safety
///
/// `PoolRegistry` is `Send + Sync`.  All shared state lives inside `Arc` values
/// or the `RingBuffer`s (which are `Send + Sync`).
pub struct PoolRegistry {
    /// Ring buffer for attention-block tensors (Q, K, V, O projections).
    attention_ring: Arc<RingBuffer>,

    /// Ring buffer for MLP-block tensors (up, gate, down projections).
    mlp_ring: Arc<RingBuffer>,

    /// Ring buffer for normalisation-layer tensors (LayerNorm weight and bias).
    norm_ring: Arc<RingBuffer>,

    /// Ring buffer for embedding-table tensors.
    embedding_ring: Arc<RingBuffer>,

    /// Invoked when any ring`s `try_claim` returns `None`.
    slow_path: SlowPathAllocator,

    /// Controls whether the embedding ring uses pinned RAM or the SSD path.
    embedding_fallback: EmbeddingFallbackMode,

    /// Eagerly-built slabs for fast startup access.
    slabs: Vec<TensorSlab>,

    /// Lazy slabs for low-RAM machines.
    lazy_slabs: Mutex<HashMap<u32, TensorSlab>>,

    /// Slab allocation strategy.
    slab_init_mode: SlabInitMode,

    /// Tensor byte threshold for slab packing.
    slab_threshold: usize,
}

impl PoolRegistry {
    /// Construct from a hardware profile and tensor location dict.
    ///
    /// Derives slot counts from `profile.{attention,mlp,norm,optimizer}_slots_needed`
    /// (minimum 1 per ring).  Derives slot sizes from `zero_copy_threshold`:
    /// attention and MLP slots use `zero_copy_threshold / 2`; norm uses 1 MiB;
    /// embedding uses `zero_copy_threshold / 4`.
    ///
    /// Pools are initialised at the Recomputation-phase profile (the worst case)
    /// so that the Phase Rebalancer only ever needs to shrink, never to
    /// emergency-grow during active training.
    ///
    /// # Errors
    ///
    /// Returns [`RamFlowError::AllocationFailed`] if any ring`s `PinnedBuffer`
    /// allocations fail.  Returns [`RamFlowError::ConfigError`] if
    /// `zero_copy_threshold` is zero.
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

        let large_slot_bytes = (zero_copy_threshold / 2).max(MINIMUM_SLOT_BYTES);
        let norm_slot_bytes = DEFAULT_NORM_SLOT_BYTES;
        let embedding_slot_bytes = (zero_copy_threshold / 4).max(MINIMUM_SLOT_BYTES);

        let attention_slot_count = (profile.attention_slots_needed as usize).max(1);
        let mlp_slot_count = (profile.mlp_slots_needed as usize).max(1);
        let norm_slot_count = (profile.norm_slots_needed as usize).max(1);
        let embedding_slot_count = (profile.optimizer_slots_needed as usize).max(1);

        let attention_ring = Arc::new(RingBuffer::new(large_slot_bytes, attention_slot_count)?);
        let mlp_ring = Arc::new(RingBuffer::new(large_slot_bytes, mlp_slot_count)?);
        let norm_ring = Arc::new(RingBuffer::new(norm_slot_bytes, norm_slot_count)?);
        let embedding_ring = Arc::new(RingBuffer::new(embedding_slot_bytes, embedding_slot_count)?);

        let slabs = build_eager_slabs(dict, zero_copy_threshold)?;

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
        })
    }

    /// Construct a registry with lazy slab allocation.
    ///
    /// Low-RAM systems can call [`Self::ensure_slab_for_layer`] for the layer
    /// being visited and [`Self::release_lazy_slab`] once that layer will not
    /// be revisited for a full pass.
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

    /// Convenience constructor with small built-in defaults (4 slots each, 64 MiB
    /// for attention/mlp, 1 MiB for norm, 32 MiB for embedding).
    ///
    /// Used for unit tests and environments without a hardware profile.
    ///
    /// # Errors
    ///
    /// Returns [`RamFlowError::AllocationFailed`] if any ring`s `PinnedBuffer`
    /// allocations fail.
    pub fn with_defaults() -> Result<Self> {
        let attention_ring = Arc::new(RingBuffer::new(
            DEFAULT_LARGE_SLOT_BYTES,
            DEFAULT_ATTENTION_SLOT_COUNT,
        )?);
        let mlp_ring = Arc::new(RingBuffer::new(
            DEFAULT_LARGE_SLOT_BYTES,
            DEFAULT_MLP_SLOT_COUNT,
        )?);
        let norm_ring = Arc::new(RingBuffer::new(
            DEFAULT_NORM_SLOT_BYTES,
            DEFAULT_NORM_SLOT_COUNT,
        )?);
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
        })
    }

    /// Claim one slot of the pool appropriate for `kind`.
    ///
    /// Fast path: calls `try_claim()` on the matching ring.
    /// Slow path (on `None`): signals the co-scheduler stall hook and blocks
    /// until a slot is returned (`SlowPathAllocator::handle_exhaustion`).
    ///
    /// # Errors
    ///
    /// Currently infallible for Sprint 3A (the slow path blocks indefinitely).
    /// Sprint 4 will propagate `PoolExhausted` when the slow path timeout fires
    /// and the stage-3 fresh allocation also fails.
    pub fn claim(&self, kind: LayerKind) -> Result<PoolSlot> {
        let ring = match kind {
            LayerKind::Attention => &self.attention_ring,
            LayerKind::Mlp => &self.mlp_ring,
            LayerKind::Norm => &self.norm_ring,
            LayerKind::Embedding => &self.embedding_ring,
        };

        // Sprint 4 interface seam: when embedding_fallback == SsdLookup,
        // this will return an SSD-backed handle instead of a pinned-RAM slot.
        // For Sprint 3A, fall through to the normal ring claim.
        if kind == LayerKind::Embedding
            && self.embedding_fallback == EmbeddingFallbackMode::SsdLookup
        {
            // Intentional fall-through; Sprint 4 inserts the SSD path here.
        }

        // Fast path: lock-free empty check + Mutex pop.
        if let Some(pool_slot) = ring.try_claim() {
            return Ok(pool_slot);
        }

        // Slow path: pool exhausted — block until a slot is returned.
        self.slow_path.handle_exhaustion(ring, kind)
    }

    /// Select the embedding pool residency strategy.
    ///
    /// `SsdLookup` is an interface seam for low-RAM machines where the
    /// embedding table exceeds a single pinned slot.  Sprint 4 wires this mode
    /// to an SSD-backed lookup-step loader; Sprint 3A preserves the call shape.
    pub fn set_embedding_fallback(&mut self, fallback_mode: EmbeddingFallbackMode) {
        self.embedding_fallback = fallback_mode;
    }

    /// Current embedding residency strategy.
    pub fn embedding_fallback(&self) -> EmbeddingFallbackMode {
        self.embedding_fallback
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
    /// Returns [`RamFlowError::PhaseTransitionError`] if any ring still has an
    /// in-flight slot. Returns [`RamFlowError::AllocationFailed`] if new pinned
    /// allocations fail.
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

