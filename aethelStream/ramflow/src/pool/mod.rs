// src/pool/mod.rs — pool module root
//
// Sprint 0: all types and submodule declarations. No logic runs yet.
// The key exported types are:
//   - PoolRegistry  — central registry routing claim() calls to the right ring
//   - PoolSlot      — RAII guard that returns its ring slot on drop
//   - LayerKind     — discriminant for which pool a tensor goes into
//   - TensorSlab    — per-layer packed small-tensor allocation (Algorithm 5)
//   - TensorLocationDict — loaded from shard_index.json (Module 1 output)

pub mod ring_buffer;
pub mod slab;
pub mod slow_path;
pub mod subpools;
pub mod tensor_location;

pub use tensor_location::{TensorInfo, TensorLocation, TensorLocationDict};
pub use slab::TensorSlab;
pub use ring_buffer::RingBuffer;

// ---------------------------------------------------------------------------
// LayerKind — discriminant for pool routing
// ---------------------------------------------------------------------------

/// Determines which pool ring receives a claim or return.
///
/// Matches the four pool categories from the spec:
/// attention, mlp, norm, embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerKind {
    Attention,
    Mlp,
    Norm,
    Embedding,
}

// ---------------------------------------------------------------------------
// PoolSlot — RAII claim on one ring-buffer slot
// ---------------------------------------------------------------------------

/// An active claim on a single pool slot.
///
/// The slot is returned to the ring automatically when this value is dropped,
/// even on panic.  This is the mechanism that prevents pool starvation from
/// leaked claims.
///
/// # Sprint 0 contract
/// Compiles; the backing `buf` pointer is null; calling any method panics.
pub struct PoolSlot {
    _opaque: (),
}

impl Drop for PoolSlot {
    fn drop(&mut self) {
        // Sprint 0: nothing to return — real impl advances ring tail and
        // decrements the active-claims counter (phase fence).
    }
}

// ---------------------------------------------------------------------------
// PoolRegistry — central allocator, owns all ring buffers and slabs
// ---------------------------------------------------------------------------

/// Central registry of all pool shards.
///
/// Routes `claim(LayerKind)` to the correct ring buffer, maintains the
/// active-claims counter used by the phase fence, and owns all [`TensorSlab`]s.
///
/// # Sprint 0 contract
/// Compiles; all methods `unimplemented!`.
pub struct PoolRegistry {
    _opaque: (),
}

impl PoolRegistry {
    /// Construct a registry from a hardware profile and tensor location dict.
    ///
    /// Pools are initialised at the *Recomputation* phase profile (the worst
    /// case) so that the rebalancer only ever needs to shrink, never to
    /// emergency-grow during active training.
    #[allow(unused_variables)]
    pub fn new(
        profile: &crate::phase::classifier::PhaseMemoryProfile,
        dict: &TensorLocationDict,
        zero_copy_threshold: usize,
    ) -> crate::Result<Self> {
        unimplemented!("PoolRegistry::new — Sprint 0 stub")
    }

    /// Convenience constructor with default (Recomputation-sized) profiles.
    pub fn with_defaults() -> crate::Result<Self> {
        unimplemented!("PoolRegistry::with_defaults — Sprint 0 stub")
    }

    /// Claim one slot of the pool appropriate for `kind`.
    ///
    /// On pool exhaustion the slow-path handler fires:
    ///   1. Signals the co-scheduler to pause prefetch immediately.
    ///   2. Attempts to reclaim the oldest reclaimable buffer.
    ///   3. Falls back to a fresh `allocate_pinned` call with a `warn!` log.
    #[allow(unused_variables)]
    pub fn claim(&self, kind: LayerKind) -> crate::Result<PoolSlot> {
        unimplemented!("PoolRegistry::claim — Sprint 0 stub")
    }

    /// Number of currently-claimed slots across all pools.
    pub fn total_claimed_slots(&self) -> usize {
        unimplemented!("PoolRegistry::total_claimed_slots — Sprint 0 stub")
    }

    /// Total slot capacity across all pools.
    pub fn total_capacity(&self) -> usize {
        unimplemented!("PoolRegistry::total_capacity — Sprint 0 stub")
    }

    /// Return a reference to the pre-built slab for `layer_idx`.
    ///
    /// All 80 layer slabs are built at startup; this is a zero-allocation
    /// index operation during training.
    #[allow(unused_variables)]
    pub fn slab_for_layer(&self, layer_idx: u32) -> Option<&TensorSlab> {
        unimplemented!("PoolRegistry::slab_for_layer — Sprint 0 stub")
    }

    /// Total bytes currently allocated (across all rings + all slabs).
    pub fn bytes_allocated(&self) -> usize {
        unimplemented!("PoolRegistry::bytes_allocated — Sprint 0 stub")
    }
}
