// src/pool/mod.rs — pool module root
//
// Sprint 3A: PoolSlot upgraded to real RAII; PoolRegistry re-exported from subpools.

/// LZ4-compressed eviction tier (feature = "lz4-cache").
#[cfg(feature = "lz4-cache")]
pub mod eviction_cache;
/// Lock-free ring buffer with atomic claim/release and phase-fence resize.
pub mod ring_buffer;
/// Slab packer: merges co-traveling small tensors into one pinned allocation.
pub mod slab;
/// Slow-path allocator: blocks the training loop on pool exhaustion.
pub mod slow_path;
/// `PoolRegistry`: central registry owning all pool rings.
pub mod subpools;
/// `TensorLocationDict`: maps `(layer, name)` → on-disk shard location.
pub mod tensor_location;

#[cfg(feature = "lz4-cache")]
pub use eviction_cache::{CachePrecision, EvictionCache, Lz4CacheTelemetry};
pub use ring_buffer::RingBuffer;
pub use slab::TensorSlab;
pub use subpools::{EmbeddingFallbackMode, PoolRegistry, SlabInitMode};
pub use tensor_location::{TensorInfo, TensorLocation, TensorLocationDict};

use std::mem::ManuallyDrop;
use std::sync::Arc;

use crate::allocator::PinnedBuffer;

// ---------------------------------------------------------------------------
// LayerKind — discriminant for pool routing (unchanged from Sprint 0)
// ---------------------------------------------------------------------------

/// Determines which pool ring receives a claim or return.
///
/// Matches the four pool categories from the AethelStream spec:
/// attention, mlp, norm, embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerKind {
    /// Attention block: Q, K, V, and output projections.
    Attention,
    /// MLP block: up-projection, gate, and down-projection weights.
    Mlp,
    /// Normalisation layers: LayerNorm weight and bias pairs.
    Norm,
    /// Embedding table tensors.
    Embedding,
}

// ---------------------------------------------------------------------------
// PoolSlot — RAII claim guard (real implementation, Sprint 3A)
// ---------------------------------------------------------------------------

/// An active claim on a single pool ring slot.
///
/// Holds the claimed [`PinnedBuffer`] and a reference to the owning
/// [`RingBuffer`].  When dropped, the buffer is returned to the ring —
/// even on panic — preventing pool starvation.
///
/// # Invariants
///
/// - `buffer` is valid for the lifetime of this `PoolSlot`.
/// - Only one `PoolSlot` for a given `slot_index` exists at any time
///   (enforced by the ring's atomic accounting).
/// - `drop` is the only code that calls `RingBuffer::release`; no other
///   code moves the buffer out of `buffer`.
pub struct PoolSlot {
    /// The ring buffer this slot belongs to.
    ///
    /// Kept as `Arc` so the ring stays alive at least as long as the slot,
    /// even if the `PoolRegistry` that created it is dropped first.
    pub(crate) ring: Option<Arc<RingBuffer>>,

    /// Stable slot index within the ring — used to re-insert the buffer on drop.
    pub(crate) slot_index: usize,

    /// The pinned buffer for this slot.
    ///
    /// Wrapped in `ManuallyDrop` so we can move it out in `drop` without a
    /// double-free.  `drop` is the only place that calls `ManuallyDrop::take`.
    pub(crate) buffer: ManuallyDrop<PinnedBuffer>,
}

impl PoolSlot {
    /// Construct a slot owned by a pool ring.
    pub(crate) fn pooled(
        ring: Arc<RingBuffer>,
        slot_index: usize,
        buffer: ManuallyDrop<PinnedBuffer>,
    ) -> Self {
        PoolSlot {
            ring: Some(ring),
            slot_index,
            buffer,
        }
    }

    /// Construct a slow-path overflow slot that is freed directly on drop.
    pub(crate) fn overflow(buffer: PinnedBuffer) -> Self {
        PoolSlot {
            ring: None,
            slot_index: usize::MAX,
            buffer: ManuallyDrop::new(buffer),
        }
    }

    /// Shared reference to the underlying pinned buffer.
    ///
    /// Valid for the lifetime of this `PoolSlot`.
    pub fn buffer(&self) -> &PinnedBuffer {
        // SAFETY: `buffer` is initialised in `try_claim` / `claim_blocking` and
        // remains valid until `drop` runs.  `ManuallyDrop` prevents automatic
        // drop; we are the only owner.
        unsafe { &*std::ptr::addr_of!(*self.buffer) }
    }

    /// Mutable reference to the underlying pinned buffer.
    ///
    /// Valid for the lifetime of this `PoolSlot`.  `&mut self` guarantees
    /// exclusive access.
    pub fn buffer_mut(&mut self) -> &mut PinnedBuffer {
        // SAFETY: same invariants as `buffer`; `&mut self` guarantees no
        // concurrent access.
        unsafe { &mut *std::ptr::addr_of_mut!(*self.buffer) }
    }

    /// Raw mutable pointer to the start of the pinned buffer (for CUDA FFI and
    /// io_uring write paths).
    ///
    /// # Safety
    ///
    /// Valid as long as this `PoolSlot` is alive.  The caller must not retain
    /// the pointer beyond the `PoolSlot`'s lifetime and must not alias it with
    /// any other reference to the same buffer.
    pub unsafe fn buffer_ptr(&self) -> *mut u8 {
        // SAFETY: buffer is valid for the lifetime of self; ManuallyDrop
        // prevents automatic drop.  We expose a raw pointer so CUDA FFI and
        // io_uring can write directly into the pinned allocation without an
        // intermediate slice.
        self.buffer.as_ptr() as *mut u8
    }

    /// Byte length of the underlying pinned buffer.
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    /// The stable slot index within the ring.
    pub fn slot_index(&self) -> usize {
        self.slot_index
    }
}

impl Drop for PoolSlot {
    fn drop(&mut self) {
        // SAFETY: `buffer` was initialised in `try_claim` / `claim_blocking`
        // exactly once.  `ManuallyDrop::take` moves the value out, transferring
        // ownership to `release`.  This is the only code that calls `take`, so
        // there is no double-take.  After `take`, the `ManuallyDrop` wrapper
        // contains uninitialised memory, but this struct is immediately dropped
        // (we are inside `drop`), so no subsequent code can access it.
        let buffer = unsafe { ManuallyDrop::take(&mut self.buffer) };
        if let Some(ring) = &self.ring {
            // Return the buffer to the ring — this is guaranteed to run even on panic.
            ring.release(self.slot_index, ManuallyDrop::new(buffer));
        } else {
            drop(buffer);
        }
    }
}
