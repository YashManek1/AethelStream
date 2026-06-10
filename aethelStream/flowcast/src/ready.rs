//! `ReadyLayer` -- the type handed to Module 5 when a prefetch completes.
//!
//! M5 receives a `ReadyLayer` from `FlowCast::wait_for_layer`.  It must call
//! `FlowCast::retire_layer` when the GPU kernel is done.  Dropping a
//! `ReadyLayer` automatically returns the pinned buffer to RamFlow pool.

use crate::config::Precision;
use ramflow::pool::PoolSlot;

/// A layer whose weights are resident in pinned RAM and ready for the GPU.
///
/// # Invariants (enforced by FlowCast, not by M5)
/// * The underlying `PinnedBuffer` remains valid until this value is dropped.
/// * `precision` matches the mode under which the shard was decoded.
/// * Dropping returns the pinned buffer to the RamFlow pool ring.
pub struct ReadyLayer {
    /// Layer index (matches `shard_index.json`).
    pub layer_idx: u32,

    /// Precision the weights were decoded into.
    pub precision: Precision,

    /// Pinned-RAM slot holding the layer weights.
    ///
    /// Returned to the pool ring when `ReadyLayer` is dropped.
    pub(crate) slot: PoolSlot,
}

impl ReadyLayer {
    /// Read-only byte slice over the layer weights in pinned RAM.
    ///
    /// The slice is valid as long as this `ReadyLayer` is alive.
    pub fn as_slice(&self) -> &[u8] {
        self.slot.buffer().as_slice()
    }

    /// Length of the pinned buffer in bytes.
    pub fn len(&self) -> usize {
        self.slot.buffer().len()
    }

    /// Returns `true` if the buffer is zero-sized (always `false` for valid layers).
    pub fn is_empty(&self) -> bool {
        self.slot.buffer().is_empty()
    }
}
