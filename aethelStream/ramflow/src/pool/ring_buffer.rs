// src/pool/ring_buffer.rs — lock-free MPSC ring buffer for pool slots
//
// Sprint 0: type declared with correct field shape; methods unimplemented!.
//
// The ring uses two AtomicUsize values:
//   head — index of next slot to CLAIM (advanced by claimer)
//   tail — index of next slot to RETURN (advanced by releaser)
//
// Empty condition:  head == tail
// Full condition:   (head + 1) % capacity == tail
//
// Ordering on claim: AcqRel — the claimer sees all writes the last releaser
//                    made to the buffer contents.
// Ordering on return: Release — the releaser's writes are visible before the
//                     slot is marked available.

use std::sync::atomic::AtomicUsize;
use std::sync::{Condvar, Mutex};

/// Lock-free MPSC ring buffer holding pinned-memory pool slots.
///
/// # Sprint 0 contract
/// Compiles; all methods `unimplemented!`.
pub struct RingBuffer {
    /// The pool slots themselves (Box<[PinnedBuffer]> in the real impl).
    _slots: (),
    /// Next slot index to claim.
    _head: AtomicUsize,
    /// Next slot index for the returner to write.
    _tail: AtomicUsize,
    /// Total number of slots (fixed after construction).
    _capacity: usize,
    /// Blocking claim uses this to sleep until a slot is returned.
    _condvar: (Mutex<()>, Condvar),
}

impl RingBuffer {
    /// Construct a ring buffer with `capacity` slots, each `slot_bytes` bytes.
    #[allow(unused_variables)]
    pub fn new(slot_bytes: usize, capacity: usize) -> crate::Result<Self> {
        unimplemented!("RingBuffer::new — Sprint 0 stub")
    }

    /// Attempt a non-blocking claim.  Returns `None` if the ring is full.
    pub fn try_claim(&self) -> Option<crate::pool::PoolSlot> {
        unimplemented!("RingBuffer::try_claim — Sprint 0 stub")
    }

    /// Blocking claim — waits on the condvar until a slot is returned.
    pub fn claim_blocking(&self) -> crate::pool::PoolSlot {
        unimplemented!("RingBuffer::claim_blocking — Sprint 0 stub")
    }

    /// Return a previously claimed slot, waking any blocked claimers.
    #[allow(unused_variables)]
    pub(crate) fn release(&self, slot_index: usize) {
        unimplemented!("RingBuffer::release — Sprint 0 stub")
    }

    /// Resize to `new_capacity` slots.
    ///
    /// **SAFETY** — must only be called when no slots are currently claimed
    /// (phase fence must be held).  Failing to uphold this invariant is a
    /// data race with in-flight DMA transfers.
    #[allow(unused_variables)]
    pub fn resize(&self, new_capacity: usize) -> crate::Result<()> {
        unimplemented!("RingBuffer::resize — Sprint 0 stub")
    }

    /// Current number of claimed (in-flight) slots.
    pub fn claimed_slots(&self) -> usize {
        unimplemented!("RingBuffer::claimed_slots — Sprint 0 stub")
    }

    /// Total slot capacity.
    pub fn total_slots(&self) -> usize {
        unimplemented!("RingBuffer::total_slots — Sprint 0 stub")
    }
}
