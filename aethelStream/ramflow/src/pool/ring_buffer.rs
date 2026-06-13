// src/pool/ring_buffer.rs — pre-allocated pinned-buffer pool ring
//
// Sprint 3A: full implementation replacing Sprint 0 stubs.
//
// ─── DESIGN ───────────────────────────────────────────────────────────────────
//
// The ring buffer maintains a free-list of pre-allocated PinnedBuffers.
// Claims and returns are mediated by two atomics:
//
//   head  — count of buffers ever claimed (monotonically increasing).
//   tail  — count of buffers ever returned (monotonically increasing).
//
// Available slot count = tail - head.  Empty when head == tail.
//
// The actual buffer data lives in `ResizableState`, guarded by a `Mutex`
// so that `resize` can swap the entire allocation atomically.
//
// ─── ORDERING RATIONALE ──────────────────────────────────────────────────────
//
//   `head` CAS: AcqRel success / Acquire failure.
//     Makes all writes by the previous owner visible to the new claimer.
//
//   `tail` fetch_add: Release.
//     Makes the returned buffer`s writes visible to the next claimer.
//
//   `active_claims` fetch_add / fetch_sub: Relaxed.
//     Best-effort gauge per AethelStream spec ss6.  The Mutex provides the
//     actual ordering guarantee when resize checks this value.

use std::mem::ManuallyDrop;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::{AcqRel, Acquire, Relaxed, Release};
use std::sync::{Arc, Condvar, Mutex};

use crate::allocator::PinnedBuffer;
use crate::{RamFlowError, Result};

// ---------------------------------------------------------------------------
// ResizableState — everything that changes on a pool resize
// ---------------------------------------------------------------------------

/// All state that must be replaced atomically on a resize operation.
///
/// Invariant: the outer `Mutex` in `RingBuffer` is held whenever this struct
/// is mutated.
struct ResizableState {
    /// Queue of free (unclaimed) buffers, each paired with its stable slot index.
    ///
    /// A slot index is assigned at allocation time (0..capacity) and returned
    /// in `PoolSlot::drop` so the ring can re-insert the buffer at the correct
    /// position.
    free_queue: std::collections::VecDeque<(usize, ManuallyDrop<PinnedBuffer>)>,

    /// Total number of pre-allocated slots (free + in-flight).
    capacity: usize,

    /// Size in bytes of each slot buffer.
    slot_bytes: usize,
}

impl ResizableState {
    /// Allocate `capacity` `PinnedBuffer`s of `slot_bytes` bytes each.
    ///
    /// Allocations are eager.  On partial failure every already-allocated
    /// buffer is freed before the error is returned.
    ///
    /// # Errors
    /// Returns [`RamFlowError::AllocationFailed`] if `capacity` or `slot_bytes`
    /// is zero, or if any `PinnedBuffer::alloc` fails.
    fn allocate(slot_bytes: usize, capacity: usize, numa_node: Option<u32>) -> Result<Self> {
        if capacity == 0 {
            return Err(RamFlowError::AllocationFailed(
                "RingBuffer capacity must be at least 1".into(),
            ));
        }
        if slot_bytes == 0 {
            return Err(RamFlowError::AllocationFailed(
                "RingBuffer slot_bytes must be at least 1".into(),
            ));
        }

        let mut free_queue = std::collections::VecDeque::with_capacity(capacity);
        for slot_index in 0..capacity {
            // Route large slots through hugepage-backed allocation when the
            // `hugepages` feature is active.  Smaller slots use standard
            // posix_memalign; there is no TLB benefit below 2 MiB.
            #[cfg(feature = "hugepages")]
            let alloc_fn = |sz: usize| {
                if sz >= crate::allocator::huge::HUGEPAGE_THRESHOLD {
                    PinnedBuffer::alloc_pinned_huge(sz)
                } else {
                    PinnedBuffer::alloc(sz)
                }
            };
            #[cfg(not(feature = "hugepages"))]
            let alloc_fn = PinnedBuffer::alloc;

            let pinned_buffer = alloc_fn(slot_bytes).inspect_err(|_| {
                // On partial failure, drop all already-inserted buffers.
                // ManuallyDrop does NOT run the inner destructor, so we must.
                //
                // SAFETY: each buffer in `free_queue` was produced by
                // `PinnedBuffer::alloc` and is exclusively owned by this
                // ResizableState — no PoolSlot references any of them yet.
                for (_, mut manually_dropped_buffer) in free_queue.drain(..) {
                    unsafe { ManuallyDrop::drop(&mut manually_dropped_buffer) };
                }
            })?;
            #[cfg(all(feature = "numa", target_os = "linux"))]
            if let Some(node) = numa_node {
                // Bind this slot to the GPU NUMA node.  Failure is non-fatal.
                let _ = crate::allocator::numa::mbind_buffer(pinned_buffer.as_ptr() as *mut u8, slot_bytes, node);
            }
            free_queue.push_back((slot_index, ManuallyDrop::new(pinned_buffer)));
        }

        // On non-Linux or without the numa feature the parameter is unused.
        #[cfg(not(all(feature = "numa", target_os = "linux")))]
        let _ = numa_node;

        Ok(ResizableState {
            free_queue,
            capacity,
            slot_bytes,
        })
    }

    /// Allocate `capacity` mmap-backed buffers of `slot_bytes` bytes each.
    ///
    /// Used by `RingBuffer::new_mmap` for graceful-degradation pool construction.
    ///
    /// # Errors
    /// Returns `AllocationFailed` if capacity or slot_bytes is zero, or if any
    /// `PinnedBuffer::alloc_mmap` fails.
    #[cfg(feature = "mmap-fallback")]
    fn allocate_mmap(slot_bytes: usize, capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(RamFlowError::AllocationFailed(
                "RingBuffer mmap capacity must be at least 1".into(),
            ));
        }
        if slot_bytes == 0 {
            return Err(RamFlowError::AllocationFailed(
                "RingBuffer mmap slot_bytes must be at least 1".into(),
            ));
        }

        let mut free_queue = std::collections::VecDeque::with_capacity(capacity);
        for slot_index in 0..capacity {
            let mmap_buf = PinnedBuffer::alloc_mmap(slot_bytes).inspect_err(|_| {
                for (_, mut manually_dropped_buffer) in free_queue.drain(..) {
                    // SAFETY: each buffer in free_queue was produced by PinnedBuffer::alloc_mmap
                    // and is exclusively owned here — no PoolSlot references any of them.
                    unsafe { ManuallyDrop::drop(&mut manually_dropped_buffer) };
                }
            })?;
            free_queue.push_back((slot_index, ManuallyDrop::new(mmap_buf)));
        }

        Ok(ResizableState { free_queue, capacity, slot_bytes })
    }
}

impl Drop for ResizableState {
    fn drop(&mut self) {
        // Explicitly drop every PinnedBuffer still in the free queue.
        // ManuallyDrop does not run the inner destructor automatically.
        //
        // SAFETY: each buffer in `free_queue` was produced by `PinnedBuffer::alloc`
        // and is exclusively owned by this ResizableState (no PoolSlot holds it).
        for (_, mut manually_dropped_buffer) in self.free_queue.drain(..) {
            unsafe { ManuallyDrop::drop(&mut manually_dropped_buffer) };
        }
    }
}

// ---------------------------------------------------------------------------
// RingBuffer — the public pool type
// ---------------------------------------------------------------------------

/// Pre-allocated pool of pinned host-memory buffers, dispensed as RAII
/// [`crate::pool::PoolSlot`] claim tokens.
///
/// # Concurrency model
///
/// - `head` and `tail` atomics allow a fast lock-free empty check in
///   [`RingBuffer::try_claim`].
/// - Actual buffer movement is guarded by the `inner` `Mutex` (which also
///   serves as the wait object for [`RingBuffer::claim_blocking`]).
/// - The `Condvar` wakes threads sleeping in `claim_blocking` when a slot is
///   returned.
///
/// # Invariants
///
/// - `tail - head == free_queue.len()` at any quiescent point.
/// - Every slot index in a [`crate::pool::PoolSlot`] refers to a buffer
///   originally allocated by this ring.  Slot indices are stable across claims.
pub struct RingBuffer {
    /// Monotonically increasing claim counter.
    ///
    /// Ordering: AcqRel on CAS success, Acquire on CAS failure and plain loads.
    head: AtomicUsize,

    /// Monotonically increasing return counter.
    ///
    /// Ordering: Release on fetch_add, Acquire on loads.
    tail: AtomicUsize,

    /// Number of slots currently claimed and in-flight (best-effort gauge).
    ///
    /// Uses Relaxed ordering per AethelStream spec ss6.  The `Mutex` provides
    /// the actual ordering guarantee when `resize` reads this value.
    active_claims: AtomicUsize,

    /// Free buffer queue, capacity, and slot_bytes — replaced atomically on resize.
    inner: Mutex<ResizableState>,

    /// Condvar woken by `release`; waited on by `claim_blocking`.
    condvar: Condvar,

    /// NUMA node for pool-slot page binding (-1 = no binding, >= 0 = node index).
    /// Set via apply_numa_binding(); read by resize() when repopulating slots.
    numa_node: std::sync::atomic::AtomicI32,
}

// SAFETY: `RingBuffer` is `Send` because all state is either atomic (inherently
// `Send`) or guarded by a `Mutex`.  `ResizableState` is `Send` because
// `ManuallyDrop<PinnedBuffer>` is `Send` (`PinnedBuffer: Send`; `ManuallyDrop`
// is `Send` when the inner type is `Send`).
unsafe impl Send for RingBuffer {}

// SAFETY: `RingBuffer` is `Sync` because all mutable access goes through the
// `Mutex`.  The atomics and `Condvar` are unconditionally `Sync`.
unsafe impl Sync for RingBuffer {}

impl RingBuffer {
    /// Construct a ring buffer with `capacity` slots of `slot_bytes` bytes each.
    ///
    /// All `PinnedBuffer` allocations happen eagerly; subsequent `try_claim` and
    /// `claim_blocking` calls are allocation-free.
    ///
    /// # Errors
    ///
    /// Returns [`RamFlowError::AllocationFailed`] if `capacity` or `slot_bytes`
    /// is zero, or if any individual `PinnedBuffer::alloc` fails.
    pub fn new(slot_bytes: usize, capacity: usize) -> Result<Self> {
        let resizable_state = ResizableState::allocate(slot_bytes, capacity, None)?;

        // tail starts at `capacity` so tail - head = capacity (all slots free).
        Ok(RingBuffer {
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(capacity),
            active_claims: AtomicUsize::new(0),
            inner: Mutex::new(resizable_state),
            condvar: Condvar::new(),
            numa_node: std::sync::atomic::AtomicI32::new(-1),
        })
    }

    /// Construct a ring buffer with mmap-backed slots (graceful-degradation mode).
    ///
    /// Uses `PinnedBuffer::alloc_mmap` instead of `PinnedBuffer::alloc`.
    /// Slots returned by `try_claim`/`claim_blocking` have `is_pinned() == false`.
    ///
    /// # Errors
    /// Returns `AllocationFailed` if capacity or slot_bytes is zero, or if any
    /// `PinnedBuffer::alloc_mmap` fails.
    #[cfg(feature = "mmap-fallback")]
    pub fn new_mmap(slot_bytes: usize, capacity: usize) -> Result<Self> {
        let resizable_state = ResizableState::allocate_mmap(slot_bytes, capacity)?;
        Ok(RingBuffer {
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(capacity),
            active_claims: AtomicUsize::new(0),
            inner: Mutex::new(resizable_state),
            condvar: Condvar::new(),
            numa_node: std::sync::atomic::AtomicI32::new(-1),
        })
    }

    /// Attempt a non-blocking claim of one free slot.
    ///
    /// Returns `None` immediately if all slots are currently in-flight.
    ///
    /// Uses a `compare_exchange` (AcqRel/Acquire) on `head` as required by the
    /// Sprint 3A specification.
    ///
    /// # Lock behaviour
    ///
    /// Holds the inner `Mutex` only while popping from the free queue.
    pub fn try_claim(self: &Arc<Self>) -> Option<crate::pool::PoolSlot> {
        // Lock-free empty check: skip the Mutex when definitely empty.
        let current_tail = self.tail.load(Acquire);
        let current_head = self.head.load(Acquire);
        if current_head == current_tail {
            return None;
        }

        let mut state_guard = self
            .inner
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        // Re-check under the lock: another thread may have taken the last slot
        // between the atomic check above and acquiring the lock.
        let (slot_index, buffer) = state_guard.free_queue.pop_front()?;

        // Advance `head` with compare_exchange (AcqRel success, Acquire failure).
        // If the CAS fails (rare concurrent race with claim_blocking), fall back
        // to fetch_add.  Correctness is guaranteed by the Mutex — only one thread
        // can pop from the queue at a time, so head must advance by exactly 1.
        let expected_head = self.head.load(Relaxed);
        let advance_result =
            self.head
                .compare_exchange(expected_head, expected_head + 1, AcqRel, Acquire);
        if advance_result.is_err() {
            self.head.fetch_add(1, AcqRel);
        }
        self.active_claims.fetch_add(1, Relaxed);
        drop(state_guard);

        Some(crate::pool::PoolSlot::pooled(
            Arc::clone(self),
            slot_index,
            buffer,
        ))
    }

    /// Blocking claim — suspends the calling thread until a free slot is available.
    ///
    /// Guaranteed to return a valid `PoolSlot`.  There is no timeout (Sprint 4
    /// adds timeout support).  Use `try_claim` in a timed loop for bounded waits.
    ///
    /// # Invariants upheld by caller
    ///
    /// The caller must not hold any other `Mutex` from this crate when calling
    /// `claim_blocking` to avoid deadlock with the internal Condvar.
    pub fn claim_blocking(self: &Arc<Self>) -> crate::pool::PoolSlot {
        let mut state_guard = self
            .inner
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        loop {
            if let Some((slot_index, buffer)) = state_guard.free_queue.pop_front() {
                let expected_head = self.head.load(Relaxed);
                let advance_result =
                    self.head
                        .compare_exchange(expected_head, expected_head + 1, AcqRel, Acquire);
                if advance_result.is_err() {
                    self.head.fetch_add(1, AcqRel);
                }
                self.active_claims.fetch_add(1, Relaxed);
                drop(state_guard);

                return crate::pool::PoolSlot::pooled(Arc::clone(self), slot_index, buffer);
            }

            // `wait` atomically releases the Mutex and parks the thread.
            // It reacquires the Mutex before returning.
            state_guard = self
                .condvar
                .wait(state_guard)
                .unwrap_or_else(|poison| poison.into_inner());
        }
    }

    /// Return slot `slot_index` with its `buffer` to the free queue.
    ///
    /// Called exclusively from [`crate::pool::PoolSlot::drop`]; the Drop impl
    /// guarantees this runs even on panic, preventing pool starvation.
    ///
    /// Advances `tail` (fetch_add, Release), decrements `active_claims` (Relaxed),
    /// then notifies one thread sleeping in `claim_blocking`.
    pub(crate) fn release(&self, slot_index: usize, buffer: ManuallyDrop<PinnedBuffer>) {
        let mut state_guard = self
            .inner
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        state_guard.free_queue.push_back((slot_index, buffer));
        // Release ordering: makes the returning thread's writes to the buffer
        // visible to the next claimer before they observe the incremented tail.
        self.tail.fetch_add(1, Release);
        self.active_claims.fetch_sub(1, Relaxed);
        drop(state_guard);

        // Wake exactly one thread waiting in claim_blocking.
        self.condvar.notify_one();
    }

    /// Number of slots currently claimed and in-flight.
    ///
    /// Best-effort gauge (Relaxed ordering); may be transiently stale under
    /// concurrent claim/release activity.
    pub fn claimed_slots(&self) -> usize {
        self.active_claims.load(Relaxed)
    }

    /// Total slot capacity (free + in-flight).
    pub fn total_slots(&self) -> usize {
        let state_guard = self
            .inner
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        state_guard.capacity
    }

    /// Size in bytes of each individual slot buffer.
    pub fn slot_bytes(&self) -> usize {
        let state_guard = self
            .inner
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        state_guard.slot_bytes
    }

    /// Number of currently-free (unclaimed) slots.
    ///
    /// Computed as `tail - head`.  May be transiently inaccurate under
    /// concurrent claim/release activity.
    pub fn available_slots(&self) -> usize {
        let current_tail = self.tail.load(Acquire);
        let current_head = self.head.load(Acquire);
        current_tail.saturating_sub(current_head)
    }

    /// Total bytes currently allocated across all slots (free + in-flight).
    pub fn bytes_allocated(&self) -> usize {
        let state_guard = self
            .inner
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        state_guard.capacity * state_guard.slot_bytes
    }

    /// Resize the pool to `new_capacity` slots.
    ///
    /// # Preconditions
    ///

    /// Bind all current and future pool slots to a NUMA node.
    ///
    /// Calls mbind(MPOL_BIND) on every free buffer in the ring and stores the
    /// node so resize() rebinds new slots too.  In-flight slots are not touched
    /// (the ring holds no references to them); in practice PoolRegistry calls
    /// this at startup before any claims are made.
    ///
    /// A mbind failure is not an error -- NUMA binding is a performance hint.
    /// No-op on non-Linux or without the `numa` feature.
    #[cfg(all(feature = "numa", target_os = "linux"))]
    pub fn apply_numa_binding(&self, node: u32) {
        use std::sync::atomic::Ordering::Relaxed;
        self.numa_node.store(node as i32, Relaxed);
        let state_guard = self
            .inner
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let slot_bytes = state_guard.slot_bytes;
        for (_, buf) in &state_guard.free_queue {
            // SAFETY: buffer is in the free queue, no PoolSlot holds it,
            // so we have exclusive access under the Mutex lock.
            if !crate::allocator::numa::mbind_buffer(
                unsafe { buf.as_ptr() as *mut u8 },
                slot_bytes,
                node,
            ) {
                break; // mbind failed (EPERM); stop -- subsequent would also fail
            }
        }
    }

    /// The phase fence MUST be held and `claimed_slots() == 0` before calling
    /// this method.  In-flight `PoolSlot`s hold slot indices that belong to the
    /// discarded `ResizableState`; their `drop` would corrupt the new pool.
    ///
    /// # Errors
    ///
    /// Returns [`RamFlowError::PhaseTransitionError`] if any slot is in-flight.
    /// Returns [`RamFlowError::AllocationFailed`] if new allocations fail.
    pub fn resize(&self, new_capacity: usize) -> Result<()> {
        // Acquire the lock first so no concurrent claim can proceed during resize.
        let mut state_guard = self
            .inner
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        // Defensive check — Mutex provides ordering so Relaxed suffices here.
        let currently_active = self.active_claims.load(Relaxed);
        if currently_active != 0 {
            return Err(RamFlowError::PhaseTransitionError(format!(
                "resize called with {currently_active} slot(s) still in-flight; \
                 the phase fence must be held before calling resize"
            )));
        }

        let current_slot_bytes = state_guard.slot_bytes;
        let numa = {
            let n = self.numa_node.load(std::sync::atomic::Ordering::Relaxed);
            if n >= 0 { Some(n as u32) } else { None }
        };
        let new_state = ResizableState::allocate(current_slot_bytes, new_capacity, numa)?;

        // Reset atomics under the Mutex so no concurrent observer sees an
        // intermediate state.
        self.head.store(0, Release);
        self.tail.store(new_capacity, Release);

        // Replace ResizableState — the old one is dropped here, freeing all
        // PinnedBuffers still in the free queue via ResizableState::drop.
        *state_guard = new_state;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    const SMALL_SLOT_BYTES: usize = 512;
    const SMALL_CAPACITY: usize = 4;

    #[test]
    fn new_ring_has_all_slots_available() {
        let ring =
            Arc::new(RingBuffer::new(SMALL_SLOT_BYTES, SMALL_CAPACITY).expect("ring alloc failed"));
        assert_eq!(ring.available_slots(), SMALL_CAPACITY);
        assert_eq!(ring.claimed_slots(), 0);
        assert_eq!(ring.total_slots(), SMALL_CAPACITY);
    }

    #[test]
    fn try_claim_reduces_available_count() {
        let ring =
            Arc::new(RingBuffer::new(SMALL_SLOT_BYTES, SMALL_CAPACITY).expect("ring alloc failed"));
        let slot = ring.try_claim().expect("first claim must succeed");
        assert_eq!(ring.available_slots(), SMALL_CAPACITY - 1);
        assert_eq!(ring.claimed_slots(), 1);
        drop(slot);
        assert_eq!(ring.available_slots(), SMALL_CAPACITY);
        assert_eq!(ring.claimed_slots(), 0);
    }

    #[test]
    fn try_claim_returns_none_when_all_slots_claimed() {
        let ring =
            Arc::new(RingBuffer::new(SMALL_SLOT_BYTES, SMALL_CAPACITY).expect("ring alloc failed"));
        let mut claims = Vec::new();
        for _ in 0..SMALL_CAPACITY {
            claims.push(ring.try_claim().expect("should claim while available"));
        }
        assert!(ring.try_claim().is_none(), "ring must be empty now");
        drop(claims);
        assert_eq!(ring.available_slots(), SMALL_CAPACITY);
    }

    #[test]
    fn drop_slot_returns_buffer_to_ring() {
        let ring =
            Arc::new(RingBuffer::new(SMALL_SLOT_BYTES, SMALL_CAPACITY).expect("ring alloc failed"));
        let slot = ring.try_claim().expect("first claim");
        drop(slot);
        assert_eq!(ring.available_slots(), SMALL_CAPACITY);
        assert_eq!(ring.claimed_slots(), 0);
    }

    #[test]
    fn resize_rejected_when_slots_in_flight() {
        let ring =
            Arc::new(RingBuffer::new(SMALL_SLOT_BYTES, SMALL_CAPACITY).expect("ring alloc failed"));
        let _slot = ring.try_claim().expect("must claim one slot");
        let resize_result = ring.resize(8);
        assert!(
            matches!(resize_result, Err(RamFlowError::PhaseTransitionError(_))),
            "resize must be rejected while slots are in-flight"
        );
    }

    #[test]
    fn resize_succeeds_when_no_slots_in_flight() {
        let ring =
            Arc::new(RingBuffer::new(SMALL_SLOT_BYTES, SMALL_CAPACITY).expect("ring alloc failed"));
        ring.resize(8)
            .expect("resize must succeed when no slots are claimed");
        assert_eq!(ring.total_slots(), 8);
        assert_eq!(ring.available_slots(), 8);
        assert_eq!(ring.claimed_slots(), 0);
    }

    #[test]
    fn slot_buffer_length_matches_slot_bytes() {
        let ring =
            Arc::new(RingBuffer::new(SMALL_SLOT_BYTES, SMALL_CAPACITY).expect("ring alloc failed"));
        let slot = ring.try_claim().expect("first claim");
        assert_eq!(slot.buffer_len(), SMALL_SLOT_BYTES);
    }

    #[test]
    fn bytes_allocated_is_capacity_times_slot_bytes() {
        let ring =
            Arc::new(RingBuffer::new(SMALL_SLOT_BYTES, SMALL_CAPACITY).expect("ring alloc failed"));
        assert_eq!(ring.bytes_allocated(), SMALL_SLOT_BYTES * SMALL_CAPACITY);
    }

    #[test]
    fn cap_plus_one_concurrent_claims_all_succeed_with_slow_path() {
        let capacity = 2;
        let ring = Arc::new(RingBuffer::new(SMALL_SLOT_BYTES, capacity).expect("ring alloc"));
        let barrier = Arc::new(std::sync::Barrier::new(capacity + 1));
        let mut handles = Vec::new();

        for _thread_index in 0..=capacity {
            let thread_ring = Arc::clone(&ring);
            let thread_barrier = Arc::clone(&barrier);
            handles.push(std::thread::spawn(move || {
                thread_barrier.wait();
                match thread_ring.try_claim() {
                    Some(pool_slot) => pool_slot.buffer_len(),
                    None => crate::pool::slow_path::SlowPathAllocator::new()
                        .handle_exhaustion(&thread_ring, crate::pool::LayerKind::Attention)
                        .expect("slow path claim failed")
                        .buffer_len(),
                }
            }));
        }

        for handle in handles {
            assert_eq!(
                handle.join().expect("claim thread panicked"),
                SMALL_SLOT_BYTES
            );
        }
    }
}
