//! VRAM double-buffer: overlaps RAM→VRAM copy with GPU compute using two
//! alternating VRAM slots and a dedicated CUDA copy stream.
//!
//! # Seam design (recorded per CLAUDE.md §DESIGN NOTE)
//!
//! **VRAM allocation is M5's responsibility.** Module 5 (the compute loop) must
//! allocate two device-memory regions of at least `slot_bytes` each before
//! training begins.  Under mock-cuda the slots are heap-allocated `Vec<u8>`
//! owned by this module — no device memory is touched.
//!
//! M5 must uphold three invariants:
//!
//! 1. **Two VRAM slots** — provided on construction (or heap-allocated under mock).
//! 2. **Event-wait before compute** — after receiving a [`crate::ready::ReadyLayer`]
//!    with `copy_event = Some(event)`, M5 must call
//!    `cuda_stream_wait_event(compute_stream, event)` before dispatching the
//!    compute kernel.  This ensures the DMA into the VRAM slot completes before
//!    the SM reads it.
//! 3. **Single-slot read window** — M5 must not hold a reference to the VRAM
//!    slot returned by call `N` after requesting call `N+2`, because the
//!    ping-pong cycle reuses that buffer.  In the sequential per-layer training
//!    loop this invariant is trivially satisfied.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

// ── CUDA stream ────────────────────────────────────────────────────────────

/// A CUDA stream handle used for RAM→VRAM copies.
///
/// Under `mock-cuda` this is a zero-sized type; all stream operations execute
/// synchronously.  Under real CUDA (feature `cuda`) this would wrap a
/// `cudaStream_t` handle obtained from the runtime.
pub struct CudaStream {
    _private: (),
}

// SAFETY: Under mock-cuda there is no real GPU resource; the struct is trivially
// Send + Sync.  Under real CUDA, separate streams may be enqueued from any
// thread — the CUDA runtime serialises per-stream operations internally.
unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl CudaStream {
    /// Create a new CUDA stream.
    ///
    /// Under mock-cuda: returns immediately (no-op).
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl Default for CudaStream {
    fn default() -> Self {
        Self::new()
    }
}

// ── CUDA event ─────────────────────────────────────────────────────────────

/// A CUDA event that signals RAM→VRAM copy completion.
///
/// Cloning shares the underlying signal (backed by `Arc<AtomicBool>` under
/// mock-cuda), so all clones observe the same `record`/`reset` transitions.
/// Module 5 uses the clone returned by [`VramDoubleBuffer::advance`] to call
/// `cuda_stream_wait_event(compute_stream, event)`.
///
/// # Mock semantics
/// `record` marks the event immediately (the simulated "copy" is synchronous),
/// so `is_ready` returns `true` and `synchronize` is always a no-op.
#[derive(Clone)]
pub struct CudaEvent {
    done: Arc<AtomicBool>,
}

impl CudaEvent {
    /// Create a new, unsignalled event.
    pub fn new() -> Self {
        Self {
            done: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Record the event at the current position on `stream`.
    ///
    /// Under mock-cuda: marks the event as signalled immediately, because all
    /// stream operations are synchronous in the mock path.
    pub fn record(&self, _stream: &CudaStream) {
        self.done.store(true, Ordering::Release);
    }

    /// Reset the event to the unsignalled state so it can be re-used.
    ///
    /// Must be called before re-recording to prevent stale clones from
    /// observing a false-ready state during the in-flight copy.
    pub fn reset(&self) {
        self.done.store(false, Ordering::Release);
    }

    /// Non-blocking query: `true` if the event has already fired.
    pub fn is_ready(&self) -> bool {
        self.done.load(Ordering::Acquire)
    }

    /// Block the calling thread until the event has fired.
    ///
    /// Under mock-cuda: asserts the event is already signalled (it always is,
    /// because `record` is synchronous) and returns immediately.
    pub fn synchronize(&self) {
        // Under mock-cuda record() runs synchronously, so this assertion always
        // holds.  Under real CUDA this would call cudaEventSynchronize().
        debug_assert!(
            self.done.load(Ordering::Acquire),
            "CudaEvent::synchronize called before record — M5 invariant violated"
        );
    }
}

impl Default for CudaEvent {
    fn default() -> Self {
        Self::new()
    }
}

// ── VRAM slot ──────────────────────────────────────────────────────────────

/// One VRAM slot — a contiguous device-memory region sized to hold one layer.
///
/// Under mock-cuda the slot is backed by a heap `Vec<u8>`.  The raw pointer
/// `ptr` remains stable across moves of the `VramSlot` struct because a
/// `Vec`'s internal heap allocation does not relocate when the `Vec` metadata
/// (pointer, length, capacity) is moved.
struct VramSlot {
    /// Raw pointer into device memory (or heap memory under mock-cuda).
    ptr: *mut u8,
    /// Allocated capacity in bytes.
    capacity: usize,
    /// Heap backing storage (mock-cuda only). Dropped together with the struct.
    _backing: Vec<u8>,
}

// SAFETY: ptr points to memory exclusively owned by `_backing`; no aliasing
// because `VramDoubleBuffer` holds exactly two slots and writes to only one
// at a time (the staging slot).
unsafe impl Send for VramSlot {}
unsafe impl Sync for VramSlot {}

impl VramSlot {
    /// Allocate a zeroed slot of `capacity` bytes.
    fn new(capacity: usize) -> Self {
        let mut backing = vec![0u8; capacity];
        let ptr = backing.as_mut_ptr();
        Self {
            ptr,
            capacity,
            _backing: backing,
        }
    }

    /// Copy `src` into this slot via `stream`.
    ///
    /// Under mock-cuda: `ptr::copy_nonoverlapping` (synchronous).
    ///
    /// # Safety
    /// `src.len()` must not exceed `self.capacity`.  Caller guarantees that
    /// `src` and `self.ptr` do not overlap (different heap allocations).
    unsafe fn copy_from_host(&self, src: &[u8], _stream: &CudaStream) {
        debug_assert!(
            src.len() <= self.capacity,
            "copy_from_host: src {} B exceeds slot capacity {} B",
            src.len(),
            self.capacity,
        );
        let len = src.len().min(self.capacity);
        // SAFETY: bounds checked above; allocations do not alias.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), self.ptr, len);
        }
    }

    /// Device pointer as `u64` (the canonical form stored in `ReadyLayer`).
    fn as_device_ptr(&self) -> u64 {
        self.ptr as u64
    }

    /// Read-only byte slice over the slot (mock-cuda / tests only).
    #[cfg(test)]
    pub(crate) fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr is valid for `capacity` bytes (owned by `_backing`).
        unsafe { std::slice::from_raw_parts(self.ptr, self.capacity) }
    }
}

// ── VramDoubleBuffer ────────────────────────────────────────────────────────

/// Two-slot ping-pong VRAM buffer with a dedicated RAM→VRAM copy stream.
///
/// Each call to [`VramDoubleBuffer::advance`]:
/// 1. Resets `event_b_ready`, then copies `src` → staging `slot_b` on
///    `copy_stream`.
/// 2. Records `event_b_ready` on `copy_stream` (fires when the copy finishes).
/// 3. Swaps `slot_a` ↔ `slot_b` (ping-pong).
/// 4. Returns `(vram_ptr, event)`:
///    * `vram_ptr` — device pointer into the new compute slot (filled `slot_b`,
///      now renamed `slot_a` after the swap).
///    * `event` — Module 5 must call
///      `cuda_stream_wait_event(compute_stream, event)` before dispatching the
///      compute kernel so the SM does not read a partially-written slot.
///
/// # Concurrency
/// `VramDoubleBuffer` is not internally synchronised — the caller (FlowCast)
/// must hold a `Mutex<VramDoubleBuffer>` and lock it for each `advance` call.
/// This serialises the slot swap and prevents two concurrent `advance` calls
/// from corrupting the `slot_a`/`slot_b` assignment.
pub struct VramDoubleBuffer {
    /// Current compute slot: the layer the GPU is (or will be) reading.
    slot_a: VramSlot,
    /// Staging slot: receives the next layer's data via DMA on `copy_stream`.
    slot_b: VramSlot,
    /// Dedicated H→D copy stream (separate from the compute stream).
    copy_stream: CudaStream,
    /// Fires when the most recent copy into `slot_b` is complete.
    event_b_ready: CudaEvent,
}

impl VramDoubleBuffer {
    /// Construct a double buffer where each slot holds at most `slot_bytes`.
    ///
    /// Under mock-cuda, two zeroed `Vec<u8>` allocations are created.
    pub fn new(slot_bytes: usize) -> Self {
        Self {
            slot_a: VramSlot::new(slot_bytes),
            slot_b: VramSlot::new(slot_bytes),
            copy_stream: CudaStream::new(),
            event_b_ready: CudaEvent::new(),
        }
    }

    /// Capacity of each slot in bytes (both slots share the same size).
    pub fn slot_capacity(&self) -> usize {
        self.slot_a.capacity
    }

    /// Advance the pipeline by one layer — the core double-buffer operation.
    ///
    /// Steps:
    /// 1. Resets `event_b_ready` to unsignalled.
    /// 2. Copies `src` into staging `slot_b` via `copy_stream`.
    /// 3. Records `event_b_ready` on `copy_stream`.
    /// 4. Swaps `slot_a` ↔ `slot_b` (the just-written buffer becomes the
    ///    compute slot; the previously-computed buffer becomes the next staging
    ///    slot).
    /// 5. Returns `(vram_ptr, event)` for the new compute slot.
    ///
    /// # Errors
    /// Returns `Err(String)` when `src.len()` exceeds the slot capacity.
    ///
    /// # Buffer-reuse invariant
    /// The physical buffer returned at call `N` is overwritten again at call
    /// `N+2` (ping-pong cycle of two slots).  Module 5 must not retain a GPU
    /// read reference to call-`N`'s buffer after making call `N+2`.  In a
    /// sequential per-layer training loop this invariant is trivially satisfied.
    pub fn advance(
        &mut self,
        layer_idx: u32,
        src: &[u8],
    ) -> Result<(u64, CudaEvent), String> {
        let _ = layer_idx;
        if src.len() > self.slot_b.capacity {
            return Err(format!(
                "layer {layer_idx}: src len {} B exceeds slot capacity {} B",
                src.len(),
                self.slot_b.capacity,
            ));
        }

        // Reset before the copy so any observer of the shared event sees the
        // correct unsignalled state during the in-flight DMA.
        self.event_b_ready.reset();

        // Copy layer bytes from pinned RAM into the staging slot.
        // SAFETY: bounds-checked above; slot_b._backing and src are distinct
        // heap allocations (no aliasing).
        unsafe { self.slot_b.copy_from_host(src, &self.copy_stream) };

        // Record the event: under real CUDA this fires when copy_stream reaches
        // this point (i.e. after the DMA completes).  Under mock-cuda it fires
        // immediately because the copy above is synchronous.
        self.event_b_ready.record(&self.copy_stream);

        // Ping-pong: the staging slot (just written) becomes the compute slot.
        // The old compute slot (being read by the GPU kernel from the previous
        // call) becomes the new staging slot for the NEXT call.  Swapping the
        // struct values is safe because `ptr` points into `_backing`, and both
        // move together so the invariant `ptr ∈ _backing` is preserved.
        std::mem::swap(&mut self.slot_a, &mut self.slot_b);

        let vram_ptr = self.slot_a.as_device_ptr();
        let event = self.event_b_ready.clone();
        Ok((vram_ptr, event))
    }
}
