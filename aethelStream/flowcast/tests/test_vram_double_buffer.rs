//! Tests for the VRAM double-buffer (mock-cuda path).
//!
//! All tests compile and run only when the `cuda-double-buffer` feature is
//! active (and implicitly under `mock-cuda` for CI).  Real-CUDA paths that
//! would require device memory are marked `#[ignore]`.
#![cfg(feature = "cuda-double-buffer")]

use flowcast::vram_double_buffer::{CudaEvent, VramDoubleBuffer};

// ── slot swap ──────────────────────────────────────────────────────────────

/// Every other `advance` call must reuse the same physical buffer (ping-pong).
#[test]
fn slot_swap_alternates_compute_slot() {
    let slot_bytes = 64usize;
    let mut vdb = VramDoubleBuffer::new(slot_bytes);

    let data_a = vec![1u8; slot_bytes];
    let data_b = vec![2u8; slot_bytes];

    let (ptr_0, _e0) = vdb.advance(0, &data_a).expect("advance 0");
    let (ptr_1, _e1) = vdb.advance(1, &data_b).expect("advance 1");
    let (ptr_2, _e2) = vdb.advance(2, &data_a).expect("advance 2");

    // Consecutive advances must use DIFFERENT physical slots to avoid
    // overwriting the buffer while the GPU is reading it.
    assert_ne!(ptr_0, ptr_1, "consecutive advances must use different slots");

    // The third advance returns to the first slot (two-slot ping-pong).
    assert_eq!(
        ptr_0, ptr_2,
        "advance(N) and advance(N+2) must reuse the same physical slot"
    );
}

// ── copy event ─────────────────────────────────────────────────────────────

/// The event returned by `advance` must be signalled (under mock-cuda the
/// simulated copy is synchronous, so the event fires immediately).
#[test]
fn copy_event_fires_after_advance() {
    let mut vdb = VramDoubleBuffer::new(32);
    let data = vec![0xABu8; 32];

    let (_ptr, event) = vdb.advance(0, &data).expect("advance");

    assert!(
        event.is_ready(),
        "event must be signalled immediately after advance (mock-cuda: synchronous copy)"
    );
}

/// `synchronize` must not panic when called after `advance`.
#[test]
fn event_synchronize_is_noop_after_advance() {
    let mut vdb = VramDoubleBuffer::new(16);
    let (_ptr, event) = vdb.advance(0, &vec![0u8; 16]).expect("advance");
    event.synchronize(); // must not panic
}

/// Cloned events share the same signal; `record` on one clone is observed by all.
#[test]
fn cloned_events_share_signal() {
    let event = CudaEvent::new();
    let clone = event.clone();

    assert!(!event.is_ready());
    assert!(!clone.is_ready());

    let stream = flowcast::vram_double_buffer::CudaStream::new();
    event.record(&stream);

    assert!(clone.is_ready(), "clone must observe parent's record");
}

// ── byte correctness ───────────────────────────────────────────────────────

/// Bytes written via `advance` must be readable from the returned device
/// pointer (mock-cuda: the pointer is a heap address).
#[test]
fn bytes_in_slot_match_source_content() {
    let slot_bytes = 16usize;
    let mut vdb = VramDoubleBuffer::new(slot_bytes);

    let pattern: Vec<u8> = (0u8..16).collect();
    let (ptr, event) = vdb.advance(0, &pattern).expect("advance");
    event.synchronize();

    // Under mock-cuda, the returned `u64` IS the heap pointer.
    let slot_slice =
        unsafe { std::slice::from_raw_parts(ptr as *const u8, slot_bytes) };
    assert_eq!(
        &slot_slice[..pattern.len()],
        pattern.as_slice(),
        "slot must contain an exact copy of the source bytes"
    );
}

/// A second advance over the same slot must contain the new data (not stale).
#[test]
fn slot_contents_update_on_reuse() {
    let slot_bytes = 8usize;
    let mut vdb = VramDoubleBuffer::new(slot_bytes);

    let (ptr_0, _) = vdb.advance(0, &vec![0xAAu8; slot_bytes]).expect("advance 0");
    let (_ptr_1, _) = vdb.advance(1, &vec![0xBBu8; slot_bytes]).expect("advance 1");
    // advance(2) reuses slot_0's physical buffer; write new data.
    let (ptr_2, _) = vdb.advance(2, &vec![0xCCu8; slot_bytes]).expect("advance 2");

    assert_eq!(ptr_0, ptr_2, "same physical slot");
    let reused_slice =
        unsafe { std::slice::from_raw_parts(ptr_2 as *const u8, slot_bytes) };
    assert!(
        reused_slice.iter().all(|&byte| byte == 0xCC),
        "reused slot must contain new data, not stale bytes from advance(0)"
    );
}

// ── error paths ────────────────────────────────────────────────────────────

/// `advance` with `src.len() > slot_capacity` must return `Err`.
#[test]
fn oversized_src_returns_error() {
    let slot_bytes = 8usize;
    let mut vdb = VramDoubleBuffer::new(slot_bytes);
    let oversized = vec![0u8; slot_bytes + 1];

    assert!(
        vdb.advance(0, &oversized).is_err(),
        "advance must error when src exceeds slot capacity"
    );
}

/// The buffer remains usable after an error (error is not fatal).
#[test]
fn buffer_usable_after_error() {
    let slot_bytes = 4usize;
    let mut vdb = VramDoubleBuffer::new(slot_bytes);

    let _ = vdb.advance(0, &vec![0u8; slot_bytes + 1]);
    let result = vdb.advance(1, &vec![0xFFu8; slot_bytes]);
    assert!(result.is_ok(), "buffer must remain usable after an error");
}

// ── event reset ────────────────────────────────────────────────────────────

/// Each `advance` call returns a fresh event; a clone from call `N` observing
/// the reset from call `N+1` is expected behaviour (shared `Arc`).
#[test]
fn event_resets_between_advances() {
    let slot_bytes = 8usize;
    let mut vdb = VramDoubleBuffer::new(slot_bytes);
    let data = vec![0u8; slot_bytes];

    let (_ptr0, event_0) = vdb.advance(0, &data).expect("advance 0");
    assert!(event_0.is_ready());

    // After the second advance the underlying AtomicBool was reset then
    // re-recorded.  The clone event_0 shares the same Arc, so it may
    // observe the intermediate reset then the re-record — in mock-cuda both
    // happen synchronously so the final state is always `true`.
    let (_ptr1, event_1) = vdb.advance(1, &data).expect("advance 1");
    assert!(event_1.is_ready(), "event from advance 1 must be ready");
}

// ── capacity accessor ──────────────────────────────────────────────────────

#[test]
fn slot_capacity_matches_constructor() {
    let slot_bytes = 1024usize;
    let vdb = VramDoubleBuffer::new(slot_bytes);
    assert_eq!(vdb.slot_capacity(), slot_bytes);
}

// ── thread safety ──────────────────────────────────────────────────────────

/// Concurrent `advance` calls serialised by `Mutex<VramDoubleBuffer>` must
/// not cause data races, panics, or incorrect events.
#[test]
fn thread_safety_concurrent_advance_via_mutex() {
    use std::sync::{Arc, Mutex};
    use std::thread;

    let slot_bytes = 64usize;
    let vdb = Arc::new(Mutex::new(VramDoubleBuffer::new(slot_bytes)));
    let data = vec![0xFFu8; slot_bytes];

    let handles: Vec<_> = (0u32..8)
        .map(|index| {
            let vdb_clone = Arc::clone(&vdb);
            let data_clone = data.clone();
            thread::spawn(move || {
                let mut guard = vdb_clone.lock().expect("mutex must not be poisoned");
                guard
                    .advance(index, &data_clone)
                    .expect("concurrent advance must not fail")
            })
        })
        .collect();

    for handle in handles {
        let (_ptr, event) = handle.join().expect("thread must not panic");
        assert!(event.is_ready(), "every event from a completed advance must be ready");
    }
}

// ── real-CUDA gate ─────────────────────────────────────────────────────────

/// Placeholder for future tests that require a real CUDA device.
/// Marked `#[ignore]` so CI (mock-cuda path) always skips them.
#[test]
#[ignore = "requires real CUDA device; run with --ignored on a GPU node"]
fn real_cuda_slot_matches_host_copy() {
    // Future: allocate two device slots via cudaMalloc, construct
    // VramDoubleBuffer with real device pointers, verify D→H copy matches src.
    unimplemented!("real-CUDA path not yet wired");
}
