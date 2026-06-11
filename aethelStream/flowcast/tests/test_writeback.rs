//! T4 — Writeback correctness.
//!
//! 1. After `on_weights_updated(i, src, lr_grad_norm)`, the write is submitted
//!    to the backend; polling completions accounts for it.
//! 2. A "re-read" of the shard (simulated via in-memory buffer copy) matches
//!    the written data within 1e-4.
//! 3. In-flight write cap is respected: inflight never exceeds `max_inflight`.
//! 4. Writes above threshold are never skipped.
//! 5. `flush_all` drains the legacy pending queue.

use flowcast::backend::mock::MockBackend;
use flowcast::backend::IoBackend;
use flowcast::writeback::{WritebackConfig, WritebackMode, WritebackScheduler, PendingWrite};
use ramflow::PinnedBuffer;

fn make_buf(value: u8, size: usize) -> PinnedBuffer {
    let mut buf = PinnedBuffer::alloc(size).expect("alloc");
    buf.as_mut_slice().fill(value);
    buf
}

// T4-1/T4-2: write submitted, re-read matches within 1e-4
#[test]
fn write_submitted_and_data_matches() {
    let backend = MockBackend::new();
    let config = WritebackConfig {
        skip_threshold: 0.0, // never skip
        max_skip_rate: 0.0,
        max_inflight_writes: 4,
        ..Default::default()
    };
    let mut sched = WritebackScheduler::with_config(WritebackMode::Immediate, config);

    let src = make_buf(0xAB, 256);
    sched
        .on_weights_updated(3, &src, 0, 1.0 /* above threshold */, &backend)
        .expect("on_weights_updated");

    // Poll completions — mock backend completed instantly.
    let completions = backend.poll_completions().expect("poll");
    assert_eq!(completions.len(), 1, "expected exactly 1 write completion");
    assert!(completions[0].result >= 0, "write should succeed");

    // Byte-identical read-back via MockBackend's written store (A4-e fix).
    let written_len = completions[0].result as usize;
    assert_eq!(written_len, 256, "written length should match src");

    let written = backend.last_written_bytes(3).expect("written bytes must be stored by mock");
    assert_eq!(written.len(), 256, "written byte count must match src");
    assert!(
        written.iter().all(|&b| b == 0xAB),
        "all written bytes must be 0xAB (byte-identical round-trip)"
    );
}

// T4-3: inflight write cap enforced
#[test]
fn inflight_write_cap_respected() {
    let backend = MockBackend::new();
    let config = WritebackConfig {
        skip_threshold: 0.0,
        max_skip_rate: 0.0,
        max_inflight_writes: 2,
        ..Default::default()
    };
    let mut sched = WritebackScheduler::with_config(WritebackMode::Immediate, config);

    // Submit 4 writes; each poll drains completions so the cap is respected.
    for layer in 0u32..4 {
        let src = make_buf(layer as u8, 64);
        sched
            .on_weights_updated(layer, &src, 0, 1.0, &backend)
            .expect("update");
        // Inflight must never exceed cap (2).
        assert!(
            sched.inflight_count() <= 2,
            "inflight {} > cap 2 after layer {layer}",
            sched.inflight_count()
        );
    }
}

// T4-4: write above threshold is never skipped
#[test]
fn above_threshold_write_never_skipped() {
    let backend = MockBackend::new();
    let config = WritebackConfig {
        skip_threshold: 1e-4,
        max_skip_rate: 0.5,
        max_inflight_writes: 8,
        ..Default::default()
    };
    let mut sched = WritebackScheduler::with_config(WritebackMode::Immediate, config);

    let src = make_buf(0xFF, 128);
    // lr_grad_norm well above threshold
    sched
        .on_weights_updated(0, &src, 0, 1.0, &backend)
        .expect("update");

    let completions = backend.poll_completions().expect("poll");
    assert_eq!(completions.len(), 1, "above-threshold write must not be skipped");
}

// T4-5: flush_all drains legacy pending queue
#[test]
fn flush_all_drains_pending_queue() {
    let backend = MockBackend::new();
    let mut sched = WritebackScheduler::new(WritebackMode::Batched { batch_size: 4 });

    for i in 0..3 {
        sched
            .enqueue(PendingWrite { layer_idx: i, byte_offset: 0, byte_length: 64 })
            .expect("enqueue");
    }
    assert_eq!(sched.pending_count(), 3);

    sched.flush_all(&backend).expect("flush_all");
    assert_eq!(sched.pending_count(), 0, "pending queue must be empty after flush_all");
}
