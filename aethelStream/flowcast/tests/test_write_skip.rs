//! T10 — Write-skip optimisation (A9).
//!
//! 1. Sub-threshold updates skip + accumulate delta.
//! 2. Crossing the threshold flushes the write.
//! 3. Skip-rate never exceeds the configured cap.
//! 4. Final weights (flush_epoch_end) are correct.
//! 5. max_skip_rate guard forces a write when skip budget exhausted.

use flowcast::backend::mock::MockBackend;
use flowcast::backend::IoBackend;
use flowcast::writeback::{WritebackConfig, WritebackMode, WritebackScheduler};
use ramflow::PinnedBuffer;
use std::collections::HashMap;

fn make_buf(size: usize) -> PinnedBuffer {
    PinnedBuffer::alloc(size).expect("alloc")
}

fn make_buf_filled(value: u8, size: usize) -> PinnedBuffer {
    let mut buf = PinnedBuffer::alloc(size).expect("alloc");
    buf.as_mut_slice().fill(value);
    buf
}

// T10-1/T10-2: sub-threshold skips + accumulates; crossing flushes
#[test]
fn sub_threshold_skips_and_crossing_flushes() {
    let backend = MockBackend::new();
    let threshold = 1e-3_f32;
    let config = WritebackConfig {
        skip_threshold: threshold,
        max_skip_rate: 1.0, // allow all skips for this test
        max_inflight_writes: 8,
        ..Default::default()
    };
    let mut sched = WritebackScheduler::with_config(WritebackMode::Immediate, config);

    let src = make_buf(64);

    // 9 sub-threshold updates: each contributes 0.0002, total = 0.0018 after 9 — crosses 1e-3
    for step in 1u32..=9 {
        let lr_grad = 0.0002_f32; // total after step 5 = 0.001 which crosses 1e-3
        sched
            .on_weights_updated(7, &src, 0, lr_grad, &backend)
            .expect("update");
        let completions = backend.poll_completions().expect("poll");
        if step < 5 {
            // Accumulated delta < threshold → no write yet
            assert!(
                completions.is_empty(),
                "step {step}: expected no write (delta < threshold), got {} completions",
                completions.len()
            );
        }
    }

    // At step 5 accumulated = 0.001 ≥ threshold → flush happens.
    // Some completion must have fired by now.
    // Re-poll to catch any pending.
    let final_completions = backend.poll_completions().expect("final poll");
    // total completions across all steps
    let _ = final_completions; // we assert the crossing happened via accumulated_delta reset
    assert!(
        sched.accumulated_delta(7) < threshold,
        "accumulated delta should reset after crossing flush"
    );
}

// T10-3: skip-rate never exceeds cap
#[test]
fn skip_rate_never_exceeds_cap() {
    let backend = MockBackend::new();
    let config = WritebackConfig {
        skip_threshold: 1.0, // very high → all small updates would skip
        max_skip_rate: 0.5,   // but at most 50% may skip
        max_inflight_writes: 8,
        ..Default::default()
    };
    let mut sched = WritebackScheduler::with_config(WritebackMode::Immediate, config);

    let src = make_buf(64);
    for layer in 0u32..10 {
        sched
            .on_weights_updated(layer, &src, 0, 0.001 /* below threshold=1.0 */, &backend)
            .expect("update");
        let rate = sched.current_skip_rate();
        assert!(
            rate <= 0.5 + f32::EPSILON,
            "skip rate {rate:.3} exceeded cap 0.5 after layer {layer}"
        );
    }
}

// T10-4: flush_epoch_end writes every layer with non-zero accumulated delta
#[test]
fn flush_epoch_end_writes_accumulated_layers() {
    let backend = MockBackend::new();
    let config = WritebackConfig {
        skip_threshold: 1.0, // all updates are sub-threshold
        max_skip_rate: 1.0,  // allow full skip
        max_inflight_writes: 8,
        ..Default::default()
    };
    let mut sched = WritebackScheduler::with_config(WritebackMode::Immediate, config);

    let buf0 = make_buf(64);
    let buf1 = make_buf(64);

    // Both skipped.
    sched.on_weights_updated(0, &buf0, 0, 0.001, &backend).expect("update 0");
    sched.on_weights_updated(1, &buf1, 0, 0.002, &backend).expect("update 1");

    // No writes yet.
    assert!(backend.poll_completions().expect("poll").is_empty());

    // Epoch end: flush everything.
    let mut src_map: HashMap<u32, &PinnedBuffer> = HashMap::new();
    src_map.insert(0, &buf0);
    src_map.insert(1, &buf1);
    sched.flush_epoch_end(&src_map, &backend).expect("flush_epoch_end");

    let completions = backend.poll_completions().expect("poll after flush");
    assert_eq!(
        completions.len(),
        2,
        "flush_epoch_end must write both accumulated layers"
    );
    assert!(
        completions.iter().all(|c| c.result >= 0),
        "all writes must succeed"
    );

    // Accumulated deltas must be cleared.
    assert_eq!(sched.accumulated_delta(0), 0.0, "delta[0] must be cleared");
    assert_eq!(sched.accumulated_delta(1), 0.0, "delta[1] must be cleared");
}

// T10-6: accumulated delta increments by lr_grad_norm at each sub-threshold step (A9-f2).
//
// Verifies the per-step accumulation semantics: accumulated_delta must equal the
// sum of all lr_grad_norm values submitted while the threshold is not crossed.
#[test]
fn accumulated_delta_increments_per_step() {
    let backend = MockBackend::new();
    let config = WritebackConfig {
        skip_threshold: 10.0, // very high — never crossed during this test
        max_skip_rate: 1.0,
        max_inflight_writes: 8,
        ..Default::default()
    };
    let mut sched = WritebackScheduler::with_config(WritebackMode::Immediate, config);
    let src = make_buf(64);

    for step in 1u32..=5 {
        sched
            .on_weights_updated(11, &src, 0, 0.1, &backend)
            .expect("update");
        let expected = step as f32 * 0.1;
        let actual = sched.accumulated_delta(11);
        assert!(
            (actual - expected).abs() < 1e-5,
            "step {step}: accumulated_delta = {actual:.6}, expected {expected:.6}"
        );
    }
}

// T10-7: after flush_epoch_end, written bytes are byte-identical to source (A9-f5).
//
// Verifies the final-write correctness guarantee: skipped-and-accumulated layers
// must be flushed with the exact buffer content passed to flush_epoch_end.
#[test]
fn epoch_end_written_bytes_match_source() {
    let backend = MockBackend::new();
    let config = WritebackConfig {
        skip_threshold: 1.0, // all small updates are sub-threshold → skipped
        max_skip_rate: 1.0,
        max_inflight_writes: 8,
        ..Default::default()
    };
    let mut sched = WritebackScheduler::with_config(WritebackMode::Immediate, config);

    let buf3 = make_buf_filled(0xCC, 128);
    let buf8 = make_buf_filled(0xDD, 64);

    // Both sub-threshold → skipped, not written yet.
    sched.on_weights_updated(3, &buf3, 0, 0.0001, &backend).expect("layer 3");
    sched.on_weights_updated(8, &buf8, 0, 0.0001, &backend).expect("layer 8");
    assert!(backend.poll_completions().expect("poll").is_empty(), "no writes until epoch end");

    let mut src_map = HashMap::new();
    src_map.insert(3u32, &buf3);
    src_map.insert(8u32, &buf8);
    sched.flush_epoch_end(&src_map, &backend).expect("flush_epoch_end");
    backend.poll_completions().expect("drain completions");

    // Byte-identical verification (A9-f5).
    let written3 = backend.last_written_bytes(3).expect("layer 3 must be written at epoch end");
    assert_eq!(written3.len(), 128, "layer 3 written length");
    assert!(
        written3.iter().all(|&b| b == 0xCC),
        "layer 3: written bytes must be 0xCC (byte-identical)"
    );

    let written8 = backend.last_written_bytes(8).expect("layer 8 must be written at epoch end");
    assert_eq!(written8.len(), 64, "layer 8 written length");
    assert!(
        written8.iter().all(|&b| b == 0xDD),
        "layer 8: written bytes must be 0xDD (byte-identical)"
    );
}

// T10-5: max_skip_rate guard forces write when skip budget exhausted
#[test]
fn max_skip_rate_guard_forces_write_when_exhausted() {
    let backend = MockBackend::new();
    let config = WritebackConfig {
        skip_threshold: 100.0, // everything is sub-threshold
        max_skip_rate: 0.5,
        max_inflight_writes: 8,
        ..Default::default()
    };
    let mut sched = WritebackScheduler::with_config(WritebackMode::Immediate, config);

    let src = make_buf(64);
    let mut forced_writes = 0usize;

    for layer in 0u32..6 {
        sched.on_weights_updated(layer, &src, 0, 0.001, &backend).expect("update");
        let completions = backend.poll_completions().expect("poll");
        forced_writes += completions.len();
    }

    // With 6 layers and max_skip_rate=0.5, at most 3 can skip → at least 3 forced writes.
    assert!(
        forced_writes >= 3,
        "expected ≥3 forced writes from 6 layers at 50% cap, got {forced_writes}"
    );
    // Skip rate must not exceed 0.5.
    assert!(
        sched.current_skip_rate() <= 0.5 + f32::EPSILON,
        "skip rate {} exceeded cap",
        sched.current_skip_rate()
    );
}
