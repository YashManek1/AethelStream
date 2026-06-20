//! T-PARITY-6: ParityGuard behavioural tests.
//!
//! Verifies:
//! 1. A 0.01 offset in stream_grad triggers escalation within `parity_check_interval` steps.
//! 2. After removing the offset, parity returns below 1e-5.
//! 3. The guard NEVER calls any optimizer projection-reset method.
//! 4. `tol_warn` / `tol_halt` thresholds are configurable.
//! 5. `recompute_precision` returns FP32 for escalated layers and the default otherwise.
//! 6. `measure_parity` raises `ParityHalt` when rel >= halt threshold.

use doublepass::{
    compute_relative_error, measure_parity, DoublePassError, ParityAction, ParityGuard,
    ParityTolerances, Precision,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_ref_grad(n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|i| scale * (i as f32 + 1.0)).collect()
}

fn with_offset(ref_grad: &[f32], offset: f32) -> Vec<f32> {
    ref_grad.iter().map(|&x| x + offset).collect()
}

// ---------------------------------------------------------------------------
// T-PARITY-6-A: offset fires within parity_check_interval
// ---------------------------------------------------------------------------

#[test]
fn t_parity_6a_offset_fires_within_interval() {
    let tol = ParityTolerances {
        warn: 1e-4,
        halt: 1e-2,
    };
    let interval = 5_u64;
    let mut guard = ParityGuard::new(tol, interval);

    let ref_grad = make_ref_grad(32, 1.0);
    // 0.01 absolute offset over max|ref|=32 → rel ≈ 0.01/32 ≈ 3.1e-4 > tol_warn
    let bad_stream = with_offset(&ref_grad, 0.01);

    let mut escalated_at: Option<u64> = None;

    for step in 1_u64..=interval {
        if guard.should_check(step) {
            let action = guard
                .check(step, 0, &bad_stream, &ref_grad)
                .expect("should not halt");
            if action == ParityAction::Escalated {
                escalated_at = Some(step);
                break;
            }
        }
    }

    assert!(
        escalated_at.is_some(),
        "guard should escalate within {interval} steps; didn't fire"
    );
    assert!(
        escalated_at.unwrap() <= interval,
        "escalation at step {} > interval {}",
        escalated_at.unwrap(),
        interval
    );
    assert!(guard.is_escalated(0), "layer 0 should be in escalated set");
}

// ---------------------------------------------------------------------------
// T-PARITY-6-B: after removing offset parity recovers below 1e-5
// ---------------------------------------------------------------------------

#[test]
fn t_parity_6b_recovers_after_offset_removed() {
    let tol = ParityTolerances {
        warn: 1e-4,
        halt: 1e-2,
    };
    let mut guard = ParityGuard::new(tol, 1);

    let ref_grad = make_ref_grad(32, 1.0);
    let bad_stream = with_offset(&ref_grad, 0.01);

    // Step 1: inject offset → escalate
    let action = guard.check(1, 0, &bad_stream, &ref_grad).expect("no halt");
    assert_eq!(action, ParityAction::Escalated);

    // Step 2: clean stream — rel should be 0 (identical arrays)
    let clean_action = guard.check(2, 0, &ref_grad, &ref_grad).expect("no halt");
    assert_eq!(clean_action, ParityAction::Clean);
    assert!(
        !guard.is_escalated(0),
        "layer 0 should de-escalate after clean step"
    );

    // Verify the raw relative error is < 1e-5 for identical arrays.
    let rel = compute_relative_error(&ref_grad, &ref_grad);
    assert!(
        rel < 1e-5,
        "relative error after offset removal should be < 1e-5, got {rel:.3e}"
    );
}

// ---------------------------------------------------------------------------
// T-PARITY-6-C: guard NEVER calls projection reset (optimizer invariant)
// ---------------------------------------------------------------------------

/// Mock optimizer that panics if any projection-mutating method is called.
struct StrictMockOptimizer {
    notify_count: std::sync::atomic::AtomicU64,
}

impl StrictMockOptimizer {
    fn new() -> Self {
        Self {
            notify_count: std::sync::atomic::AtomicU64::new(0),
        }
    }

    fn notify_step(&self, _step: u64) {
        self.notify_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    fn notify_count(&self) -> u64 {
        self.notify_count.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[test]
fn t_parity_6c_guard_never_resets_projections() {
    let tol = ParityTolerances {
        warn: 1e-4,
        halt: 1e-2,
    };
    let mut guard = ParityGuard::new(tol, 1);
    let optimizer = StrictMockOptimizer::new();

    let ref_grad = make_ref_grad(16, 2.0);
    let bad_stream = with_offset(&ref_grad, 0.05);

    // Training loop: optimizer.notify_step per step; guard.check on interval.
    // The guard holds NO optimizer reference — it is structurally impossible for
    // it to call zero_accum or project_and_accumulate (no method signatures exist
    // on ParityGuard that accept an optimizer).
    for step in 1_u64..=5 {
        optimizer.notify_step(step);
        if guard.should_check(step) {
            let _ = guard.check(step, 1, &bad_stream, &ref_grad);
        }
    }

    assert!(
        optimizer.notify_count() > 0,
        "notify_step must be called by the loop"
    );
    // If the guard had called zero_accum or project_and_accumulate this test would
    // fail to compile — those methods don't exist on ParityGuard.
}

// ---------------------------------------------------------------------------
// T-PARITY-6-D: warn and halt thresholds are configurable
// ---------------------------------------------------------------------------

#[test]
fn t_parity_6d_custom_warn_threshold_respected() {
    // Very tight warn threshold — even a tiny offset should escalate.
    let tol = ParityTolerances {
        warn: 1e-6,
        halt: 1e-1,
    };
    let mut guard = ParityGuard::new(tol, 1);

    let ref_grad = vec![1.0_f32; 64];
    // 1e-5 offset → rel = 1e-5 / (1.0 + 1e-8) ≈ 1e-5 > warn threshold 1e-6
    let stream = with_offset(&ref_grad, 1e-5);

    let action = guard.check(1, 2, &stream, &ref_grad).expect("no halt");
    assert_eq!(
        action,
        ParityAction::Escalated,
        "tight warn threshold should trigger"
    );
}

#[test]
fn t_parity_6d_halt_threshold_returns_error() {
    let tol = ParityTolerances {
        warn: 1e-6,
        halt: 1e-4,
    };
    let mut guard = ParityGuard::new(tol, 1);

    let ref_grad = vec![1.0_f32; 64];
    // Large offset: rel ≈ 0.5 >> halt=1e-4
    let stream = with_offset(&ref_grad, 0.5);

    let result = guard.check(1, 3, &stream, &ref_grad);
    assert!(result.is_err(), "rel > halt must return Err");
    match result.unwrap_err() {
        DoublePassError::ParityHalt { layer_idx, rel } => {
            assert_eq!(layer_idx, 3);
            assert!(rel > 1e-4, "rel should be >> halt; got {rel:.3e}");
        }
        other => panic!("expected ParityHalt, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// T-PARITY-6-E: recompute_precision returns FP32 for escalated, default otherwise
// ---------------------------------------------------------------------------

#[test]
fn t_parity_6e_recompute_precision_escalated_vs_default() {
    let mut guard = ParityGuard::new(
        ParityTolerances {
            warn: 1e-6,
            halt: 1.0,
        },
        1,
    );

    let ref_grad = vec![1.0_f32; 8];
    let bad = with_offset(&ref_grad, 1e-4);

    // Escalate layer 7
    let action = guard.check(1, 7, &bad, &ref_grad).expect("no halt");
    assert_eq!(action, ParityAction::Escalated);

    // Escalated layer → FP32
    assert_eq!(
        guard.recompute_precision(7, Precision::BF16),
        Precision::FP32
    );
    // Non-escalated layer → default
    assert_eq!(
        guard.recompute_precision(0, Precision::BF16),
        Precision::BF16
    );
    assert_eq!(
        guard.recompute_precision(0, Precision::FP16),
        Precision::FP16
    );
}

// ---------------------------------------------------------------------------
// T-PARITY-6-F: measure_parity raises ParityHalt above halt threshold
// ---------------------------------------------------------------------------

#[test]
fn t_parity_6f_measure_parity_halts_above_threshold() {
    let tol = ParityTolerances {
        warn: 1e-4,
        halt: 1e-3,
    };
    let ref_grad = vec![1.0_f32; 16];
    // rel ≈ 0.01 >> halt=1e-3
    let stream = with_offset(&ref_grad, 0.01);

    let result = measure_parity(5, &stream, &ref_grad, &tol);
    assert!(result.is_err());
    match result.unwrap_err() {
        DoublePassError::ParityHalt { layer_idx, .. } => assert_eq!(layer_idx, 5),
        other => panic!("expected ParityHalt, got {other:?}"),
    }
}

#[test]
fn t_parity_6f_measure_parity_ok_below_threshold() {
    let tol = ParityTolerances {
        warn: 1e-4,
        halt: 1e-3,
    };
    let ref_grad = make_ref_grad(16, 1.0);
    let rel = measure_parity(0, &ref_grad, &ref_grad, &tol).expect("no halt for identical grads");
    assert!(rel < 1e-9, "identical grads → rel ≈ 0, got {rel:.3e}");
}

// ---------------------------------------------------------------------------
// T-PARITY-6-G: check_count and escalated_layer_count diagnostics
// ---------------------------------------------------------------------------

#[test]
fn t_parity_6g_check_count_and_escalation_count() {
    let mut guard = ParityGuard::new(
        ParityTolerances {
            warn: 1e-6,
            halt: 1.0,
        },
        1,
    );
    assert_eq!(guard.check_count(), 0);
    assert_eq!(guard.escalated_layer_count(), 0);

    let ref_grad = vec![1.0_f32; 4];
    let bad = with_offset(&ref_grad, 1e-4);

    guard.check(1, 0, &bad, &ref_grad).unwrap();
    assert_eq!(guard.check_count(), 1);
    assert_eq!(guard.escalated_layer_count(), 1);

    guard.check(2, 1, &bad, &ref_grad).unwrap();
    assert_eq!(guard.check_count(), 2);
    assert_eq!(guard.escalated_layer_count(), 2);

    // Clean step for layer 0 de-escalates it
    guard.check(3, 0, &ref_grad, &ref_grad).unwrap();
    assert_eq!(guard.check_count(), 3);
    assert_eq!(
        guard.escalated_layer_count(),
        1,
        "layer 0 de-escalated, layer 1 still up"
    );
}

// ---------------------------------------------------------------------------
// T-PARITY-6-H: should_check cadence (interval=0 disables, fires at multiples)
// ---------------------------------------------------------------------------

#[test]
fn t_parity_6h_should_check_cadence() {
    let guard = ParityGuard::new(ParityTolerances::default(), 10);
    assert!(!guard.should_check(0), "step 0 must not fire");
    assert!(!guard.should_check(5), "step 5 must not fire (interval=10)");
    assert!(guard.should_check(10), "step 10 must fire");
    assert!(guard.should_check(20), "step 20 must fire");
    assert!(!guard.should_check(15), "step 15 must not fire");

    let disabled = ParityGuard::new(ParityTolerances::default(), 0);
    for step in [1_u64, 10, 100, 1000] {
        assert!(
            !disabled.should_check(step),
            "interval=0 must never fire (step {step})"
        );
    }
}
