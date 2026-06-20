//! Integration test: construct a DoublePass with mock FlowCast + canned plan,
//! assert the crate builds and the mock cycle compiles.
//!
//! No real GPU or NVMe required. Run with:
//!   cargo test --features mock-cuda

// Allow unimplemented!() panics in test scaffolding — we are asserting structure,
// not logic, at S0.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use doublepass::{
    Batch, DoublePass, FlowCast, FlowCastConfig, StepMetrics,
    TrainingPlan, TrainingTier,
};
use flowcast::backend::mock::MockBackend;

/// Verify that `TrainingPlan::default()` constructs without error.
#[test]
fn plan_default_constructs() {
    let plan = TrainingPlan::default();
    assert_eq!(plan.checkpoint_freq, 4);
    assert_eq!(plan.grad_accum, 2);
    assert!(matches!(plan.tier, TrainingTier::LoraOnly));
}

/// Verify that `StepMetrics::default()` initializes correctly.
#[test]
fn step_metrics_default_is_zero() {
    let m = StepMetrics::default();
    assert_eq!(m.weight_bytes_streamed, 0);
    assert_eq!(m.prefetch_misses, 0);
    // parity_rel_error defaults to 0.0 (f64::default()), not NAN
    assert_eq!(m.parity_rel_error, 0.0);
}

/// Verify that `Batch` can be serialized and deserialized.
#[test]
fn batch_serializes() {
    let batch = Batch {
        input_ids: vec![vec![1u32, 2, 3], vec![4, 5, 6]],
    };
    let json = serde_json::to_string(&batch).expect("serialize");
    let back: Batch = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back.input_ids, batch.input_ids);
}

/// Verify that FlowCastError::PrefetchMiss wraps correctly into DoublePassError.
#[test]
fn error_wraps_prefetch_miss() {
    use doublepass::error::DoublePassError;
    use flowcast::FlowCastError;
    let fc_err = FlowCastError::PrefetchMiss { layer_idx: 7 };
    let dp_err = DoublePassError::from(fc_err);
    let msg = format!("{dp_err}");
    assert!(
        msg.contains("prefetch miss") || msg.contains("layer 7") || msg.contains("flowcast"),
        "error message was: {msg}"
    );
}

/// The full DoublePass construction is behind unimplemented!(), so assert
/// it panics at S0 (it will stop panicking when S1 implements the body).
#[test]
#[should_panic]
fn doublepass_new_is_s0_stub() {
    let config = FlowCastConfig {
        num_shards: 2,
        ..FlowCastConfig::default()
    };
    let backend = Box::new(MockBackend::new());
    let fc = FlowCast::new(config, backend).expect("FlowCast::new");
    let _ = DoublePass::new(fc, None, None).expect("DoublePass::new");
}
