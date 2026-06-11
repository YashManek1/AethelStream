//! T9 — Priority queue and adaptive precision.
//!
//! 1. High-variance layers get higher importance → popped first.
//! 2. High-variance layers get FP16 precision.
//! 3. Low-variance layers get INT4 precision.
//! 4. Every layer eventually ready — no starvation (importance ≥ 1.0).
//! 5. update_importances changes scores correctly.
//! 6. rebuild_from_scale_table populates queue from PerLayerScaleTable.
//! 7. Middle-variance layers keep default precision.
//! 8. Aging prevents starvation: zero-variance layer pops before high-variance (A8-e2).

use flowcast::priority::{
    PriorityQueue, PrefetchRequest,
    HIGH_VARIANCE_THRESHOLD, LOW_VARIANCE_THRESHOLD, MAX_AGE_STEPS,
    precision_for_variance, importance_for_variance,
};
use flowcast::config::Precision;
use ramflow::PerLayerScaleTable;

// T9-1/T9-2: high-variance layer popped before low-variance, gets FP16
#[test]
fn high_variance_layers_ready_earlier_and_fp16() {
    let mut queue = PriorityQueue::new();

    queue.push(PrefetchRequest {
        layer_idx: 0,
        importance: importance_for_variance(LOW_VARIANCE_THRESHOLD * 0.5),
        precision: precision_for_variance(LOW_VARIANCE_THRESHOLD * 0.5, Precision::FP16),
        enqueue_step: 0,
    }).unwrap();

    let high_var = HIGH_VARIANCE_THRESHOLD * 2.0;
    queue.push(PrefetchRequest {
        layer_idx: 1,
        importance: importance_for_variance(high_var),
        precision: precision_for_variance(high_var, Precision::FP16),
        enqueue_step: 0,
    }).unwrap();

    let first = queue.pop().expect("first pop");
    assert_eq!(first.layer_idx, 1, "high-variance layer must pop first");
    assert_eq!(first.precision, Precision::FP16, "high-variance must be FP16");
}

// T9-3: low-variance layer gets INT4
#[test]
fn low_variance_layers_get_int4() {
    let low_var = LOW_VARIANCE_THRESHOLD * 0.1;
    let precision = precision_for_variance(low_var, Precision::FP16);
    assert_eq!(precision, Precision::INT4, "low-variance must select INT4");
}

// T9-4: no starvation — every layer has importance ≥ 1.0
#[test]
fn no_starvation_importance_always_positive() {
    for raw_var in [0.0_f32, 0.001, 0.05, 0.2, 1.0, 100.0] {
        let importance = importance_for_variance(raw_var);
        assert!(
            importance >= 1.0,
            "importance {importance} < 1.0 for variance {raw_var} — starvation possible"
        );
    }
}

// T9-5: update_importances changes scores
#[test]
fn update_importances_changes_scores() {
    let mut queue = PriorityQueue::new();
    queue.push(PrefetchRequest {
        layer_idx: 0, importance: 1.0, precision: Precision::FP16, enqueue_step: 0,
    }).unwrap();
    queue.push(PrefetchRequest {
        layer_idx: 1, importance: 2.0, precision: Precision::FP16, enqueue_step: 0,
    }).unwrap();

    queue.update_importances(&[(0, 99.0), (1, 0.5)]).unwrap();

    let first = queue.pop().expect("pop");
    assert_eq!(first.layer_idx, 0, "layer 0 should now be highest after update");
    assert!((first.importance - 99.0).abs() < 0.01);
}

// T9-6: rebuild_from_scale_table populates correctly
#[test]
fn rebuild_from_scale_table_populates_queue() {
    const NUM_LAYERS: usize = 8;
    let mut scale_table = PerLayerScaleTable::new(NUM_LAYERS, 0.05);

    for _ in 0..10 {
        scale_table.update_gradient_variance(3, 1.0);
        scale_table.update_gradient_variance(5, 0.0);
    }

    let mut queue = PriorityQueue::new();
    queue.rebuild_from_scale_table(0..NUM_LAYERS as u32, &scale_table, Precision::BF16).unwrap();

    assert_eq!(queue.len(), NUM_LAYERS, "all layers must be enqueued");

    let first = queue.pop().expect("first");
    assert_eq!(first.layer_idx, 3, "layer 3 (high variance) must pop first");
    assert_eq!(first.precision, Precision::FP16);
    assert_eq!(queue.len(), NUM_LAYERS - 1);
}

// T9-7: middle-variance layers keep default precision
#[test]
fn middle_variance_keeps_default_precision() {
    let mid_var = (HIGH_VARIANCE_THRESHOLD + LOW_VARIANCE_THRESHOLD) / 2.0;
    assert_eq!(precision_for_variance(mid_var, Precision::BF16), Precision::BF16);
    assert_eq!(precision_for_variance(mid_var, Precision::FP16), Precision::FP16);
}

// T9-8: aging forces zero-variance layer to pop before high-variance after MAX_AGE_STEPS (A8-e2).
//
// Verifies that the starvation-prevention mechanism actually works: a zero-importance
// layer enqueued at step 0 must pop before a high-importance layer enqueued later,
// once the low-importance layer's age exceeds MAX_AGE_STEPS.
#[test]
fn aging_prevents_starvation_of_zero_variance_layer() {
    let mut queue = PriorityQueue::new();

    // Enqueue a zero-variance (floor importance = 1.0) layer first.
    // After MAX_AGE_STEPS pushes its effective importance = f32::MAX.
    queue.push(PrefetchRequest {
        layer_idx: 0,
        importance: 1.0,
        precision: Precision::INT4,
        enqueue_step: 0,
    }).unwrap();

    // Enqueue MAX_AGE_STEPS high-variance layers so the step counter advances.
    for step in 1..=(MAX_AGE_STEPS as u32 + 1) {
        queue.push(PrefetchRequest {
            layer_idx: step,
            importance: importance_for_variance(HIGH_VARIANCE_THRESHOLD * 10.0),
            precision: Precision::FP16,
            enqueue_step: step as u64,
        }).unwrap();
    }

    // Layer 0 was enqueued at step 0; current_step is now MAX_AGE_STEPS+2.
    // Its age ≥ MAX_AGE_STEPS → effective importance = f32::MAX → must pop first.
    let first = queue.pop().expect("pop");
    assert_eq!(
        first.layer_idx, 0,
        "zero-variance layer 0 should be force-popped by aging, got layer {}",
        first.layer_idx
    );
}
