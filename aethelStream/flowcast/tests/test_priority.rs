//! T9 — Priority queue and adaptive precision.
//!
//! 1. High-variance layers get higher importance → popped first.
//! 2. High-variance layers get FP16 precision.
//! 3. Low-variance layers get INT4 precision.
//! 4. Every layer eventually ready — no starvation (importance ≥ 1.0).
//! 5. update_importances changes scores correctly.
//! 6. rebuild_from_scale_table populates queue from PerLayerScaleTable.

use flowcast::priority::{
    PriorityQueue, PrefetchRequest,
    HIGH_VARIANCE_THRESHOLD, LOW_VARIANCE_THRESHOLD, precision_for_variance, importance_for_variance,
};
use flowcast::config::Precision;
use ramflow::PerLayerScaleTable;

// T9-1/T9-2: high-variance layer popped before low-variance, gets FP16
#[test]
fn high_variance_layers_ready_earlier_and_fp16() {
    let mut queue = PriorityQueue::new();

    // Low-variance layer enqueued first.
    queue.push(PrefetchRequest {
        layer_idx: 0,
        importance: importance_for_variance(LOW_VARIANCE_THRESHOLD * 0.5),
        precision: precision_for_variance(LOW_VARIANCE_THRESHOLD * 0.5, Precision::FP16),
    }).unwrap();

    // High-variance layer enqueued second.
    let high_var = HIGH_VARIANCE_THRESHOLD * 2.0;
    queue.push(PrefetchRequest {
        layer_idx: 1,
        importance: importance_for_variance(high_var),
        precision: precision_for_variance(high_var, Precision::FP16),
    }).unwrap();

    // High-variance must pop first.
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
    queue.push(PrefetchRequest { layer_idx: 0, importance: 1.0, precision: Precision::FP16 }).unwrap();
    queue.push(PrefetchRequest { layer_idx: 1, importance: 2.0, precision: Precision::FP16 }).unwrap();

    // Demote layer 1 and promote layer 0.
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

    // Inject gradient variance: layer 3 high, layer 5 low.
    for _ in 0..10 {
        scale_table.update_gradient_variance(3, 1.0);  // high
        scale_table.update_gradient_variance(5, 0.0);  // low
    }

    let mut queue = PriorityQueue::new();
    queue.rebuild_from_scale_table(0..NUM_LAYERS as u32, &scale_table, Precision::BF16).unwrap();

    assert_eq!(queue.len(), NUM_LAYERS, "all layers must be enqueued");

    // Layer 3 (high variance) should pop first.
    let first = queue.pop().expect("first");
    assert_eq!(first.layer_idx, 3, "layer 3 (high variance) must pop first");
    assert_eq!(first.precision, Precision::FP16);

    // All remaining layers must still be present (no starvation).
    assert_eq!(queue.len(), NUM_LAYERS - 1);
}

// T9-7: middle-variance layers keep default precision
#[test]
fn middle_variance_keeps_default_precision() {
    let mid_var = (HIGH_VARIANCE_THRESHOLD + LOW_VARIANCE_THRESHOLD) / 2.0;
    assert_eq!(precision_for_variance(mid_var, Precision::BF16), Precision::BF16);
    assert_eq!(precision_for_variance(mid_var, Precision::FP16), Precision::FP16);
}
