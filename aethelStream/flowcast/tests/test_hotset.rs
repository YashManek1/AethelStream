//! T7 — Hot-set / residency tests.
//!
//! 1. Resident layers issue zero prefetch I/O.
//! 2. Non-resident layers do issue prefetch I/O.
//! 3. `is_resident` matches actual residency.
//! 4. LFU promotion works when RAM headroom > threshold.
//! 5. LFU eviction works when full and a higher-count layer arrives.
//! 6. `seed_static` marks layers resident in `PerLayerScaleTable`.

use flowcast::hotset::HotSet;
use flowcast::state_machine::PrefetchStateMachine;
use flowcast::backend::mock::MockBackend;
use flowcast::backend::IoBackend;
use flowcast::config::Precision;
use ramflow::{PoolRegistry, PerLayerScaleTable};
use ramflow::phase::Direction;

const NUM_LAYERS: u32 = 16;

// ---------------------------------------------------------------------------
// T7-1 & T7-2: resident → zero I/O, non-resident → I/O issued
// ---------------------------------------------------------------------------

#[test]
fn resident_layers_issue_zero_prefetch_io() {
    let pool = PoolRegistry::with_defaults().expect("pool");
    let backend = MockBackend::new();
    let sm = PrefetchStateMachine::new(NUM_LAYERS, 2, Precision::FP16);

    let mut hotset = HotSet::new(8);
    let mut scale_table = PerLayerScaleTable::new(NUM_LAYERS as usize, 0.05);

    // Seed layers 0 and 1 as resident.
    hotset.seed_static(NUM_LAYERS, 2, &[], &mut scale_table);
    let residents: Vec<u32> = hotset.resident_layers();
    assert!(!residents.is_empty(), "seed must produce at least one resident layer");

    // Count prefetch calls for residents vs non-residents.
    let resident_set: std::collections::HashSet<u32> = residents.iter().copied().collect();

    // Use on_layer_start_with_residency for current layer 0 (forward).
    // Targets are layers 1 and 2. Layer 1 is resident; layer 2 is not.
    sm.on_layer_start_with_residency(
        0,
        Direction::Forward,
        &pool,
        &backend,
        |idx| resident_set.contains(&idx),
    )
    .expect("on_layer_start_with_residency");

    let completions = backend.poll_completions().expect("poll");
    // Only non-resident targets should have fired a prefetch.
    for c in &completions {
        // Token is not directly tied to layer_idx here, but every completion
        // means a prefetch was submitted. We verify count: resident layer 1
        // must have been skipped, so only layer 2 (non-resident) should appear.
        let _ = c;
    }
    // Layer 1 is resident → skipped. Layer 2 is not → prefetched.
    // Expected: 1 completion for layer 2 only (lookahead=2 from layer 0 = targets 1,2).
    let non_resident_targets: Vec<u32> = (1u32..=2)
        .filter(|idx| !resident_set.contains(idx))
        .collect();
    assert_eq!(
        completions.len(),
        non_resident_targets.len(),
        "expected {} prefetch(es) for non-resident targets, got {}",
        non_resident_targets.len(),
        completions.len()
    );
}

// T7-3: is_resident matches actual residency
#[test]
fn is_resident_matches_residency() {
    let mut hotset = HotSet::new(4);
    let mut scale_table = PerLayerScaleTable::new(NUM_LAYERS as usize, 0.05);

    hotset.seed_static(NUM_LAYERS, 1, &[], &mut scale_table);
    let residents = hotset.resident_layers();

    for layer in 0..NUM_LAYERS {
        let expected = residents.contains(&layer);
        assert_eq!(
            hotset.is_resident(layer),
            expected,
            "is_resident({layer}) mismatch"
        );
        assert_eq!(
            scale_table.is_resident(layer as usize),
            expected,
            "scale_table.is_resident({layer}) mismatch"
        );
    }
}

// T7-4: LFU promotion when headroom available
#[test]
fn lfu_promotion_with_headroom() {
    let mut hotset = HotSet::new(4);
    let mut scale_table = PerLayerScaleTable::new(NUM_LAYERS as usize, 0.05);

    // Seed 1 layer (layer 0). Capacity = 4, so there is room.
    hotset.seed_static(NUM_LAYERS, 0, &[], &mut scale_table);

    // Access layer 5 with plenty of headroom → should be promoted.
    let large_headroom: u64 = 10 * 1024 * 1024 * 1024; // 10 GiB
    let threshold: u64 = 1 * 1024 * 1024 * 1024;       // 1 GiB

    hotset.record_access(5, large_headroom, threshold, &mut scale_table);
    assert!(hotset.is_resident(5), "layer 5 should be promoted with headroom");
    assert!(scale_table.is_resident(5), "scale_table should reflect promotion");
}

// T7-5: LFU eviction when hotset full and higher-count layer arrives
#[test]
fn lfu_eviction_when_full() {
    let mut hotset = HotSet::new(2);
    let mut scale_table = PerLayerScaleTable::new(NUM_LAYERS as usize, 0.05);

    let large_headroom: u64 = 10 * 1024 * 1024 * 1024;
    let threshold: u64 = 1;

    // Fill hotset: layer 3 (1 access), layer 4 (1 access).
    hotset.record_access(3, large_headroom, threshold, &mut scale_table);
    hotset.record_access(4, large_headroom, threshold, &mut scale_table);
    assert_eq!(hotset.len(), 2);

    // Layer 3 gets 1 more access → count=2.
    hotset.record_access(3, large_headroom, threshold, &mut scale_table);

    // Layer 7 arrives with 3 accesses (promote 3 times to get count=3).
    hotset.record_access(7, large_headroom, threshold, &mut scale_table);
    hotset.record_access(7, large_headroom, threshold, &mut scale_table);
    hotset.record_access(7, large_headroom, threshold, &mut scale_table);

    // Layer 7 (count=3) > layer 4 (count=1) → layer 4 evicted, layer 7 promoted.
    assert!(hotset.is_resident(7), "layer 7 should be promoted");
    assert!(!hotset.is_resident(4), "layer 4 (LFU) should be evicted");
    assert!(!scale_table.is_resident(4), "scale_table should reflect eviction of layer 4");
    assert!(scale_table.is_resident(7), "scale_table should reflect promotion of layer 7");
}

// T7-6: seed_static marks layers resident in scale_table
#[test]
fn seed_static_marks_scale_table_resident() {
    let mut hotset = HotSet::new(8);
    let mut scale_table = PerLayerScaleTable::new(NUM_LAYERS as usize, 0.05);

    hotset.seed_static(NUM_LAYERS, 2, &[6, 7], &mut scale_table);

    // All entries in hotset must be resident in scale_table.
    for layer in hotset.resident_layers() {
        assert!(
            scale_table.is_resident(layer as usize),
            "scale_table not resident for seeded layer {layer}"
        );
    }
}
