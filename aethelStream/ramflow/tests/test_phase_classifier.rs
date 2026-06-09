// tests/test_phase_classifier.rs — Sprint 3A integration tests for phase/classifier.rs
//
// These tests drive the TDD loop for TierClassifier and DefaultPhaseClassifier.
// Each test is numbered to match the sprint task list.

use ramflow::phase::classifier::DefaultPhaseClassifier;
use ramflow::phase::{
    Direction, PhaseClassifier, PhaseMemoryProfile, PhaseRebalancer, Tier, TierClassifier,
    TrainingPhase,
};
use ramflow::pool::{LayerKind, PoolRegistry, TensorLocationDict};
use ramflow::RamFlowError;
use std::path::PathBuf;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Test 1 — TierClassifier: unseen tensor → Cold
// ---------------------------------------------------------------------------
#[test]
fn test01_tier_unseen_tensor_is_cold() {
    let classifier = TierClassifier::new();
    assert_eq!(classifier.classify(42), Tier::Cold);
}

#[test]
fn test02_default_phase_classifier_forward_backward_recompute_transitions() {
    let classifier =
        DefaultPhaseClassifier::new(PathBuf::from("hardware_profile.json")).expect("classifier");

    assert_eq!(
        classifier.current_phase(),
        TrainingPhase::Forward {
            layers_in_flight: 0
        }
    );

    classifier.notify_layer_start(0, Direction::Forward);
    assert_eq!(
        classifier.current_phase(),
        TrainingPhase::Forward {
            layers_in_flight: 1
        }
    );

    classifier.notify_layer_start(9, Direction::Forward);
    assert_eq!(
        classifier.current_phase(),
        TrainingPhase::Forward {
            layers_in_flight: 2
        },
        "forward phase should cap simultaneous streaming layers at current+prefetch"
    );

    classifier.notify_layer_start(9, Direction::Backward);
    assert_eq!(
        classifier.current_phase(),
        TrainingPhase::Backward {
            checkpoint_interval: 1
        }
    );

    classifier.notify_backward_recompute_start(4, 7);
    assert_eq!(
        classifier.current_phase(),
        TrainingPhase::Recomputation {
            window_start: 4,
            window_end: 7
        }
    );
}

#[test]
fn test03_rebalancer_fence_times_out_when_slot_is_held() {
    std::env::set_var("RAMFLOW_PHASE_FENCE_TIMEOUT_MS", "5");

    let forward_profile = profile_for_phase(
        TrainingPhase::Forward {
            layers_in_flight: 1,
        },
        1,
    );
    let registry = Arc::new(
        PoolRegistry::new(&forward_profile, &TensorLocationDict::empty(), 1024).expect("registry"),
    );
    let held_slot = registry
        .claim(LayerKind::Attention)
        .expect("claim held slot");
    let rebalancer = PhaseRebalancer::with_registry(Arc::clone(&registry));
    let recompute_profile = profile_for_phase(
        TrainingPhase::Recomputation {
            window_start: 0,
            window_end: 2,
        },
        recompute_attention_slots(0, 2),
    );

    let result = rebalancer.rebalance_to_profile(&recompute_profile);
    std::env::remove_var("RAMFLOW_PHASE_FENCE_TIMEOUT_MS");
    drop(held_slot);

    assert!(
        matches!(result, Err(RamFlowError::PhaseTransitionError(ref message)) if message.contains("phase fence timed out")),
        "expected phase fence timeout, got {result:?}"
    );
}

#[test]
fn test04_recompute_window_boundaries_drive_inclusive_slot_counts() {
    assert_eq!(
        recompute_attention_slots(3, 3),
        2,
        "single-layer recompute still needs current layer + prefetch slot"
    );
    assert_eq!(
        recompute_attention_slots(3, 5),
        4,
        "inclusive [start,end] window should need window_len + 1 prefetch slot"
    );

    let single_layer_profile = profile_for_phase(
        TrainingPhase::Recomputation {
            window_start: 3,
            window_end: 3,
        },
        recompute_attention_slots(3, 3),
    );
    let single_layer_registry =
        PoolRegistry::new(&single_layer_profile, &TensorLocationDict::empty(), 1024)
            .expect("single-layer registry");
    assert_eq!(single_layer_registry.capacity_for(LayerKind::Attention), 2);

    let multi_layer_profile = profile_for_phase(
        TrainingPhase::Recomputation {
            window_start: 3,
            window_end: 5,
        },
        recompute_attention_slots(3, 5),
    );
    let multi_layer_registry =
        PoolRegistry::new(&multi_layer_profile, &TensorLocationDict::empty(), 1024)
            .expect("multi-layer registry");
    assert_eq!(multi_layer_registry.capacity_for(LayerKind::Attention), 4);
}

fn recompute_attention_slots(window_start: u32, window_end: u32) -> u32 {
    window_end.saturating_sub(window_start).saturating_add(2)
}

fn profile_for_phase(phase: TrainingPhase, attention_slots_needed: u32) -> PhaseMemoryProfile {
    PhaseMemoryProfile {
        phase,
        expected_peak_bytes: 0,
        attention_slots_needed,
        mlp_slots_needed: 1,
        norm_slots_needed: 1,
        optimizer_slots_needed: 1,
    }
}
