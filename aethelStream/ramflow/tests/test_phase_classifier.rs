// tests/test_phase_classifier.rs — Sprint 3A integration tests for phase/classifier.rs
//
// These tests drive the TDD loop for TierClassifier and DefaultPhaseClassifier.
// Each test is numbered to match the sprint task list.

use ramflow::phase::classifier::DefaultPhaseClassifier;
use ramflow::phase::{Direction, PhaseClassifier, Tier, TierClassifier, TrainingPhase};
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Test 1 — TierClassifier: unseen tensor → Cold
// ---------------------------------------------------------------------------
#[test]
fn test01_tier_unseen_tensor_is_cold() {
    let classifier = TierClassifier::new();
    assert_eq!(classifier.classify(42), Tier::Cold);
}
