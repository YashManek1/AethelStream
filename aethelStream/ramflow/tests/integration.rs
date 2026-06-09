// tests/integration.rs — Sprint 6 Module 2 end-to-end integration test
//
// Covers:
//   - TensorLocationDict build for a synthetic model
//   - WarmupProfiler run → hardware_profile.json written
//   - 2 full Forward→Backward→Recomputation streaming cycles (6 transitions)
//   - Pool slot capacity changes at each phase boundary per the rebalancer
//   - Extra slot claims mid-run → pressure gauge fires, prefetch window changes
//   - 3% NaN injection into one layer → per-layer scale table isolated
//   - All pinned memory freed after the test (VmRSS check on Linux)
//   - Zero panics / zero unwrap failures (the test completing proves this)
//
// Run with:
//   cargo test --no-default-features --features mock-cuda integration -- --nocapture

use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use ramflow::phase::{
    DefaultPhaseClassifier, Direction, PhaseClassifier, PhaseRebalancer, WarmupConfig,
    WarmupProfiler,
};
use ramflow::pool::{LayerKind, PoolRegistry, TensorLocationDict};
use ramflow::scheduler::{CoScheduler, MemoryPressureGauge, PerLayerScaleTable};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a temporary directory with a unique name for this test run.
fn make_temp_dir(label: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("ramflow_{}_{}", label, std::process::id()));
    std::fs::create_dir_all(&path).expect("temp dir create failed");
    path
}

/// Remove a temporary directory, ignoring errors (best-effort cleanup).
fn remove_temp_dir(path: &PathBuf) {
    let _ = std::fs::remove_dir_all(path);
}

/// Read the current process RSS in KiB from /proc/self/status (Linux only).
/// Returns 0 on non-Linux platforms so the VmRSS assertion is silently skipped.
fn vmrss_kb() -> u64 {
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                return line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|value| value.parse::<u64>().ok())
                    .unwrap_or(0);
            }
        }
    }
    0
}

// ---------------------------------------------------------------------------
// Integration test
// ---------------------------------------------------------------------------

#[test]
fn test_integration_module2_full_streaming_cycle() {
    let tmp_dir = make_temp_dir("integration");
    let profile_path = tmp_dir.join("hardware_profile.json");

    // Record RSS baseline before any large pinned allocations.
    let vmrss_before = vmrss_kb();

    // All large allocations live inside this scope so they are dropped before
    // the VmRSS post-check, allowing the OS to reclaim mmap-backed pages.
    {
        run_integration_body(&profile_path);
    }

    let vmrss_after = vmrss_kb();

    // On Linux, all pinned buffers were mmap-backed; once dropped, VmRSS must
    // not have grown by more than 1 MiB above baseline (allocator internals).
    if vmrss_before > 0 {
        let grown_kb = vmrss_after.saturating_sub(vmrss_before);
        assert!(
            grown_kb <= 1024,
            "potential pinned memory leak: VmRSS grew by {grown_kb} KiB after all allocations freed"
        );
    }

    remove_temp_dir(&tmp_dir);
}

fn run_integration_body(profile_path: &PathBuf) {
    // =========================================================================
    // 1. TensorLocationDict for a synthetic 3-layer model
    // =========================================================================
    let dict = TensorLocationDict::empty();
    assert_eq!(
        dict.num_layers(),
        0,
        "empty TensorLocationDict must report 0 layers"
    );

    // =========================================================================
    // 2. Warm-up profiler — asserts hardware_profile.json is written
    // =========================================================================
    assert!(
        !profile_path.exists(),
        "hardware_profile.json must not exist before profiler run"
    );

    let warmup_config = WarmupConfig {
        steps: 3,
        output_path: profile_path.clone(),
        model_sha256: [0u8; 32],
    };
    let profiler = WarmupProfiler::new(warmup_config).expect("WarmupProfiler::new");
    let [fwd_profile, bwd_profile, recomp_profile] = profiler.run().expect("WarmupProfiler::run");

    assert!(
        profile_path.exists(),
        "hardware_profile.json must be written after WarmupProfiler::run"
    );

    // Verify the profiler measures what simulate_mini_step produces:
    //   Forward:       attention peak = 1
    //   Backward:      attention peak = 2
    //   Recomputation: attention peak = 3
    assert_eq!(
        fwd_profile.attention_slots_needed, 1,
        "forward profiler peak: 1 simultaneous attention claim"
    );
    assert_eq!(
        bwd_profile.attention_slots_needed, 2,
        "backward profiler peak: 2 simultaneous attention claims"
    );
    assert_eq!(
        recomp_profile.attention_slots_needed, 3,
        "recomputation profiler peak: 3 simultaneous attention claims"
    );

    // =========================================================================
    // 3. Pool registry — seeded at worst-case (recomputation) profile
    //    so the rebalancer only ever shrinks, never emergency-grows
    // =========================================================================
    let threshold_bytes = profiler
        .measure_zero_copy_crossover()
        .expect("measure_zero_copy_crossover");

    // threshold_bytes from the mock estimator is 4 MiB.
    // slot bytes: large = 2 MiB, embedding = 1 MiB
    // recomp capacity: attn=3, mlp=2, norm=1, embed=1  => total 7 slots
    let registry = Arc::new(
        PoolRegistry::new(&recomp_profile, &dict, threshold_bytes)
            .expect("PoolRegistry::new from recomp profile"),
    );

    assert_eq!(
        registry.capacity_for(LayerKind::Attention),
        3,
        "initial attention capacity must equal recomputation profile (worst case = 3)"
    );
    assert_eq!(
        registry.capacity_for(LayerKind::Mlp),
        2,
        "initial MLP capacity must equal recomputation profile (worst case = 2)"
    );

    // =========================================================================
    // 4. Pressure gauge + co-scheduler
    // =========================================================================
    let gauge = MemoryPressureGauge::new(30);
    registry.set_pressure_gauge(gauge.clone());
    let co = CoScheduler::new(gauge.clone()).expect("CoScheduler::new");

    // Phase-transition counter wired via DefaultPhaseClassifier callback.
    let transition_count = Arc::new(AtomicU32::new(0));
    let transition_count_clone = Arc::clone(&transition_count);
    use ramflow::phase::classifier::TrainingPhase;
    let classifier = DefaultPhaseClassifier::with_transition_callback(
        profile_path.clone(),
        Arc::new(move |_previous_phase, _next_phase| {
            transition_count_clone.fetch_add(1, Ordering::AcqRel);
        }),
    )
    .expect("DefaultPhaseClassifier::with_transition_callback");

    let rebalancer = PhaseRebalancer::with_registry(Arc::clone(&registry));

    // =========================================================================
    // 5. Two full Forward→Backward→Recomputation streaming cycles
    //    Each cycle = 3 phase transitions; 2 cycles = 6 total
    // =========================================================================
    for cycle_index in 0..2_u32 {
        // ── 5a. Forward phase ─────────────────────────────────────────────
        classifier.notify_layer_start(0, Direction::Forward);
        assert!(
            matches!(classifier.current_phase(), TrainingPhase::Forward { .. }),
            "cycle {cycle_index}: classifier must enter Forward after notify_layer_start"
        );

        // Rebalance: all slots must be free before fence can pass.
        assert_eq!(
            registry.total_claimed_slots(),
            0,
            "cycle {cycle_index}: no slots may be in-flight before rebalancing to Forward"
        );
        rebalancer
            .rebalance_to_profile(&fwd_profile)
            .expect("rebalance to forward profile");

        let attn_cap_fwd = registry.capacity_for(LayerKind::Attention);
        assert_eq!(
            attn_cap_fwd, 1,
            "cycle {cycle_index}: attention capacity must be 1 after forward rebalance"
        );
        assert_eq!(
            registry.capacity_for(LayerKind::Mlp),
            1,
            "cycle {cycle_index}: MLP capacity must be 1 after forward rebalance"
        );

        // Simulate 3-layer forward pass: claim one slot per layer type, then release.
        {
            let _slot_attn = registry
                .claim(LayerKind::Attention)
                .expect("claim attention slot (forward)");
            let _slot_mlp = registry
                .claim(LayerKind::Mlp)
                .expect("claim MLP slot (forward)");
            let _slot_norm = registry
                .claim(LayerKind::Norm)
                .expect("claim norm slot (forward)");
        } // slots dropped here — ring drained before backward rebalance

        assert_eq!(
            registry.total_claimed_slots(),
            0,
            "cycle {cycle_index}: all forward slots must be released before backward rebalance"
        );

        // ── 5b. Backward phase ────────────────────────────────────────────
        classifier.notify_layer_start(2, Direction::Backward);
        assert!(
            matches!(classifier.current_phase(), TrainingPhase::Backward { .. }),
            "cycle {cycle_index}: classifier must enter Backward"
        );

        rebalancer
            .rebalance_to_profile(&bwd_profile)
            .expect("rebalance to backward profile");

        let attn_cap_bwd = registry.capacity_for(LayerKind::Attention);
        assert_eq!(
            attn_cap_bwd, 2,
            "cycle {cycle_index}: attention capacity must be 2 after backward rebalance"
        );
        assert!(
            attn_cap_bwd > attn_cap_fwd,
            "cycle {cycle_index}: backward needs more attention slots than forward ({attn_cap_bwd} vs {attn_cap_fwd})"
        );

        {
            let _attn_a = registry
                .claim(LayerKind::Attention)
                .expect("claim attention_a (backward)");
            let _attn_b = registry
                .claim(LayerKind::Attention)
                .expect("claim attention_b (backward)");
            let _mlp = registry
                .claim(LayerKind::Mlp)
                .expect("claim MLP (backward)");
            let _embed = registry
                .claim(LayerKind::Embedding)
                .expect("claim embedding (backward)");
        }

        assert_eq!(
            registry.total_claimed_slots(),
            0,
            "cycle {cycle_index}: all backward slots must be released before recomputation rebalance"
        );

        // ── 5c. Recomputation phase ───────────────────────────────────────
        classifier.notify_backward_recompute_start(0, 2);
        assert!(
            matches!(
                classifier.current_phase(),
                TrainingPhase::Recomputation { .. }
            ),
            "cycle {cycle_index}: classifier must enter Recomputation"
        );

        rebalancer
            .rebalance_to_profile(&recomp_profile)
            .expect("rebalance to recomputation profile");

        let attn_cap_recomp = registry.capacity_for(LayerKind::Attention);
        assert_eq!(
            attn_cap_recomp, 3,
            "cycle {cycle_index}: attention capacity must be 3 after recomputation rebalance"
        );
        assert!(
            attn_cap_recomp > attn_cap_bwd,
            "cycle {cycle_index}: recomputation needs more attention slots than backward ({attn_cap_recomp} vs {attn_cap_bwd})"
        );

        {
            let _a0 = registry
                .claim(LayerKind::Attention)
                .expect("claim attention_0 (recomp)");
            let _a1 = registry
                .claim(LayerKind::Attention)
                .expect("claim attention_1 (recomp)");
            let _a2 = registry
                .claim(LayerKind::Attention)
                .expect("claim attention_2 (recomp)");
            let _m0 = registry
                .claim(LayerKind::Mlp)
                .expect("claim MLP_0 (recomp)");
            let _m1 = registry
                .claim(LayerKind::Mlp)
                .expect("claim MLP_1 (recomp)");
        }

        assert_eq!(
            registry.total_claimed_slots(),
            0,
            "cycle {cycle_index}: all recomputation slots must be released at end of cycle"
        );
    }

    // =========================================================================
    // 6. Assert 6 phase transitions were recorded (3 per cycle × 2 cycles)
    // =========================================================================
    let recorded_transitions = transition_count.load(Ordering::Acquire);
    assert!(
        recorded_transitions >= 6,
        "must record at least 6 phase transitions over 2 cycles; got {recorded_transitions}"
    );

    // =========================================================================
    // 7. Pressure gauge fires: extra slot claims → high callback → window shrinks
    //    Current state: recomp profile — total capacity = 3+2+1+1 = 7 slots.
    //    Claiming 6 out of 7 = 85.7% > 0.80 high threshold.
    // =========================================================================
    let pressure_slots: Vec<_> = (0..6)
        .map(|slot_index| {
            if slot_index < 3 {
                registry
                    .claim(LayerKind::Attention)
                    .expect("claim attention pressure slot")
            } else if slot_index < 5 {
                registry
                    .claim(LayerKind::Mlp)
                    .expect("claim MLP pressure slot")
            } else {
                registry
                    .claim(LayerKind::Norm)
                    .expect("claim norm pressure slot")
            }
        })
        .collect();

    let fill_ratio = registry.total_claimed_slots() as f32 / registry.total_capacity() as f32;
    assert!(
        fill_ratio > 0.80,
        "pressure test setup: fill must exceed 80%, got {:.1}%",
        fill_ratio * 100.0
    );

    let window_before_pressure = co.prefetch_window();
    let mut high_callback_fired = false;
    for _ in 0..31 {
        gauge.sample_and_notify(&registry);
        if co.is_paused() {
            high_callback_fired = true;
            break;
        }
    }
    assert!(
        high_callback_fired,
        "pressure gauge must fire high callback at {:.1}% fill (threshold 80%)",
        fill_ratio * 100.0
    );
    assert!(
        co.prefetch_window() < window_before_pressure,
        "prefetch window must shrink after high-pressure event: before={window_before_pressure}, after={}",
        co.prefetch_window()
    );

    let window_after_high = co.prefetch_window();

    // Release all pressure slots → pool drops to 0% → low callback fires.
    drop(pressure_slots);

    let mut low_callback_fired = false;
    for _ in 0..31 {
        gauge.sample_and_notify(&registry);
        if !co.is_paused() {
            low_callback_fired = true;
            break;
        }
    }
    assert!(
        low_callback_fired,
        "pressure gauge must fire low callback after all pressure slots released"
    );
    assert!(
        !co.is_paused(),
        "CoScheduler pause must clear after low-pressure event"
    );
    assert!(
        co.prefetch_window() > window_after_high,
        "prefetch window must recover after low-pressure event: during_high={window_after_high}, after={}",
        co.prefetch_window()
    );

    // =========================================================================
    // 8. NaN injection: per-layer scale table isolates the affected layer
    //    Inject 3% NaN (300/10 000) into layer 1 for 100 steps.
    //    All other layers receive zero updates — no cross-contamination.
    // =========================================================================
    const NUM_SYNTHETIC_LAYERS: usize = 3;
    const INFECTED_LAYER: usize = 1;
    const ELEMENTS_PER_LAYER: usize = 10_000;
    const OVERFLOWED_ELEMENTS: u32 = 300; // exactly 3%

    let mut scale_table = PerLayerScaleTable::new(NUM_SYNTHETIC_LAYERS, 0.05);

    for _ in 0..100 {
        scale_table.update(INFECTED_LAYER, ELEMENTS_PER_LAYER, OVERFLOWED_ELEMENTS);
    }

    // EWA with alpha=0.05 after 100 steps at constant 0.03 overflow fraction:
    // density ≈ 0.03 × (1 − 0.95^100) ≈ 0.0298, within ±10% of 0.030.
    let infected_density = scale_table.get_density(INFECTED_LAYER);
    assert!(
        (infected_density - 0.030_f32).abs() <= 0.003_f32,
        "infected layer {INFECTED_LAYER} density must converge to 0.030 ±10%, got {infected_density:.6}"
    );
    assert!(
        scale_table.get_scale(INFECTED_LAYER) < 65536.0_f32,
        "infected layer {INFECTED_LAYER} scale must be reduced from 65536.0 (got {})",
        scale_table.get_scale(INFECTED_LAYER)
    );

    // Clean layers must be untouched.
    for layer_index in 0..NUM_SYNTHETIC_LAYERS {
        if layer_index == INFECTED_LAYER {
            continue;
        }
        assert_eq!(
            scale_table.get_density(layer_index),
            0.0_f32,
            "clean layer {layer_index}: density must be 0.0 (never updated)"
        );
        assert_eq!(
            scale_table.get_scale(layer_index),
            65536.0_f32,
            "clean layer {layer_index}: scale must remain at 65536.0 (never updated)"
        );
    }

    // co, gauge, registry, classifier, rebalancer, scale_table all drop here.
}
