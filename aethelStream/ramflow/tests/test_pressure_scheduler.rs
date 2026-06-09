// tests/test_pressure_scheduler.rs — Sprint 4 Module 2 Tests 8 and 9
//
// Run with:
//   cargo test --no-default-features --features mock-cuda test_pressure_scheduler -- --nocapture
//
// Test 8: MemoryPressureGauge + CoScheduler high/low pressure callbacks.
//   Claim 85% of pool slots; within 31 sample calls the high-pressure callback
//   fires and CoScheduler pauses prefetch (window shrinks, pause_signal = true).
//   Release all slots; low-pressure callback fires, window grows, pause clears.
//
// Test 9: PerLayerScaleTable EWA density convergence and scale adjustment.
//   Layer 5 receives exactly 3% NaN over 100 update steps.
//   density[5] converges to ~0.030 +/-10%.  scale[5] < 65536.
//   All other layers (0..4, 6..79) remain at density = 0.0, scale = 65536.0
//   — no cross-contamination.

use ramflow::pool::{LayerKind, PoolRegistry};
use ramflow::scheduler::{CoScheduler, MemoryPressureGauge, PerLayerScaleTable};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Test 8 — high/low pressure triggers CoScheduler callback
// ---------------------------------------------------------------------------

#[test]
fn test_8_high_pressure_triggers_coscheduler_pause_and_window_shrink() {
    let registry = Arc::new(PoolRegistry::with_defaults().expect("pool registry init"));

    let total_capacity = registry.total_capacity();
    // Target 85% fill — enough to exceed the 0.80 high threshold.
    let target_claims = ((total_capacity as f32) * 0.85).ceil() as usize;

    let gauge = MemoryPressureGauge::new(30);
    // CoScheduler::new registers high and low callbacks on the *shared* gauge
    // inner state (both gauge and co.gauge hold Arc to the same GaugeInner).
    let co = CoScheduler::new(gauge.clone()).expect("coscheduler new");

    // ── Fill pool to ≥ 85% ────────────────────────────────────────────────
    let kinds = [
        LayerKind::Attention,
        LayerKind::Mlp,
        LayerKind::Norm,
        LayerKind::Embedding,
    ];
    let mut claims = Vec::new();
    'fill: for kind in kinds {
        let ring_cap = registry.capacity_for(kind);
        for _ in 0..ring_cap {
            if claims.len() >= target_claims {
                break 'fill;
            }
            match registry.claim(kind) {
                Ok(slot) => claims.push(slot),
                Err(_) => break,
            }
        }
    }

    let actual_fill = registry.total_claimed_slots() as f32 / total_capacity as f32;
    assert!(
        actual_fill > 0.80,
        "test setup: fill must exceed 80%, got {:.1}%",
        actual_fill * 100.0
    );

    // ── High-pressure event must fire within 31 sample calls ──────────────
    let max_steps: usize = 31;
    let mut high_fired = false;
    for _ in 0..max_steps {
        gauge.sample_and_notify(&registry);
        if co.is_paused() {
            high_fired = true;
            break;
        }
    }

    assert!(
        high_fired,
        "high-pressure callback must fire within {max_steps} steps at {:.1}% fill",
        actual_fill * 100.0
    );
    let initial_window = 2_i32; // CoScheduler initialises prefetch_window to 2.
    assert!(
        co.prefetch_window() < initial_window,
        "prefetch window must shrink below {initial_window}, got {}",
        co.prefetch_window()
    );

    // ── Release all slots — pool drops to 0% (below 0.40 low threshold) ──
    drop(claims);

    let window_after_high = co.prefetch_window();
    let mut low_fired = false;
    for _ in 0..max_steps {
        gauge.sample_and_notify(&registry);
        if !co.is_paused() {
            low_fired = true;
            break;
        }
    }

    assert!(
        low_fired,
        "low-pressure callback must fire within {max_steps} steps after pool is empty"
    );
    assert!(!co.is_paused(), "pause signal must be cleared after low-pressure event");
    assert!(
        co.prefetch_window() > window_after_high,
        "prefetch window must grow after low-pressure event: before={window_after_high}, after={}",
        co.prefetch_window()
    );
}

// ---------------------------------------------------------------------------
// Test 9 — PerLayerScaleTable EWA convergence + no cross-contamination
// ---------------------------------------------------------------------------

#[test]
fn test_9_per_layer_scale_table_no_cross_contamination() {
    const NUM_LAYERS: usize = 80;
    const TARGET_LAYER: usize = 5;
    const N_TOTAL: usize = 10_000;
    // Exactly 3% NaN: 300 out of 10 000 elements.
    const N_OVERFLOW: u32 = 300;
    const STEPS: usize = 100;

    let mut table = PerLayerScaleTable::new(NUM_LAYERS, 0.05);

    for _ in 0..STEPS {
        table.update(TARGET_LAYER, N_TOTAL, N_OVERFLOW);
    }

    // EWA with alpha=0.05 and constant fraction=0.03 after 100 steps:
    //   d = 0.03 * (1 - 0.95^100) ~= 0.0298  (within +/-10% of 0.03)
    let density = table.get_density(TARGET_LAYER);
    assert!(
        (density - 0.03_f32).abs() <= 0.003_f32,
        "density[{TARGET_LAYER}] must converge to 0.030 +/-10%, got {density:.6}"
    );

    // density > 0.001 from step 1 -> scale is halved every step -> scale = 1.0.
    assert!(
        table.get_scale(TARGET_LAYER) < 65536.0_f32,
        "scale[{TARGET_LAYER}] must be reduced from 65536.0, got {}",
        table.get_scale(TARGET_LAYER)
    );

    // No other layer was ever updated — zero cross-contamination.
    for layer_idx in 0..NUM_LAYERS {
        if layer_idx == TARGET_LAYER {
            continue;
        }
        assert_eq!(
            table.get_density(layer_idx),
            0.0_f32,
            "density[{layer_idx}] must be 0.0 (never updated)"
        );
        assert_eq!(
            table.get_scale(layer_idx),
            65536.0_f32,
            "scale[{layer_idx}] must remain at 65536.0 (never updated)"
        );
    }
}
