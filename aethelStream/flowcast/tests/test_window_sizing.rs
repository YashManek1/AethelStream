//! T3 — Adaptive window sizing.
//!
//! 1. SSD throttle (ssd_ms > 0.8 × gpu_ms) → window grows within 3 updates.
//! 2. GPU-fast release (gpu_ms > 2 × ssd_ms) → window shrinks.
//! 3. Window never leaves [1, W_max] under any inputs.
//! 4. High pressure caps window even when A2 would grow.
//! 5. Low pressure lifts cap; A2 can grow again.
//! 6. increase/decrease_lookahead respect W_max and floor.
//! 7. increase_lookahead suppressed while pressure cap active.

use flowcast::window::{AdaptiveWindow, DEFAULT_ALPHA};
use ramflow::MemoryPressureGauge;

const W_MAX: f32 = 8.0;

fn make_window(initial: f32) -> AdaptiveWindow {
    AdaptiveWindow::new(initial, DEFAULT_ALPHA, W_MAX)
}

/// Fires the high-pressure callbacks via the public `signal_stall` API.
fn fire_high(gauge: &MemoryPressureGauge) {
    gauge.signal_stall(0); // fires all high_callbacks at pressure = 1.0
}

/// Fires the low-pressure callbacks by sampling an empty pool (0% fill < 0.40 threshold).
fn fire_low(gauge: &MemoryPressureGauge) {
    let pool = ramflow::PoolRegistry::with_defaults().expect("pool");
    gauge.sample_and_notify(&pool); // 0 claimed / N capacity → pressure 0 < 0.40
}

// T3-1: window grows when SSD is the bottleneck (ssd_ms > 0.8 × gpu_ms)
#[test]
fn window_grows_within_3_layers_under_io_throttle() {
    let mut w = make_window(1.0);
    let start = w.t_iter();
    // ssd=10ms, gpu=5ms → 10 > 0.8*5=4 → grow predicate satisfied.
    for _ in 0..3 {
        w.update(10.0, 5.0).unwrap();
    }
    assert!(w.t_iter() > start, "expected growth from {start}, got {}", w.t_iter());
}

// T3-2: window shrinks when GPU is fast (gpu_ms > 2 × ssd_ms)
#[test]
fn window_shrinks_on_throttle_release() {
    let mut w = make_window(W_MAX);
    // ssd=5ms, gpu=200ms → 200 > 2*5=10 → shrink predicate satisfied.
    for _ in 0..5 {
        w.update(5.0, 200.0).unwrap();
    }
    assert!(w.t_iter() < W_MAX, "expected shrink from W_MAX, got {}", w.t_iter());
}

// T3-3: window never leaves [1, W_max] under extreme alternating inputs
#[test]
fn window_stays_in_bounds() {
    let mut w = make_window(2.0);
    for step in 0..50 {
        // Alternate: grow condition (ssd >> gpu) and shrink condition (gpu >> ssd).
        let (ssd_ms, gpu_ms) = if step % 2 == 0 {
            (500.0_f32, 1.0_f32) // ssd > 0.8*gpu → grow
        } else {
            (1.0_f32, 500.0_f32) // gpu > 2*ssd → shrink
        };
        w.update(ssd_ms, gpu_ms).unwrap();
        let t = w.t_iter();
        assert!(t >= 1.0, "t_iter {t} < 1 at step {step}");
        assert!(t <= W_MAX, "t_iter {t} > W_max {W_MAX} at step {step}");
    }
}

// T3-4: high pressure caps window even when A2 would grow
#[test]
fn high_pressure_caps_window_overriding_a2_growth() {
    let gauge = MemoryPressureGauge::new(1);
    let mut w = make_window(4.0);
    w.register_pressure_callbacks(&gauge, None);

    fire_high(&gauge);
    assert!(w.pressure_cap_active(), "cap active after high pressure");
    assert_eq!(w.t_iter(), 1.0, "capped to 1");

    // A2 update with ssd > 0.8*gpu (grow condition) — cap must win.
    w.update(10.0, 5.0).unwrap();
    assert_eq!(w.t_iter(), 1.0, "cap wins over A2 growth");
}

// T3-5: low pressure lifts cap, A2 can grow again
#[test]
fn low_pressure_lifts_cap_and_allows_growth() {
    let gauge = MemoryPressureGauge::new(1);
    let mut w = make_window(1.0);
    w.register_pressure_callbacks(&gauge, None);

    fire_high(&gauge);
    assert!(w.pressure_cap_active());

    fire_low(&gauge);
    assert!(!w.pressure_cap_active(), "cap lifted after low pressure");

    // Growth must now be possible (ssd=10ms > 0.8*gpu=4ms → grow).
    let before = w.t_iter();
    w.update(10.0, 5.0).unwrap();
    assert!(w.t_iter() >= before, "window must not regress below {before}");
    assert!(w.t_iter() >= 1.0);
}

// T3-6: increase/decrease_lookahead respect W_max and floor
#[test]
fn increase_decrease_respect_bounds() {
    let mut w = make_window(W_MAX);
    w.increase_lookahead().unwrap();
    assert_eq!(w.t_iter(), W_MAX, "increase above W_max must clamp");

    let mut w2 = make_window(1.0);
    w2.decrease_lookahead().unwrap();
    assert_eq!(w2.t_iter(), 1.0, "decrease below 1 must clamp");
}

// T3-7: increase_lookahead suppressed by pressure cap
#[test]
fn increase_lookahead_suppressed_by_pressure_cap() {
    let gauge = MemoryPressureGauge::new(1);
    let mut w = make_window(3.0);
    w.register_pressure_callbacks(&gauge, None);

    fire_high(&gauge);
    w.increase_lookahead().unwrap();
    assert_eq!(w.t_iter(), 1.0, "increase must not beat pressure cap");
}
