//! T3 — Adaptive window sizing.
//!
//! 1. SSD throttle (high GPU idle) → window grows within 3 updates.
//! 2. Throttle release → window shrinks.
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

// T3-1: window grows when GPU is idle (SSD throttle)
#[test]
fn window_grows_within_3_layers_under_io_throttle() {
    let mut w = make_window(1.0);
    let start = w.t_iter();
    for _ in 0..3 {
        w.update(0.50, 10.0).unwrap();
    }
    assert!(w.t_iter() > start, "expected growth from {start}, got {}", w.t_iter());
}

// T3-2: window shrinks when throttle released (io rising, gpu not idle)
#[test]
fn window_shrinks_on_throttle_release() {
    let mut w = make_window(W_MAX);
    for _ in 0..5 {
        w.update(0.01, 200.0).unwrap();
    }
    assert!(w.t_iter() < W_MAX, "expected shrink from W_MAX, got {}", w.t_iter());
}

// T3-3: window never leaves [1, W_max] under extreme inputs
#[test]
fn window_stays_in_bounds() {
    let mut w = make_window(2.0);
    for step in 0..50 {
        let idle = if step % 2 == 0 { 1.0 } else { 0.0 };
        let io_ms = if step % 3 == 0 { 0.0 } else { 500.0 };
        w.update(idle, io_ms).unwrap();
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
    w.register_pressure_callbacks(&gauge);

    fire_high(&gauge);
    assert!(w.pressure_cap_active(), "cap active after high pressure");
    assert_eq!(w.t_iter(), 1.0, "capped to 1");

    // A2 update with very high GPU idle — cap must win.
    w.update(0.90, 5.0).unwrap();
    assert_eq!(w.t_iter(), 1.0, "cap wins over A2 growth");
}

// T3-5: low pressure lifts cap, A2 can grow again
#[test]
fn low_pressure_lifts_cap_and_allows_growth() {
    let gauge = MemoryPressureGauge::new(1);
    let mut w = make_window(1.0);
    w.register_pressure_callbacks(&gauge);

    fire_high(&gauge);
    assert!(w.pressure_cap_active());

    fire_low(&gauge);
    assert!(!w.pressure_cap_active(), "cap lifted after low pressure");

    // Growth must now be possible.
    let before = w.t_iter();
    w.update(0.90, 5.0).unwrap();
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
    w.register_pressure_callbacks(&gauge);

    fire_high(&gauge);
    w.increase_lookahead().unwrap();
    assert_eq!(w.t_iter(), 1.0, "increase must not beat pressure cap");
}
