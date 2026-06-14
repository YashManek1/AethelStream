// Integration tests for the SSD thermal monitor (A3-T).
//
// Only compiled on Linux with the `ssd-thermal` Cargo feature active.
#![cfg(all(target_os = "linux", feature = "ssd-thermal"))]

use flowcast::smart_monitor::{
    FixedTempSource, ThermalMonitor, ThermalState, UnavailableTempSource,
    parse_temperature_from_log,
};
use std::path::PathBuf;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Helper: temp dir that always exists (shard_dir need not have real shards)
// ---------------------------------------------------------------------------
fn dummy_shard_dir() -> PathBuf {
    std::env::temp_dir()
}

// ---------------------------------------------------------------------------
// 1. Temperature log parsing from known raw bytes
// ---------------------------------------------------------------------------

#[test]
fn test_parse_temperature_from_known_bytes() {
    // Composite Temperature at bytes 1-2: 318 K = 45 C
    let mut log = [0u8; 512];
    let kelvin: u16 = 318;
    let bytes = kelvin.to_le_bytes();
    log[1] = bytes[0];
    log[2] = bytes[1];
    let result = parse_temperature_from_log(&log);
    assert!(result.is_ok(), "expected Ok, got {result:?}");
    let celsius = result.unwrap();
    assert!((celsius - 45.0).abs() < 0.5, "expected ~45 C, got {celsius}");
}

#[test]
fn test_parse_temperature_zero_returns_unavailable() {
    // raw_kelvin == 0 means no sensor data
    let log = [0u8; 512];
    let result = parse_temperature_from_log(&log);
    assert!(result.is_err(), "expected Err for zero raw_kelvin");
}

#[test]
fn test_parse_temperature_high_value() {
    // 348 K = 75 C (throttle threshold)
    let mut log = [0u8; 512];
    let kelvin: u16 = 348;
    let bytes = kelvin.to_le_bytes();
    log[1] = bytes[0];
    log[2] = bytes[1];
    let celsius = parse_temperature_from_log(&log).unwrap();
    assert!((celsius - 75.0).abs() < 0.5, "expected ~75 C, got {celsius}");
}

// ---------------------------------------------------------------------------
// 2. ThermalState classification
// ---------------------------------------------------------------------------

#[test]
fn test_thermal_state_normal() {
    let state = ThermalState::classify(50.0, 65.0, 75.0);
    assert_eq!(state, ThermalState::Normal);
}

#[test]
fn test_thermal_state_warm_boundary() {
    let state = ThermalState::classify(65.0, 65.0, 75.0);
    assert_eq!(state, ThermalState::Warm);
}

#[test]
fn test_thermal_state_throttling_boundary() {
    let state = ThermalState::classify(75.0, 65.0, 75.0);
    assert_eq!(state, ThermalState::Throttling);
}

#[test]
fn test_thermal_state_u8_round_trip() {
    for state in [ThermalState::Normal, ThermalState::Warm, ThermalState::Throttling] {
        assert_eq!(ThermalState::from_u8(state.as_u8()), state);
    }
}

// ---------------------------------------------------------------------------
// 3. ThermalMonitor with UnavailableTempSource — tick is a no-op
// ---------------------------------------------------------------------------

#[test]
fn test_tick_no_op_on_unavailable_source() {
    let monitor = ThermalMonitor::with_source(
        Box::new(UnavailableTempSource),
        dummy_shard_dir(),
        32,
    );
    monitor.tick(5000, 5000);
    assert_eq!(monitor.reprofiling_events(), 0, "no events when SMART unavailable");
    assert!(monitor.poll_outcome().is_none());
}

// ---------------------------------------------------------------------------
// 4. Re-profiling trigger fires at the right step
// ---------------------------------------------------------------------------

#[test]
fn test_reprofile_trigger_fires_at_interval() {
    // 80 C > throttle threshold (75): re-profile should fire at step 5000.
    let monitor = ThermalMonitor::with_source(
        Box::new(FixedTempSource::new(80.0)),
        dummy_shard_dir(),
        4,  // 4 layers — small so probe_layers is fast
    );
    // Step not at interval — no reprofile.
    monitor.tick(1, 5000);
    assert_eq!(monitor.reprofiling_events(), 0);

    // Step at interval — reprofile should be spawned.
    monitor.tick(5000, 5000);
    assert_eq!(monitor.reprofiling_events(), 1, "reprofile should be spawned at step 5000");
}

// ---------------------------------------------------------------------------
// 5. W_max shrinks immediately when reprofile returns lower ceiling
// ---------------------------------------------------------------------------

#[test]
fn test_w_max_update_shrinks_immediately() {
    use flowcast::window::AdaptiveWindow;
    let window = AdaptiveWindow::new(8.0, 0.3, 10.0);
    assert!((window.t_iter() - 8.0).abs() < 0.1, "initial t_iter should be 8");
    // A reprofile finds that W_max should drop to 3 (SSD is throttling)
    window.apply_w_max_update(3);
    // Window should have been clamped down to 3.
    assert!(
        window.t_iter() <= 3.5,
        "t_iter should be clamped to new W_max, got {}",
        window.t_iter()
    );
    assert!((window.w_max() - 3.0).abs() < 0.1, "w_max should be 3.0");
}

// ---------------------------------------------------------------------------
// 6. W_max grows — EWMA rises naturally (no instant jump)
// ---------------------------------------------------------------------------

#[test]
fn test_w_max_update_does_not_jump_up() {
    use flowcast::window::AdaptiveWindow;
    let window = AdaptiveWindow::new(2.0, 0.3, 4.0);
    // Raise ceiling to 8 — the window should NOT immediately jump to 8.
    window.apply_w_max_update(8);
    assert!(
        window.t_iter() <= 4.5,
        "t_iter should not jump above old value immediately, got {}",
        window.t_iter()
    );
    assert!((window.w_max() - 8.0).abs() < 0.1, "w_max ceiling should be updated to 8.0");
}

// ---------------------------------------------------------------------------
// 7. Background reprofile completes and poll_outcome returns result
// ---------------------------------------------------------------------------

#[test]
fn test_background_reprofile_outcome_is_polled() {
    let monitor = ThermalMonitor::with_source(
        Box::new(FixedTempSource::new(76.0)),  // above throttle threshold
        dummy_shard_dir(),
        4,  // tiny model — probe completes quickly
    );
    monitor.tick(5000, 5000);
    assert_eq!(monitor.reprofiling_events(), 1);

    // Wait for the background thread to finish (generous timeout for CI).
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(outcome) = monitor.poll_outcome() {
            assert!(outcome.w_max > 0, "w_max should be positive");
            assert!(outcome.mean_t_ssd_ms >= 0.0, "t_ssd should be non-negative");
            assert!((outcome.ssd_temp_celsius - 76.0).abs() < 0.5);
            return;
        }
        if std::time::Instant::now() > deadline {
            panic!("background reprofile did not complete within 5 s");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

// ---------------------------------------------------------------------------
// 8. Real SMART ioctl (skipped unless /dev/nvme0 is present and readable)
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires a readable /dev/nvme0 with SMART support"]
fn test_real_nvme_smart_ioctl() {
    use flowcast::smart_monitor::SmartTempReader;
    use flowcast::smart_monitor::TemperatureSource;
    let reader = SmartTempReader::new(std::path::PathBuf::from("/dev/nvme0"));
    match reader.read_celsius() {
        Ok(temp) => {
            assert!(temp > 0.0 && temp < 200.0, "unexpected temperature {temp}");
        }
        Err(err) => {
            panic!("real SMART read failed: {err}");
        }
    }
}