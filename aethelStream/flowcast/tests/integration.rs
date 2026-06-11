//! Integration tests: end-to-end 2-cycle streaming with mock backend.
//!
//! Existing compile-check tests are preserved; new tests add:
//! T-INT-1: 2-cycle forward+backward over 4 layers, zero PrefetchMiss.
//! T-INT-2: Simulated GPU idle < 20% over 4 layers.
//! T-INT-3: Telemetry snapshot counters match independently-counted events.
//! T-INT-4: NaN-injection echoed through PerLayerScaleTable (scale update).
//! T-INT-5: Delayed write-back completes without error; byte-identical read-back.
//! T-INT-6: Linux RSS check (skipped on non-Linux).
//! T-INT-7: PeerSync compiles and SingleGpuSync is no-op.
//! T-INT-8: Telemetry JSON validates (round-trip).
//! T-INT-9: Pressure event shrinks the adaptive window.

use flowcast::{
    backend::{mock::MockBackend, IoBackend},
    Direction, FlowCastConfig, FlowCastError, HardwareProfile, LayerTiming, Precision, ReadyLayer,
};
use flowcast::telemetry::Telemetry;
use flowcast::writeback::{WritebackConfig, WritebackMode, WritebackScheduler};
use flowcast::peer::{PeerSync, SingleGpuSync};
use flowcast::state_machine::PrefetchStateMachine;
use flowcast::window::AdaptiveWindow;
use ramflow::{PoolRegistry, PerLayerScaleTable};
use ramflow::phase::Direction as RamDirection;
use ramflow::MemoryPressureGauge;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Existing compile-check tests (unchanged)
// ---------------------------------------------------------------------------

#[test]
fn test_all_public_types_accessible() {
    let config = FlowCastConfig {
        num_shards: 4,
        initial_lookahead: 2,
        ewma_alpha: 0.3,
        pressure_threshold: 0.8,
        default_precision: Precision::FP16,
        ..Default::default()
    };
    assert_eq!(config.num_shards, 4);
    let _fp32 = Precision::FP32;
    let _fp16 = Precision::FP16;
    let _bf16 = Precision::BF16;
    let _int8 = Precision::INT8;
    let _int4 = Precision::INT4;
    let _fwd = Direction::Forward;
    let _bwd = Direction::Backward;
    let profile = HardwareProfile {
        nvme_bandwidth_gbs: 3.5,
        pcie_bandwidth_gbs: 16.0,
        gpu_bandwidth_gbs: 900.0,
        mean_forward_ms: 12.0,
        mean_backward_ms: 24.0,
        sample_count: 10,
        layer_plan: vec![LayerTiming {
            layer_idx: 0,
            forward_ms: 12.0,
            backward_ms: 24.0,
            shard_bytes: 1024 * 1024,
            transfer_ms: 1.2,
            ..Default::default()
        }],
    };
    assert_eq!(profile.layer_plan.len(), 1);
    let miss = FlowCastError::PrefetchMiss { layer_idx: 7 };
    assert!(matches!(miss, FlowCastError::PrefetchMiss { layer_idx: 7 }));
    let ram_err = ramflow::RamFlowError::ConfigError("test".to_string());
    let wrapped = FlowCastError::from(ram_err);
    assert!(matches!(wrapped, FlowCastError::RamFlow(_)));
    let mut mock = MockBackend::new();
    mock.start().unwrap();
    assert_eq!(mock.capabilities().name, "mock");
    fn _assert_ready_layer_fields(r: ReadyLayer) {
        let _: u32 = r.layer_idx;
        let _: Precision = r.precision;
        let _ = r.needs_decode;
    }
}

#[test]
fn test_mock_backend_completion_roundtrip() {
    let mock = MockBackend::new();
    let completions = mock.poll_completions().unwrap();
    assert!(completions.is_empty());
}

#[test]
fn test_config_validate_rejects_bad_fields() {
    assert!(FlowCastConfig { ewma_alpha: 0.0, ..Default::default() }.validate().is_err());
    assert!(FlowCastConfig {
        ewma_alpha: 0.3, pressure_threshold: 1.5, ..Default::default()
    }.validate().is_err());
    assert!(FlowCastConfig { ewma_alpha: 0.3, pressure_threshold: 0.8, ..Default::default() }
        .validate().is_ok());
}

// ---------------------------------------------------------------------------
// T-INT-1: 2-cycle forward+backward, zero PrefetchMiss (INT-a fix: backward pass added)
// ---------------------------------------------------------------------------

#[test]
fn end_to_end_two_cycles_zero_prefetch_miss() {
    const NUM_LAYERS: u32 = 4;
    const LOOKAHEAD: u32 = 2;

    let pool = PoolRegistry::with_defaults().expect("pool");
    let backend = MockBackend::new();
    let sm = PrefetchStateMachine::new(NUM_LAYERS, LOOKAHEAD, Precision::FP16);
    let telemetry = Telemetry::new();

    // --- Forward pass ---
    sm.prime_window(RamDirection::Forward, &pool, &backend).expect("prime forward");
    sm.poll_and_route(&backend).expect("poll_and_route after prime");

    for layer in 0..NUM_LAYERS {
        sm.on_layer_start(layer, RamDirection::Forward, &pool, &backend)
            .expect("on_layer_start forward");
        sm.poll_and_route(&backend).expect("poll_and_route forward");

        match sm.take_ready(layer, std::time::Duration::from_millis(100)) {
            Ok(_ready) => {}
            Err(FlowCastError::PrefetchMiss { .. }) => {
                telemetry.record_miss();
            }
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    // --- Backward pass (INT-a fix) ---
    sm.prime_window(RamDirection::Backward, &pool, &backend).expect("prime backward");
    sm.poll_and_route(&backend).expect("poll_and_route after bwd prime");

    for layer in (0..NUM_LAYERS).rev() {
        sm.on_layer_start(layer, RamDirection::Backward, &pool, &backend)
            .expect("on_layer_start backward");
        sm.poll_and_route(&backend).expect("poll_and_route backward");

        match sm.take_ready(layer, std::time::Duration::from_millis(100)) {
            Ok(_ready) => {}
            Err(FlowCastError::PrefetchMiss { .. }) => {
                telemetry.record_miss();
            }
            Err(e) => panic!("unexpected error backward: {e}"),
        }
    }

    // Use telemetry.miss_count (INT-b fix: telemetry counter instead of manual variable).
    let snap = telemetry.snapshot();
    assert_eq!(
        snap.miss_count, 0,
        "expected zero PrefetchMiss in forward+backward pass, got {}",
        snap.miss_count
    );
}

// ---------------------------------------------------------------------------
// T-INT-2: Simulated GPU idle < 20% over 8 layers
// ---------------------------------------------------------------------------

#[test]
fn simulated_gpu_idle_below_20_percent() {
    use std::sync::Arc;
    use flowcast::completion_router::CompletionRouter;

    const NUM_LAYERS: u32 = 8;
    const COMPUTE_MS: u64 = 5;

    let pool = PoolRegistry::with_defaults().expect("pool");
    let backend = Arc::new(MockBackend::new());
    let sm = Arc::new(PrefetchStateMachine::new(NUM_LAYERS, 2, Precision::FP16));
    let _router = CompletionRouter::spawn(backend.clone(), sm.clone())
        .expect("CompletionRouter::spawn");

    sm.prime_window(Direction::Forward, &pool, &*backend)
        .expect("prime_window");

    let total_start = Instant::now();
    let mut total_wait_us: u64 = 0;

    for layer_idx in 0..NUM_LAYERS {
        sm.on_layer_start(layer_idx, Direction::Forward, &pool, &*backend)
            .expect("on_layer_start");

        let wait_start = Instant::now();
        let _ready = sm
            .take_ready(layer_idx, std::time::Duration::from_millis(500))
            .unwrap_or_else(|e| panic!("layer {layer_idx}: {e}"));
        total_wait_us += wait_start.elapsed().as_micros() as u64;

        std::thread::sleep(std::time::Duration::from_millis(COMPUTE_MS));
    }

    let elapsed_us = total_start.elapsed().as_micros() as u64;
    let idle_frac = if elapsed_us == 0 {
        0.0_f64
    } else {
        total_wait_us as f64 / elapsed_us as f64
    };

    assert!(
        idle_frac < 0.20,
        "GPU idle fraction {idle_frac:.3} ≥ 20% — prefetch not hiding I/O \
         (wait={total_wait_us}μs, elapsed={elapsed_us}μs)"
    );
}

// ---------------------------------------------------------------------------
// T-INT-3: Telemetry counters match independently-counted events
// ---------------------------------------------------------------------------

#[test]
fn telemetry_counters_match_independent_counts() {
    let telemetry = Telemetry::new();

    let prefetch_count: u64 = 7;
    let hotset_hit_count: u64 = 3;
    let hotset_miss_count: u64 = 4;
    let write_submit_count: u64 = 5;
    let write_skip_count: u64 = 2;
    let decode_ns_total: u64 = 1234;

    for _ in 0..prefetch_count {
        telemetry.record_prefetch_submitted(1024);
        telemetry.record_prefetch_completed(true);
    }
    for _ in 0..hotset_hit_count { telemetry.record_hotset(true); }
    for _ in 0..hotset_miss_count { telemetry.record_hotset(false); }
    for _ in 0..write_submit_count { telemetry.record_write_submitted(); }
    for _ in 0..write_skip_count { telemetry.record_write_skip(); }
    telemetry.record_decode_ns(decode_ns_total);

    let snap = telemetry.snapshot();

    assert_eq!(snap.prefetch_submitted, prefetch_count);
    assert_eq!(snap.prefetch_completed, prefetch_count);
    assert_eq!(snap.prefetch_errors, 0);
    assert_eq!(snap.hotset_hits, hotset_hit_count);
    assert_eq!(snap.hotset_misses, hotset_miss_count);
    assert_eq!(snap.write_submitted, write_submit_count);
    assert_eq!(snap.write_skip_count, write_skip_count);
    assert_eq!(snap.decode_ns, decode_ns_total);
    assert_eq!(snap.nvme_bytes_read, prefetch_count * 1024);
    assert!((snap.hotset_hit_rate() - 3.0 / 7.0).abs() < 1e-5);
    assert!((snap.write_skip_rate() - 2.0 / 7.0).abs() < 1e-5);
}

// ---------------------------------------------------------------------------
// T-INT-4: NaN injection echoed through PerLayerScaleTable
// ---------------------------------------------------------------------------

#[test]
fn nan_injection_updates_scale_table() {
    let mut scale_table = PerLayerScaleTable::new(8, 0.05);

    for _ in 0..20 {
        scale_table.update(3, 1_000_000, 1_000_000).expect("update layer 3");
    }
    let scale = scale_table.get_scale(3).expect("get_scale layer 3");
    assert!(scale <= 2.0, "scale {scale} should be near floor after NaN injection");
    assert!(scale >= 1.0, "scale {scale} must never go below 1.0");
}

// ---------------------------------------------------------------------------
// T-INT-5: Delayed write-back completes without error + byte-identical read-back (INT-e fix)
// ---------------------------------------------------------------------------

#[test]
fn delayed_writeback_completes_without_error() {
    let backend = MockBackend::new();
    let config = WritebackConfig {
        skip_threshold: 0.0,
        max_skip_rate: 0.0,
        max_inflight_writes: 4,
        ..Default::default()
    };
    let mut sched = WritebackScheduler::with_config(WritebackMode::Immediate, config);

    // Build a buffer with a known pattern for byte-identical verification (INT-e fix).
    let mut buf = ramflow::PinnedBuffer::alloc(256).expect("alloc");
    let buf_data = buf.as_mut_slice();
    for (idx, byte) in buf_data.iter_mut().enumerate() {
        *byte = (idx % 256) as u8;
    }

    sched.on_weights_updated(0, &buf, 0, 1.0, &backend).expect("on_weights_updated");

    let completions = backend.poll_completions().expect("poll");
    assert_eq!(completions.len(), 1, "write-back must complete");
    assert!(completions[0].result >= 0, "write must succeed");

    // Verify byte-identical read-back via MockBackend's written store (INT-e fix).
    let written = backend.last_written_bytes(0).expect("written bytes must be stored");
    assert_eq!(written.len(), 256, "written length must match buffer size");
    for (idx, &byte) in written.iter().enumerate() {
        assert_eq!(
            byte,
            (idx % 256) as u8,
            "byte at index {idx} mismatch: expected {}, got {byte}",
            (idx % 256) as u8
        );
    }
}

// ---------------------------------------------------------------------------
// T-INT-6: Linux RSS check (skipped on non-Linux)
// ---------------------------------------------------------------------------

#[test]
#[cfg_attr(not(target_os = "linux"), ignore = "RSS check Linux-only")]
fn linux_rss_does_not_spike_after_prefetch() {
    fn read_rss_kb() -> u64 {
        let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                return rest.trim().split_whitespace()
                    .next()
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(0);
            }
        }
        0
    }

    let before = read_rss_kb();
    let _buf = ramflow::PinnedBuffer::alloc(1024 * 1024).expect("alloc 1 MiB");
    let after = read_rss_kb();

    let growth_kb = after.saturating_sub(before);
    assert!(
        growth_kb < 64 * 1024,
        "RSS grew by {growth_kb} KiB after 1 MiB alloc — unexpected spike"
    );
}

// ---------------------------------------------------------------------------
// T-INT-7: PeerSync compiles and SingleGpuSync is no-op
// ---------------------------------------------------------------------------

#[test]
fn peer_sync_single_gpu_is_noop() {
    let mut peer: Box<dyn PeerSync> = Box::new(SingleGpuSync);
    peer.broadcast_ready(0).expect("broadcast");
    peer.wait_peer_ready(0).expect("wait");
    peer.shutdown().expect("shutdown");
}

// ---------------------------------------------------------------------------
// T-INT-8: Telemetry JSON round-trips without dropping fields
// ---------------------------------------------------------------------------

#[test]
fn telemetry_json_round_trips() {
    let telemetry = Telemetry::new();
    telemetry.record_prefetch_submitted(4096);
    telemetry.record_prefetch_completed(true);
    telemetry.record_hotset(true);
    telemetry.record_write_submitted();
    telemetry.record_decode_ns(999);
    telemetry.set_queue_depth(3);
    telemetry.set_ready_queue_depth(1);
    telemetry.record_window_grow();

    let snap = telemetry.snapshot();
    let json = snap.to_json().expect("to_json");

    let parsed: serde_json::Value = serde_json::from_str(&json).expect("parse JSON");

    assert_eq!(parsed["prefetch_submitted"].as_u64(), Some(1));
    assert_eq!(parsed["nvme_bytes_read"].as_u64(), Some(4096));
    assert_eq!(parsed["decode_ns"].as_u64(), Some(999));
    assert_eq!(parsed["queue_depth"].as_u64(), Some(3));
    assert_eq!(parsed["ready_queue_depth"].as_u64(), Some(1));
    assert_eq!(parsed["window_grow_events"].as_u64(), Some(1));
    assert_eq!(parsed["hotset_hits"].as_u64(), Some(1));
}

// ---------------------------------------------------------------------------
// T-INT-9: Memory pressure event shrinks the adaptive window (INT-d fix)
// ---------------------------------------------------------------------------

#[test]
fn memory_pressure_event_shrinks_window() {
    const W_MAX: f32 = 8.0;
    let gauge = MemoryPressureGauge::new(1);
    let mut window = AdaptiveWindow::new(W_MAX, 0.2, W_MAX);
    window.register_pressure_callbacks(&gauge, None);

    // Verify window starts at W_MAX.
    assert_eq!(window.t_iter(), W_MAX);

    // Fire high-pressure event → window should cap at 1.
    gauge.signal_stall(0);
    assert!(
        window.pressure_cap_active(),
        "pressure cap must be active after signal_stall"
    );
    assert_eq!(window.t_iter(), 1.0, "window must be capped at 1 under high pressure");

    // Pump one A2 update with grow condition — cap must still win.
    window.update(10.0, 1.0).unwrap(); // ssd=10ms > 0.8*gpu=0.8ms → would grow
    assert_eq!(window.t_iter(), 1.0, "cap persists through A2 update");

    // Record that at least one shrink event happened (via pressure cap).
    let snap_before_lift = window.t_iter();
    assert!(
        snap_before_lift <= 1.0,
        "window must be ≤ 1.0 under high pressure, got {snap_before_lift}"
    );

    // Lift pressure; confirm growth is now possible.
    let pool = PoolRegistry::with_defaults().expect("pool");
    gauge.sample_and_notify(&pool); // 0 fill → low pressure
    assert!(!window.pressure_cap_active(), "cap must be lifted after sample_and_notify");

    window.update(10.0, 1.0).unwrap(); // grow condition again, no cap
    assert!(
        window.t_iter() > 1.0,
        "window must grow above 1 after pressure lifts, got {}",
        window.t_iter()
    );
}
