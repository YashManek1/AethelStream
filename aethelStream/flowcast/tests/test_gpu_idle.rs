//! T1: GPU-idle fraction test.
//!
//! Simulates 80 forward layers with a fixed per-layer compute time.
//! Asserts that the mean ready-queue wait is < 5% and the max < 20%
//! of the compute time over all layers.
//!
//! Uses MockBackend (instant completions) with the CompletionRouter thread
//! so completions arrive asynchronously as they would in production.

use std::sync::Arc;
use std::time::{Duration, Instant};

use flowcast::{
    backend::mock::MockBackend,
    completion_router::CompletionRouter,
    state_machine::PrefetchStateMachine,
    Direction, Precision,
};
use ramflow::PoolRegistry;

const TOTAL_LAYERS: u32 = 80;
const LOOKAHEAD: u32 = 3;
/// Simulated GPU compute time per layer.
const COMPUTE_MS: u64 = 5;
/// Maximum fraction of compute time the ready queue may block.
const MAX_MEAN_FRACTION: f64 = 0.05;
const MAX_PEAK_FRACTION: f64 = 0.20;

#[test]
fn test_ready_queue_wait_fraction() {
    let pool = PoolRegistry::with_defaults().expect("PoolRegistry::with_defaults");
    let backend = Arc::new(MockBackend::new());
    let sm = Arc::new(PrefetchStateMachine::new(TOTAL_LAYERS, LOOKAHEAD, Precision::FP16));

    let _router = CompletionRouter::spawn(backend.clone(), sm.clone())
        .expect("CompletionRouter::spawn");

    sm.prime_window(Direction::Forward, &pool, &*backend)
        .expect("prime_window");

    let mut wait_durations: Vec<Duration> = Vec::with_capacity(TOTAL_LAYERS as usize);

    for layer_idx in 0..TOTAL_LAYERS {
        sm.on_layer_start(layer_idx, Direction::Forward, &pool, &*backend)
            .expect("on_layer_start");

        // Measure how long take_ready blocks.
        let wait_start = Instant::now();
        let _ready = sm
            .take_ready(layer_idx, Duration::from_millis(500))
            .unwrap_or_else(|e| panic!("layer {layer_idx}: {e}"));
        let wait_elapsed = wait_start.elapsed();
        wait_durations.push(wait_elapsed);

        // Simulate GPU compute.
        std::thread::sleep(Duration::from_millis(COMPUTE_MS));
    }

    let compute_ns = Duration::from_millis(COMPUTE_MS).as_nanos() as f64;
    let mean_wait_ns = wait_durations
        .iter()
        .map(|d| d.as_nanos() as f64)
        .sum::<f64>()
        / TOTAL_LAYERS as f64;
    let max_wait_ns = wait_durations
        .iter()
        .map(|d| d.as_nanos())
        .max()
        .unwrap_or(0) as f64;

    let mean_fraction = mean_wait_ns / compute_ns;
    let max_fraction = max_wait_ns / compute_ns;

    assert!(
        mean_fraction < MAX_MEAN_FRACTION,
        "mean ready-queue wait {:.1}% exceeds {:.1}% threshold (mean={:.2}ms, compute={}ms)",
        mean_fraction * 100.0,
        MAX_MEAN_FRACTION * 100.0,
        mean_wait_ns / 1_000_000.0,
        COMPUTE_MS,
    );

    assert!(
        max_fraction < MAX_PEAK_FRACTION,
        "peak ready-queue wait {:.1}% exceeds {:.1}% threshold (peak={:.2}ms, compute={}ms)",
        max_fraction * 100.0,
        MAX_PEAK_FRACTION * 100.0,
        max_wait_ns / 1_000_000.0,
        COMPUTE_MS,
    );
}
