//! Pipeline idle test: GPU ready-wait is negligible vs. simulated compute time.
//!
//! Under mock-cuda, `take_ready` is instant (no real I/O). We verify zero PrefetchMiss
//! and that operations complete in reasonable time.
//!
//! Run: cargo test --features mock-cuda -p doublepass test_pipeline_idle -- --nocapture
#![allow(clippy::unwrap_used, clippy::expect_used)]

use doublepass::backward::full_backward;
use doublepass::forward::{full_forward, BlockConfig, Model};
use doublepass::plan::TrainingPlan;
use doublepass::rng;
use doublepass::OptimizerBackend;
use std::time::Instant;

const CFG: BlockConfig = BlockConfig {
    d_model: 8,
    n_heads: 2,
    d_ff: 16,
    seq_len: 4,
    batch: 1,
    dropout_p: 0.0,
};

/// Minimal no-op optimizer for idle test.
struct NullOpt;
impl OptimizerBackend for NullOpt {
    fn project_and_accumulate(&self, _: &[f32], _: u32, _: &str) {}
    fn lowrank_grad_sqnorm(&self, _: u32, _: &str) -> f64 {
        0.0
    }
    fn apply_update(&self, _: u32, _: &str, _: f32) {}
    fn zero_accum(&self, _: u32, _: &str) {}
    fn notify_step(&self, _: u64) {}
}

/// Test that full forward+backward complete with no PrefetchMiss errors.
#[test]
#[cfg(feature = "mock-cuda")]
fn test_pipeline_idle_and_zero_prefetch_miss() {
    const N_LAYERS: usize = 6;
    const G: usize = 2;

    rng::set_step_seed(0xc0de_feed_1234_5678);

    let model = Model::new(N_LAYERS, CFG);
    let inputs: Vec<Vec<f32>> = (0..G)
        .map(|m| {
            let n = CFG.batch * CFG.seq_len * CFG.d_model;
            (0..n)
                .map(|i| ((i * 3 + m * 5) as f64 * 0.09).sin() as f32)
                .collect()
        })
        .collect();
    let mut plan = TrainingPlan::default();
    plan.checkpoint_freq = 2;

    // Measure wall time of full forward.
    let t_start = Instant::now();
    let fwd = full_forward(&model, &inputs, &plan, false).expect("full_forward — no PrefetchMiss");
    let t_fwd = t_start.elapsed();

    let upstream: Vec<Vec<f32>> = (0..G)
        .map(|_| vec![1.0f32; CFG.batch * CFG.seq_len * CFG.d_model])
        .collect();

    let t_bwd_start = Instant::now();
    let _bwd = full_backward(&model, &fwd, &inputs, &upstream, &plan, true, &NullOpt)
        .expect("full_backward — no PrefetchMiss");
    let t_bwd = t_bwd_start.elapsed();

    let total_wall_ms = (t_fwd + t_bwd).as_millis();

    // Under mock-cuda, all operations complete fast. We verify:
    // 1. No errors (PrefetchMiss check — both functions return Ok)
    // 2. Total time is reasonable (< 5 seconds for 6 layers each direction)
    println!("\nPipeline idle test (mock):");
    println!("  Total wall time:  {} ms", total_wall_ms);
    println!(
        "  Layers processed: {} forward + {} backward",
        N_LAYERS, N_LAYERS
    );
    println!("  Zero PrefetchMiss:       PASS (no errors returned)");

    assert!(
        total_wall_ms < 5000,
        "pipeline idle FAILED: total wall time {} ms is too slow",
        total_wall_ms
    );

    println!(
        "test_pipeline_idle PASSED — completed in {} ms",
        total_wall_ms
    );
}

/// Zero PrefetchMiss: full step returns Ok on all layers.
///
/// Under mock-cuda, MockBackend never emits PrefetchMiss, so any returned
/// Err(FlowCast(PrefetchMiss)) would indicate a bug in the mock wiring.
#[test]
#[cfg(feature = "mock-cuda")]
fn test_zero_prefetch_miss() {
    const N_LAYERS: usize = 4;
    rng::set_step_seed(0xdeadbeef_cafef00d);

    let model = Model::new(N_LAYERS, CFG);
    let inputs: Vec<Vec<f32>> = vec![(0..CFG.batch * CFG.seq_len * CFG.d_model)
        .map(|i| (i as f32 * 0.05).sin())
        .collect()];
    let mut plan = TrainingPlan::default();
    plan.checkpoint_freq = 2;

    let fwd = full_forward(&model, &inputs, &plan, false);
    assert!(
        fwd.is_ok(),
        "full_forward returned error: {:?}",
        fwd.as_ref().err()
    );

    let upstream = vec![vec![1.0f32; CFG.batch * CFG.seq_len * CFG.d_model]];
    let bwd = full_backward(
        &model,
        &fwd.unwrap(),
        &inputs,
        &upstream,
        &plan,
        true,
        &NullOpt,
    );
    assert!(
        bwd.is_ok(),
        "full_backward returned error: {:?}",
        bwd.as_ref().err()
    );

    println!(
        "\ntest_zero_prefetch_miss PASSED — no PrefetchMiss on {} layers",
        N_LAYERS
    );
}
