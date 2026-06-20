//! T-AMORT — Layer-major amortization: weight bytes streamed per step are constant in G.
//!
//! Asserts `weight_bytes_streamed == 1 × ΣW_i` (forward-only S3 target) and that
//! the byte count is **independent of G** (grad-accum depth). The `2×` target
//! arrives when the S4 backward re-streams weights.
//!
//! Run: cargo test --features mock-cuda -p doublepass test_amortization -- --nocapture
#![allow(clippy::unwrap_used, clippy::expect_used)]

use doublepass::forward::{full_forward, BlockConfig, Model};
use doublepass::plan::TrainingPlan;
use doublepass::rng;

const CFG: BlockConfig = BlockConfig {
    d_model: 8,
    n_heads: 2,
    d_ff: 16,
    seq_len: 4,
    batch: 1,
    dropout_p: 0.0,
};

fn make_inputs(g: usize) -> Vec<Vec<f32>> {
    let n = CFG.batch * CFG.seq_len * CFG.d_model;
    (0..g)
        .map(|m| (0..n).map(|i| ((i + m * 17) as f64 * 0.05).sin() as f32).collect())
        .collect()
}

/// T-AMORT: weight_bytes_streamed is constant across G in {1, 2, 4, 8}.
#[test]
#[cfg(feature = "mock-cuda")]
fn test_weight_bytes_constant_in_g() {
    const N_LAYERS: usize = 5;
    let model = Model::new(N_LAYERS, CFG);
    let plan = TrainingPlan::default();

    let expected_bytes = (N_LAYERS * CFG.bytes_per_layer()) as u64;
    println!("\nT-AMORT: expected bytes = {} (= {} layers × {} bytes/layer)",
             expected_bytes, N_LAYERS, CFG.bytes_per_layer());
    println!("  {:>3}   {:>16}", "G", "weight_bytes");

    let mut prev_bytes: Option<u64> = None;
    for &g in &[1usize, 2, 4, 8] {
        rng::set_step_seed(0xfeed_face_dead_beef);
        let inputs = make_inputs(g);
        let result = full_forward(&model, &inputs, &plan, false).expect("full_forward");
        let bytes = result.weight_bytes_streamed;
        println!("  {:>3}   {:>16}", g, bytes);

        // Assert == expected (forward-only: 1 × ΣW_i)
        assert_eq!(
            bytes, expected_bytes,
            "T-AMORT FAILED for G={}: got {} expected {}",
            g, bytes, expected_bytes
        );

        // Assert constant across G values.
        if let Some(prev) = prev_bytes {
            assert_eq!(
                bytes, prev,
                "T-AMORT FAILED: bytes differ between G values ({} vs {})",
                bytes, prev
            );
        }
        prev_bytes = Some(bytes);
    }
    println!("T-AMORT PASSED — weight_bytes_streamed is {expected_bytes} for all G in {{1,2,4,8}}");
}

/// Verify that the number of checkpoint entries scales with G, not L.
///
/// checkpoint_freq = 2, L = 4: checkpoints at layers 0, 2 → 2 per micro-batch.
/// With G micro-batches: 2 × G entries total.
#[test]
#[cfg(feature = "mock-cuda")]
fn test_checkpoint_count_scales_with_g() {
    const N_LAYERS: usize = 4;
    let model = Model::new(N_LAYERS, CFG);
    let mut plan = TrainingPlan::default();
    plan.checkpoint_freq = 2;

    for &g in &[1usize, 2, 4] {
        let inputs = make_inputs(g);
        let result = full_forward(&model, &inputs, &plan, false).expect("full_forward");
        // Checkpoints at layers 0 and 2, for each of G micro-batches.
        let expected_count = 2 * g;
        assert_eq!(
            result.checkpoints.len(),
            expected_count,
            "expected {} checkpoints for G={}, got {}",
            expected_count, g, result.checkpoints.len()
        );
    }
}

/// T-AMORT full step: forward + backward = 2·ΣW_i, constant in G.
///
/// keep_resident=true: fwd(1·L) + bwd(1·L) = 2·L total weight streams.
#[test]
#[cfg(feature = "mock-cuda")]
fn test_full_step_two_x_sum_weights() {
    use doublepass::backward::full_backward;
    use doublepass::OptimizerBackend;

    struct NullOpt;
    impl OptimizerBackend for NullOpt {
        fn project_and_accumulate(&self, _: &[f32], _: u32, _: &str) {}
        fn lowrank_grad_sqnorm(&self, _: u32, _: &str) -> f64 { 0.0 }
        fn apply_update(&self, _: u32, _: &str, _: f32) {}
        fn zero_accum(&self, _: u32, _: &str) {}
        fn notify_step(&self, _: u64) {}
    }

    const N_LAYERS: usize = 5;
    let model = Model::new(N_LAYERS, CFG);
    let bpl = CFG.bytes_per_layer() as u64;
    let expected_total = 2 * N_LAYERS as u64 * bpl;

    println!("\nT-AMORT full step: expected 2·ΣW_i = {expected_total}");
    println!("  {:>3}   {:>18}   {:>18}   {:>14}", "G", "fwd_bytes", "bwd_bytes", "total");

    let mut prev: Option<u64> = None;
    for &g in &[1usize, 2, 4, 8] {
        rng::set_step_seed(0x1234_5678_abcd_ef00);
        let inputs = make_inputs(g);
        let mut plan = TrainingPlan::default();
        plan.checkpoint_freq = 2;

        let fwd = full_forward(&model, &inputs, &plan, false).expect("fwd");
        let upstream: Vec<Vec<f32>> = (0..g)
            .map(|_| vec![1.0f32; CFG.batch * CFG.seq_len * CFG.d_model])
            .collect();

        let bwd = full_backward(&model, &fwd, &inputs, &upstream, &plan, true, &NullOpt)
            .expect("bwd");

        let total = fwd.weight_bytes_streamed + bwd.weight_loads;
        println!("  {:>3}   {:>18}   {:>18}   {:>14}", g, fwd.weight_bytes_streamed, bwd.weight_loads, total);

        assert_eq!(total, expected_total,
            "T-AMORT full step G={g}: total={total} expected={expected_total}");

        if let Some(p) = prev {
            assert_eq!(total, p, "T-AMORT full step: total not constant across G");
        }
        prev = Some(total);
    }
    println!("T-AMORT full step PASSED — 2·ΣW_i={expected_total} constant for G in {{1,2,4,8}}");
}
