//! T-RNG — Deterministic dropout: recompute with RNG restore reproduces forward
//! activations exactly; negative control WITHOUT restore must differ.
//!
//! Run: cargo test --features mock-cuda -p doublepass test_rng_determinism -- --nocapture
#![allow(clippy::unwrap_used, clippy::expect_used)]

use doublepass::checkpoint::read_checkpoint;
use doublepass::forward::{full_forward, single_layer_forward, BlockConfig, Model};
use doublepass::plan::TrainingPlan;
use doublepass::rng;

const CFG: BlockConfig = BlockConfig {
    d_model: 8,
    n_heads: 2,
    d_ff: 16,
    seq_len: 4,
    batch: 1,
    dropout_p: 0.3,
};

fn init_input() -> Vec<f32> {
    let n = CFG.batch * CFG.seq_len * CFG.d_model;
    (0..n)
        .map(|i| ((i as f64 * 0.11).cos() * 0.4) as f32)
        .collect()
}

/// T-RNG (part a): recompute WITH RNG restore reproduces forward activations exactly.
///
/// Strategy: store intermediate forward outputs, then check that restore reproduces them.
#[test]
#[cfg(feature = "mock-cuda")]
fn test_rng_restore_exact_reproduction() {
    rng::set_step_seed(0xdeadbeef_12345678);

    let model = Model::new(3, CFG);
    let inputs = vec![init_input()];
    let mut plan = TrainingPlan::default();
    plan.checkpoint_freq = 2; // checkpoint at layers 0, 2

    // Full forward pass — captures RNG states and checkpoints.
    let result = full_forward(&model, &inputs, &plan, false).expect("full_forward");

    // The final output is at result.outputs[0].
    // Let's recompute the last layer with RNG restore and check it matches.
    // Layer 2 checkpoint = INPUT to layer 2. We need to capture from layer 1 output first.

    // Strategy: recompute layer 1 and layer 2 independently, comparing with full forward.
    // Start with checkpoint at layer 0 (= input to layer 0).
    let ckpt_l0 = result
        .checkpoints
        .iter()
        .find(|(li, mi, _)| *li == 0 && *mi == 0)
        .map(|(_, _, buf)| buf)
        .expect("ckpt layer 0");
    let act = read_checkpoint(ckpt_l0).expect("read ckpt");

    // Recompute layer 0 with RNG restore
    let rng_state_l0 = &result.rng_states[0];
    rng::restore(rng_state_l0).expect("restore");
    let fwd_l0 = single_layer_forward(&CFG, &model.layers[0], &act);
    let output_l0 = fwd_l0.output.clone();

    // Recompute layer 1 with RNG restore
    let rng_state_l1 = &result.rng_states[1];
    rng::restore(rng_state_l1).expect("restore");
    let fwd_l1 = single_layer_forward(&CFG, &model.layers[1], &output_l0);
    let output_l1 = fwd_l1.output.clone();

    // Recompute layer 2 with RNG restore
    let rng_state_l2 = &result.rng_states[2];
    rng::restore(rng_state_l2).expect("restore");
    let fwd_l2 = single_layer_forward(&CFG, &model.layers[2], &output_l1);
    let recomputed_output = fwd_l2.output.clone();

    // Compare with golden output from full_forward
    let golden_output = &result.outputs[0];

    let max_diff: f32 = recomputed_output
        .iter()
        .zip(golden_output)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!(
        "\nT-RNG (restore): max|recomputed - golden| = {:.3e}  (expected 0)",
        max_diff
    );
    assert_eq!(
        recomputed_output, *golden_output,
        "T-RNG FAILED: recompute WITH restore must be bit-identical to golden"
    );
    println!("T-RNG (a) PASSED — recompute with restore is bit-identical");
}

/// T-RNG (part b): negative control — WITHOUT RNG restore, activations MUST differ.
///
/// This proves the test has teeth: if dropout is disabled or the RNG is always
/// the same regardless of state, this test would incorrectly pass part (a).
#[test]
#[cfg(feature = "mock-cuda")]
fn test_rng_no_restore_differs() {
    rng::set_step_seed(0xcafe_f00d_1234_5678);

    let model = Model::new(3, CFG);
    let inputs = vec![init_input()];
    let mut plan = TrainingPlan::default();
    plan.checkpoint_freq = 2;

    let result = full_forward(&model, &inputs, &plan, false).expect("full_forward");

    // Get checkpoint at layer 0 (= input to layer 0)
    let ckpt_l0 = result
        .checkpoints
        .iter()
        .find(|(li, mi, _)| *li == 0 && *mi == 0)
        .map(|(_, _, buf)| buf)
        .expect("ckpt layer 0");
    let act = read_checkpoint(ckpt_l0).expect("read ckpt");

    // Recompute layers 0, 1, 2 WITHOUT restore (fresh RNG state)
    let fwd_l0_no_restore = single_layer_forward(&CFG, &model.layers[0], &act);
    let fwd_l1_no_restore = single_layer_forward(&CFG, &model.layers[1], &fwd_l0_no_restore.output);
    let fwd_l2_no_restore = single_layer_forward(&CFG, &model.layers[2], &fwd_l1_no_restore.output);

    let golden_output = &result.outputs[0];

    let max_diff: f32 = fwd_l2_no_restore
        .output
        .iter()
        .zip(golden_output)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!(
        "\nT-RNG (no restore): max|recomputed - golden| = {:.3e}  (expected > 0)",
        max_diff
    );
    assert!(
        max_diff > 0.0,
        "T-RNG negative control FAILED: outputs must differ when RNG is not restored \
         (dropout_p = {}, diff = {:.3e})",
        CFG.dropout_p,
        max_diff
    );
    println!("T-RNG (b) PASSED — no-restore outputs differ as expected");
}
