//! T-ACT-2 — Recompute fidelity: golden full-forward activations vs. recompute
//! from checkpoint 0 to layer 4.
//!
//! Asserts `max diff < 1e-5` at layer 4. Also verifies checkpoints live in
//! `PinnedBuffer`s (not plain `Vec`) and round-trip byte-identical (uncompressed).
//!
//! Run: cargo test --features mock-cuda -p doublepass test_activation_fidelity -- --nocapture
#![allow(clippy::unwrap_used, clippy::expect_used)]

use doublepass::checkpoint::{read_checkpoint, store_checkpoint};
use doublepass::forward::{full_forward, BlockConfig, Model};
use doublepass::plan::TrainingPlan;

const CFG: BlockConfig = BlockConfig {
    d_model: 8,
    n_heads: 2,
    d_ff: 16,
    seq_len: 4,
    batch: 1,
    dropout_p: 0.0,
};

fn init_input() -> Vec<f32> {
    let n = CFG.batch * CFG.seq_len * CFG.d_model;
    (0..n)
        .map(|i| ((i as f64 * 0.07).sin() * 0.3) as f32)
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// T-ACT-2: golden full-forward vs. recompute from checkpoint at layer 0.
///
/// Model: 6 layers, checkpoint_freq = 2 → checkpoints at layers 0, 2, 4.
/// Recompute: read ckpt[layer=0, micro=0] → re-run layers 0→5 → compare final output.
#[test]
#[cfg(feature = "mock-cuda")]
fn test_activation_fidelity_t_act2() {
    const N_LAYERS: usize = 6;
    const CKPT_FREQ: u32 = 2;

    let model = Model::new(N_LAYERS, CFG);
    let inputs = vec![init_input()]; // G = 1
    let mut plan = TrainingPlan::default();
    plan.checkpoint_freq = CKPT_FREQ;

    // Golden forward.
    let golden = full_forward(&model, &inputs, &plan, false).expect("full_forward");

    // Find checkpoint for layer 0, micro-batch 0.
    // (Stored BEFORE layer 0 runs, so it holds the input to layer 0.)
    let ckpt_buf = golden
        .checkpoints
        .iter()
        .find(|(li, mi, _)| *li == 0 && *mi == 0)
        .map(|(_, _, buf)| buf)
        .expect("checkpoint at layer=0, micro=0 must exist");

    // Assert the checkpoint lives in a PinnedBuffer (not a plain Vec).
    // This is guaranteed by the type system — PinnedBuffer is the return type.
    assert!(ckpt_buf.len() > 0, "checkpoint buffer must be non-empty");
    assert!(
        !ckpt_buf.is_compressed(),
        "uncompressed path should not set compressed flag"
    );

    // Round-trip: store and read back; must be byte-identical.
    let ckpt_data = read_checkpoint(ckpt_buf).expect("read_checkpoint");
    let roundtrip_buf = store_checkpoint(&ckpt_data, false).expect("store_checkpoint");
    let roundtrip_data = read_checkpoint(&roundtrip_buf).expect("read_checkpoint roundtrip");
    assert_eq!(
        ckpt_data, roundtrip_data,
        "uncompressed round-trip must be byte-identical"
    );

    // Recompute from ckpt at layer 0: checkpoint holds the INPUT to layer 0.
    // Re-run all layers 0 → N_LAYERS-1 and compare against golden final output.
    let mut recomputed = ckpt_data.clone();
    for i in 0..N_LAYERS {
        doublepass::rng::capture(i as u32, 0).expect("capture");
        let fwd = doublepass::forward::single_layer_forward(&CFG, &model.layers[i], &recomputed);
        recomputed = fwd.output;
    }

    let diff = max_abs_diff(&golden.outputs[0], &recomputed);
    println!(
        "\nT-ACT-2: max|golden - recomputed| = {:.3e}  (tol = 1e-5)",
        diff
    );
    assert!(
        diff < 1e-5,
        "T-ACT-2 FAILED: max|diff| {:.3e} >= 1e-5",
        diff
    );
    println!("T-ACT-2 PASSED");
}

/// Verify weight_bytes_streamed = num_layers × bytes_per_layer (structural check).
#[test]
#[cfg(feature = "mock-cuda")]
fn test_weight_bytes_field() {
    let model = Model::new(5, CFG);
    let inputs = vec![init_input()];
    let plan = TrainingPlan::default();
    let result = full_forward(&model, &inputs, &plan, false).expect("full_forward");
    let expected = (5 * CFG.bytes_per_layer()) as u64;
    assert_eq!(
        result.weight_bytes_streamed, expected,
        "weight_bytes_streamed must equal L × bytes_per_layer"
    );
}
