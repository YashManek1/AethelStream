//! T-SELREC — Selective recompute tests.
//!
//! Verifies that selective-region recompute (attention-interior-only) produces:
//! 1. Bit-identical gradients compared to full recompute (T-SELREC-A).
//! 2. Lower FLOP fraction than full recompute (T-SELREC-B).
//! 3. Identical outputs at the op level as full forward (T-SELREC-C).
//! 4. Upward knee shift in the DoublePass formula (T-SELREC-D).
//! 5. Retain-all mask produces unchanged output (T-SELREC-E).
//!
//! Run: cargo test --features mock-cuda -p doublepass test_selective_recompute -- --nocapture

#![allow(clippy::unwrap_used, clippy::expect_used)]

use doublepass::backward::{full_backward_sarp, ParamGrads};
use doublepass::forward::{
    full_forward_with_retention, single_layer_forward, BlockConfig, BlockWeights, Model,
    SingleLayerFwdOut,
};
use doublepass::plan::{ActivationAction, SegmentPlan, SelectiveRecomputeMask, TrainingPlan};
use doublepass::OptimizerBackend;
use std::sync::Mutex;

// ============================================================================
// FIXTURE HELPERS
// ============================================================================

/// Tiny config for selective recompute tests: d_model=16, n_heads=2, d_ff=32, seq_len=4, batch=1, dropout_p=0.0
/// Slightly larger than HAM tests to show meaningful FLOP differences.
fn tiny_cfg() -> BlockConfig {
    BlockConfig {
        d_model: 16,
        n_heads: 2,
        d_ff: 32,
        seq_len: 4,
        batch: 1,
        dropout_p: 0.0,
    }
}

/// Create a model with `n_layers` transformer blocks using distinct weights per layer.
fn make_model(n_layers: usize) -> Model {
    let cfg = tiny_cfg();
    Model::new(n_layers, cfg)
}

/// Create `n` micro-batches of input activations (deterministic, no randomness).
fn make_inputs(n: usize, cfg: &BlockConfig) -> Vec<Vec<f32>> {
    let bs = cfg.bs();
    let mut inputs = Vec::with_capacity(n);
    for i in 0..n {
        let act: Vec<f32> = (0..bs * cfg.d_model)
            .map(|j| {
                let seed = i as f64 * 1000.0 + j as f64;
                ((seed * 0.137).sin() * 0.1) as f32
            })
            .collect();
        inputs.push(act);
    }
    inputs
}

/// Create an upstream gradient (one per micro-batch).
fn make_upstream(cfg: &BlockConfig) -> Vec<f32> {
    let bs = cfg.bs();
    (0..bs * cfg.d_model)
        .map(|i| ((i as f64 * 0.17 + 1.0).sin() * 0.1) as f32)
        .collect()
}

/// No-op optimizer: all methods do nothing.
struct NoOpOptimizer;

impl OptimizerBackend for NoOpOptimizer {
    fn project_and_accumulate(&self, _grad: &[f32], _layer_idx: u32, _param_name: &str) {}
    fn lowrank_grad_sqnorm(&self, _layer_idx: u32, _param_name: &str) -> f64 {
        0.0
    }
    fn apply_update(&self, _layer_idx: u32, _param_name: &str, _clip_scale: f32) {}
    fn zero_accum(&self, _layer_idx: u32, _param_name: &str) {}
    fn notify_step(&self, _step: u64) {}
}

/// Build a training plan with full recompute for one segment.
fn make_plan_full(checkpoint_freq: u32) -> TrainingPlan {
    TrainingPlan {
        checkpoint_freq,
        activation_schedule: vec![SegmentPlan::with_full_recompute(0)],
        ..TrainingPlan::default()
    }
}

/// Build a training plan with selective recompute (attn-interior-only) for one segment.
fn make_plan_selective(checkpoint_freq: u32) -> TrainingPlan {
    TrainingPlan {
        checkpoint_freq,
        activation_schedule: vec![SegmentPlan::with_selective_recompute(0)],
        ..TrainingPlan::default()
    }
}

/// Compute max absolute difference between two gradient fields.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Compare two ParamGrads structs for bit-identical equality.
fn assert_grads_equal(a: &ParamGrads, b: &ParamGrads, layer_idx: usize) {
    let fields = [
        ("d_rms1_w", (&a.d_rms1_w, &b.d_rms1_w)),
        ("d_wq", (&a.d_wq, &b.d_wq)),
        ("d_wk", (&a.d_wk, &b.d_wk)),
        ("d_wv", (&a.d_wv, &b.d_wv)),
        ("d_wo", (&a.d_wo, &b.d_wo)),
        ("d_rms2_w", (&a.d_rms2_w, &b.d_rms2_w)),
        ("d_wg", (&a.d_wg, &b.d_wg)),
        ("d_wu", (&a.d_wu, &b.d_wu)),
        ("d_wd", (&a.d_wd, &b.d_wd)),
        ("d_input", (&a.d_input, &b.d_input)),
    ];

    for (field_name, (field_a, field_b)) in &fields {
        let diff = max_abs_diff(field_a, field_b);
        assert!(
            diff == 0.0,
            "T-SELREC: selective and full recompute must be bit-identical (layer {}, field {}): max|Δ|={}",
            layer_idx, field_name, diff
        );
    }
}

// ============================================================================
// TEST T-SELREC-A: Selective and full recompute give bit-identical gradients
// ============================================================================

#[test]
fn test_selrec_a_bit_identical_gradients() {
    let cfg = tiny_cfg();
    let model = make_model(4);
    let inputs = make_inputs(1, &cfg);
    let upstream_grads: Vec<Vec<f32>> = vec![make_upstream(&cfg)];

    // Run forward with retention to populate retained_activations
    let plan_any = make_plan_full(4);
    let fwd = full_forward_with_retention(&model, &inputs, &plan_any, false)
        .expect("forward must succeed");

    // Run backward with selective recompute mask
    let plan_selective = make_plan_selective(4);
    let result_selective = full_backward_sarp(
        &model,
        &fwd,
        &inputs,
        &upstream_grads,
        &plan_selective,
        false,
        &NoOpOptimizer,
    )
    .expect("selective backward must succeed");

    // Run backward with full recompute mask
    let plan_full = make_plan_full(4);
    let result_full = full_backward_sarp(
        &model,
        &fwd,
        &inputs,
        &upstream_grads,
        &plan_full,
        false,
        &NoOpOptimizer,
    )
    .expect("full backward must succeed");

    // Assert all layers have identical gradients
    assert_eq!(
        result_selective.layer_grads.len(),
        result_full.layer_grads.len(),
        "T-SELREC-A: same number of layers"
    );

    for (i, (grads_selective, grads_full)) in result_selective
        .layer_grads
        .iter()
        .zip(&result_full.layer_grads)
        .enumerate()
    {
        assert_grads_equal(grads_selective, grads_full, i);
    }

    eprintln!(
        "[T-SELREC-A] ✓ selective and full recompute bit-identical over {} layers",
        model.num_layers()
    );
}

// ============================================================================
// TEST T-SELREC-B: Selective recompute FLOP fraction is less than full
// ============================================================================

#[test]
fn test_selrec_b_flop_fractions() {
    let frac_selective = SelectiveRecomputeMask::attn_interior_only().recompute_flop_fraction();
    let frac_full = SelectiveRecomputeMask::full_recompute().recompute_flop_fraction();

    assert!(
        frac_selective < frac_full,
        "selective fraction {frac_selective} must be strictly less than full {frac_full}"
    );

    assert!(
        frac_selective < 0.5,
        "attn-interior-only must recompute less than 50% of FLOPs; got {frac_selective}"
    );

    assert!(
        (frac_full - 1.0).abs() < 1e-9,
        "full recompute fraction must be 1.0; got {frac_full}"
    );

    eprintln!(
        "[T-SELREC-B] ✓ selective={:.4} < full={:.4}",
        frac_selective, frac_full
    );
}

// ============================================================================
// TEST T-SELREC-C: selective_layer_forward output matches full forward
// ============================================================================

#[test]
fn test_selrec_c_layer_forward_identical() {
    let cfg = tiny_cfg();
    let weights = BlockWeights::from_formula(&cfg);
    let input_data = make_inputs(1, &cfg).into_iter().next().expect("one input");

    // Run full forward to get retained values
    let retained = single_layer_forward(&cfg, &weights, &input_data);

    // Run selective forward with attn-interior-only mask
    let selective_out = doublepass::forward::selective_layer_forward(
        &cfg,
        &weights,
        &retained,
        &SelectiveRecomputeMask::attn_interior_only(),
    );

    // Run full forward again as reference
    let full_out = single_layer_forward(&cfg, &weights, &input_data);

    // Assert every field matches
    assert_eq!(selective_out.x_in, full_out.x_in, "x_in must match");
    assert_eq!(selective_out.rms1, full_out.rms1, "rms1 must match");
    assert_eq!(
        selective_out.x_norm1, full_out.x_norm1,
        "x_norm1 must match"
    );
    assert_eq!(selective_out.h1, full_out.h1, "h1 must match");
    assert_eq!(
        selective_out.q_heads, full_out.q_heads,
        "q_heads must match"
    );
    assert_eq!(
        selective_out.k_heads, full_out.k_heads,
        "k_heads must match"
    );
    assert_eq!(
        selective_out.v_heads, full_out.v_heads,
        "v_heads must match"
    );
    assert_eq!(
        selective_out.attn_scores, full_out.attn_scores,
        "attn_scores must match"
    );
    assert_eq!(
        selective_out.attn_weights, full_out.attn_weights,
        "attn_weights must match"
    );
    assert_eq!(
        selective_out.attn_out, full_out.attn_out,
        "attn_out must match"
    );
    assert_eq!(
        selective_out.out_proj, full_out.out_proj,
        "out_proj must match"
    );
    assert_eq!(selective_out.x2, full_out.x2, "x2 must match");
    assert_eq!(selective_out.rms2, full_out.rms2, "rms2 must match");
    assert_eq!(
        selective_out.x_norm2, full_out.x_norm2,
        "x_norm2 must match"
    );
    assert_eq!(selective_out.h2, full_out.h2, "h2 must match");
    assert_eq!(selective_out.gate, full_out.gate, "gate must match");
    assert_eq!(selective_out.up, full_out.up, "up must match");
    assert_eq!(
        selective_out.silu_gate, full_out.silu_gate,
        "silu_gate must match"
    );
    assert_eq!(selective_out.hidden, full_out.hidden, "hidden must match");
    assert_eq!(
        selective_out.mlp_out, full_out.mlp_out,
        "mlp_out must match"
    );
    assert_eq!(selective_out.output, full_out.output, "output must match");

    eprintln!(
        "[T-SELREC-C] ✓ selective_layer_forward bit-identical to full forward on all 20 fields"
    );
}

// ============================================================================
// TEST T-SELREC-D: Knee G* shifts up under selective recompute
// ============================================================================

#[test]
fn test_selrec_d_knee_shift() {
    // From DoublePass_Engine.md §3 worked example: 7B model, PCIe-4
    let t_io_s = 4.7_f64; // seconds per 100GB: hand-set from paper
    let flop_s = 50e12_f64; // RTX 4090 realized BF16 FLOP/s
    let p_total = 7e9_f64; // 7B parameters
    let s_tokens = 2048_f64; // sequence length in tokens

    // Recompute coefficients from DoublePass formula:
    // G* = T_io * F / (recompute_coeff * P_total * s)
    //
    // Full recompute:    coeff = 8.0  (fwd 2P + recompute 2P + bwd 4P)
    // Selective:         coeff = 6.0 + 2*alpha
    //   where alpha = fraction of FLOPs in selective mask
    let alpha = SelectiveRecomputeMask::attn_interior_only().recompute_flop_fraction();

    let g_star_full = t_io_s * flop_s / (8.0 * p_total * s_tokens);
    let g_star_selective = t_io_s * flop_s / ((6.0 + 2.0 * alpha) * p_total * s_tokens);

    assert!(
        g_star_selective > g_star_full,
        "selective G* ({g_star_selective:.4}) must exceed full G* ({g_star_full:.4}) — knee shifts up"
    );

    eprintln!(
        "[T-SELREC-D] ✓ G*_full={g_star_full:.4} < G*_selective={g_star_selective:.4}, alpha={alpha:.4}"
    );
}

// ============================================================================
// TEST T-SELREC-E: Retain-all mask leaves output unchanged
// ============================================================================

#[test]
fn test_selrec_e_retain_all_unchanged() {
    let cfg = tiny_cfg();
    let weights = BlockWeights::from_formula(&cfg);
    let input_vec = make_inputs(1, &cfg).into_iter().next().expect("one input");

    // Run full forward to get a reference
    let original = single_layer_forward(&cfg, &weights, &input_vec);

    // Apply retain_all mask (should recompute nothing)
    let result = doublepass::forward::selective_layer_forward(
        &cfg,
        &weights,
        &original,
        &SelectiveRecomputeMask::retain_all(),
    );

    // All fields should be unchanged
    assert_eq!(
        original.attn_scores, result.attn_scores,
        "attn_scores unchanged"
    );
    assert_eq!(
        original.attn_weights, result.attn_weights,
        "attn_weights unchanged"
    );
    assert_eq!(original.attn_out, result.attn_out, "attn_out unchanged");
    assert_eq!(original.output, result.output, "output unchanged");
    assert_eq!(original.mlp_out, result.mlp_out, "mlp_out unchanged");
    assert_eq!(original.x2, result.x2, "x2 unchanged");
    assert_eq!(original.q_heads, result.q_heads, "q_heads unchanged");
    assert_eq!(original.k_heads, result.k_heads, "k_heads unchanged");
    assert_eq!(original.v_heads, result.v_heads, "v_heads unchanged");

    eprintln!("[T-SELREC-E] ✓ retain_all mask leaves all fields unchanged");
}
