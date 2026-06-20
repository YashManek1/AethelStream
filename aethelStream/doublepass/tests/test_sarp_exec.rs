//! T-SARP — SARP Executor integration tests.
//!
//! Comprehensive tests for the Segment Activation Recompute Planner (SARP):
//! - T-SARP-A: All four actions (RetainVram, PageCompressedRam, PageNvme, Recompute)
//!   produce bit-identical gradients (FP32, deterministic, no dropout).
//! - T-SARP-B: Reference DP logic selects minimum-T_iter action on known cost grids.
//! - T-SARP-C: SarpExecutor honors the specified action from the plan.
//! - T-SARP-D: Fallback (empty schedule) reproduces full_backward bit-identically.
//! - T-SARP-E: SarpExecutor without M9 schedule defaults to Recompute (no ham-offload).
//!
//! Run: cargo test --features mock-cuda -p doublepass test_sarp_exec -- --nocapture

#![allow(clippy::unwrap_used, clippy::expect_used)]

use doublepass::backward::{full_backward, full_backward_sarp};
use doublepass::forward::{full_forward_with_retention, BlockConfig, Model};
use doublepass::plan::{ActivationAction, SegmentPlan, TrainingPlan, NUM_OPS};
use doublepass::sarp::SarpExecutor;
use doublepass::{HardwareProfile, OptimizerBackend};

// ============================================================================
// FIXTURE HELPERS
// ============================================================================

/// Tiny config for fast tests: d_model=8, n_heads=2, d_ff=16, seq_len=2, batch=1, dropout_p=0.0
fn tiny_cfg() -> BlockConfig {
    BlockConfig {
        d_model: 8,
        n_heads: 2,
        d_ff: 16,
        seq_len: 2,
        batch: 1,
        dropout_p: 0.0,
    }
}

/// Create a model with `n_layers` transformer blocks using distinct weights per layer.
fn make_model(n_layers: usize) -> Model {
    let cfg = tiny_cfg();
    Model::new(n_layers, cfg)
}

/// Create `n` micro-batches of input activations (random-seeded, deterministic).
fn make_inputs(n: usize, cfg: &BlockConfig) -> Vec<Vec<f32>> {
    let bs = cfg.bs();
    let d = cfg.d_model;
    let size = bs * d;
    let mut inputs = Vec::with_capacity(n);
    for i in 0..n {
        let act: Vec<f32> = (0..size)
            .map(|j| {
                let seed = i as f64 * 1000.0 + j as f64;
                ((seed * 0.137).sin() * 0.1) as f32
            })
            .collect();
        inputs.push(act);
    }
    inputs
}

/// Create an upstream gradient for one micro-batch (size = batch * seq_len * d_model).
fn make_upstream(cfg: &BlockConfig) -> Vec<f32> {
    let bs = cfg.bs();
    let d = cfg.d_model;
    let size = bs * d;
    (0..size)
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

/// Build a training plan with specific actions for each segment.
fn make_plan_with_actions(
    checkpoint_freq: u32,
    actions: &[(u32, ActivationAction)],
) -> TrainingPlan {
    let mut plan = TrainingPlan::default();
    plan.checkpoint_freq = checkpoint_freq;

    plan.activation_schedule = actions
        .iter()
        .map(|(seg_idx, action)| {
            let mut seg_plan = SegmentPlan {
                segment_index: *seg_idx,
                action: *action,
                recompute_ops: match action {
                    ActivationAction::Recompute => vec![true; NUM_OPS],
                    _ => vec![false; NUM_OPS],
                },
            };
            // Full recompute by default; non-Recompute actions use no recompute mask
            if !matches!(action, ActivationAction::Recompute) {
                seg_plan.recompute_ops = vec![false; NUM_OPS];
            }
            seg_plan
        })
        .collect();

    plan
}

/// Compute max absolute difference between two gradient fields.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .max_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0.0)
}

// ============================================================================
// T-SARP-A: All four actions produce bit-identical gradients
// ============================================================================

#[test]
fn test_sarp_a_bit_identical_gradients() {
    let model = make_model(4);
    let cfg = tiny_cfg();
    let inputs = make_inputs(1, &cfg);
    let upstream = make_upstream(&cfg);
    let upstreams = vec![upstream.clone()];
    let optimizer = NoOpOptimizer;

    // Baseline: Recompute (full recompute plan)
    let plan_baseline = make_plan_with_actions(4, &[(0, ActivationAction::Recompute)]);
    let fwd_baseline = full_forward_with_retention(&model, &inputs, &plan_baseline, false)
        .expect("forward baseline failed");
    let result_baseline = full_backward_sarp(
        &model,
        &fwd_baseline,
        &inputs,
        &upstreams,
        &plan_baseline,
        true,
        &optimizer,
    )
    .expect("backward baseline failed");

    let baseline_grads = &result_baseline.layer_grads;

    // Test each of the four actions
    let actions = [
        ActivationAction::RetainVram,
        ActivationAction::PageCompressedRam,
        ActivationAction::PageNvme,
        ActivationAction::Recompute,
    ];

    for action in actions.iter() {
        let plan_test = make_plan_with_actions(4, &[(0, *action)]);
        let fwd_test = full_forward_with_retention(&model, &inputs, &plan_test, false)
            .expect(&format!("forward failed for action {:?}", action));
        let result_test = full_backward_sarp(
            &model, &fwd_test, &inputs, &upstreams, &plan_test, true, &optimizer,
        )
        .expect(&format!("backward failed for action {:?}", action));

        let test_grads = &result_test.layer_grads;

        // Compare all 9 gradient fields for each layer
        for (layer_idx, (baseline_grad, test_grad)) in
            baseline_grads.iter().zip(test_grads.iter()).enumerate()
        {
            let max_d_rms1_w = max_abs_diff(&baseline_grad.d_rms1_w, &test_grad.d_rms1_w);
            assert_eq!(
                max_d_rms1_w, 0.0,
                "gradient d_rms1_w mismatch at layer {} for action {:?}: max_diff={}",
                layer_idx, action, max_d_rms1_w
            );

            let max_d_wq = max_abs_diff(&baseline_grad.d_wq, &test_grad.d_wq);
            assert_eq!(
                max_d_wq, 0.0,
                "gradient d_wq mismatch at layer {} for action {:?}: max_diff={}",
                layer_idx, action, max_d_wq
            );

            let max_d_wk = max_abs_diff(&baseline_grad.d_wk, &test_grad.d_wk);
            assert_eq!(
                max_d_wk, 0.0,
                "gradient d_wk mismatch at layer {} for action {:?}: max_diff={}",
                layer_idx, action, max_d_wk
            );

            let max_d_wv = max_abs_diff(&baseline_grad.d_wv, &test_grad.d_wv);
            assert_eq!(
                max_d_wv, 0.0,
                "gradient d_wv mismatch at layer {} for action {:?}: max_diff={}",
                layer_idx, action, max_d_wv
            );

            let max_d_wo = max_abs_diff(&baseline_grad.d_wo, &test_grad.d_wo);
            assert_eq!(
                max_d_wo, 0.0,
                "gradient d_wo mismatch at layer {} for action {:?}: max_diff={}",
                layer_idx, action, max_d_wo
            );

            let max_d_rms2_w = max_abs_diff(&baseline_grad.d_rms2_w, &test_grad.d_rms2_w);
            assert_eq!(
                max_d_rms2_w, 0.0,
                "gradient d_rms2_w mismatch at layer {} for action {:?}: max_diff={}",
                layer_idx, action, max_d_rms2_w
            );

            let max_d_wg = max_abs_diff(&baseline_grad.d_wg, &test_grad.d_wg);
            assert_eq!(
                max_d_wg, 0.0,
                "gradient d_wg mismatch at layer {} for action {:?}: max_diff={}",
                layer_idx, action, max_d_wg
            );

            let max_d_wu = max_abs_diff(&baseline_grad.d_wu, &test_grad.d_wu);
            assert_eq!(
                max_d_wu, 0.0,
                "gradient d_wu mismatch at layer {} for action {:?}: max_diff={}",
                layer_idx, action, max_d_wu
            );

            let max_d_wd = max_abs_diff(&baseline_grad.d_wd, &test_grad.d_wd);
            assert_eq!(
                max_d_wd, 0.0,
                "gradient d_wd mismatch at layer {} for action {:?}: max_diff={}",
                layer_idx, action, max_d_wd
            );
        }
    }
}

// ============================================================================
// T-SARP-B: Reference DP picks T_iter-minimizing action on known cost grids
// ============================================================================

/// Reference DP logic: select action with minimum total time.
///
/// For a segment, compute:
///   T_iter(RetainVram)      = t_retain
///   T_iter(PageCompressedRam) = t_page_lz4
///   T_iter(PageNvme)        = t_page_nvme
///   T_iter(Recompute)       = t_recompute
///
/// Return the argmin.
fn reference_dp_best_action(
    t_retain: f64,
    t_page_lz4: f64,
    t_page_nvme: f64,
    t_recompute: f64,
) -> ActivationAction {
    let mut costs = vec![
        (t_retain, ActivationAction::RetainVram),
        (t_page_lz4, ActivationAction::PageCompressedRam),
        (t_page_nvme, ActivationAction::PageNvme),
        (t_recompute, ActivationAction::Recompute),
    ];

    costs.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("cost comparison"));
    costs[0].1
}

#[test]
fn test_sarp_b_dp_retain_cheapest() {
    // Producer-side cross-check — not M5's responsibility; marks M9 SARP DP sanity

    let t_retain = 0.001;
    let t_page_lz4 = 0.010;
    let t_page_nvme = 0.050;
    let t_recompute = 0.100;

    let action = reference_dp_best_action(t_retain, t_page_lz4, t_page_nvme, t_recompute);
    assert_eq!(
        action,
        ActivationAction::RetainVram,
        "Expected RetainVram to be cheapest but got {:?} instead",
        action
    );
}

#[test]
fn test_sarp_b_dp_recompute_cheapest() {
    // Producer-side cross-check — not M5's responsibility; marks M9 SARP DP sanity

    let t_retain = 0.100;
    let t_page_lz4 = 0.050;
    let t_page_nvme = 0.030;
    let t_recompute = 0.001;

    let action = reference_dp_best_action(t_retain, t_page_lz4, t_page_nvme, t_recompute);
    assert_eq!(
        action,
        ActivationAction::Recompute,
        "Expected Recompute to be cheapest but got {:?} instead",
        action
    );
}

// ============================================================================
// T-SARP-C: Executor honors the specified action
// ============================================================================

#[test]
fn test_sarp_c_executor_honors_action() {
    let profile = HardwareProfile {
        nvme_bandwidth_gbs: 4.0,
        pcie_bandwidth_gbs: 16.0,
        gpu_bandwidth_gbs: 900.0,
        mean_forward_ms: 1.0,
        mean_backward_ms: 1.0,
        sample_count: 100,
        layer_plan: Vec::new(),
        optimal_super_shard_bytes: 0,
    };

    let plan = make_plan_with_actions(4, &[(0, ActivationAction::PageNvme)]);
    let executor = SarpExecutor::new(&plan, profile);

    let action = executor.action_for_segment(0, 1_000_000, 4);
    assert_eq!(
        action,
        ActivationAction::PageNvme,
        "Expected executor to return PageNvme but got {:?}",
        action
    );
}

// ============================================================================
// T-SARP-D: Fallback — absent schedule reproduces full_backward bit-identically
// ============================================================================

#[test]
fn test_sarp_d_fallback_reproduces_full_backward() {
    let model = make_model(4);
    let cfg = tiny_cfg();
    let inputs = make_inputs(1, &cfg);
    let upstream = make_upstream(&cfg);
    let upstreams = vec![upstream.clone()];
    let optimizer = NoOpOptimizer;

    // Empty schedule plan (no M9 SARP schedule)
    let plan_empty = TrainingPlan {
        checkpoint_freq: 4,
        micro_batch: 1,
        grad_accum: 1,
        precision_schedule: Vec::new(),
        optimizer_rank: 64,
        tier: doublepass::plan::TrainingTier::LoraOnly,
        w_max_hint: 4,
        activation_schedule: Vec::new(), // Empty!
        parity_check_interval: 500,
        projection_refresh_interval: 200,
        max_grad_norm: 1.0,
    };

    // Full forward with retention (needed for both)
    let fwd =
        full_forward_with_retention(&model, &inputs, &plan_empty, false).expect("forward failed");

    // Run full_backward_sarp with empty schedule
    let result_sarp = full_backward_sarp(
        &model,
        &fwd,
        &inputs,
        &upstreams,
        &plan_empty,
        true,
        &optimizer,
    )
    .expect("backward_sarp failed");

    // Run baseline full_backward
    let result_baseline = full_backward(
        &model,
        &fwd,
        &inputs,
        &upstreams,
        &plan_empty,
        true,
        &optimizer,
    )
    .expect("backward baseline failed");

    let sarp_grads = &result_sarp.layer_grads;
    let baseline_grads = &result_baseline.layer_grads;

    // Compare bit-identically
    for (layer_idx, (sarp_grad, baseline_grad)) in
        sarp_grads.iter().zip(baseline_grads.iter()).enumerate()
    {
        let max_d_rms1_w = max_abs_diff(&sarp_grad.d_rms1_w, &baseline_grad.d_rms1_w);
        assert_eq!(
            max_d_rms1_w, 0.0,
            "fallback: d_rms1_w mismatch at layer {}: max_diff={}",
            layer_idx, max_d_rms1_w
        );

        let max_d_wq = max_abs_diff(&sarp_grad.d_wq, &baseline_grad.d_wq);
        assert_eq!(
            max_d_wq, 0.0,
            "fallback: d_wq mismatch at layer {}: max_diff={}",
            layer_idx, max_d_wq
        );

        let max_d_wk = max_abs_diff(&sarp_grad.d_wk, &baseline_grad.d_wk);
        assert_eq!(
            max_d_wk, 0.0,
            "fallback: d_wk mismatch at layer {}: max_diff={}",
            layer_idx, max_d_wk
        );

        let max_d_wv = max_abs_diff(&sarp_grad.d_wv, &baseline_grad.d_wv);
        assert_eq!(
            max_d_wv, 0.0,
            "fallback: d_wv mismatch at layer {}: max_diff={}",
            layer_idx, max_d_wv
        );

        let max_d_wo = max_abs_diff(&sarp_grad.d_wo, &baseline_grad.d_wo);
        assert_eq!(
            max_d_wo, 0.0,
            "fallback: d_wo mismatch at layer {}: max_diff={}",
            layer_idx, max_d_wo
        );

        let max_d_rms2_w = max_abs_diff(&sarp_grad.d_rms2_w, &baseline_grad.d_rms2_w);
        assert_eq!(
            max_d_rms2_w, 0.0,
            "fallback: d_rms2_w mismatch at layer {}: max_diff={}",
            layer_idx, max_d_rms2_w
        );

        let max_d_wg = max_abs_diff(&sarp_grad.d_wg, &baseline_grad.d_wg);
        assert_eq!(
            max_d_wg, 0.0,
            "fallback: d_wg mismatch at layer {}: max_diff={}",
            layer_idx, max_d_wg
        );

        let max_d_wu = max_abs_diff(&sarp_grad.d_wu, &baseline_grad.d_wu);
        assert_eq!(
            max_d_wu, 0.0,
            "fallback: d_wu mismatch at layer {}: max_diff={}",
            layer_idx, max_d_wu
        );

        let max_d_wd = max_abs_diff(&sarp_grad.d_wd, &baseline_grad.d_wd);
        assert_eq!(
            max_d_wd, 0.0,
            "fallback: d_wd mismatch at layer {}: max_diff={}",
            layer_idx, max_d_wd
        );
    }
}

// ============================================================================
// T-SARP-E: SarpExecutor fallback path (no ham-offload feature)
// ============================================================================

#[test]
#[cfg(not(feature = "ham-offload"))]
fn test_sarp_e_no_schedule_defaults_to_recompute() {
    let profile = HardwareProfile {
        nvme_bandwidth_gbs: 4.0,
        pcie_bandwidth_gbs: 16.0,
        gpu_bandwidth_gbs: 900.0,
        mean_forward_ms: 1.0,
        mean_backward_ms: 1.0,
        sample_count: 100,
        layer_plan: Vec::new(),
        optimal_super_shard_bytes: 0,
    };

    // Empty activation_schedule
    let plan = TrainingPlan {
        checkpoint_freq: 4,
        micro_batch: 1,
        grad_accum: 1,
        precision_schedule: Vec::new(),
        optimizer_rank: 64,
        tier: doublepass::plan::TrainingTier::LoraOnly,
        w_max_hint: 4,
        activation_schedule: Vec::new(),
        parity_check_interval: 500,
        projection_refresh_interval: 200,
        max_grad_norm: 1.0,
    };

    let executor = SarpExecutor::new(&plan, profile);
    assert!(
        !executor.has_m9_schedule(),
        "Expected executor to report no M9 schedule"
    );

    let action = executor.action_for_segment(0, 1_000_000, 4);
    assert_eq!(
        action,
        ActivationAction::Recompute,
        "Expected fallback without ham-offload to default to Recompute, but got {:?}",
        action
    );
}
