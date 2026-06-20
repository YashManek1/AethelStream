//! T-SEG — Segment-wise recompute: multi-segment model backward grads vs. full
//! per-layer reference backward. Asserts max|Δ| < 1e-5.
//! Also asserts weight-load count == 2·L (keep_resident) and 3·L (not).
//!
//! Run: cargo test --features mock-cuda -p doublepass test_segment_recompute -- --nocapture
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Mutex;
use doublepass::forward::{BlockConfig, Model, full_forward, single_layer_forward};
use doublepass::backward::{single_layer_backward, full_backward};
use doublepass::plan::TrainingPlan;
use doublepass::OptimizerBackend;
use doublepass::rng;

const CFG: BlockConfig = BlockConfig {
    d_model: 8,
    n_heads: 2,
    d_ff: 16,
    seq_len: 4,
    batch: 1,
    dropout_p: 0.0,
};

/// A mock optimizer that records project_and_accumulate calls.
struct MockOptimizer {
    calls: Mutex<Vec<(u32, String)>>,
}

impl MockOptimizer {
    fn new() -> Self {
        Self { calls: Mutex::new(Vec::new()) }
    }
    fn call_count(&self) -> usize {
        self.calls.lock().expect("lock").len()
    }
}

impl OptimizerBackend for MockOptimizer {
    fn project_and_accumulate(&self, _grad: &[f32], layer_idx: u32, param_name: &str) {
        if let Ok(mut c) = self.calls.lock() {
            c.push((layer_idx, param_name.to_string()));
        }
    }
    fn lowrank_grad_sqnorm(&self, _layer_idx: u32, _param_name: &str) -> f64 { 0.0 }
    fn apply_update(&self, _layer_idx: u32, _param_name: &str, _clip_scale: f32) {}
    fn zero_accum(&self, _layer_idx: u32, _param_name: &str) {}
    fn notify_step(&self, _step: u64) {}
}

fn init_inputs(g: usize) -> Vec<Vec<f32>> {
    let n = CFG.batch * CFG.seq_len * CFG.d_model;
    (0..g)
        .map(|m| (0..n).map(|i| ((i * 7 + m * 13) as f64 * 0.07).sin() as f32).collect())
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

/// Build the reference gradients by running the full forward (storing ALL layer fwd
/// states) then backward descending layer-by-layer.
///
/// Returns `(layer_grads, final_outputs)` where `layer_grads[i]` is the grad for layer i
/// accumulated over G micro-batches (summed).
fn reference_backward(
    model: &Model,
    inputs: &[Vec<f32>],
    upstream_grads: &[Vec<f32>],
) -> Vec<doublepass::backward::ParamGrads> {
    let l = model.num_layers();
    let g = inputs.len();

    // Full forward storing ALL layer outputs and fwd states per micro-batch.
    // all_fwd[m][i] = SingleLayerFwdOut for layer i, micro-batch m.
    let mut all_fwd: Vec<Vec<doublepass::forward::SingleLayerFwdOut>> =
        (0..g).map(|_| Vec::with_capacity(l)).collect();

    for m in 0..g {
        let mut act = inputs[m].clone();
        for i in 0..l {
            let fwd_out = single_layer_forward(&CFG, &model.layers[i], &act);
            act = fwd_out.output.clone();
            all_fwd[m].push(fwd_out);
        }
    }

    // Backward descending: accumulate grads over G for each layer.
    let mut upstreams: Vec<Vec<f32>> = upstream_grads.to_vec();
    let mut layer_grads: Vec<Option<doublepass::backward::ParamGrads>> =
        (0..l).map(|_| None).collect();

    for i in (0..l).rev() {
        let mut acc: Option<doublepass::backward::ParamGrads> = None;
        for m in 0..g {
            let grads = single_layer_backward(&CFG, &model.layers[i], &all_fwd[m][i], &upstreams[m]);
            upstreams[m] = grads.d_input.clone();
            match &mut acc {
                None => acc = Some(grads),
                Some(a) => {
                    for (x, y) in a.d_rms1_w.iter_mut().zip(&grads.d_rms1_w) { *x += y; }
                    for (x, y) in a.d_wq.iter_mut().zip(&grads.d_wq) { *x += y; }
                    for (x, y) in a.d_wk.iter_mut().zip(&grads.d_wk) { *x += y; }
                    for (x, y) in a.d_wv.iter_mut().zip(&grads.d_wv) { *x += y; }
                    for (x, y) in a.d_wo.iter_mut().zip(&grads.d_wo) { *x += y; }
                    for (x, y) in a.d_rms2_w.iter_mut().zip(&grads.d_rms2_w) { *x += y; }
                    for (x, y) in a.d_wg.iter_mut().zip(&grads.d_wg) { *x += y; }
                    for (x, y) in a.d_wu.iter_mut().zip(&grads.d_wu) { *x += y; }
                    for (x, y) in a.d_wd.iter_mut().zip(&grads.d_wd) { *x += y; }
                }
            }
        }
        layer_grads[i] = acc;
    }

    layer_grads.into_iter().map(|o| o.expect("layer visited")).collect()
}

/// T-SEG: segment-wise backward grads == per-layer reference backward.
/// Uses 6 layers, checkpoint_freq=2 → 3 segments.
#[test]
#[cfg(feature = "mock-cuda")]
fn test_segment_backward_matches_reference() {
    const N_LAYERS: usize = 6;
    const G: usize = 2;
    rng::set_step_seed(0xabcd_ef12_3456_7890);

    let model = Model::new(N_LAYERS, CFG);
    let inputs = init_inputs(G);
    let mut plan = TrainingPlan::default();
    plan.checkpoint_freq = 2;

    // Full forward (stores checkpoint = INPUT to each segment start).
    let fwd_result = full_forward(&model, &inputs, &plan, false).expect("full_forward");

    // Upstream grads = ones (summing all outputs).
    let upstream: Vec<Vec<f32>> = (0..G)
        .map(|_| vec![1.0f32; CFG.batch * CFG.seq_len * CFG.d_model])
        .collect();

    // Reference: naive per-layer backward.
    let ref_grads = reference_backward(&model, &inputs, &upstream);

    // Under test: segment-wise backward (keep_resident=true).
    let optimizer = MockOptimizer::new();
    let bwd_result = full_backward(
        &model,
        &fwd_result,
        &inputs,
        &upstream,
        &plan,
        true,
        &optimizer,
    ).expect("full_backward");

    let tol = 1e-5f32;
    println!("\nT-SEG — max|Δ| per layer and parameter (tol = {:.0e}):", tol);

    macro_rules! cmp_field {
        ($li:expr, $name:expr, $ref_f:ident, $test_f:ident) => {{
            let diff = max_abs_diff(&ref_grads[$li].$ref_f, &bwd_result.layer_grads[$li].$test_f);
            if diff >= tol {
                panic!("T-SEG FAILED layer={} {}: max|Δ|={:.3e} >= {:.3e}", $li, $name, diff, tol);
            }
            diff
        }};
    }

    let mut max_overall = 0.0f32;
    for i in 0..N_LAYERS {
        let d = [
            cmp_field!(i, "d_rms1_w", d_rms1_w, d_rms1_w),
            cmp_field!(i, "d_wq",     d_wq,     d_wq),
            cmp_field!(i, "d_wk",     d_wk,     d_wk),
            cmp_field!(i, "d_wv",     d_wv,     d_wv),
            cmp_field!(i, "d_wo",     d_wo,     d_wo),
            cmp_field!(i, "d_rms2_w", d_rms2_w, d_rms2_w),
            cmp_field!(i, "d_wg",     d_wg,     d_wg),
            cmp_field!(i, "d_wu",     d_wu,     d_wu),
            cmp_field!(i, "d_wd",     d_wd,     d_wd),
        ];
        let layer_max = d.iter().copied().fold(0.0f32, f32::max);
        max_overall = max_overall.max(layer_max);
        println!("  layer={}  max|Δ|={:.3e}", i, layer_max);
    }
    println!("T-SEG overall max|Δ|={:.3e}  PASSED", max_overall);

    // Hook was called: 9 params × N_LAYERS calls expected.
    let expected_calls = 9 * N_LAYERS;
    assert_eq!(
        optimizer.call_count(), expected_calls,
        "hook call count: expected {expected_calls}, got {}", optimizer.call_count()
    );
    println!("Hook calls: {} (expected {})", optimizer.call_count(), expected_calls);
}

/// T-SEG weight-load count: 2·L (keep_resident) and 3·L (not).
#[test]
#[cfg(feature = "mock-cuda")]
fn test_weight_load_count() {
    const N_LAYERS: usize = 6;
    rng::set_step_seed(0xfeed_c0de_dead_beef);

    let model = Model::new(N_LAYERS, CFG);
    let inputs = init_inputs(1);
    let mut plan = TrainingPlan::default();
    plan.checkpoint_freq = 2;

    let fwd_result = full_forward(&model, &inputs, &plan, false).expect("full_forward");
    let upstream = vec![vec![1.0f32; CFG.batch * CFG.seq_len * CFG.d_model]];
    let optimizer = MockOptimizer::new();

    let bpl = CFG.bytes_per_layer() as u64;
    let expected_resident = N_LAYERS as u64 * bpl;
    let expected_restream = 2 * N_LAYERS as u64 * bpl;

    // keep_resident = true → 2·L
    let res_true = full_backward(&model, &fwd_result, &inputs, &upstream, &plan, true, &optimizer)
        .expect("backward keep_resident=true");
    println!("\nWeight loads (keep_resident=true):  {} (expected {})", res_true.weight_loads, expected_resident);
    assert_eq!(res_true.weight_loads, expected_resident,
        "keep_resident=true: expected {expected_resident}, got {}", res_true.weight_loads);

    // keep_resident = false → 3·L
    let fwd_result2 = full_forward(&model, &inputs, &plan, false).expect("full_forward2");
    let res_false = full_backward(&model, &fwd_result2, &inputs, &upstream, &plan, false, &optimizer)
        .expect("backward keep_resident=false");
    println!("Weight loads (keep_resident=false): {} (expected {})", res_false.weight_loads, expected_restream);
    assert_eq!(res_false.weight_loads, expected_restream,
        "keep_resident=false: expected {expected_restream}, got {}", res_false.weight_loads);

    println!("T-SEG weight-load counts PASSED  (2·L={expected_resident}, 3·L={expected_restream})");
}
