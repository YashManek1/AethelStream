//! A1 → A8 → A2′ → A6 wired training step.

use crate::{
    backward::{full_backward_sarp, ParamGrads},
    forward::{full_forward_with_retention, Model},
    hook::{deferred_apply_with_clip, ClipResult},
    loss::streaming_cut_ce,
    plan::TrainingPlan,
    OptimizerBackend, Result,
};

/// Per-step configuration knobs not stored in TrainingPlan.
#[derive(Debug, Clone)]
pub struct StepConfig {
    /// Vocabulary size for streaming_cut_ce.
    pub vocab_size: usize,
    /// Vocabulary tile width for the streaming two-pass softmax (O(batch_seq × chunk_size) peak).
    pub chunk_size: usize,
    /// When true, weight-loads for the recompute pass are counted as zero.
    pub keep_resident: bool,
    /// When true, sparse checkpoints are INT8-compressed at rest.
    pub compress_checkpoints: bool,
}

impl Default for StepConfig {
    fn default() -> Self {
        Self {
            vocab_size: 0,
            chunk_size: 256,
            keep_resident: false,
            compress_checkpoints: false,
        }
    }
}

/// Output of one complete A1 → A8 → A2′ → A6 step.
#[derive(Debug, Clone)]
pub struct StepOut {
    /// Scalar mean cross-entropy loss (natural log, mean over batch_seq).
    pub loss: f32,
    /// Per-layer parameter gradients accumulated across micro-batches by A2′.
    pub layer_grads: Vec<ParamGrads>,
    /// Raw (pre-clip) global gradient L2 norm computed from layer_grads.
    pub global_grad_norm: f64,
    /// Diagnostic output from A6 deferred_apply_with_clip.
    pub clip_result: ClipResult,
    /// Weight bytes streamed during forward + backward.
    pub weight_loads: u64,
}

/// Wire A1 (forward) → A8 (loss) → A2′ (backward/SARP) → A6 (clip+apply).
///
/// - `model`: Transformer layers; read-only.
/// - `lm_head`: `[vocab_size × d_model]` row-major LM-head projection.
/// - `inputs`: One `Vec<f32>` per micro-batch; each has shape `[batch * seq_len * d_model]`.
/// - `labels`: Token ids concatenated across all micro-batches; length `G * batch * seq_len`.
/// - `plan`: Frozen training plan (checkpoint freq, precision schedule, etc.).
/// - `cfg`: Step-level knobs (vocab_size, chunk_size, keep_resident, compress).
/// - `optimizer`: M4 backend; receives LM-head tile grads via A8 hook, per-layer grads via A3
///   hook inside A2′, then A6 calls apply_update + zero_accum per parameter.
/// - `trainable_layers`: `(layer_idx, param_name)` pairs for A6 global-norm clip.
///   Pass an empty slice to skip A6 (clip_coeff = 1.0, no apply_update calls).
#[allow(clippy::too_many_arguments)]
pub fn full_training_step(
    model: &Model,
    lm_head: &[f32],
    inputs: &[Vec<f32>],
    labels: &[u32],
    plan: &TrainingPlan,
    cfg: &StepConfig,
    optimizer: &dyn OptimizerBackend,
    trainable_layers: &[(u32, String)],
) -> Result<StepOut> {
    let d_model = model.cfg.d_model;
    let g = inputs.len();

    // A1 — full forward with activation retention for SARP segment dispatch.
    let fwd = full_forward_with_retention(model, inputs, plan, cfg.compress_checkpoints)?;

    // A8 — flatten per-micro-batch outputs into a single hidden tensor for loss.
    let hidden: Vec<f32> = fwd.outputs.iter().flat_map(|o| o.iter().copied()).collect();
    let loss_out = streaming_cut_ce(
        &hidden,
        lm_head,
        d_model,
        cfg.vocab_size,
        cfg.chunk_size,
        labels,
        Some(optimizer),
    )?;

    // Partition grad_hidden back into one slice per micro-batch.
    let tokens_per_mb = loss_out.grad_hidden.len().checked_div(g).unwrap_or(0);
    let upstream_grads: Vec<Vec<f32>> = (0..g)
        .map(|i| loss_out.grad_hidden[i * tokens_per_mb..(i + 1) * tokens_per_mb].to_vec())
        .collect();

    // A2′ — backward with SARP segment dispatch (falls back to full_backward when no M9 schedule).
    let bwd = full_backward_sarp(
        model,
        &fwd,
        inputs,
        &upstream_grads,
        plan,
        cfg.keep_resident,
        optimizer,
    )?;

    // Pre-clip global gradient L2 norm.
    let global_grad_norm = grad_l2_norm(&bwd.layer_grads);

    // A6 — exact (or grouped-fallback) global-norm clip + optimizer apply.
    let clip_result = if trainable_layers.is_empty() {
        ClipResult {
            global_grad_norm,
            clip_coeff: 1.0,
            clipped: false,
            used_grouped_fallback: false,
            frobenius_exact_layers: 0,
        }
    } else {
        deferred_apply_with_clip(optimizer, plan, trainable_layers)?
    };

    Ok(StepOut {
        loss: loss_out.loss,
        layer_grads: bwd.layer_grads,
        global_grad_norm,
        clip_result,
        weight_loads: bwd.weight_loads,
    })
}

/// L2 norm of all parameter gradients across all layers.
fn grad_l2_norm(grads: &[ParamGrads]) -> f64 {
    let mut sq = 0.0_f64;
    for g in grads {
        for &x in &g.d_rms1_w {
            sq += f64::from(x) * f64::from(x);
        }
        for &x in &g.d_wq {
            sq += f64::from(x) * f64::from(x);
        }
        for &x in &g.d_wk {
            sq += f64::from(x) * f64::from(x);
        }
        for &x in &g.d_wv {
            sq += f64::from(x) * f64::from(x);
        }
        for &x in &g.d_wo {
            sq += f64::from(x) * f64::from(x);
        }
        for &x in &g.d_rms2_w {
            sq += f64::from(x) * f64::from(x);
        }
        for &x in &g.d_wg {
            sq += f64::from(x) * f64::from(x);
        }
        for &x in &g.d_wu {
            sq += f64::from(x) * f64::from(x);
        }
        for &x in &g.d_wd {
            sq += f64::from(x) * f64::from(x);
        }
    }
    sq.sqrt()
}
