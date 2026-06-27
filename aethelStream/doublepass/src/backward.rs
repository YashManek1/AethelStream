//! A2 backward

use crate::math::silu_grad_f;

use crate::OptimizerBackend;

use crate::forward::{BlockConfig, BlockWeights, SingleLayerFwdOut};


use crate::plan::TrainingPlan;

#[derive(Debug, Clone)]

/// Grads.
pub struct ParamGrads {
    /// d.
    pub d_rms1_w: Vec<f32>,

    /// d.
    pub d_wq: Vec<f32>,

    /// d.
    pub d_wk: Vec<f32>,

    /// d.
    pub d_wv: Vec<f32>,

    /// d.
    pub d_wo: Vec<f32>,

    /// d.
    pub d_rms2_w: Vec<f32>,

    /// d.
    pub d_wg: Vec<f32>,

    /// d.
    pub d_wu: Vec<f32>,

    /// d.
    pub d_wd: Vec<f32>,

    /// Gradient with respect to the layer input, for propagation to the next lower layer.
    pub d_input: Vec<f32>,
}

/// Back.
pub fn single_layer_backward(
    cfg: &BlockConfig,

    w: &BlockWeights,

    fwd: &SingleLayerFwdOut,

    upstream: &[f32],
) -> ParamGrads {
    let bs = cfg.bs();

    let d = cfg.d_model;

    let h = cfg.n_heads;

    let dh = cfg.d_head();

    let bh = cfg.batch * cfg.n_heads;

    let s = cfg.seq_len;

    let ff = cfg.d_ff;

    let mut d_x2 = upstream.to_vec();

    let d_mlp_out = upstream.to_vec();

    let mut d_hidden = vec![0.0f32; bs * ff];

    let mut d_wd = vec![0.0f32; d * ff];

    for i in 0..bs {
        for j in 0..ff {
            for k in 0..d {
                d_hidden[i * ff + j] += d_mlp_out[i * d + k] * w.wd[k * ff + j];
            }
        }
    }

    for k in 0..d {
        for j in 0..ff {
            for i in 0..bs {
                d_wd[k * ff + j] += d_mlp_out[i * d + k] * fwd.hidden[i * ff + j];
            }
        }
    }

    let mut d_silu_gate = vec![0.0f32; bs * ff];

    let mut d_up = vec![0.0f32; bs * ff];

    for i in 0..bs * ff {
        d_silu_gate[i] = d_hidden[i] * fwd.up[i];

        d_up[i] = d_hidden[i] * fwd.silu_gate[i];
    }

    let mut d_gate = vec![0.0f32; bs * ff];

    for i in 0..bs * ff {
        d_gate[i] = d_silu_gate[i] * silu_grad_f(fwd.gate[i]);
    }

    let mut d_h2 = vec![0.0f32; bs * d];

    let mut d_wg = vec![0.0f32; ff * d];

    let mut d_wu = vec![0.0f32; ff * d];

    for i in 0..bs {
        for j in 0..ff {
            for k in 0..d {
                d_h2[i * d + k] += d_gate[i * ff + j] * w.wg[j * d + k];

                d_wg[j * d + k] += d_gate[i * ff + j] * fwd.h2[i * d + k];

                d_h2[i * d + k] += d_up[i * ff + j] * w.wu[j * d + k];

                d_wu[j * d + k] += d_up[i * ff + j] * fwd.h2[i * d + k];
            }
        }
    }

    let mut d_rms2_w = vec![0.0f32; d];

    let eps = 1e-6f32;

    for b in 0..bs {
        for i in 0..d {
            d_rms2_w[i] += d_h2[b * d + i] * fwd.x_norm2[b * d + i];
        }

        let d_x_norm2: Vec<f32> = (0..d).map(|i| d_h2[b * d + i] * w.rms2_w[i]).collect();

        let dot: f32 = (0..d)
            .map(|i| d_x_norm2[i] * fwd.x_norm2[b * d + i])
            .sum::<f32>()
            / d as f32;

        for i in 0..d {
            let rms2_plus_eps = fwd.rms2[b].max(eps);

            d_x2[b * d + i] += (d_x_norm2[i] - fwd.x_norm2[b * d + i] * dot) / rms2_plus_eps;
        }
    }

    let d_out_proj = d_x2.clone();

    let mut d_attn_out = vec![0.0f32; bs * d];

    let mut d_wo = vec![0.0f32; d * d];

    for i in 0..bs {
        for j in 0..d {
            for k in 0..d {
                d_attn_out[i * d + k] += d_out_proj[i * d + j] * w.wo[j * d + k];

                d_wo[j * d + k] += d_out_proj[i * d + j] * fwd.attn_out[i * d + k];
            }
        }
    }

    let mut d_attn_out_heads = vec![0.0f32; bh * s * dh];

    for b in 0..cfg.batch {
        for hh in 0..h {
            for ss in 0..s {
                for t in 0..dh {
                    d_attn_out_heads[((b * h + hh) * s + ss) * dh + t] =
                        d_attn_out[(b * s + ss) * d + hh * dh + t];
                }
            }
        }
    }

    let mut d_v_heads = vec![0.0f32; bh * s * dh];

    for bh_i in 0..bh {
        for s2 in 0..s {
            for t in 0..dh {
                for s1 in 0..s {
                    d_v_heads[bh_i * s * dh + s2 * dh + t] += fwd.attn_weights
                        [bh_i * s * s + s1 * s + s2]
                        * d_attn_out_heads[bh_i * s * dh + s1 * dh + t];
                }
            }
        }
    }

    let mut d_attn_logits = vec![0.0f32; bh * s * s];

    for bh_i in 0..bh {
        for s1 in 0..s {
            for s2 in 0..s {
                for t in 0..dh {
                    d_attn_logits[bh_i * s * s + s1 * s + s2] += d_attn_out_heads
                        [bh_i * s * dh + s1 * dh + t]
                        * fwd.v_heads[bh_i * s * dh + s2 * dh + t];
                }
            }
        }
    }

    let mut d_scores = vec![0.0f32; bh * s * s];

    for bh_i in 0..bh {
        for s1 in 0..s {
            let dot: f32 = (0..s)
                .map(|s2| {
                    d_attn_logits[bh_i * s * s + s1 * s + s2]
                        * fwd.attn_weights[bh_i * s * s + s1 * s + s2]
                })
                .sum();

            for s2 in 0..s {
                d_scores[bh_i * s * s + s1 * s + s2] = fwd.attn_weights[bh_i * s * s + s1 * s + s2]
                    * (d_attn_logits[bh_i * s * s + s1 * s + s2] - dot);
            }
        }
    }

    let scale = (dh as f32).sqrt();

    for v in &mut d_scores {
        *v /= scale;
    }

    let mut d_q_heads = vec![0.0f32; bh * s * dh];

    let mut d_k_heads = vec![0.0f32; bh * s * dh];

    for bh_i in 0..bh {
        for s1 in 0..s {
            for t in 0..dh {
                for s2 in 0..s {
                    d_q_heads[bh_i * s * dh + s1 * dh + t] += d_scores[bh_i * s * s + s1 * s + s2]
                        * fwd.k_heads[bh_i * s * dh + s2 * dh + t];

                    d_k_heads[bh_i * s * dh + s2 * dh + t] += d_scores[bh_i * s * s + s1 * s + s2]
                        * fwd.q_heads[bh_i * s * dh + s1 * dh + t];
                }
            }
        }
    }

    let reshape_from_heads = |heads: &[f32]| -> Vec<f32> {
        let mut flat = vec![0.0f32; bs * d];

        for b in 0..cfg.batch {
            for hh in 0..h {
                for ss in 0..s {
                    for t in 0..dh {
                        flat[(b * s + ss) * d + hh * dh + t] =
                            heads[((b * h + hh) * s + ss) * dh + t];
                    }
                }
            }
        }

        flat
    };

    let d_q_flat = reshape_from_heads(&d_q_heads);

    let d_k_flat = reshape_from_heads(&d_k_heads);

    let d_v_flat = reshape_from_heads(&d_v_heads);

    let mut d_wq = vec![0.0f32; d * d];

    let mut d_wk = vec![0.0f32; d * d];

    let mut d_wv = vec![0.0f32; d * d];

    let mut d_h1 = vec![0.0f32; bs * d];

    for i in 0..bs {
        for j in 0..d {
            for k in 0..d {
                d_h1[i * d + k] += d_q_flat[i * d + j] * w.wq[j * d + k];

                d_wq[j * d + k] += d_q_flat[i * d + j] * fwd.h1[i * d + k];

                d_h1[i * d + k] += d_k_flat[i * d + j] * w.wk[j * d + k];

                d_wk[j * d + k] += d_k_flat[i * d + j] * fwd.h1[i * d + k];

                d_h1[i * d + k] += d_v_flat[i * d + j] * w.wv[j * d + k];

                d_wv[j * d + k] += d_v_flat[i * d + j] * fwd.h1[i * d + k];
            }
        }
    }

    let mut d_rms1_w = vec![0.0f32; d];

    for b in 0..bs {
        for i in 0..d {
            d_rms1_w[i] += d_h1[b * d + i] * fwd.x_norm1[b * d + i];
        }
    }

    // Backward through RMSNorm1 to propagate gradient to layer input.
    // d_x_in = d_x2 (residual skip) + d_from_attn_through_norm1.
    let eps = 1e-6f32;
    let mut d_x_in_from_attn = vec![0.0f32; bs * d];
    for b in 0..bs {
        let rms1_eps = fwd.rms1[b].max(eps);
        let dot: f32 = (0..d)
            .map(|i| (d_h1[b * d + i] * w.rms1_w[i]) * fwd.x_norm1[b * d + i])
            .sum::<f32>()
            / d as f32;
        for i in 0..d {
            let d_h1_norm_i = d_h1[b * d + i] * w.rms1_w[i];
            d_x_in_from_attn[b * d + i] = (d_h1_norm_i - fwd.x_norm1[b * d + i] * dot) / rms1_eps;
        }
    }
    // Total: residual skip (d_x2) + attention branch (d_x_in_from_attn).
    let d_input: Vec<f32> = d_x2
        .iter()
        .zip(&d_x_in_from_attn)
        .map(|(a, b)| a + b)
        .collect();

    ParamGrads {
        d_rms1_w,

        d_wq,

        d_wk,

        d_wv,

        d_wo,

        d_rms2_w,

        d_wg,

        d_wu,

        d_wd,

        d_input,
    }
}

/// Result of [`full_backward`]: accumulated parameter gradients and weight-load count.
pub struct FullBackwardResult {
    /// Per-layer parameter gradients, indexed by `layer_idx`.
    ///
    /// Accumulated (summed) over all `G` micro-batches.
    pub layer_grads: Vec<ParamGrads>,
    /// Weight bytes loaded during this backward pass.
    ///
    /// Equals `L × bytes_per_layer` when `keep_resident = true` (backward only).
    /// Equals `2 × L × bytes_per_layer` when `keep_resident = false` (recompute + backward).
    pub weight_loads: u64,
}

/// Segment-wise backward pass (A2): pure math, no FlowCast I/O.
///
/// Processes segments in **reverse** order. For each segment:
/// 1. **Recompute-forward** (ascending): restore RNG, re-run each layer to recover
///    intermediate activations needed for backward.
/// 2. **Backward** (descending): call [`single_layer_backward`] per layer, accumulate
///    weight gradients over `G` micro-batches, fire the A3 hook
///    (`optimizer.project_and_accumulate`) once per layer.
///
/// `keep_resident` controls whether weight loads are counted for the recompute pass
/// (false = re-stream, adds `L × bytes_per_layer`; true = weights already in VRAM).
///
/// `upstream_grads` is the gradient of the loss w.r.t. the model's final output,
/// one `Vec<f32>` per micro-batch (`G = upstream_grads.len()`).
///
/// # Errors
/// Propagates checkpoint read errors as [`crate::error::DoublePassError::Checkpoint`].
pub fn full_backward(
    model: &crate::forward::Model,
    fwd: &crate::forward::FullForwardResult,
    inputs: &[Vec<f32>],
    upstream_grads: &[Vec<f32>],
    plan: &TrainingPlan,
    keep_resident: bool,
    optimizer: &dyn OptimizerBackend,
) -> crate::Result<FullBackwardResult> {
    let l = model.layers.len();
    let g = inputs.len();
    let k = plan.checkpoint_freq as usize;
    let bpl = model.cfg.bytes_per_layer() as u64;

    let schedule = crate::schedule::LayerSchedule::new(l as u32, k as u32);

    // One upstream gradient per micro-batch; updated in-place as backward propagates.
    let mut upstreams: Vec<Vec<f32>> = upstream_grads.to_vec();

    // Accumulate weight grads per layer (initialise to zero-length, fill on first visit).
    let mut layer_grads: Vec<Option<ParamGrads>> = (0..l).map(|_| None).collect();

    // Weight loads: always count backward (1×L). Add recompute if !keep_resident.
    let backward_loads = bpl.saturating_mul(l as u64);
    let recompute_loads = if keep_resident { 0 } else { backward_loads };
    let weight_loads = backward_loads.saturating_add(recompute_loads);

    for seg in schedule.backward_segments() {
        let seg_layers = &seg.layers_ascending;
        let seg_start = seg_layers[0] as usize;

        // ── Recompute-forward within this segment ────────────────────────────
        // For each micro-batch: read checkpoint (= input to seg_start), restore RNG,
        // re-run each layer in the segment ascending, collect SingleLayerFwdOut.
        let mut seg_fwd: Vec<Vec<crate::forward::SingleLayerFwdOut>> = (0..g)
            .map(|_| Vec::with_capacity(seg_layers.len()))
            .collect();

        #[allow(clippy::needless_range_loop)]
        #[allow(clippy::needless_range_loop)]
        for m in 0..g {
            // Checkpoint at (seg_start, m) = input activation to layer seg_start.
            let ckpt_buf = fwd
                .checkpoints
                .iter()
                .find(|(li, mi, _)| *li == seg_start as u32 && *mi == m as u32)
                .map(|(_, _, buf)| buf)
                .ok_or_else(|| {
                    crate::error::DoublePassError::Checkpoint(format!(
                        "checkpoint missing for layer={seg_start} micro_batch={m}"
                    ))
                })?;

            let mut act = crate::checkpoint::read_checkpoint(ckpt_buf)?;

            for &layer_idx in seg_layers {
                let layer_idx_usize = layer_idx as usize;
                // Restore RNG so dropout masks match the original forward exactly.
                let rng_idx = layer_idx_usize * g + m;
                if let Some(rng_state) = fwd.rng_states.get(rng_idx) {
                    crate::rng::restore(rng_state)?;
                }
                let fwd_out = crate::forward::single_layer_forward(
                    &model.cfg,
                    &model.layers[layer_idx_usize],
                    &act,
                );
                act = fwd_out.output.clone();
                seg_fwd[m].push(fwd_out);
            }
        }

        // ── Backward within this segment (descending) ────────────────────────
        for (i_in_seg, &layer_idx) in seg_layers.iter().enumerate().rev() {
            let layer_idx_usize = layer_idx as usize;
            let mut acc_grads: Option<ParamGrads> = None;

            #[allow(clippy::needless_range_loop)]
            for m in 0..g {
                let fwd_out = &seg_fwd[m][i_in_seg];
                let grads = single_layer_backward(
                    &model.cfg,
                    &model.layers[layer_idx_usize],
                    fwd_out,
                    &upstreams[m],
                );
                // Propagate gradient to next lower layer.
                upstreams[m] = grads.d_input.clone();

                // Accumulate weight grads over G micro-batches.
                match &mut acc_grads {
                    None => acc_grads = Some(grads),
                    Some(acc) => {
                        for (a, b) in acc.d_rms1_w.iter_mut().zip(&grads.d_rms1_w) {
                            *a += b;
                        }
                        for (a, b) in acc.d_wq.iter_mut().zip(&grads.d_wq) {
                            *a += b;
                        }
                        for (a, b) in acc.d_wk.iter_mut().zip(&grads.d_wk) {
                            *a += b;
                        }
                        for (a, b) in acc.d_wv.iter_mut().zip(&grads.d_wv) {
                            *a += b;
                        }
                        for (a, b) in acc.d_wo.iter_mut().zip(&grads.d_wo) {
                            *a += b;
                        }
                        for (a, b) in acc.d_rms2_w.iter_mut().zip(&grads.d_rms2_w) {
                            *a += b;
                        }
                        for (a, b) in acc.d_wg.iter_mut().zip(&grads.d_wg) {
                            *a += b;
                        }
                        for (a, b) in acc.d_wu.iter_mut().zip(&grads.d_wu) {
                            *a += b;
                        }
                        for (a, b) in acc.d_wd.iter_mut().zip(&grads.d_wd) {
                            *a += b;
                        }
                    }
                }
            }

            // Fire A3 hook: project-and-accumulate per parameter (DO NOT apply yet).
            if let Some(ref acc) = acc_grads {
                let li = layer_idx_usize as u32;
                optimizer.project_and_accumulate(&acc.d_rms1_w, li, "d_rms1_w");
                optimizer.project_and_accumulate(&acc.d_wq, li, "d_wq");
                optimizer.project_and_accumulate(&acc.d_wk, li, "d_wk");
                optimizer.project_and_accumulate(&acc.d_wv, li, "d_wv");
                optimizer.project_and_accumulate(&acc.d_wo, li, "d_wo");
                optimizer.project_and_accumulate(&acc.d_rms2_w, li, "d_rms2_w");
                optimizer.project_and_accumulate(&acc.d_wg, li, "d_wg");
                optimizer.project_and_accumulate(&acc.d_wu, li, "d_wu");
                optimizer.project_and_accumulate(&acc.d_wd, li, "d_wd");
            }

            layer_grads[layer_idx_usize] = acc_grads;
        }
    }

    // Unwrap Option<ParamGrads> → ParamGrads (every layer must have been visited).
    let mut result_grads: Vec<ParamGrads> = Vec::with_capacity(l);
    for (idx, opt) in layer_grads.into_iter().enumerate() {
        match opt {
            Some(g) => result_grads.push(g),
            None => {
                return Err(crate::error::DoublePassError::Checkpoint(format!(
                    "layer {idx} was never reached during backward"
                )))
            }
        }
    }

    Ok(FullBackwardResult {
        layer_grads: result_grads,
        weight_loads,
    })
}

/// A2 SARP.
pub fn full_backward_sarp(
    model: &crate::forward::Model,
    fwd: &crate::forward::FullForwardResult,
    inputs: &[Vec<f32>],
    upstream_grads: &[Vec<f32>],
    plan: &TrainingPlan,
    keep_resident: bool,
    optimizer: &dyn OptimizerBackend,
) -> crate::Result<FullBackwardResult> {
    if !plan.has_sarp_schedule() {
        return full_backward(
            model,
            fwd,
            inputs,
            upstream_grads,
            plan,
            keep_resident,
            optimizer,
        );
    }
    let l = model.layers.len();
    let g = inputs.len();
    let bpl = model.cfg.bytes_per_layer() as u64;
    let schedule = crate::schedule::LayerSchedule::new(l as u32, plan.checkpoint_freq);
    let mut upstreams: Vec<Vec<f32>> = upstream_grads.to_vec();
    let mut layer_grads: Vec<Option<ParamGrads>> = (0..l).map(|_| None).collect();
    let backward_loads = bpl.saturating_mul(l as u64);
    let recompute_loads = if keep_resident { 0 } else { backward_loads };
    let weight_loads = backward_loads.saturating_add(recompute_loads);
    for seg in schedule.backward_segments() {
        let seg_layers = &seg.layers_ascending;
        let seg_start = seg_layers[0] as usize;
        let seg_index = seg.segment_index;
        let seg_plan = plan.segment_plan(seg_index);
        let mut seg_fwd: Vec<Vec<crate::forward::SingleLayerFwdOut>> = (0..g)
            .map(|_| Vec::with_capacity(seg_layers.len()))
            .collect();
        use crate::plan::ActivationAction;
        let action = seg_plan
            .map(|sp| sp.action)
            .unwrap_or(ActivationAction::Recompute);
        match action {
            ActivationAction::RetainVram
            | ActivationAction::PageCompressedRam
            | ActivationAction::PageNvme => {
                let mut pcie_bytes: u64 = 0;
                let mut ssd_bytes: u64 = 0;
                #[allow(clippy::needless_range_loop)]
                for m in 0..g {
                    for &layer_idx in seg_layers {
                        let fwd_out = fwd
                            .retained_activations
                            .iter()
                            .find(|(li, mi, _)| *li == layer_idx && *mi == m as u32)
                            .map(|(_, _, out)| out)
                            .ok_or_else(|| {
                                crate::error::DoublePassError::Checkpoint(
                                    "retained missing".to_string(),
                                )
                            })?;
                        let ec = count_activation_elements(fwd_out);
                        match action {
                            ActivationAction::PageCompressedRam => {
                                pcie_bytes = pcie_bytes.saturating_add(ec as u64 * 4)
                            }
                            ActivationAction::PageNvme => {
                                ssd_bytes = ssd_bytes.saturating_add(ec as u64 * 4)
                            }
                            _ => {}
                        }
                        seg_fwd[m].push(fwd_out.clone());
                    }
                }
                eprintln!("[sarp-bwd] seg={}", seg_index);
            }
            ActivationAction::Recompute => {
                let mask = seg_plan.map(|sp| sp.selective_mask()).unwrap_or_default();
                let is_full = mask.is_full_recompute();
                #[allow(clippy::needless_range_loop)]
                for m in 0..g {
                    let ckpt_buf = fwd
                        .checkpoints
                        .iter()
                        .find(|(li, mi, _)| *li == seg_start as u32 && *mi == m as u32)
                        .map(|(_, _, buf)| buf)
                        .ok_or_else(|| {
                            crate::error::DoublePassError::Checkpoint(
                                "checkpoint missing".to_string(),
                            )
                        })?;
                    let mut act = crate::checkpoint::read_checkpoint(ckpt_buf)?;
                    for &layer_idx in seg_layers {
                        let layer_idx_usize = layer_idx as usize;
                        let rng_idx = layer_idx_usize * g + m;
                        if let Some(rng_state) = fwd.rng_states.get(rng_idx) {
                            crate::rng::restore(rng_state)?;
                        }
                        let fwd_out = if is_full {
                            crate::forward::single_layer_forward(
                                &model.cfg,
                                &model.layers[layer_idx_usize],
                                &act,
                            )
                        } else {
                            let retained = fwd
                                .retained_activations
                                .iter()
                                .find(|(li, mi, _)| {
                                    *li == layer_idx_usize as u32 && *mi == m as u32
                                })
                                .map(|(_, _, out)| out)
                                .ok_or_else(|| {
                                    crate::error::DoublePassError::Checkpoint(
                                        "retained missing".to_string(),
                                    )
                                })?;
                            crate::forward::selective_layer_forward(
                                &model.cfg,
                                &model.layers[layer_idx_usize],
                                retained,
                                &mask,
                            )
                        };
                        act = fwd_out.output.clone();
                        seg_fwd[m].push(fwd_out);
                    }
                }
            }
        }
        for (i_in_seg, &layer_idx) in seg_layers.iter().enumerate().rev() {
            let layer_idx_usize = layer_idx as usize;
            let mut acc_grads: Option<ParamGrads> = None;
            for m in 0..g {
                let fwd_out = &seg_fwd[m][i_in_seg];
                let grads = single_layer_backward(
                    &model.cfg,
                    &model.layers[layer_idx_usize],
                    fwd_out,
                    &upstreams[m],
                );
                upstreams[m] = grads.d_input.clone();
                match &mut acc_grads {
                    None => acc_grads = Some(grads),
                    Some(acc) => {
                        for (a, b) in acc.d_rms1_w.iter_mut().zip(&grads.d_rms1_w) {
                            *a += b;
                        }
                        for (a, b) in acc.d_wq.iter_mut().zip(&grads.d_wq) {
                            *a += b;
                        }
                        for (a, b) in acc.d_wk.iter_mut().zip(&grads.d_wk) {
                            *a += b;
                        }
                        for (a, b) in acc.d_wv.iter_mut().zip(&grads.d_wv) {
                            *a += b;
                        }
                        for (a, b) in acc.d_wo.iter_mut().zip(&grads.d_wo) {
                            *a += b;
                        }
                        for (a, b) in acc.d_rms2_w.iter_mut().zip(&grads.d_rms2_w) {
                            *a += b;
                        }
                        for (a, b) in acc.d_wg.iter_mut().zip(&grads.d_wg) {
                            *a += b;
                        }
                        for (a, b) in acc.d_wu.iter_mut().zip(&grads.d_wu) {
                            *a += b;
                        }
                        for (a, b) in acc.d_wd.iter_mut().zip(&grads.d_wd) {
                            *a += b;
                        }
                    }
                }
            }
            if let Some(ref acc) = acc_grads {
                let li = layer_idx_usize as u32;
                optimizer.project_and_accumulate(&acc.d_rms1_w, li, "d_rms1_w");
                optimizer.project_and_accumulate(&acc.d_wq, li, "d_wq");
                optimizer.project_and_accumulate(&acc.d_wk, li, "d_wk");
                optimizer.project_and_accumulate(&acc.d_wv, li, "d_wv");
                optimizer.project_and_accumulate(&acc.d_wo, li, "d_wo");
                optimizer.project_and_accumulate(&acc.d_rms2_w, li, "d_rms2_w");
                optimizer.project_and_accumulate(&acc.d_wg, li, "d_wg");
                optimizer.project_and_accumulate(&acc.d_wu, li, "d_wu");
                optimizer.project_and_accumulate(&acc.d_wd, li, "d_wd");
            }
            layer_grads[layer_idx_usize] = acc_grads;
        }
    }
    let mut result_grads: Vec<ParamGrads> = Vec::with_capacity(l);
    for (idx, opt) in layer_grads.into_iter().enumerate() {
        match opt {
            Some(g) => result_grads.push(g),
            None => {
                return Err(crate::error::DoublePassError::Checkpoint(format!(
                    "layer {} unreached",
                    idx
                )))
            }
        }
    }
    Ok(FullBackwardResult {
        layer_grads: result_grads,
        weight_loads,
    })
}

fn count_activation_elements(fwd: &crate::forward::SingleLayerFwdOut) -> usize {
    fwd.x_in.len()
        + fwd.rms1.len()
        + fwd.x_norm1.len()
        + fwd.h1.len()
        + fwd.q_heads.len()
        + fwd.k_heads.len()
        + fwd.v_heads.len()
        + fwd.attn_scores.len()
        + fwd.attn_weights.len()
        + fwd.attn_out.len()
        + fwd.out_proj.len()
        + fwd.x2.len()
        + fwd.rms2.len()
        + fwd.x_norm2.len()
        + fwd.h2.len()
        + fwd.gate.len()
        + fwd.up.len()
        + fwd.silu_gate.len()
        + fwd.hidden.len()
        + fwd.mlp_out.len()
        + fwd.output.len()
}


