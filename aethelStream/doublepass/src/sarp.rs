//! M9 SARP Executor — Segment Activation Recompute Planner.
//!
//! # Design
//!
//! The **Segment Activation Recompute Planner (SARP)** executor consults the
//! optimal activation schedule produced by M9's dynamic programming solver.
//! When the schedule is absent (M9 not yet run), it falls back to the A9 HAM
//! greedy heuristic.
//!
//! # STOP-and-note — M9 SARP DP sprint not yet run
//!
//! The optimal schedule is dark until M9 emits `activation_schedule`. All four
//! `ActivationAction` paths in `sarp_backward_segment` are tested (T-SARP) and
//! ready; the DP solver that fills the schedule belongs to the M9 sprint.

use crate::forward::{
    selective_layer_forward, single_layer_forward, FullForwardResult, SingleLayerFwdOut,
};
use crate::plan::{ActivationAction, SegmentPlan, SelectiveRecomputeMask};
use crate::schedule::SegmentRecomputeOrder;
use crate::{DoublePassError, HardwareProfile, Result};

/// Executor for the M9 SARP schedule.
///
/// Per-segment, this struct consults the [`crate::plan::TrainingPlan::activation_schedule`]
/// produced by the M9 SARP DP. When the schedule is absent (M9 not yet run),
/// it falls back to the A9 HAM greedy heuristic ([`crate::ham::Sars`] under the
/// `ham-offload` feature, or `ActivationAction::Recompute` otherwise).
#[derive(Debug, Clone)]
pub struct SarpExecutor {
    /// Per-segment plans from M9, indexed by `segment_index`.
    plans: Vec<SegmentPlan>,
    /// Hardware profile for the HAM fallback (`ham-offload` feature).
    #[allow(dead_code)]
    profile: HardwareProfile,
}

impl SarpExecutor {
    /// Construct from a [`crate::plan::TrainingPlan`] and a warmup [`HardwareProfile`].
    pub fn new(plan: &crate::plan::TrainingPlan, profile: HardwareProfile) -> Self {
        Self {
            plans: plan.activation_schedule.clone(),
            profile,
        }
    }

    /// Return `true` when M9 has provided a schedule (not falling back to HAM).
    pub fn has_m9_schedule(&self) -> bool {
        !self.plans.is_empty()
    }

    /// Return the [`SegmentPlan`] for `segment_index`, if M9 provided one.
    pub fn plan_for_segment(&self, segment_index: u32) -> Option<&SegmentPlan> {
        self.plans.iter().find(|p| p.segment_index == segment_index)
    }

    /// Select the action for `segment_index`.
    ///
    /// Priority: M9 plan > HAM greedy (`ham-offload`) > `Recompute` (default).
    pub fn action_for_segment(
        &self,
        segment_index: u32,
        _segment_weight_bytes: u64,
        _num_layers: usize,
    ) -> ActivationAction {
        // Priority 1: M9 plan
        if let Some(plan) = self.plan_for_segment(segment_index) {
            return plan.action;
        }

        // Priority 2: HAM fallback (if feature enabled)
        #[cfg(feature = "ham-offload")]
        {
            crate::ham::select_action(
                segment_index,
                &self.profile,
                0.0,
                _segment_weight_bytes,
                _num_layers,
            )
        }

        // Priority 3: Default to Recompute
        #[cfg(not(feature = "ham-offload"))]
        {
            ActivationAction::Recompute
        }
    }
}

/// Per-segment telemetry logged by [`sarp_backward_segment`] for paper plots.
#[derive(Debug, Clone)]
pub struct SarpSegmentStats {
    /// Segment index.
    pub segment_index: u32,
    /// Action actually taken.
    pub action: ActivationAction,
    /// Full-layer recompute FLOPs (non-zero only for Recompute + full mask).
    pub recompute_flops: f64,
    /// FLOPs saved by selective-region recompute vs full recompute (N1″).
    pub selective_flops_saved: f64,
    /// PCIe bytes moved (non-zero for PageCompressedRam).
    pub pcie_bytes: usize,
    /// SSD bytes moved (non-zero for PageNvme).
    pub ssd_bytes: usize,
}

/// Count total f32 scalars in a [`SingleLayerFwdOut`].
fn fwd_out_element_count(fwd: &SingleLayerFwdOut) -> usize {
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

/// Recover the forward activations for `seg` according to the SARP action.
///
/// Returns `(seg_fwd, stats)` where `seg_fwd[m][i_in_seg]` is the
/// [`SingleLayerFwdOut`] for micro-batch `m`, layer position `i_in_seg` within
/// the segment (ascending). The caller passes this to `single_layer_backward`.
///
/// # Actions
/// - `RetainVram` / `PageCompressedRam` / `PageNvme`: load from `fwd.retained_activations`.
///   In mock these are identical; they differ only in I/O accounting.
/// - `Recompute` (full mask): re-run forward from checkpoint (standard A2).
/// - `Recompute` (selective mask): load retained + call `selective_layer_forward`.
///
/// # Errors
/// Missing retained activations → `DoublePassError::Checkpoint`.
pub fn sarp_backward_segment(
    executor: &SarpExecutor,
    seg: &SegmentRecomputeOrder,
    model: &crate::forward::Model,
    fwd: &FullForwardResult,
    inputs: &[Vec<f32>],
    segment_weight_bytes: u64,
) -> Result<(Vec<Vec<SingleLayerFwdOut>>, SarpSegmentStats)> {
    let g = inputs.len();
    let seg_len = seg.layers_ascending.len();

    // Select action
    let action = executor.action_for_segment(seg.segment_index, segment_weight_bytes, seg_len);

    // Fetch the plan for this segment (for selective recompute mask)
    let seg_plan = executor.plan_for_segment(seg.segment_index);

    let (seg_fwd, recompute_flops, selective_flops_saved, pcie_bytes, ssd_bytes) = match action {
        ActivationAction::RetainVram => {
            let seg_fwd = load_from_retained(fwd, seg, g)?;
            (seg_fwd, 0.0, 0.0, 0, 0)
        }
        ActivationAction::PageCompressedRam => {
            let seg_fwd = load_from_retained(fwd, seg, g)?;
            // Calculate element count for I/O accounting
            let element_count: usize = seg_fwd
                .iter()
                .flat_map(|mb| mb.iter())
                .map(fwd_out_element_count)
                .sum();
            let pcie_bytes = element_count * std::mem::size_of::<f32>();
            (seg_fwd, 0.0, 0.0, pcie_bytes, 0)
        }
        ActivationAction::PageNvme => {
            let seg_fwd = load_from_retained(fwd, seg, g)?;
            // Calculate element count for I/O accounting
            let element_count: usize = seg_fwd
                .iter()
                .flat_map(|mb| mb.iter())
                .map(fwd_out_element_count)
                .sum();
            let ssd_bytes = element_count * std::mem::size_of::<f32>();
            (seg_fwd, 0.0, 0.0, 0, ssd_bytes)
        }
        ActivationAction::Recompute => {
            // Build mask for selective vs full recompute
            let mask = seg_plan
                .map(|p| p.selective_mask())
                .unwrap_or_else(SelectiveRecomputeMask::full_recompute);

            let is_full = mask.is_full_recompute();
            let selective_fraction = mask.recompute_flop_fraction();

            // Compute FLOPs for this segment
            // Rough model: 2 FLOPs per f32 parameter per forward pass
            let layer_flops = (model.cfg.bytes_per_layer() / 4) as f64 * 2.0;
            let recompute_flops = layer_flops * selective_fraction * seg_len as f64 * g as f64;
            let selective_flops_saved =
                layer_flops * (1.0 - selective_fraction) * seg_len as f64 * g as f64;

            if is_full {
                // Full recompute: restore from checkpoint and re-run forward
                let seg_fwd = recompute_full_segment(model, fwd, seg, g)?;
                (seg_fwd, recompute_flops, selective_flops_saved, 0, 0)
            } else {
                // Selective recompute: load retained + call selective_layer_forward
                let seg_fwd = recompute_selective_segment(model, fwd, seg, &mask, g)?;
                (seg_fwd, recompute_flops, selective_flops_saved, 0, 0)
            }
        }
    };

    let stats = SarpSegmentStats {
        segment_index: seg.segment_index,
        action,
        recompute_flops,
        selective_flops_saved,
        pcie_bytes,
        ssd_bytes,
    };

    eprintln!(
        "[sarp] seg={} action={:?} recompute_flops={:.3e} selective_flops_saved={:.3e} pcie_bytes={} ssd_bytes={}",
        seg.segment_index, action, recompute_flops, selective_flops_saved, pcie_bytes, ssd_bytes
    );

    Ok((seg_fwd, stats))
}

/// Load activations from `retained_activations` for a segment.
///
/// Returns `seg_fwd[m][i_in_seg]` for all micro-batches `m` and positions
/// `i_in_seg` within the segment.
fn load_from_retained(
    fwd: &FullForwardResult,
    seg: &SegmentRecomputeOrder,
    g: usize,
) -> Result<Vec<Vec<SingleLayerFwdOut>>> {
    let mut seg_fwd: Vec<Vec<SingleLayerFwdOut>> = vec![Vec::new(); g];

    #[allow(clippy::needless_range_loop)]
    for m in 0..g {
        for &layer_idx in &seg.layers_ascending {
            let mb_as_u32 = m as u32;
            let fwd_out = fwd
                .retained_activations
                .iter()
                .find(|(l, mb, _)| *l == layer_idx && *mb == mb_as_u32)
                .map(|(_, _, fwd_out)| fwd_out.clone())
                .ok_or_else(|| {
                    DoublePassError::Checkpoint(format!(
                        "retained activation missing for (layer={}, micro_batch={})",
                        layer_idx, m
                    ))
                })?;
            seg_fwd[m].push(fwd_out);
        }
    }

    Ok(seg_fwd)
}

/// Full recompute: restore from checkpoint and re-run forward for each layer.
fn recompute_full_segment(
    model: &crate::forward::Model,
    fwd: &FullForwardResult,
    seg: &SegmentRecomputeOrder,
    g: usize,
) -> Result<Vec<Vec<SingleLayerFwdOut>>> {
    let mut seg_fwd: Vec<Vec<SingleLayerFwdOut>> = vec![Vec::new(); g];

    #[allow(clippy::needless_range_loop)]
    for m in 0..g {
        // Restore checkpoint for this segment and micro-batch
        let seg_start = seg.segment_index * model.cfg.batch as u32;
        let mb_as_u32 = m as u32;
        let checkpoint = fwd
            .checkpoints
            .iter()
            .find(|(l, mb, _)| *l >= seg_start && *mb == mb_as_u32)
            .ok_or_else(|| {
                DoublePassError::Checkpoint(format!(
                    "checkpoint missing for segment {}, micro_batch {}",
                    seg.segment_index, m
                ))
            })?;

        // Read checkpoint buffer
        let ckpt_act = crate::checkpoint::read_checkpoint(&checkpoint.2)?;
        let mut act = ckpt_act.clone();

        for &layer_idx in &seg.layers_ascending {
            let rng_idx = layer_idx as usize * g + m;
            if let Some(rng) = fwd.rng_states.get(rng_idx) {
                crate::rng::restore(rng)?;
            }

            let fwd_out = single_layer_forward(&model.cfg, &model.layers[layer_idx as usize], &act);
            act = fwd_out.output.clone();
            seg_fwd[m].push(fwd_out);
        }
    }

    Ok(seg_fwd)
}

/// Selective recompute: load retained and call `selective_layer_forward` per layer.
fn recompute_selective_segment(
    model: &crate::forward::Model,
    fwd: &FullForwardResult,
    seg: &SegmentRecomputeOrder,
    mask: &SelectiveRecomputeMask,
    g: usize,
) -> Result<Vec<Vec<SingleLayerFwdOut>>> {
    let mut seg_fwd: Vec<Vec<SingleLayerFwdOut>> = vec![Vec::new(); g];

    #[allow(clippy::needless_range_loop)]
    for m in 0..g {
        for &layer_idx in &seg.layers_ascending {
            let rng_idx = layer_idx as usize * g + m;
            if let Some(rng) = fwd.rng_states.get(rng_idx) {
                crate::rng::restore(rng)?;
            }

            // Load retained activation
            let mb_as_u32 = m as u32;
            let retained = fwd
                .retained_activations
                .iter()
                .find(|(l, mb, _)| *l == layer_idx && *mb == mb_as_u32)
                .map(|(_, _, fwd_out)| fwd_out.clone())
                .ok_or_else(|| {
                    DoublePassError::Checkpoint(format!(
                        "retained activation missing for (layer={}, micro_batch={})",
                        layer_idx, m
                    ))
                })?;

            let sel_out = selective_layer_forward(
                &model.cfg,
                &model.layers[layer_idx as usize],
                &retained,
                mask,
            );
            seg_fwd[m].push(sel_out);
        }
    }

    Ok(seg_fwd)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sarp_executor_no_schedule() {
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
        let plan = crate::plan::TrainingPlan::default();
        let executor = SarpExecutor::new(&plan, profile);

        assert!(!executor.has_m9_schedule());
        assert!(executor.plan_for_segment(0).is_none());
    }

    #[test]
    fn test_sarp_executor_with_schedule() {
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
        let mut plan = crate::plan::TrainingPlan::default();
        plan.activation_schedule = vec![SegmentPlan::with_full_recompute(0)];
        let executor = SarpExecutor::new(&plan, profile);

        assert!(executor.has_m9_schedule());
        assert_eq!(
            executor.plan_for_segment(0).map(|p| p.segment_index),
            Some(0)
        );
    }

    #[test]
    fn test_sarp_executor_action_selection() {
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
        let mut plan = crate::plan::TrainingPlan::default();
        plan.activation_schedule = vec![SegmentPlan::page_compressed_ram(0)];
        let executor = SarpExecutor::new(&plan, profile);

        let action = executor.action_for_segment(0, 1000, 4);
        assert_eq!(action, ActivationAction::PageCompressedRam);
    }

    #[test]
    fn test_fwd_out_element_count() {
        let fwd_out = SingleLayerFwdOut {
            x_in: vec![1.0; 10],
            rms1: vec![2.0; 5],
            x_norm1: vec![3.0; 10],
            h1: vec![4.0; 10],
            q_heads: vec![5.0; 8],
            k_heads: vec![6.0; 8],
            v_heads: vec![7.0; 8],
            attn_scores: vec![8.0; 16],
            attn_weights: vec![9.0; 16],
            attn_out: vec![10.0; 8],
            out_proj: vec![11.0; 10],
            x2: vec![12.0; 10],
            rms2: vec![13.0; 5],
            x_norm2: vec![14.0; 10],
            h2: vec![15.0; 10],
            gate: vec![16.0; 20],
            up: vec![17.0; 20],
            silu_gate: vec![18.0; 20],
            hidden: vec![19.0; 20],
            mlp_out: vec![20.0; 10],
            output: vec![21.0; 10],
        };

        let count = fwd_out_element_count(&fwd_out);
        assert_eq!(
            count,
            10 + 5
                + 10
                + 10
                + 8
                + 8
                + 8
                + 16
                + 16
                + 8
                + 10
                + 10
                + 5
                + 10
                + 10
                + 20
                + 20
                + 20
                + 20
                + 10
                + 10
        );
    }

    #[test]
    fn test_sarp_segment_stats_construction() {
        let stats = SarpSegmentStats {
            segment_index: 0,
            action: ActivationAction::Recompute,
            recompute_flops: 1e9,
            selective_flops_saved: 5e8,
            pcie_bytes: 0,
            ssd_bytes: 0,
        };

        assert_eq!(stats.segment_index, 0);
        assert_eq!(stats.recompute_flops, 1e9);
        assert_eq!(stats.selective_flops_saved, 5e8);
    }
}
