//! A9: Hybrid Activation Materialization (HAM) — feature `ham-offload`.
//!
//! # Design
//!
//! The **Segment Activation Recompute Selector (SARS)** is a greedy, self-contained
//! fallback that fires when M9's optimal planner (S9.5) has not emitted a
//! `SegmentPlan` override.  Keep this module small: one decision per segment, two
//! backward execution paths, one shared descending backward kernel.
//!
//! ## Decision rule (SARS)
//!
//! ```text
//! t_io  = segment_weight_bytes / (pcie_bandwidth_gbs x 1e9)        [seconds]
//! t_cmp = (mean_forward_ms + mean_backward_ms) x 1e-3 x num_layers [seconds]
//!
//! I/O-bound  (t_io > t_cmp): compute is idle -> RECOMPUTE is free.
//! Compute-bound (t_io <= t_cmp): PCIe is idle -> OFFLOAD via idle PCIe link.
//! ```
//!
//! ## Gradient parity guarantee (T-HAM)
//!
//! Both paths call `single_layer_backward` with the **same** `SingleLayerFwdOut`.
//! - RECOMPUTE: restores the saved `RngState` before re-running forward, reproducing
//!   dropout masks exactly -- the resulting `SingleLayerFwdOut` is numerically
//!   identical to the original.
//! - OFFLOAD: uses the stored `SingleLayerFwdOut` directly (no re-run, no RNG).
//!
//! Because the backward call is identical in both cases, gradients are
//! **bit-for-bit equal**.

use crate::forward::{single_layer_forward, BlockConfig, BlockWeights, SingleLayerFwdOut};
use crate::backward::{single_layer_backward, ParamGrads};
use crate::plan::ActivationAction;
use crate::state::RngState;
use crate::{HardwareProfile, Result};

// ---------------------------------------------------------------------------
// HamAction
// ---------------------------------------------------------------------------

/// Per-segment decision produced by [`Sars`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HamAction {
    /// Re-run the segment forward during backward; relies on idle compute.
    Recompute,
    /// Store activations in pinned RAM during forward; reload during backward.
    Offload,
}

// ---------------------------------------------------------------------------
// SARS -- Segment Activation Recompute Selector
// ---------------------------------------------------------------------------

/// Greedy fallback selector: compares estimated PCIe I/O time vs GPU compute time.
///
/// Instantiate once per training session from the `HardwareProfile` produced by
/// `FlowCast::warmup`.  Call [`Sars::select`] once per segment at the start of
/// each forward pass; the result is logged for the paper's I/O-vs-compute plots.
pub struct Sars {
    profile: HardwareProfile,
}

impl Sars {
    /// Construct from a warm-up-measured hardware profile.
    pub fn new(profile: HardwareProfile) -> Self {
        Self { profile }
    }

    /// Return `true` when the segment's estimated PCIe transfer time exceeds its
    /// GPU compute time (I/O-bound -> RECOMPUTE is the better choice).
    ///
    /// `segment_weight_bytes` -- bytes to DMA for all layers in this segment.
    /// `num_layers` -- number of layers in the segment (scales compute estimate).
    pub fn is_io_bound(&self, segment_weight_bytes: u64, num_layers: usize) -> bool {
        let pcie_bps = self.profile.pcie_bandwidth_gbs as f64 * 1e9;
        let t_io_s = if pcie_bps > 0.0 {
            segment_weight_bytes as f64 / pcie_bps
        } else {
            f64::INFINITY
        };
        let t_compute_s =
            (self.profile.mean_forward_ms + self.profile.mean_backward_ms) as f64
            * 1e-3
            * num_layers as f64;
        t_io_s > t_compute_s
    }

    /// Select and log the action for `segment_index`.
    ///
    /// The `eprintln!` line is parsed by the paper's plotting pipeline
    /// (`scripts/plot_ham_decisions.py`).
    pub fn select(
        &self,
        segment_index: u32,
        segment_weight_bytes: u64,
        num_layers: usize,
    ) -> HamAction {
        let io_bound = self.is_io_bound(segment_weight_bytes, num_layers);
        let action = if io_bound { HamAction::Recompute } else { HamAction::Offload };
        eprintln!(
            "[ham] seg={segment_index} layers={num_layers} \
             weight_bytes={segment_weight_bytes} \
             pcie_bw={:.3}GBs t_compute_ms={:.3} io_bound={io_bound} action={action:?}",
            self.profile.pcie_bandwidth_gbs,
            (self.profile.mean_forward_ms + self.profile.mean_backward_ms)
                * num_layers as f32,
        );
        action
    }
}

// ---------------------------------------------------------------------------
// Legacy entry point (matches S0 stub signature, plus num_layers extension)
// ---------------------------------------------------------------------------

/// Greedy segment selector -- fallback when M9 has not emitted a `SegmentPlan`.
///
/// Maps to [`ActivationAction::Recompute`] (I/O-bound) or
/// [`ActivationAction::PageCompressedRam`] (compute-bound / OFFLOAD).
///
/// `_segment_flops` is accepted for API symmetry with the S0 stub but is not
/// consumed; compute time is estimated from `HardwareProfile::mean_forward_ms`
/// and `mean_backward_ms` (measured values that include all overhead).
pub fn select_action(
    segment_index: u32,
    profile: &HardwareProfile,
    _segment_flops: f64,
    segment_weight_bytes: u64,
    num_layers: usize,
) -> ActivationAction {
    let sars = Sars::new(profile.clone());
    match sars.select(segment_index, segment_weight_bytes, num_layers) {
        HamAction::Recompute => ActivationAction::Recompute,
        HamAction::Offload => ActivationAction::PageCompressedRam,
    }
}

// ---------------------------------------------------------------------------
// Offload store
// ---------------------------------------------------------------------------

/// Per-segment activation store written during the OFFLOAD forward pass.
///
/// In production each `SingleLayerFwdOut` is serialised into a `PinnedBuffer`
/// (DMA-aligned pinned memory) and transferred via the idle PCIe link.  Under
/// `mock-cuda` the struct holds heap-allocated `Vec<f32>` fields directly.
pub struct SegmentActivationStore {
    /// `activations[micro_batch][layer_in_seg]` -- full forward intermediates.
    pub activations: Vec<Vec<SingleLayerFwdOut>>,
    /// Total f32 scalar count (`pcie_bytes = element_count x 4`).
    pub element_count: usize,
}

impl SegmentActivationStore {
    /// Build a store from the forward-pass outputs of one segment.
    ///
    /// `fwd_per_micro_batch[m][i]` is `SingleLayerFwdOut` for micro-batch `m`,
    /// layer position `i` in ascending order within the segment.
    pub fn new(fwd_per_micro_batch: Vec<Vec<SingleLayerFwdOut>>) -> Self {
        let element_count: usize = fwd_per_micro_batch
            .iter()
            .flat_map(|mb| mb.iter())
            .map(fwd_out_element_count)
            .sum();
        Self { activations: fwd_per_micro_batch, element_count }
    }

    /// Estimated bytes moved over PCIe to store and later reload these activations.
    pub fn pcie_bytes(&self) -> usize {
        self.element_count * std::mem::size_of::<f32>()
    }
}

// ---------------------------------------------------------------------------
// Per-segment telemetry
// ---------------------------------------------------------------------------

/// Stats for one segment -- logged by both backward functions for paper plots.
#[derive(Debug, Clone)]
pub struct HamSegmentStats {
    /// Checkpoint segment index.
    pub segment_index: u32,
    /// Action actually taken.
    pub action: HamAction,
    /// Forward-recompute FLOPs: non-zero only for `Recompute`.
    pub recompute_flops: f64,
    /// PCIe/RAM bytes: non-zero only for `Offload`.
    pub pcie_bytes: usize,
}

// ---------------------------------------------------------------------------
// RECOMPUTE backward path
// ---------------------------------------------------------------------------

/// Run the RECOMPUTE backward for one segment.
///
/// Re-runs `single_layer_forward` (with RNG restore) per layer per micro-batch
/// to recover `SingleLayerFwdOut`, then descends with `single_layer_backward`.
///
/// `layer_indices` -- absolute layer indices ascending within the segment.
/// `checkpoint_bufs` -- one `Vec<f32>` per micro-batch: the input activation to
/// `layer_indices[0]` (the segment checkpoint read from `FullForwardResult`).
/// `rng_states` -- indexed as `layer_idx * num_micro_batches + m`; `None` where
/// no dropout was applied.
/// `upstreams` -- one upstream gradient per micro-batch; updated in-place.
///
/// # Errors
/// Propagates RNG restore errors.
pub fn backward_recompute(
    cfg: &BlockConfig,
    weights: &[BlockWeights],
    layer_indices: &[usize],
    checkpoint_bufs: &[Vec<f32>],
    rng_states: &[Option<RngState>],
    upstreams: &mut Vec<Vec<f32>>,
    segment_index: u32,
) -> Result<(Vec<ParamGrads>, HamSegmentStats)> {
    let num_micro_batches = checkpoint_bufs.len();
    let seg_len = layer_indices.len();

    let mut seg_fwd: Vec<Vec<SingleLayerFwdOut>> =
        (0..num_micro_batches).map(|_| Vec::with_capacity(seg_len)).collect();

    for m in 0..num_micro_batches {
        let mut act = checkpoint_bufs[m].clone();
        for &layer_idx in layer_indices {
            let rng_flat = layer_idx * num_micro_batches + m;
            if let Some(Some(rng_state)) = rng_states.get(rng_flat) {
                crate::rng::restore(rng_state)?;
            }
            let fwd_out = single_layer_forward(cfg, &weights[layer_idx], &act);
            act = fwd_out.output.clone();
            seg_fwd[m].push(fwd_out);
        }
    }

    let recompute_flops = estimate_recompute_flops(cfg, seg_len, num_micro_batches);
    let layer_grads = descend_backward(
        cfg, weights, layer_indices, &seg_fwd, upstreams, num_micro_batches,
    );
    let stats = HamSegmentStats {
        segment_index,
        action: HamAction::Recompute,
        recompute_flops,
        pcie_bytes: 0,
    };
    eprintln!(
        "[ham-bwd] seg={segment_index} action=Recompute \
         recompute_flops={recompute_flops:.3e} pcie_bytes=0"
    );
    Ok((layer_grads, stats))
}

// ---------------------------------------------------------------------------
// OFFLOAD backward path
// ---------------------------------------------------------------------------

/// Run the OFFLOAD backward for one segment using pre-stored activations.
///
/// Uses `store.activations` directly -- no forward re-run, no RNG restore.
/// The stored `SingleLayerFwdOut` values were produced by the same forward pass
/// (identical dropout masks) so `single_layer_backward` receives numerically
/// identical inputs -> **bit-identical gradients** vs RECOMPUTE (T-HAM).
///
/// `upstreams` is updated in-place as backward propagates through the segment.
///
/// # Errors
/// Currently infallible; returns `Result` for API uniformity with `backward_recompute`.
pub fn backward_offload(
    cfg: &BlockConfig,
    weights: &[BlockWeights],
    layer_indices: &[usize],
    store: &SegmentActivationStore,
    upstreams: &mut Vec<Vec<f32>>,
    segment_index: u32,
) -> Result<(Vec<ParamGrads>, HamSegmentStats)> {
    let num_micro_batches = store.activations.len();
    let pcie_bytes = store.pcie_bytes();

    let layer_grads = descend_backward(
        cfg, weights, layer_indices, &store.activations, upstreams, num_micro_batches,
    );
    let stats = HamSegmentStats {
        segment_index,
        action: HamAction::Offload,
        recompute_flops: 0.0,
        pcie_bytes,
    };
    eprintln!(
        "[ham-bwd] seg={segment_index} action=Offload \
         recompute_flops=0 pcie_bytes={pcie_bytes}"
    );
    Ok((layer_grads, stats))
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Shared descending backward loop for both RECOMPUTE and OFFLOAD paths.
///
/// `seg_fwd[m][i]` is `SingleLayerFwdOut` for micro-batch `m`, segment position `i`.
fn descend_backward(
    cfg: &BlockConfig,
    weights: &[BlockWeights],
    layer_indices: &[usize],
    seg_fwd: &[Vec<SingleLayerFwdOut>],
    upstreams: &mut Vec<Vec<f32>>,
    num_micro_batches: usize,
) -> Vec<ParamGrads> {
    let seg_len = layer_indices.len();
    let mut layer_grads: Vec<Option<ParamGrads>> = (0..seg_len).map(|_| None).collect();

    for (i_in_seg, &layer_idx) in layer_indices.iter().enumerate().rev() {
        let mut acc: Option<ParamGrads> = None;
        for m in 0..num_micro_batches {
            let fwd_out = &seg_fwd[m][i_in_seg];
            let grads = single_layer_backward(cfg, &weights[layer_idx], fwd_out, &upstreams[m]);
            upstreams[m] = grads.d_input.clone();
            match &mut acc {
                None => acc = Some(grads),
                Some(a) => accumulate_grads(a, &grads),
            }
        }
        layer_grads[i_in_seg] = acc;
    }

    layer_grads.into_iter().flatten().collect()
}

/// In-place addition of `src` parameter gradients into `dst`.
fn accumulate_grads(dst: &mut ParamGrads, src: &ParamGrads) {
    for (a, b) in dst.d_rms1_w.iter_mut().zip(&src.d_rms1_w) { *a += b; }
    for (a, b) in dst.d_wq.iter_mut().zip(&src.d_wq) { *a += b; }
    for (a, b) in dst.d_wk.iter_mut().zip(&src.d_wk) { *a += b; }
    for (a, b) in dst.d_wv.iter_mut().zip(&src.d_wv) { *a += b; }
    for (a, b) in dst.d_wo.iter_mut().zip(&src.d_wo) { *a += b; }
    for (a, b) in dst.d_rms2_w.iter_mut().zip(&src.d_rms2_w) { *a += b; }
    for (a, b) in dst.d_wg.iter_mut().zip(&src.d_wg) { *a += b; }
    for (a, b) in dst.d_wu.iter_mut().zip(&src.d_wu) { *a += b; }
    for (a, b) in dst.d_wd.iter_mut().zip(&src.d_wd) { *a += b; }
}

/// Count f32 scalars in a `SingleLayerFwdOut` (for PCIe-bytes telemetry).
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

/// Approximate recompute FLOPs: 8 dominant matmuls per layer x segment x micro-batches.
///
/// `2 x bs x (4 x d^2 + 3 x d x ff) x seg_len x G`
fn estimate_recompute_flops(
    cfg: &BlockConfig,
    seg_len: usize,
    num_micro_batches: usize,
) -> f64 {
    let bs = cfg.bs() as f64;
    let d = cfg.d_model as f64;
    let ff = cfg.d_ff as f64;
    2.0 * bs * (4.0 * d * d + 3.0 * d * ff) * seg_len as f64 * num_micro_batches as f64
}
