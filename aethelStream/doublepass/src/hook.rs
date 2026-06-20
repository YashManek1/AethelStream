//! A3 LRDA hook + A6 deferred apply: gradient projection to low rank and clipped apply.
//!
//! # A3: LRDA hook
//! During backward, `project_and_accumulate` fires per layer × micro-batch via the
//! [`crate::OptimizerBackend`] trait. The full gradient never persists beyond one
//! layer's backward pass (LOMO guarantee, Lv et al. 2023 arXiv 2306.09782).
//!
//! # A6: Deferred exact global-norm clipping (N3 / N3′)
//!
//! After the full backward, all low-rank accumulators remain resident (tens of MB for
//! the whole model). This lets M5 compute the **exact global gradient norm** and apply
//! a single clipped Adam step per parameter, removing LOMO's grouped/local approximation
//! on the GaLore/LoRA path (Zhao et al. 2024, arXiv 2403.03507).
//!
//! ## Norm-preservation guarantee (conditional, per N3′)
//!
//! The guarantee is stated **conditionally** because M4 may use different projector types:
//!
//! * **Orthonormal projection** (SVD GaLore, arXiv 2403.03507 §3.2): P ∈ ℝ^{m×r} has
//!   orthonormal columns (P^T P = I_r), so P acts as an isometry on its column span:
//!   `‖P·(P^T G)‖_F = ‖P^T G‖_F` exactly. Therefore `lowrank_grad_sqnorm` equals the
//!   squared Frobenius norm of the back-projected gradient and global clipping is
//!   **exact without forming the full gradient**.
//!
//! * **Random projection** (APOLLO arXiv 2412.05270, Q-GaLore, Fira): P is not
//!   orthonormal, so `‖P·(P^T G)‖_F ≠ ‖P^T G‖_F` in general. The Johnson–Lindenstrauss
//!   lemma guarantees concentration: for P ∈ ℝ^{m×r} with i.i.d. Gaussian entries scaled
//!   by `1/√r`, `E[‖P^T G‖_F^2] = ‖G‖_F^2` and the relative error is O(1/√r). For
//!   clip-critical layers the optional O(1)-per-layer true-Frobenius accumulator (captured
//!   during A3 before projecting) restores exactness.
//!
//! * **No projection** ([`ProjectorKind::None`]): grouped/local-clip fallback per LOMO
//!   §3.3 (Lv et al. 2023, arXiv 2306.09782). Each layer is clipped independently to
//!   `max_norm`; the effective global norm may exceed `max_norm` by up to √L. LOMO's
//!   smooth-loss-surface argument justifies this approximation.

use std::collections::BTreeMap;

use crate::error::DoublePassError;
use crate::plan::TrainingPlan;
use crate::OptimizerBackend;
use crate::Result;

/// How M4 projects gradients to low rank — determines the A6 clipping strategy (N3′).
///
/// Returned by [`crate::OptimizerBackend::projector_kind`] and treated as opaque by M5.
/// Adding a new variant here requires a matching arm in [`deferred_apply_with_clip`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectorKind {
    /// SVD-based orthonormal projection (GaLore, arXiv 2403.03507).
    ///
    /// P has orthonormal columns (P^T P = I_r), so `‖P·(P^T G)‖_F = ‖P^T G‖_F` exactly.
    /// A6 therefore achieves **exact** global-norm clipping without forming the full gradient.
    Orthonormal,
    /// Random projection (APOLLO arXiv 2412.05270, Q-GaLore, Fira).
    ///
    /// Norm preserved only **approximately** via the Johnson–Lindenstrauss lemma.
    /// Use [`crate::OptimizerBackend::true_frobenius_sqnorm`] to restore exactness on
    /// clip-critical layers; otherwise A6 uses the JL-approximate norm from
    /// `lowrank_grad_sqnorm`.
    Random,
    /// No low-rank projection; full gradient stored by the optimizer.
    ///
    /// Triggers the grouped/local-clip fallback (LOMO §3.3, arXiv 2306.09782).
    /// Per-layer norms are read from `lowrank_grad_sqnorm` (which returns the full
    /// gradient norm in this case) and each layer is clipped independently.
    None,
}

/// Diagnostic output of [`deferred_apply_with_clip`].
#[derive(Debug, Clone)]
pub struct ClipResult {
    /// Square root of the summed squared norms across all trainable parameters.
    ///
    /// Under a [`ProjectorKind::Random`] projector without a Frobenius accumulator
    /// this is JL-approximate. In the grouped-fallback path it is the sqrt of the
    /// sum of all per-layer squared norms (not a true global norm for clipping purposes).
    pub global_grad_norm: f64,
    /// Applied clip coefficient: `min(1.0, max_norm / gnorm)`.
    ///
    /// In the grouped-fallback path this is the **minimum** per-layer coefficient;
    /// individual layers may have received a larger (less clipping) scale.
    pub clip_coeff: f32,
    /// `true` when clipping actually triggered, i.e. `clip_coeff < 1.0`.
    pub clipped: bool,
    /// `true` when at least one parameter used the grouped/local-clip fallback.
    ///
    /// Engages only when [`crate::OptimizerBackend::projector_kind`] returns
    /// [`ProjectorKind::None`] for at least one parameter.
    pub used_grouped_fallback: bool,
    /// Number of parameters for which the O(1) Frobenius accumulator restored
    /// exact clipping under a [`ProjectorKind::Random`] projector.
    pub frobenius_exact_layers: u32,
}

/// Apply exact (or JL-approximate) global-norm gradient clipping and the M4 Adam
/// update for all trainable parameters (A6 / N3 / N3′).
///
/// Must be called **once** immediately after [`crate::backward::full_backward`]
/// returns. All per-layer low-rank accumulators must still be resident in M4 memory;
/// they are zeroed by this function via `zero_accum`.
///
/// # Clipping strategy (per N3′)
///
/// If **any** parameter reports [`ProjectorKind::None`] (no low-rank projection),
/// the entire call falls back to grouped/local clipping (LOMO §3.3). Otherwise a
/// single global-norm clip is computed:
/// - [`ProjectorKind::Orthonormal`]: exact norm via `lowrank_grad_sqnorm`.
/// - [`ProjectorKind::Random`]: JL-approximate norm, optionally restored to exact
///   for clip-critical layers via `true_frobenius_sqnorm`.
///
/// # Errors
///
/// Returns [`DoublePassError::Config`] if `trainable_layers` is empty.
pub fn deferred_apply_with_clip(
    optimizer: &dyn OptimizerBackend,
    plan: &TrainingPlan,
    trainable_layers: &[(u32, String)],
) -> Result<ClipResult> {
    if trainable_layers.is_empty() {
        return Err(DoublePassError::Config(
            "deferred_apply_with_clip: trainable_layers must not be empty".into(),
        ));
    }

    // Gate: any parameter without a low-rank projection requires grouped fallback.
    let any_no_proj = trainable_layers
        .iter()
        .any(|(li, pn)| optimizer.projector_kind(*li, pn.as_str()) == ProjectorKind::None);

    if any_no_proj {
        return grouped_clip_fallback(optimizer, plan, trainable_layers);
    }

    global_clip(optimizer, plan, trainable_layers)
}

/// Global-norm clip path: exact under `Orthonormal`, JL-approximate under `Random`.
fn global_clip(
    optimizer: &dyn OptimizerBackend,
    plan: &TrainingPlan,
    trainable_layers: &[(u32, String)],
) -> Result<ClipResult> {
    let max_norm = plan.max_grad_norm as f64;
    let mut gsq = 0.0_f64;
    let mut frobenius_exact_layers = 0_u32;

    // Pass 1 — accumulate the squared global gradient norm.
    for (layer_idx, param_name) in trainable_layers {
        let pn = param_name.as_str();
        let sq = match optimizer.projector_kind(*layer_idx, pn) {
            ProjectorKind::Orthonormal => {
                // Exact: P has orthonormal columns ⟹ ‖P·(P^T G)‖_F = ‖P^T G‖_F.
                // GaLore arXiv 2403.03507 §3.2: the SVD projection is isometric on
                // its column span, so multiplying by P preserves the Frobenius norm.
                optimizer.lowrank_grad_sqnorm(*layer_idx, pn)
            }
            ProjectorKind::Random => {
                // Prefer the O(1) pre-projection Frobenius accumulator when the
                // optimizer captured it for this clip-critical layer.
                match optimizer.true_frobenius_sqnorm(*layer_idx, pn) {
                    Some(fsq) => {
                        // Exact: norm was captured before random projection was applied.
                        frobenius_exact_layers += 1;
                        fsq
                    }
                    // JL-approximate: E[‖P^T G‖_F^2] = ‖G‖_F^2 for scaled Gaussian P,
                    // relative error O(1/√r) by the Johnson–Lindenstrauss lemma.
                    // (APOLLO arXiv 2412.05270, JL lemma §A.1)
                    None => optimizer.lowrank_grad_sqnorm(*layer_idx, pn),
                }
            }
            // Unreachable: the `any_no_proj` guard above routes None-projector calls
            // to `grouped_clip_fallback` before `global_clip` is entered.
            ProjectorKind::None => {
                return Err(DoublePassError::Config(
                    "global_clip reached ProjectorKind::None — use grouped fallback".into(),
                ));
            }
        };
        gsq += sq;
    }

    let gnorm = gsq.sqrt();
    // Clip coefficient: ≤ 1.0; the 1e-6 floor prevents division by zero.
    let clip_coeff = ((max_norm / (gnorm + 1e-6)) as f32).min(1.0_f32);
    let clipped = clip_coeff < 1.0_f32;

    // Pass 2 — apply the clipped Adam step and zero accumulators.
    for (layer_idx, param_name) in trainable_layers {
        let pn = param_name.as_str();
        optimizer.apply_update(*layer_idx, pn, clip_coeff);
        optimizer.zero_accum(*layer_idx, pn);
    }

    Ok(ClipResult {
        global_grad_norm: gnorm,
        clip_coeff,
        clipped,
        used_grouped_fallback: false,
        frobenius_exact_layers,
    })
}

/// Grouped/local-clip fallback for the no-projection case (LOMO §3.3, arXiv 2306.09782).
///
/// Each layer is clipped independently to `max_norm` using the per-layer gradient norm
/// from `lowrank_grad_sqnorm`. The effective global norm may exceed `max_norm` by up to
/// `√L`, which is the known approximation cost. LOMO's smooth-loss-surface argument
/// justifies this approximation (Lv et al. 2023, arXiv 2306.09782 §3.3).
///
/// Called only from [`deferred_apply_with_clip`] after it detects at least one
/// [`ProjectorKind::None`] parameter.
fn grouped_clip_fallback(
    optimizer: &dyn OptimizerBackend,
    plan: &TrainingPlan,
    trainable_layers: &[(u32, String)],
) -> Result<ClipResult> {
    let max_norm = plan.max_grad_norm as f64;

    // Group parameters by layer index; BTreeMap gives deterministic ordering.
    let mut layer_params: BTreeMap<u32, Vec<&str>> = BTreeMap::new();
    for (li, pn) in trainable_layers {
        layer_params.entry(*li).or_default().push(pn.as_str());
    }

    let mut total_gsq = 0.0_f64;
    let mut min_clip_coeff = 1.0_f32;

    for (&layer_idx, params) in &layer_params {
        // Per-layer squared norm: sum the squared norms of all params in this layer.
        let layer_sq: f64 = params
            .iter()
            .map(|pn| optimizer.lowrank_grad_sqnorm(layer_idx, pn))
            .sum();
        let layer_norm = layer_sq.sqrt();
        let layer_clip = ((max_norm / (layer_norm + 1e-6)) as f32).min(1.0_f32);

        for pn in params {
            optimizer.apply_update(layer_idx, pn, layer_clip);
            optimizer.zero_accum(layer_idx, pn);
        }

        total_gsq += layer_sq;
        if layer_clip < min_clip_coeff {
            min_clip_coeff = layer_clip;
        }
    }

    let gnorm = total_gsq.sqrt();

    Ok(ClipResult {
        global_grad_norm: gnorm,
        clip_coeff: min_clip_coeff,
        clipped: min_clip_coeff < 1.0_f32,
        used_grouped_fallback: true,
        frobenius_exact_layers: 0,
    })
}
