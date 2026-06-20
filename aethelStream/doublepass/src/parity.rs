//! A7 parity diagnostic: pure diagnostic, never mutates optimizer to mask a bug.
//!
//! Compares one micro-step gradient (M5 path) against an in-memory FP32 autograd
//! reference every `parity_check_interval` steps.
//!
//! ```text
//! rel = max|stream_grad - ref_grad| / (max|ref_grad| + ε)
//! rel >= tol_warn  → log warning + escalate layer recompute precision to FP32
//! rel >= tol_halt  → return Err(ParityHalt)  (caller must snapshot before acting)
//! ```
//!
//! # Optimizer projection invariant
//!
//! **M4 owns projection refresh on its own cadence.**  The training loop calls
//! `optimizer.notify_step(step)` once per step; M4 decides internally when to
//! refresh its projection matrix.  [`ParityGuard`] does NOT receive an optimizer
//! reference and therefore cannot call `zero_accum`, `project_and_accumulate`,
//! or any other mutation method — this is enforced by design, not convention.

use crate::{DoublePassError, Precision, Result};
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Tolerance thresholds for the parity guard.
///
/// Configurable via `TrainingPlan` fields or CLI flags.
#[derive(Debug, Clone, Copy)]
pub struct ParityTolerances {
    /// Warn and escalate recompute precision when relative error exceeds this.
    pub warn: f64,
    /// Halt training when relative error exceeds this.
    pub halt: f64,
}

impl Default for ParityTolerances {
    fn default() -> Self {
        Self {
            warn: 1e-4,
            halt: 1e-3,
        }
    }
}

/// Action taken by [`ParityGuard::check`] after comparing gradients.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParityAction {
    /// Relative error is below `tol_warn` — no action required.
    ///
    /// If the layer was previously escalated it is de-escalated back to the
    /// schedule default precision.
    Clean,
    /// Relative error exceeded `tol_warn`; layer recompute precision escalated to FP32.
    Escalated,
}

// ---------------------------------------------------------------------------
// ParityGuard
// ---------------------------------------------------------------------------

/// A7 parity guard.
///
/// Maintains a per-layer escalation set and checks one layer's gradient against
/// a full-precision FP32 reference every `parity_check_interval` steps.
///
/// # Optimizer contract
/// The guard holds **no** optimizer reference.  The training loop must call
/// `optimizer.notify_step(step)` independently.  The guard never calls
/// `zero_accum`, `project_and_accumulate`, or any projection-management method.
pub struct ParityGuard {
    tolerances: ParityTolerances,
    /// Cadence: fires when `step > 0 && step % parity_check_interval == 0`.
    /// 0 = disabled.
    parity_check_interval: u64,
    /// Layers whose recompute precision has been elevated to FP32.
    escalated_layers: HashSet<u32>,
    /// Cumulative parity checks performed (diagnostic counter).
    check_count: u64,
}

impl ParityGuard {
    /// Construct a parity guard with the given tolerances and check interval.
    ///
    /// `parity_check_interval = 0` disables all checks (`should_check` always returns `false`).
    pub fn new(tolerances: ParityTolerances, parity_check_interval: u64) -> Self {
        Self {
            tolerances,
            parity_check_interval,
            escalated_layers: HashSet::new(),
            check_count: 0,
        }
    }

    /// Returns `true` when a parity check should run at `step`.
    ///
    /// Fires at steps `interval, 2×interval, 3×interval, …` (never at step 0).
    pub fn should_check(&self, step: u64) -> bool {
        self.parity_check_interval > 0
            && step > 0
            && step.is_multiple_of(self.parity_check_interval)
    }

    /// Run one parity check for `layer_idx`.
    ///
    /// Compares `stream_grad` (M5 path) against `reference_grad` (FP32 autograd).
    ///
    /// | Condition                  | Return            | Side effect                      |
    /// |----------------------------|-------------------|----------------------------------|
    /// | rel < tol_warn             | `Ok(Clean)`       | De-escalates layer if escalated  |
    /// | tol_warn ≤ rel < tol_halt  | `Ok(Escalated)`   | Adds layer to escalation set     |
    /// | rel ≥ tol_halt             | `Err(ParityHalt)` | Caller must snapshot first       |
    ///
    /// This method accepts no optimizer reference — it cannot reset projections.
    ///
    /// # Errors
    /// Returns [`DoublePassError::ParityHalt`] when `rel >= tol_halt`.
    pub fn check(
        &mut self,
        step: u64,
        layer_idx: u32,
        stream_grad: &[f32],
        reference_grad: &[f32],
    ) -> Result<ParityAction> {
        let rel = compute_relative_error(stream_grad, reference_grad);
        self.check_count += 1;

        if rel >= self.tolerances.halt {
            eprintln!(
                "[parity] HALT  step={step} layer={layer_idx} rel={rel:.3e} \
                 >= tol_halt={:.3e}",
                self.tolerances.halt
            );
            return Err(DoublePassError::ParityHalt { layer_idx, rel });
        }

        if rel >= self.tolerances.warn {
            eprintln!(
                "[parity] WARN  step={step} layer={layer_idx} rel={rel:.3e} \
                 >= tol_warn={:.3e}  → escalating to FP32 recompute",
                self.tolerances.warn
            );
            self.escalated_layers.insert(layer_idx);
            return Ok(ParityAction::Escalated);
        }

        // Clean: de-escalate if previously elevated.
        self.escalated_layers.remove(&layer_idx);
        Ok(ParityAction::Clean)
    }

    /// Whether `layer_idx` is currently escalated to FP32 recompute.
    pub fn is_escalated(&self, layer_idx: u32) -> bool {
        self.escalated_layers.contains(&layer_idx)
    }

    /// Effective recompute precision for `layer_idx`.
    ///
    /// Returns `Precision::FP32` for escalated layers; otherwise returns `default`
    /// (typically from `TrainingPlan::precision_schedule`).
    pub fn recompute_precision(&self, layer_idx: u32, default: Precision) -> Precision {
        if self.escalated_layers.contains(&layer_idx) {
            Precision::FP32
        } else {
            default
        }
    }

    /// Total parity checks performed since construction.
    pub fn check_count(&self) -> u64 {
        self.check_count
    }

    /// Number of layers currently in the escalated set.
    pub fn escalated_layer_count(&self) -> usize {
        self.escalated_layers.len()
    }
}

// ---------------------------------------------------------------------------
// Metric helpers
// ---------------------------------------------------------------------------

/// Compute `max|a - b| / (max|b| + ε)`.
///
/// Operates on the common prefix when lengths differ.
/// `ε = 1e-8` prevents division by zero on all-zero references.
pub fn compute_relative_error(stream_grad: &[f32], reference_grad: &[f32]) -> f64 {
    const EPS: f64 = 1e-8;
    let max_delta = stream_grad
        .iter()
        .zip(reference_grad.iter())
        .map(|(&a, &b)| ((a as f64) - (b as f64)).abs())
        .fold(0.0_f64, f64::max);
    let max_ref = reference_grad
        .iter()
        .map(|&x| (x as f64).abs())
        .fold(0.0_f64, f64::max);
    max_delta / (max_ref + EPS)
}

// ---------------------------------------------------------------------------
// Legacy / DoublePass::parity_probe entry point
// ---------------------------------------------------------------------------

/// Run the parity diagnostic for `ref_layer` and return `max|Δgrad| / (max|ref_grad| + ε)`.
///
/// # Errors
/// Returns [`DoublePassError::ParityHalt`] when `rel >= tolerances.halt`.
pub fn measure_parity(
    ref_layer: u32,
    stream_grad: &[f32],
    reference_grad: &[f32],
    tolerances: &ParityTolerances,
) -> Result<f64> {
    let rel = compute_relative_error(stream_grad, reference_grad);
    if rel >= tolerances.halt {
        return Err(DoublePassError::ParityHalt {
            layer_idx: ref_layer,
            rel,
        });
    }
    Ok(rel)
}
