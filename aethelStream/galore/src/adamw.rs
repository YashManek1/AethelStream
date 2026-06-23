//! AdamW update in low-rank space (Algorithm 2).

use crate::project::{project_backward_f32, project_forward_f32};
use crate::quantize::{dequantize_absmax, quantize_absmax};

/// AdamW hyperparameters.
#[derive(Debug, Clone, Copy)]
pub struct AdamWConfig {
    /// First moment decay β1.
    pub beta1: f32,
    /// Second moment decay β2.
    pub beta2: f32,
    /// Numerical stability ε.
    pub eps: f32,
    /// Learning rate α (NOT divided by rank unless [`scale_lr_by_rank`] is set).
    pub lr: f32,
    /// Weight decay λ (AdamW decoupled).
    pub weight_decay: f32,
    /// When true, effective step size is `lr / r` (α/r scaling variant for ablation).
    pub scale_lr_by_rank: bool,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            lr: 1e-3,
            weight_decay: 0.0,
            scale_lr_by_rank: false,
        }
    }
}

/// Effective learning rate after optional rank scaling.
pub fn effective_lr(cfg: &AdamWConfig, rank: usize) -> f32 {
    if cfg.scale_lr_by_rank {
        cfg.lr / rank.max(1) as f32
    } else {
        cfg.lr
    }
}

/// Low-rank AdamW state for one parameter matrix (r×r momentum/variance).
pub struct LowRankAdamState {
    /// First moment M (r×r) in FP32 during update.
    pub momentum: Vec<f32>,
    /// Second moment V (r×r) in FP32 during update.
    pub variance: Vec<f32>,
    /// 8-bit compressed momentum.
    pub momentum_i8: Vec<i8>,
    /// 8-bit compressed variance.
    pub variance_i8: Vec<i8>,
    /// Absmax scale for momentum.
    pub scale_m: f32,
    /// Absmax scale for variance.
    pub scale_v: f32,
    /// Accumulated low-rank gradient R (r×r).
    pub grad_accum: Vec<f32>,
}

impl LowRankAdamState {
    /// Create zero-initialized state for rank `r`.
    pub fn new(r: usize) -> Self {
        let n = r * r;
        Self {
            momentum: vec![0.0; n],
            variance: vec![0.0; n],
            momentum_i8: vec![0; n],
            variance_i8: vec![0; n],
            scale_m: 1.0,
            scale_v: 1.0,
            grad_accum: vec![0.0; n],
        }
    }

    /// Load 8-bit states from System RAM into FP32 VRAM-resident buffers.
    pub fn dequantize_from_ram(&mut self) {
        dequantize_absmax(&self.momentum_i8, self.scale_m, &mut self.momentum);
        dequantize_absmax(&self.variance_i8, self.scale_v, &mut self.variance);
    }

    /// Write FP32 states back to 8-bit System RAM buffers.
    pub fn quantize_to_ram(&mut self) {
        self.scale_m = quantize_absmax(&self.momentum, &mut self.momentum_i8);
        self.scale_v = quantize_absmax(&self.variance, &mut self.variance_i8);
    }

    /// Accumulate projected gradient into low-rank accumulator.
    pub fn accumulate_grad(&mut self, projected: &[f32]) {
        assert_eq!(projected.len(), self.grad_accum.len());
        for (a, g) in self.grad_accum.iter_mut().zip(projected.iter()) {
            *a += g;
        }
    }

    /// Squared Frobenius norm of accumulated gradient.
    pub fn grad_accum_sqnorm(&self) -> f64 {
        self.grad_accum
            .iter()
            .map(|v| f64::from(*v) * f64::from(*v))
            .sum()
    }

    /// Zero the gradient accumulator.
    pub fn zero_accum(&mut self) {
        self.grad_accum.fill(0.0);
    }
}

/// Result of one AdamW step in low-rank space.
pub struct AdamWStepResult {
    /// Full-size weight update G̃ (m×n) to apply: w -= α * G̃.
    pub weight_delta: Vec<f32>,
    /// Normalized update N (r×r) before back-projection.
    pub normalized: Vec<f32>,
}

/// Execute Algorithm 2: AdamW in low-rank space with back-projection.
///
/// 1. Project gradient: R_t = P^T @ G @ Q  (caller provides `grad` full or pre-projected)
/// 2. AdamW on r×r: M, V, N = M / (sqrt(V) + ε)
/// 3. Back-project: G̃ = P @ N @ Q^T
/// 4. Quantize M, V to 8-bit
pub fn adamw_lowrank_step(
    grad_full: &[f32],
    p: &[f32],
    q: &[f32],
    state: &mut LowRankAdamState,
    m: usize,
    n: usize,
    r: usize,
    cfg: &AdamWConfig,
    clip_scale: f32,
) -> AdamWStepResult {
    let mut projected = vec![0.0f32; r * r];
    project_forward_f32(grad_full, p, q, &mut projected, m, n, r);

    // Add any accumulated gradient from backward hook.
    for (a, g) in projected.iter_mut().zip(state.grad_accum.iter()) {
        *a += g;
    }
    if clip_scale != 1.0 {
        for v in projected.iter_mut() {
            *v *= clip_scale;
        }
    }

    let rr = r * r;
    for i in 0..rr {
        state.momentum[i] = cfg.beta1 * state.momentum[i] + (1.0 - cfg.beta1) * projected[i];
        state.variance[i] =
            cfg.beta2 * state.variance[i] + (1.0 - cfg.beta2) * projected[i] * projected[i];
    }

    let mut normalized = vec![0.0f32; rr];
    for i in 0..rr {
        normalized[i] = state.momentum[i] / (state.variance[i].sqrt() + cfg.eps);
    }

    let mut weight_delta = vec![0.0f32; m * n];
    project_backward_f32(&normalized, p, q, &mut weight_delta, m, n, r);

    let lr = effective_lr(cfg, r);
    for v in weight_delta.iter_mut() {
        *v *= lr;
    }

    state.quantize_to_ram();
    state.zero_accum();

    AdamWStepResult {
        weight_delta,
        normalized,
    }
}

/// Apply weight update: w -= delta (delta already includes lr).
pub fn apply_weight_delta(weights: &mut [f32], delta: &[f32], weight_decay: f32, lr: f32) {
    assert_eq!(weights.len(), delta.len());
    for (w, d) in weights.iter_mut().zip(delta.iter()) {
        *w -= d;
        if weight_decay > 0.0 {
            *w *= 1.0 - lr * weight_decay;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowrank_step_runs() {
        let m = 8usize;
        let n = 8usize;
        let r = 2usize;
        let g: Vec<f32> = (0..m * n).map(|i| 0.001 * (i as f32)).collect();
        let p: Vec<f32> = (0..m * r).map(|i| 0.1 * ((i % 3) as f32)).collect();
        let q: Vec<f32> = (0..n * r).map(|i| 0.1 * ((i % 4) as f32)).collect();
        let mut state = LowRankAdamState::new(r);
        state.dequantize_from_ram();
        let result = adamw_lowrank_step(&g, &p, &q, &mut state, m, n, r, &AdamWConfig::default(), 1.0);
        assert_eq!(result.weight_delta.len(), m * n);
    }
}
