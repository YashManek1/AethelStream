//! Standard full-rank AdamW baseline for GaLore validation tests.

use crate::adamw::AdamWConfig;
use doublepass::hook::ProjectorKind;
use doublepass::OptimizerBackend;
use std::collections::HashMap;
use std::sync::{Mutex, RwLock};

type ParamKey = (u32, String);

struct ParamState {
    momentum: Vec<f32>,
    variance: Vec<f32>,
    grad_accum: Vec<f32>,
}

/// Full-rank AdamW optimizer (baseline for Theorem 3.8 comparison).
pub struct StandardAdamW {
    adam: AdamWConfig,
    params: RwLock<HashMap<ParamKey, ParamState>>,
    pending_deltas: Mutex<HashMap<ParamKey, Vec<f32>>>,
    step: Mutex<u64>,
}

impl StandardAdamW {
    /// Create optimizer for `(layer_idx, param_name, num_elements)` specs.
    pub fn new(adam: AdamWConfig, param_specs: &[(u32, &str, usize)]) -> Self {
        let mut params = HashMap::new();
        for &(layer_idx, name, n) in param_specs {
            params.insert(
                (layer_idx, name.to_string()),
                ParamState {
                    momentum: vec![0.0; n],
                    variance: vec![0.0; n],
                    grad_accum: vec![0.0; n],
                },
            );
        }
        Self {
            adam,
            params: RwLock::new(params),
            pending_deltas: Mutex::new(HashMap::new()),
            step: Mutex::new(0),
        }
    }

    /// Take computed weight delta for `(layer_idx, param_name)`.
    pub fn take_weight_delta(&self, layer_idx: u32, param_name: &str) -> Option<Vec<f32>> {
        self.pending_deltas
            .lock()
            .ok()?
            .remove(&(layer_idx, param_name.to_string()))
    }

    /// Current training step.
    pub fn step_count(&self) -> u64 {
        self.step.lock().map(|s| *s).unwrap_or(0)
    }
}

impl OptimizerBackend for StandardAdamW {
    fn project_and_accumulate(&self, grad: &[f32], layer_idx: u32, param_name: &str) {
        let key = (layer_idx, param_name.to_string());
        if let Ok(mut params) = self.params.write() {
            if let Some(ps) = params.get_mut(&key) {
                if grad.len() == ps.grad_accum.len() {
                    for (a, g) in ps.grad_accum.iter_mut().zip(grad.iter()) {
                        *a += g;
                    }
                }
            }
        }
    }

    fn lowrank_grad_sqnorm(&self, layer_idx: u32, param_name: &str) -> f64 {
        let key = (layer_idx, param_name.to_string());
        self.params
            .read()
            .ok()
            .and_then(|p| p.get(&key).map(|ps| ps.grad_accum.iter().map(|v| f64::from(*v) * f64::from(*v)).sum()))
            .unwrap_or(0.0)
    }

    fn apply_update(&self, layer_idx: u32, param_name: &str, clip_scale: f32) {
        let key = (layer_idx, param_name.to_string());
        let delta = {
            let mut params = match self.params.write() {
                Ok(p) => p,
                Err(_) => return,
            };
            let Some(ps) = params.get_mut(&key) else {
                return;
            };

            if clip_scale != 1.0 {
                for v in ps.grad_accum.iter_mut() {
                    *v *= clip_scale;
                }
            }

            let cfg = &self.adam;
            let n = ps.grad_accum.len();
            let mut delta = vec![0.0f32; n];
            for i in 0..n {
                let g = ps.grad_accum[i];
                ps.momentum[i] = cfg.beta1 * ps.momentum[i] + (1.0 - cfg.beta1) * g;
                ps.variance[i] = cfg.beta2 * ps.variance[i] + (1.0 - cfg.beta2) * g * g;
                delta[i] = cfg.lr * ps.momentum[i] / (ps.variance[i].sqrt() + cfg.eps);
            }
            delta
        };

        if let Ok(mut pending) = self.pending_deltas.lock() {
            pending.insert(key, delta);
        }
    }

    fn zero_accum(&self, layer_idx: u32, param_name: &str) {
        let key = (layer_idx, param_name.to_string());
        if let Ok(mut params) = self.params.write() {
            if let Some(ps) = params.get_mut(&key) {
                ps.grad_accum.fill(0.0);
            }
        }
    }

    fn notify_step(&self, step: u64) {
        if let Ok(mut s) = self.step.lock() {
            *s = step;
        }
    }

    fn projector_kind(&self, _layer_idx: u32, _param_name: &str) -> ProjectorKind {
        // Full-rank baseline uses global-norm clip (same path as GaLore in Theorem 3.8 tests).
        ProjectorKind::Orthonormal
    }
}
