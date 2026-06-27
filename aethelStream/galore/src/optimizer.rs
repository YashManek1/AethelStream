//! GaLore heterogeneous optimizer implementing [`OptimizerBackend`].

use crate::adamw::{effective_lr, AdamWConfig, LowRankAdamState};
use crate::error::{GaLoreError, Result};
use crate::layer_rank::LayerRankConfig;
use crate::project::{project_backward_f32, project_forward_f32, projection_roundtrip_error};
use crate::randomized_svd::{randomized_svd_on_device, should_switch_subspace, RandomizedSvdConfig};
use crate::state_file::OptimizerStateFile;
use doublepass::hook::ProjectorKind;
use doublepass::OptimizerBackend;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, RwLock};

/// Key for per-parameter optimizer state.
type ParamKey = (u32, String);

/// Per-parameter GaLore state held in System RAM (8-bit compressed m/v + FP16 P/Q on disk).
struct ParamState {
    m: usize,
    n: usize,
    rank: usize,
    /// Left projection P (m×r) — resident in System RAM when not on GPU.
    p: Vec<f32>,
    /// Right projection Q (n×r).
    q: Vec<f32>,
    /// Low-rank Adam state (8-bit m/v in RAM between steps).
    adam: LowRankAdamState,
    /// Layer index in optimizer_states.bin (for O(1) seek).
    file_layer_index: usize,
}

/// Configuration for [`GaLoreOptimizer`].
#[derive(Debug, Clone)]
pub struct GaLoreConfig {
    /// Default projection rank r.
    pub rank: usize,
    /// Per-layer-type rank overrides.
    pub layer_ranks: LayerRankConfig,
    /// Subspace switch interval T_switch (default 200). Set to 0 to hold projection constant.
    pub switch_interval: u64,
    /// Randomized SVD oversampling p.
    pub oversampling: usize,
    /// AdamW hyperparameters.
    pub adam: AdamWConfig,
    /// Path to memory-mapped optimizer_states.bin.
    pub state_file_path: PathBuf,
}

impl Default for GaLoreConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            layer_ranks: LayerRankConfig::default(),
            switch_interval: 200,
            oversampling: 10,
            adam: AdamWConfig::default(),
            state_file_path: PathBuf::from("optimizer_states.bin"),
        }
    }
}

/// Heterogeneous GaLore optimizer: 8-bit compressed AdamW state in System RAM,
/// FP16 projection matrices, periodic randomized SVD subspace refresh.
pub struct GaLoreOptimizer {
    config: GaLoreConfig,
    params: RwLock<HashMap<ParamKey, ParamState>>,
    step: Mutex<u64>,
    version: Mutex<u64>,
    state_file: Mutex<Option<OptimizerStateFile>>,
    pending_svd: Mutex<Vec<(u32, String, Vec<f32>)>>,
    /// Full gradients accumulated on subspace-refresh steps (before projection).
    pending_svd_grads: Mutex<HashMap<ParamKey, Vec<f32>>>,
    pending_deltas: Mutex<HashMap<ParamKey, Vec<f32>>>,
}

impl GaLoreOptimizer {
    /// Create optimizer and register parameter shapes.
    ///
    /// `param_specs`: `(layer_idx, param_name, m, n)` for each trainable matrix.
    pub fn new(config: GaLoreConfig, param_specs: &[(u32, &str, usize, usize)]) -> Result<Self> {
        let dims: Vec<(u32, u32)> = param_specs
            .iter()
            .map(|&(_, _, m, n)| (m as u32, n as u32))
            .collect();
        let layer_ranks: Vec<u32> = param_specs
            .iter()
            .map(|&(_, name, m, n)| config.layer_ranks.rank_for_param(name, m, n) as u32)
            .collect();

        let state_file = OptimizerStateFile::create(
            &config.state_file_path,
            &dims,
            &layer_ranks,
            &config.adam,
        )?;

        let mut params = HashMap::new();
        for (i, &(layer_idx, name, m, n)) in param_specs.iter().enumerate() {
            let r = config.layer_ranks.rank_for_param(name, m, n);
            let p = init_random_projection(m, r, layer_idx.wrapping_mul(17));
            let q = init_random_projection(n, r, layer_idx.wrapping_mul(31));
            let mut adam = LowRankAdamState::new(r);
            adam.quantize_to_ram();

            params.insert(
                (layer_idx, name.to_string()),
                ParamState {
                    m,
                    n,
                    rank: r,
                    p,
                    q,
                    adam,
                    file_layer_index: i,
                },
            );
        }

        let opt = Self {
            config,
            params: RwLock::new(params),
            step: Mutex::new(0),
            version: Mutex::new(0),
            state_file: Mutex::new(Some(state_file)),
            pending_svd: Mutex::new(Vec::new()),
            pending_svd_grads: Mutex::new(HashMap::new()),
            pending_deltas: Mutex::new(HashMap::new()),
        };

        opt.flush_all_to_file()?;
        Ok(opt)
    }

    /// Open existing optimizer from state file; param_specs must match creation order.
    pub fn open(config: GaLoreConfig, param_specs: &[(u32, &str, usize, usize)]) -> Result<Self> {
        let state_file = OptimizerStateFile::open(&config.state_file_path)?;
        let resumed_step = state_file.step_count();
        let mut header_adam = config.adam;
        header_adam.beta1 = state_file.header().beta1;
        header_adam.beta2 = state_file.header().beta2;
        header_adam.eps = state_file.header().eps;

        let mut params = HashMap::new();
        for (i, &(layer_idx, name, m, n)) in param_specs.iter().enumerate() {
            let r = state_file.layer_rank(i)? as usize;
            let p = state_file.read_p_f32(i)?;
            let q = state_file.read_q_f32(i)?;
            let adam = state_file.load_layer_state(i)?;
            params.insert(
                (layer_idx, name.to_string()),
                ParamState {
                    m,
                    n,
                    rank: r,
                    p,
                    q,
                    adam,
                    file_layer_index: i,
                },
            );
        }

        let mut cfg = config;
        cfg.adam = header_adam;

        Ok(Self {
            config: cfg,
            params: RwLock::new(params),
            step: Mutex::new(resumed_step),
            version: Mutex::new(0),
            state_file: Mutex::new(Some(state_file)),
            pending_svd: Mutex::new(Vec::new()),
            pending_svd_grads: Mutex::new(HashMap::new()),
            pending_deltas: Mutex::new(HashMap::new()),
        })
    }

    /// Flush all parameter states to mmap file.
    pub fn flush_all_to_file(&self) -> Result<()> {
        let params = self.params.read().map_err(|_| GaLoreError::Config("lock poison".into()))?;
        let step = self.step.lock().map(|s| *s).unwrap_or(0);
        let mut file_guard = self.state_file.lock().map_err(|_| GaLoreError::Config("lock poison".into()))?;
        if let Some(ref mut file) = *file_guard {
            file.write_header(&self.config.adam, step)?;
            for ps in params.values() {
                file.write_layer_state(ps.file_layer_index, &ps.p, &ps.q, &ps.adam)?;
            }
            file.flush()?;
        }
        Ok(())
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

    /// Process pending SVD jobs on device (never on CPU hot path).
    pub fn process_pending_svd(&self) -> Result<()> {
        let jobs: Vec<_> = {
            let mut pending = self.pending_svd.lock().map_err(|_| GaLoreError::Config("lock".into()))?;
            std::mem::take(&mut *pending)
        };

        if jobs.is_empty() {
            return Ok(());
        }

        let svd_cfg = RandomizedSvdConfig {
            rank: self.config.rank,
            oversampling: self.config.oversampling,
            seed: self.step.lock().map(|s| *s).unwrap_or(0).wrapping_mul(0x9E37_79B9),
            ..Default::default()
        };

        let _refresh_step = self.step.lock().map(|s| *s).unwrap_or(0);

        let mut params = self.params.write().map_err(|_| GaLoreError::Config("lock".into()))?;
        for (layer_idx, name, grad) in jobs {
            let key = (layer_idx, name);
            if let Some(ps) = params.get_mut(&key) {
                let mut cfg = svd_cfg;
                cfg.rank = ps.rank;
                let proj = randomized_svd_on_device(&grad, ps.m, ps.n, &cfg)?;
                ps.p = proj.p;
                ps.q = proj.q;
                // Reset Adam state on subspace change (coordinates are not portable).
                ps.adam = LowRankAdamState::new(ps.rank);
                ps.adam.quantize_to_ram();
            }
        }

        if let Ok(mut ver) = self.version.lock() {
            *ver += 1;
        }
        Ok(())
    }

    /// Current optimizer version (incremented on subspace switch).
    pub fn optimizer_version(&self) -> u64 {
        self.version.lock().map(|v| *v).unwrap_or(0)
    }

    /// Compute projection round-trip error for validation (Test 1 helper).
    pub fn projection_error(layer_idx: u32, param_name: &str, g: &[f32], opt: &Self) -> Result<f64> {
        let params = opt.params.read().map_err(|_| GaLoreError::Config("lock".into()))?;
        let ps = params
            .get(&(layer_idx, param_name.to_string()))
            .ok_or_else(|| GaLoreError::Config("unknown param".into()))?;
        Ok(projection_roundtrip_error(g, &ps.p, &ps.q, ps.m, ps.n, ps.rank))
    }
}

impl OptimizerBackend for GaLoreOptimizer {
    fn project_and_accumulate(&self, grad: &[f32], layer_idx: u32, param_name: &str) {
        let key = (layer_idx, param_name.to_string());
        let this_step = self.step.lock().map(|s| *s + 1).unwrap_or(1);
        let capture_for_svd = this_step == 1
            || should_switch_subspace(this_step, self.config.switch_interval);

        if let Ok(mut params) = self.params.write() {
            if let Some(ps) = params.get_mut(&key) {
                if grad.len() == ps.m * ps.n {
                    if capture_for_svd {
                        if let Ok(mut pending) = self.pending_svd_grads.lock() {
                            pending
                                .entry(key.clone())
                                .and_modify(|acc| {
                                    for (a, g) in acc.iter_mut().zip(grad.iter()) {
                                        *a += g;
                                    }
                                })
                                .or_insert_with(|| grad.to_vec());
                        }
                    }

                    let mut projected = vec![0.0f32; ps.rank * ps.rank];
                    project_forward_f32(grad, &ps.p, &ps.q, &mut projected, ps.m, ps.n, ps.rank);
                    ps.adam.accumulate_grad(&projected);
                }
            }
        }
    }

    fn lowrank_grad_sqnorm(&self, layer_idx: u32, param_name: &str) -> f64 {
        let key = (layer_idx, param_name.to_string());
        self.params
            .read()
            .ok()
            .and_then(|p| p.get(&key).map(|ps| ps.adam.grad_accum_sqnorm()))
            .unwrap_or(0.0)
    }

    fn apply_update(&self, layer_idx: u32, param_name: &str, clip_scale: f32) {
        let key = (layer_idx, param_name.to_string());

        let update_info = {
            let mut params = match self.params.write() {
                Ok(p) => p,
                Err(_) => return,
            };
            let Some(ps) = params.get_mut(&key) else {
                return;
            };
            ps.adam.dequantize_from_ram();

            // Apply clipping to a temporary -- never mutate grad_accum in-place
            // so that zero_accum() is the sole owner of clearing the accumulator.
            let accum: Vec<f32> = if clip_scale != 1.0 {
                ps.adam.grad_accum.iter().map(|v| v * clip_scale).collect()
            } else {
                ps.adam.grad_accum.clone()
            };

            let cfg = &self.config.adam;
            let normalized = crate::adamw::adamw_inner_update(
                &accum,
                &mut ps.adam.momentum,
                &mut ps.adam.variance,
                cfg,
            );

            let mut weight_delta = vec![0.0f32; ps.m * ps.n];
            project_backward_f32(&normalized, &ps.p, &ps.q, &mut weight_delta, ps.m, ps.n, ps.rank);

            let lr = effective_lr(cfg, ps.rank);
            for v in weight_delta.iter_mut() {
                *v *= lr;
            }

            ps.adam.quantize_to_ram();

            Some((
                key.clone(),
                weight_delta,
                ps.file_layer_index,
                ps.p.clone(),
                ps.q.clone(),
                ps.adam.clone_for_writeback(),
            ))
        };

        if let Some((key, weight_delta, file_idx, p, q, adam)) = update_info {
            if let Ok(mut pending) = self.pending_deltas.lock() {
                pending.insert(key, weight_delta);
            }
            if let Ok(mut file_guard) = self.state_file.lock() {
                if let Some(ref mut file) = *file_guard {
                    let _ = file.write_layer_state(file_idx, &p, &q, &adam);
                }
            }
        }
    }

    fn zero_accum(&self, layer_idx: u32, param_name: &str) {
        let key = (layer_idx, param_name.to_string());
        if let Ok(mut params) = self.params.write() {
            if let Some(ps) = params.get_mut(&key) {
                ps.adam.zero_accum();
            }
        }
    }

    fn notify_step(&self, step: u64) {
        if let Ok(mut s) = self.step.lock() {
            *s = step;
        }

        // Move captured full gradients into SVD job queue.
        if let Ok(mut grads) = self.pending_svd_grads.lock() {
            if !grads.is_empty() {
                if let Ok(mut pending) = self.pending_svd.lock() {
                    for ((layer_idx, name), grad) in grads.drain() {
                        pending.push((layer_idx, name, grad));
                    }
                }
            }
        }

        let has_pending = self
            .pending_svd
            .lock()
            .map(|p| !p.is_empty())
            .unwrap_or(false);
        if has_pending {
            let _ = self.process_pending_svd();
            let _ = self.flush_all_to_file();
        }
    }

    fn projector_kind(&self, _layer_idx: u32, _param_name: &str) -> ProjectorKind {
        ProjectorKind::Orthonormal
    }
}

impl LowRankAdamState {
    fn clone_for_writeback(&self) -> Self {
        Self {
            momentum: self.momentum.clone(),
            variance: self.variance.clone(),
            momentum_i8: self.momentum_i8.clone(),
            variance_i8: self.variance_i8.clone(),
            scale_m: self.scale_m,
            scale_v: self.scale_v,
            grad_accum: self.grad_accum.clone(),
        }
    }
}

fn init_random_projection(rows: usize, cols: usize, seed: u32) -> Vec<f32> {
    let mut rng_state = seed as u64 + 1;
    let mut data = vec![0.0f32; rows * cols];
    for v in data.iter_mut() {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        *v = ((rng_state >> 40) as f32 / (1u32 << 24) as f32) - 0.5;
    }
    for j in 0..cols {
        for i in 0..j {
            let mut dot = 0.0f32;
            for row in 0..rows {
                dot += data[row * cols + i] * data[row * cols + j];
            }
            for row in 0..rows {
                data[row * cols + j] -= dot * data[row * cols + i];
            }
        }
        let mut norm = 0.0f32;
        for row in 0..rows {
            norm += data[row * cols + j] * data[row * cols + j];
        }
        norm = norm.max(1e-12).sqrt();
        for row in 0..rows {
            data[row * cols + j] /= norm;
        }
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimizer_registers_params() {
        let cfg = GaLoreConfig {
            state_file_path: std::env::temp_dir().join("galore_opt_test.bin"),
            rank: 4,
            ..Default::default()
        };
        let specs = [(0u32, "d_wq", 32usize, 32usize)];
        let opt = GaLoreOptimizer::new(cfg, &specs).expect("new");
        assert_eq!(opt.lowrank_grad_sqnorm(0, "d_wq"), 0.0);
        let _ = std::fs::remove_file(std::env::temp_dir().join("galore_opt_test.bin"));
    }
}


