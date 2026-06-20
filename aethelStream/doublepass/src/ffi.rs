//! PyO3 FFI surface for M7 (Python training loop driver).
//!
//! When built with `--features python-ffi`, this module exports a
//! `doublepass_py` Python extension module containing [`PyDoublePass`].
//!
//! # Building the Python extension
//!
//! ```bash
//! pip install maturin
//! # In aethelStream/doublepass/:
//! maturin develop --features python-ffi
//! ```
//!
//! Then from Python:
//! ```python
//! import doublepass_py
//! dp = doublepass_py.PyDoublePass(
//!     n_layers=2, d_model=32, n_heads=2, d_ff=64,
//!     seq_len=4, batch=1, vocab_size=256
//! )
//! metrics_json = dp.step([[0.1]*128], list(range(4)))
//! print(metrics_json)
//! snapshot = dp.snapshot()
//! print(snapshot)
//! ```
//!
//! Without `--features python-ffi` this file compiles as an empty module so
//! the `pub mod ffi;` declaration in `lib.rs` always succeeds.

#[cfg(feature = "python-ffi")]
use pyo3::prelude::*;

// ── No-op optimizer used by PyDoublePass ────────────────────────────────────

#[cfg(feature = "python-ffi")]
struct FfiNoOpOpt;

#[cfg(feature = "python-ffi")]
impl crate::OptimizerBackend for FfiNoOpOpt {
    fn project_and_accumulate(&self, _grad: &[f32], _layer_idx: u32, _param_name: &str) {}
    fn lowrank_grad_sqnorm(&self, _layer_idx: u32, _param_name: &str) -> f64 {
        0.0
    }
    fn apply_update(&self, _layer_idx: u32, _param_name: &str, _clip_scale: f32) {}
    fn zero_accum(&self, _layer_idx: u32, _param_name: &str) {}
    fn notify_step(&self, _step: u64) {}
    fn projector_kind(
        &self,
        _layer_idx: u32,
        _param_name: &str,
    ) -> crate::hook::ProjectorKind {
        crate::hook::ProjectorKind::None
    }
}

// ── PyDoublePass ─────────────────────────────────────────────────────────────

/// Python-facing M5 DoublePass engine.
///
/// Wraps [`crate::train_step::full_training_step`] (A1→A8→A2'→A6) for use in
/// HuggingFace-style Python training loops. All tensor I/O uses Python lists
/// (no NumPy dependency) for maximum portability.
///
/// # Mock path
///
/// When compiled with `--features mock-cuda` (the default for CI), all CUDA
/// calls are CPU f32 stubs. Numbers returned by `step()` reflect code-path
/// correctness, not real GPU performance.
#[cfg(feature = "python-ffi")]
#[pyclass]
pub struct PyDoublePass {
    model: crate::forward::Model,
    lm_head: Vec<f32>,
    plan: crate::plan::TrainingPlan,
    cfg: crate::train_step::StepConfig,
    step_count: u64,
    parity_tols: crate::parity::ParityTolerances,
}

#[cfg(feature = "python-ffi")]
#[pymethods]
impl PyDoublePass {
    /// Construct a new `PyDoublePass` engine with a randomly initialised model.
    ///
    /// Parameters match those of [`crate::forward::BlockConfig`]:
    /// - `n_layers`: number of transformer blocks
    /// - `d_model`: residual-stream width
    /// - `n_heads`: number of attention heads
    /// - `d_ff`: feed-forward intermediate width
    /// - `seq_len`: sequence length per micro-batch
    /// - `batch`: micro-batch size
    /// - `vocab_size`: vocabulary size for the LM head
    /// - `chunk_size`: vocabulary tile width for streaming CE (default 256)
    #[new]
    #[pyo3(signature = (n_layers, d_model, n_heads, d_ff, seq_len, batch, vocab_size, chunk_size=256))]
    pub fn new(
        n_layers: usize,
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        seq_len: usize,
        batch: usize,
        vocab_size: usize,
        chunk_size: usize,
    ) -> Self {
        let blk_cfg = crate::forward::BlockConfig {
            d_model,
            n_heads,
            d_ff,
            seq_len,
            batch,
            dropout_p: 0.0,
        };
        let model = crate::forward::Model::new(n_layers, blk_cfg);
        let lm_head: Vec<f32> = (0..vocab_size * d_model)
            .map(|i| ((i as f64 * 0.137_f64).sin() * 0.02_f64) as f32)
            .collect();
        Self {
            model,
            lm_head,
            plan: crate::plan::TrainingPlan {
                checkpoint_freq: 1,
                ..Default::default()
            },
            cfg: crate::train_step::StepConfig {
                vocab_size,
                chunk_size,
                ..Default::default()
            },
            step_count: 0,
            parity_tols: crate::parity::ParityTolerances::default(),
        }
    }

    /// Run one full training step (A1 forward → A8 loss → A2' backward → A6 clip).
    ///
    /// - `inputs`: list of micro-batch tensors; each inner list has length
    ///   `batch * seq_len * d_model` (flat row-major f32).
    /// - `labels`: flat list of token-id labels with length `G * batch * seq_len`.
    ///
    /// Returns a JSON string containing [`crate::metrics::StepMetrics`].
    pub fn step(&mut self, inputs: Vec<Vec<f32>>, labels: Vec<u32>) -> PyResult<String> {
        let opt = FfiNoOpOpt;
        let out =
            crate::train_step::full_training_step(
                &self.model,
                &self.lm_head,
                &inputs,
                &labels,
                &self.plan,
                &self.cfg,
                &opt,
                &[], // empty trainable_layers → skip A6 apply_update
            )
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let g = inputs.len() as u32;
        let tokens = (g as u64) * (self.model.cfg.bs() as u64);
        let bpl = self.model.cfg.bytes_per_layer() as u64;
        let l = self.model.num_layers() as u64;
        let bwd = l * bpl;
        let fwd = l * bpl;

        let metrics = crate::metrics::StepMetrics {
            step_index: self.step_count,
            weight_bytes_streamed: out.weight_loads,
            forward_weight_bytes: fwd,
            backward_weight_bytes: bwd,
            grad_accum_steps: g,
            tokens_processed: tokens,
            prefetch_misses: 0,
            parity_rel_error: f64::NAN,
            ..crate::metrics::StepMetrics::default()
        };

        self.step_count += 1;
        Ok(metrics.to_json())
    }

    /// Install a new [`crate::plan::TrainingPlan`] from a JSON string.
    ///
    /// The plan controls checkpoint frequency, precision schedule, SARP activation
    /// schedule, and grad-norm clipping threshold.
    pub fn set_plan(&mut self, plan_json: &str) -> PyResult<()> {
        let plan: crate::plan::TrainingPlan = serde_json::from_str(plan_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.plan = plan;
        Ok(())
    }

    /// Apply a partial plan update from a JSON [`crate::plan::PlanDelta`].
    ///
    /// Supports hot-patching `checkpoint_freq`, `w_max_hint`, and
    /// `precision_overrides` without a full `set_plan` round-trip.
    pub fn apply_delta(&mut self, delta_json: &str) -> PyResult<()> {
        let delta: crate::plan::PlanDelta = serde_json::from_str(delta_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        if let Some(freq) = delta.checkpoint_freq {
            self.plan.checkpoint_freq = freq;
        }
        if let Some(w) = delta.w_max_hint {
            self.plan.w_max_hint = w;
        }
        for (li, prec) in delta.precision_overrides {
            if let Some(p) = self.plan.precision_schedule.get_mut(li as usize) {
                *p = prec;
            }
        }
        Ok(())
    }

    /// Return a JSON [`crate::state::ConsistentState`] snapshot for M10 resume.
    ///
    /// Safe to call only immediately after `step()` returns (at a step boundary).
    pub fn snapshot(&self) -> PyResult<String> {
        let state = crate::state::ConsistentState {
            step: self.step_count,
            optimizer_version: 0,
            rng_states: Vec::new(),
            data_position: 0,
        };
        serde_json::to_string_pretty(&state)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Run the A7 parity diagnostic on one layer and return the relative error.
    ///
    /// - `ref_layer`: layer index to probe
    /// - `stream_grad`: gradient from the M5 streaming path (flat f32 list)
    /// - `ref_grad`: reference FP32 autograd gradient (same shape)
    ///
    /// Returns `rel = max|stream - ref| / (max|ref| + eps)`.
    /// Raises `RuntimeError` if `rel >= tol_halt`.
    pub fn parity_probe(
        &mut self,
        ref_layer: u32,
        stream_grad: Vec<f32>,
        ref_grad: Vec<f32>,
    ) -> PyResult<f64> {
        crate::parity::measure_parity(ref_layer, &stream_grad, &ref_grad, &self.parity_tols)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}

// ── Python module registration ───────────────────────────────────────────────

/// Register the `doublepass_py` Python extension module.
///
/// Exports: `PyDoublePass`
#[cfg(feature = "python-ffi")]
#[pymodule]
fn doublepass_py(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDoublePass>()?;
    Ok(())
}
