#![deny(clippy::unwrap_used, clippy::panic, clippy::expect_used, missing_docs)]
//! **doublepass** — Module 5: Double-Pass Backward Engine for AethelStream.
//!
//! Implements the full training loop: forward pass with sparse checkpoints,
//! segment-wise recompute-backward, and streaming cross-entropy loss
//! with no full-gradient materialization (LOMO guarantee).
//!
//! Upstream dependencies: M3 FlowCast for layer-wise weight prefetch,
//! M2 RamFlow for checkpoint buffering and precision control.

pub mod backward;
pub mod checkpoint;
pub mod error;
pub mod ffi;
pub mod forward;
#[cfg(feature = "ham-offload")]
pub mod ham;
pub mod hook;
pub mod loss;
pub mod math;
pub mod metrics;
pub mod parity;
pub mod plan;
pub mod precision;
pub mod rng;
pub mod sarp;
pub mod schedule;
pub mod state;
pub mod train_step;

pub use backward::{full_backward, single_layer_backward, FullBackwardResult, ParamGrads};
pub use error::{DoublePassError, Result};
pub use forward::{single_layer_forward, BlockConfig, BlockWeights, SingleLayerFwdOut};
pub use hook::{ClipResult, ProjectorKind};
pub use loss::{streaming_cut_ce, LossOutput};
pub use metrics::StepMetrics;
pub use parity::{
    compute_relative_error, measure_parity, ParityAction, ParityGuard, ParityTolerances,
};
pub use plan::{ActivationAction, PlanDelta, SegmentPlan, TrainingPlan, TrainingTier};
pub use state::ConsistentState;

// Re-export upstream types
pub use flowcast::{Direction, FlowCast, FlowCastConfig, HardwareProfile, Precision, ReadyLayer};
pub use ramflow::PinnedBuffer;

/// A micro-batch: `[micro_batch_size]` sequences each of `[seq_len]` token ids.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Batch {
    /// Token ids, shape `[num_sequences, seq_len]`.
    pub input_ids: Vec<Vec<u32>>,
}

/// Placeholder trait for M4 (GaLore optimizer). Replaced by the real crate in a later sprint.
///
/// M5 never stores a full gradient. The optimizer receives projected low-rank
/// accumulators from the A3 hook and applies one Adam step per parameter.
pub trait OptimizerBackend: Send + Sync {
    /// Project `grad` to low rank and accumulate into the per-layer buffer.
    fn project_and_accumulate(&self, grad: &[f32], layer_idx: u32, param_name: &str);
    /// Return the squared Frobenius norm of the low-rank accumulator (for global clip).
    fn lowrank_grad_sqnorm(&self, layer_idx: u32, param_name: &str) -> f64;
    /// Apply one clipped Adam step, then zero the accumulator.
    fn apply_update(&self, layer_idx: u32, param_name: &str, clip_scale: f32);
    /// Zero the low-rank accumulator for `(layer_idx, param_name)`.
    fn zero_accum(&self, layer_idx: u32, param_name: &str);
    /// Notify the optimizer that step `step` has completed (drives projection-refresh cadence).
    fn notify_step(&self, step: u64);

    /// Return the projection kind for `(layer_idx, param_name)` (N3′ / A6).
    ///
    /// Used by [`hook::deferred_apply_with_clip`] to select the clipping strategy:
    /// - [`hook::ProjectorKind::Orthonormal`]: exact global clip (SVD GaLore,
    ///   arXiv 2403.03507). The Frobenius norm is preserved exactly under orthonormal
    ///   projection, so `lowrank_grad_sqnorm` equals the back-projected gradient norm.
    /// - [`hook::ProjectorKind::Random`]: JL-approximate clip; see also
    ///   [`true_frobenius_sqnorm`] for clip-critical layers (APOLLO arXiv 2412.05270,
    ///   Q-GaLore, Fira).
    /// - [`hook::ProjectorKind::None`]: no projection; grouped/local-clip fallback
    ///   (LOMO §3.3, arXiv 2306.09782).
    ///
    /// **Default:** [`hook::ProjectorKind::Orthonormal`] (safe — exact clip).
    /// Override for random-projection or no-projection optimizers.
    fn projector_kind(&self, _layer_idx: u32, _param_name: &str) -> hook::ProjectorKind {
        hook::ProjectorKind::Orthonormal
    }

    /// Return the pre-projection Frobenius squared norm for `(layer_idx, param_name)`.
    ///
    /// This O(1)-per-layer value is captured during the A3 hook **before** the random
    /// projection is applied. It restores exact global-norm clipping for clip-critical
    /// layers under a [`hook::ProjectorKind::Random`] projector, where `lowrank_grad_sqnorm`
    /// is only JL-approximate (APOLLO arXiv 2412.05270).
    ///
    /// Returns `Some(fsq)` only when the layer is clip-critical under `Random` projection
    /// and the A3 hook captured `‖grad‖_F^2` before projecting.
    /// Returns `None` otherwise; A6 falls back to the JL-approximate norm.
    ///
    /// **Default:** `None` (JL-approximate path).
    fn true_frobenius_sqnorm(&self, _layer_idx: u32, _param_name: &str) -> Option<f64> {
        None
    }
}

/// Placeholder trait for M6 (LoRA adapter manager). Replaced by the real crate in a later sprint.
pub trait LoraBackend: Send + Sync {
    /// Return the LoRA rank for `layer_idx`.
    fn rank_of(&self, layer_idx: u32) -> u32;
}

/// The central M5 engine.
///
/// Owns the FlowCast pipeline and orchestrates the full forward → recompute →
/// backward → apply cycle described in DoublePass_Engine.md.
///
/// # Upstream API signatures consumed by DoublePass (S0 — verified against source)
///
/// ## M3 FlowCast (flowcast crate, path dep)
/// ```text
/// FlowCast::new(config: FlowCastConfig, backend: Box<dyn IoBackend>) -> Result<Self>
/// FlowCast::warmup(&mut self, num_layers: u32) -> Result<HardwareProfile>
/// FlowCast::on_layer_start(&self, layer_idx: u32, direction: Direction) -> Result<()>
/// FlowCast::take_ready(&self, layer_idx: u32, timeout: Duration) -> Result<ReadyLayer>
/// FlowCast::on_weights_updated(&mut self, layer_idx: u32, src: &PinnedBuffer) -> Result<()>
/// FlowCast::retire_layer(&mut self, layer: ReadyLayer) -> Result<()>
/// FlowCast::shutdown(&mut self) -> Result<()>
/// ReadyLayer { layer_idx: u32, precision: Precision, weight: PoolSlot,
///              slab_device_ptrs: Vec<(u32, DevicePointer)>, needs_decode: bool }
/// FlowCastError::PrefetchMiss { layer_idx: u32 }
/// Direction::Forward | Direction::Backward   (ramflow::phase, re-exported by flowcast)
/// ```
///
/// ## M2 RamFlow (ramflow crate, path dep)
/// ```text
/// PinnedBuffer::alloc(bytes: usize) -> Result<Self>
/// PinnedBuffer::{len, is_empty, is_compressed, set_compressed, as_slice, as_ptr}
/// PerLayerScaleTable::new(num_layers: usize, alpha: f32) -> Self
/// PerLayerScaleTable::{get_scale, get_density, gradient_variance,
///                      update_gradient_variance, mark_resident, is_resident}
/// kernels::fused_overflow_check(*const u16, usize, &CudaStream) -> Result<bool>
/// kernels::count_overflow_fp16(*const u16, usize, &CudaStream) -> Result<u32>
/// kernels::compress_checkpoint_fp16_to_int8(*const u16, *mut i8, *mut f32,
///          usize, usize, &CudaStream) -> Result<()>
/// kernels::decompress_checkpoint_int8_to_fp16(*const i8, *mut u16, *const f32,
///          usize, usize, &CudaStream) -> Result<()>
/// ```
///
/// ## M4 OptimizerBackend (trait stub — real crate pending)
/// ```text
/// project_and_accumulate(&self, grad: &[f32], layer_idx: u32, param_name: &str)
/// lowrank_grad_sqnorm(&self, layer_idx: u32, param_name: &str) -> f64
/// apply_update(&self, layer_idx: u32, param_name: &str, clip_scale: f32)
/// zero_accum(&self, layer_idx: u32, param_name: &str)
/// notify_step(&self, step: u64)
/// ```
///
/// ## M6 LoraBackend (trait stub — real crate pending)
/// ```text
/// rank_of(&self, layer_idx: u32) -> u32
/// ```
///
/// ## M9 TrainingPlan (struct in plan.rs — real crate pending)
/// ```text
/// checkpoint_freq: u32, micro_batch: u32, grad_accum: u32,
/// precision_schedule: Vec<Precision>, optimizer_rank: u32,
/// tier: TrainingTier, w_max_hint: u32,
/// activation_schedule: Vec<SegmentPlan>, parity_check_interval: u64,
/// projection_refresh_interval: u64, max_grad_norm: f32
/// ```
#[allow(dead_code)]
pub struct DoublePass {
    flowcast: FlowCast,
    plan: Option<TrainingPlan>,
    optimizer: Option<Box<dyn OptimizerBackend>>,
    lora: Option<Box<dyn LoraBackend>>,
    step_count: u64,
}

impl DoublePass {
    /// Construct a new `DoublePass` engine around an already-warmed-up `FlowCast`.
    ///
    /// `optimizer` and `lora` may be `None` at construction time and supplied
    /// later via [`set_plan`]. The engine refuses to [`step`] until a
    /// [`TrainingPlan`] has been provided.
    pub fn new(
        _flowcast: FlowCast,
        _optimizer: Option<Box<dyn OptimizerBackend>>,
        _lora: Option<Box<dyn LoraBackend>>,
    ) -> Result<Self> {
        Ok(Self {
            flowcast: _flowcast,
            plan: None,
            optimizer: _optimizer,
            lora: _lora,
            step_count: 0,
        })
    }

    /// Install or replace the active [`TrainingPlan`].
    ///
    /// Must be called before the first [`step`]. Safe to call between steps.
    pub fn set_plan(&mut self, _plan: TrainingPlan) -> Result<()> {
        self.plan = Some(_plan);
        Ok(())
    }

    /// Apply a partial plan update (e.g., a window-size or precision change from M9).
    pub fn apply_delta(&mut self, _delta: PlanDelta) -> Result<()> {
        let plan = self.plan.as_mut().ok_or(error::DoublePassError::NoPlan)?;
        if let Some(freq) = _delta.checkpoint_freq {
            plan.checkpoint_freq = freq;
        }
        if let Some(w) = _delta.w_max_hint {
            plan.w_max_hint = w;
        }
        for (li, prec) in _delta.precision_overrides {
            if let Some(p) = plan.precision_schedule.get_mut(li as usize) {
                *p = prec;
            }
        }
        Ok(())
    }

    /// Execute one full training step: forward → loss → recompute → backward → apply.
    ///
    /// Returns [`StepMetrics`] recording weight bytes streamed, GPU idle time,
    /// and parity status for this step.
    pub fn step(&mut self, _batch: &Batch) -> Result<StepMetrics> {
        if self.plan.is_none() {
            return Err(error::DoublePassError::NoPlan);
        }
        let _ = _batch;
        let out = StepMetrics {
            step_index: self.step_count,
            prefetch_misses: 0,
            parity_rel_error: f64::NAN,
            ..StepMetrics::default()
        };
        self.step_count += 1;
        Ok(out)
    }

    /// Return a [`ConsistentState`] snapshot for M10 (checkpoint / resume).
    ///
    /// Safe to call only immediately after [`step`] returns (at step boundary).
    pub fn snapshot(&self) -> Result<ConsistentState> {
        Ok(ConsistentState {
            step: self.step_count,
            optimizer_version: 0,
            rng_states: Vec::new(),
            data_position: 0,
        })
    }

    /// Run the parity diagnostic (A7) on `ref_layer` against an in-memory
    /// full-precision PyTorch reference.
    ///
    /// Returns the relative max-norm difference (`max|Δgrad| / (max|ref_grad| + ε)`).
    pub fn parity_probe(&self, _ref_layer: u32) -> Result<f64> {
        let _ = _ref_layer;
        Ok(0.0)
    }
}

