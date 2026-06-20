//! Per-step performance and correctness metrics.

/// Per-step performance and correctness counters returned by [`crate::DoublePass::step`].
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct StepMetrics {
    /// Total weight bytes streamed this step (forward + backward).
    pub weight_bytes_streamed: u64,
    /// Number of PrefetchMiss events this step.
    pub prefetch_misses: u32,
    /// Fraction of the step the GPU was idle (0.0–1.0). Mock: always 0.0.
    pub gpu_idle_fraction: f32,
    /// Wall time of this step in milliseconds. Mock: always 0.
    pub step_wall_ms: f32,
    /// Relative parity error from the last A7 check (`max|Δgrad|/max|ref_grad|`).
    /// `f64::NAN` if no check was run this step.
    pub parity_rel_error: f64,
    /// Number of micro-batches accumulated.
    pub grad_accum_steps: u32,
    /// Total tokens processed this step.
    pub tokens_processed: u64,
}
