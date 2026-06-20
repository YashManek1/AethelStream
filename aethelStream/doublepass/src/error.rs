//! Error types for the DoublePass engine.

/// Errors that can occur during double-pass training.
#[derive(Debug, thiserror::Error)]
pub enum DoublePassError {
    /// Transparent pass-through from FlowCast (including PrefetchMiss).
    #[error("flowcast: {0}")]
    FlowCast(#[from] flowcast::FlowCastError),

    /// No training plan has been set; [`step`] cannot execute.
    ///
    /// [`step`]: crate::DoublePass::step
    #[error("plan not set")]
    NoPlan,

    /// Parity diagnostic detected unacceptable gradient divergence and halted training.
    #[error("parity halt on layer {layer_idx}: rel={rel:.2e}")]
    ParityHalt {
        /// Layer index where parity failed.
        layer_idx: u32,
        /// Relative error (`max|Δgrad| / max|ref_grad|`).
        rel: f64,
    },

    /// Invalid configuration or plan parameters.
    #[error("invalid configuration: {0}")]
    Config(String),

    /// RNG state is missing for a given layer and micro-batch.
    #[error("rng state missing for (layer={layer_idx}, micro_batch={micro_batch})")]
    RngStateMissing {
        /// Layer index.
        layer_idx: u32,
        /// Micro-batch index.
        micro_batch: u32,
    },

    /// Checkpoint store/read failure (PinnedBuffer allocation or kernel error).
    #[error("checkpoint: {0}")]
    Checkpoint(String),
}

impl From<ramflow::RamFlowError> for DoublePassError {
    fn from(e: ramflow::RamFlowError) -> Self {
        Self::Checkpoint(e.to_string())
    }
}

/// Convenience type alias for `Result<T, DoublePassError>`.
pub type Result<T> = std::result::Result<T, DoublePassError>;
