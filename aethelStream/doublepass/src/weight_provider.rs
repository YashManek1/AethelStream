//! WeightProvider — dyn-safe trait bridging any weight source to M5 BlockWeights.
//!
//! Implementations:
//! - [`SyntheticProvider`]: formula weights for tests (no file I/O).
//! - [`crate::shard_provider::ShardEngineProvider`]: loads from M1 ShardLoader.

use crate::forward::{BlockConfig, BlockWeights};

/// Error variants for [`WeightProvider`] operations.
#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    /// Underlying I/O or mmap failure.
    #[error("I/O error: {0}")]
    IoError(String),
    /// Tensor shape does not match expected BlockConfig dimensions.
    #[error("Shape mismatch: {0}")]
    ShapeError(String),
    /// FP16 byte conversion failed.
    #[error("Conversion error: {0}")]
    ConversionError(String),
    /// Parameter key absent from the shard index.
    #[error("Parameter not found: {0}")]
    NotFound(String),
}

/// Result alias for [`ProviderError`].
pub type ProviderResult<T> = std::result::Result<T, ProviderError>;

/// Dyn-safe trait bridging any weight source to M5's [`BlockWeights`].
///
/// The trait is object-safe so it can be stored as `Box<dyn WeightProvider>` and
/// passed across the M1→M5 boundary without monomorphisation.
pub trait WeightProvider: Send + Sync {
    /// Load weights for transformer layer `layer_idx` given the block geometry.
    fn load_layer_weights(
        &mut self,
        layer_idx: u32,
        cfg: &BlockConfig,
    ) -> ProviderResult<BlockWeights>;

    /// Total number of transformer layers in this model.
    fn num_layers(&self) -> u32;

    /// Residual-stream width (`d_model`).
    fn d_model(&self) -> usize;

    /// Feed-forward hidden dimension (`d_ff`).
    fn d_ff(&self) -> usize;

    /// Number of attention heads.
    fn n_heads(&self) -> usize;
}

/// Formula-based provider that generates deterministic sine weights (no file I/O).
///
/// Used in tests and CI where no model checkpoint is available.
/// Each layer gets a distinct offset so adjacent layers have different weights.
pub struct SyntheticProvider {
    /// Total number of transformer layers.
    pub num_layers: u32,
    /// `d_model` dimension.
    pub d_model: usize,
    /// Feed-forward hidden dimension.
    pub d_ff: usize,
    /// Number of attention heads.
    pub n_heads: usize,
}

impl WeightProvider for SyntheticProvider {
    fn load_layer_weights(
        &mut self,
        layer_idx: u32,
        cfg: &BlockConfig,
    ) -> ProviderResult<BlockWeights> {
        Ok(BlockWeights::from_formula_layered(cfg, layer_idx as usize))
    }

    fn num_layers(&self) -> u32 {
        self.num_layers
    }

    fn d_model(&self) -> usize {
        self.d_model
    }

    fn d_ff(&self) -> usize {
        self.d_ff
    }

    fn n_heads(&self) -> usize {
        self.n_heads
    }
}
