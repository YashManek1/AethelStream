#![deny(missing_docs)]
//! **galore** — Module 4: Heterogeneous GaLore Optimizer State for AethelStream.
//!
//! Stores AdamW momentum and variance in 8-bit compressed format in System RAM.
//! Implements GaLore gradient projection/back-projection, randomized SVD subspace
//! switching, and memory-mapped `optimizer_states.bin` persistence.

pub mod adamw;
pub mod error;
pub mod kernels;
pub mod layer_rank;
pub mod optimizer;
pub mod project;
pub mod quantize;
pub mod randomized_svd;
pub mod standard_adamw;
pub mod state_file;

pub use adamw::{effective_lr, AdamWConfig, AdamWStepResult, LowRankAdamState};
pub use error::{GaLoreError, Result};
pub use layer_rank::LayerRankConfig;
pub use optimizer::{GaLoreConfig, GaLoreOptimizer};
pub use project::{
    project_backward_f32, project_forward_f32, projection_roundtrip_error, validate_projection_dims,
};
pub use quantize::{absmax_scale, dequantize_absmax, quantize_absmax, quantize_relative_error};
pub use randomized_svd::{
    randomized_svd_on_device, randomized_svd_projections, should_switch_subspace, RandomizedSvdConfig,
    SubspaceProjections,
};
pub use standard_adamw::StandardAdamW;
pub use state_file::{
    build_layer_descriptors, layer_layout, layer_state_size, LayerDescriptor, LayerLayout,
    OptimizerStateFile, OptimizerStateHeader, PrecisionMeta, HEADER_SIZE, LAYER_DESC_SIZE, MAGIC,
    VERSION,
};
