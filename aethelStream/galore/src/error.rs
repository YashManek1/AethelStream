//! Structured errors for the GaLore optimizer crate.

use thiserror::Error;

/// Result alias for galore operations.
pub type Result<T> = std::result::Result<T, GaLoreError>;

/// Errors emitted by M4 optimizer state management.
#[derive(Debug, Error)]
pub enum GaLoreError {
    /// Invalid configuration or layout parameters.
    #[error("config error: {0}")]
    Config(String),

    /// Shape mismatch between tensors.
    #[error("shape mismatch: {0}")]
    Shape(String),

    /// CUDA kernel or runtime failure.
    #[error("cuda error: {0}")]
    Cuda(String),

    /// Memory-mapped optimizer state file I/O error.
    #[error("state file error: {0}")]
    StateFile(String),

    /// Linear algebra failure (SVD/QR).
    #[error("linalg error: {0}")]
    Linalg(String),
}
