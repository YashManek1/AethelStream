// src/error.rs — RamFlow unified error type

use std::io;
use thiserror::Error;

/// Every failure mode that a RamFlow subsystem can surface.
#[derive(Debug, Error)]
pub enum RamFlowError {
    /// `cudaMallocHost` (or equivalent) returned out-of-memory.
    #[error("pinned allocation failed: {0}")]
    AllocationFailed(String),

    /// A pool has no free slots and slow-path borrowing was not possible.
    #[error("pool exhausted: {0}")]
    PoolExhausted(String),

    /// An io_uring submission or completion returned an OS error.
    #[error("io_uring error: {0}")]
    IoUringError(#[from] io::Error),

    /// CUDA driver or runtime returned a non-zero error code.
    /// The `i32` is the raw `cudaError_t` / `CUresult` value.
    #[error("CUDA error code {0}")]
    CudaError(i32),

    /// A phase-transition operation failed (wrong tensor state, in-flight
    /// references, phase fence not held before resize, etc.).
    #[error("phase transition failed: {0}")]
    PhaseTransitionError(String),

    /// The NVMe write-budget (Idea 4 / `ssd-wear` feature) was exceeded.
    /// Writing is halted to protect drive endurance.
    #[error("NVMe write budget exceeded: {0}")]
    WearBudgetExceeded(String),

    /// The co-scheduler signalled a pause due to high memory pressure.
    ///
    /// Returned by `DirectNvmeEngine::prefetch` when `pause_signal` is set.
    /// The prefetcher should treat this as a transient back-pressure signal
    /// and retry after sleeping for one layer's compute time.
    #[error("prefetch paused by memory pressure (layer {0})")]
    PressurePause(u32),

    /// Shard index or hardware profile JSON could not be parsed.
    #[error("configuration error: {0}")]
    ConfigError(String),
}

/// Crate-wide `Result` alias.  Every public API that can fail returns this.
pub type Result<T> = std::result::Result<T, RamFlowError>;
