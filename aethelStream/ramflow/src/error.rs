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

    /// A requested tensor name was not present in the layer index or slab.
    #[error("tensor not found: layer {layer_idx}, tensor {name}")]
    TensorNotFound {
        /// Layer index whose tensor table was queried.
        layer_idx: u32,
        /// Tensor name that was requested.
        name: String,
    },

    /// A completed NVMe read produced bytes whose xxHash3 digest did not match
    /// the value stored in `shard_index.json`.  The shard file is corrupted or
    /// the hardware silently flipped bits mid-transfer.
    #[error("shard {shard_id} corrupted: expected xxh3={expected:#018x}, got {got:#018x}")]
    ShardCorrupted {
        /// Shard file index (matches the numeric suffix of `shard_NNNN.bin`).
        shard_id: u32,
        /// xxHash3-64 digest from `shard_index.json`.
        expected: u64,
        /// xxHash3-64 digest computed from received bytes.
        got: u64,
    },
}

/// Crate-wide `Result` alias.  Every public API that can fail returns this.
pub type Result<T> = std::result::Result<T, RamFlowError>;
