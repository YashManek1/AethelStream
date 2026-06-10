//! FlowCast error types.
//!
//! `FlowCastError` wraps every upstream `RamFlowError` via `#[from]` and adds
//! pipeline-specific failures.  The most critical variant is `PrefetchMiss`:
//! M5 (the compute loop) must receive this and handle the stall explicitly
//! rather than silently blocking.

use thiserror::Error;

/// Every failure mode that a FlowCast pipeline stage can surface.
#[derive(Debug, Error)]
pub enum FlowCastError {
    /// Upstream RamFlow mechanism error — transparently forwarded.
    #[error("ramflow: {0}")]
    RamFlow(#[from] ramflow::RamFlowError),

    /// Layer was requested by M5 but the prefetch has not yet completed.
    ///
    /// Callers must not silently stall; they must act on this error
    /// (retry after one GPU kernel boundary or adjust lookahead).
    #[error("prefetch miss on layer {layer_idx}")]
    PrefetchMiss {
        /// The layer index whose buffer was not ready.
        layer_idx: u32,
    },

    /// The state machine received a transition that is illegal given current state.
    #[error("invalid state transition: {0}")]
    InvalidTransition(String),

    /// An I/O backend operation failed.
    #[error("backend I/O error: {0}")]
    BackendIo(String),

    /// Hardware-profile JSON could not be read or written.
    #[error("profile I/O error: {0}")]
    ProfileIo(String),

    /// FlowCastConfig contained an out-of-range or contradictory field.
    #[error("configuration error: {0}")]
    Config(String),
}

/// Result alias for all FlowCast public APIs.
pub type Result<T> = std::result::Result<T, FlowCastError>;
