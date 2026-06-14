//! Prefetch submission schedulers for FlowCast.
//!
//! Exposes two schedulers:
//! - EDF (Earliest-Deadline-First) ordering (A5-EDF) that replaces the scalar
//!   window ordering in [`crate::state_machine::PrefetchStateMachine`].
//! - [`DuplexBudget`] token-bucket that splits NVMe bandwidth between the
//!   prefetch-read path and the write-back path, preventing write bursts from
//!   starving EDF-scheduled reads.
pub mod edf;
pub mod token_bucket;
pub use edf::EdfScheduler;
pub use token_bucket::{BandwidthExhausted, DuplexBudget, DEFAULT_READ_FRACTION};
