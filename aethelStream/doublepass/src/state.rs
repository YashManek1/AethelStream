//! Consistent step-boundary snapshot for M10 (checkpoint / resume).

/// Consistent step-boundary snapshot handed to M10 for checkpoint / resume.
///
/// Captured immediately after [`crate::DoublePass::step`] applies all updates and
/// before any next-step prefetch begins.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConsistentState {
    /// The step index this snapshot was taken at.
    pub step: u64,
    /// M4 optimizer version counter (to detect stale checkpoints).
    pub optimizer_version: u64,
    /// Per-(layer, micro_batch) RNG seeds captured at forward pass start.
    pub rng_states: Vec<RngState>,
    /// Data loader position (token index in the dataset).
    pub data_position: u64,
}

/// Per-(layer, micro-batch) RNG state captured at forward for deterministic recompute.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RngState {
    /// Layer index.
    pub layer_idx: u32,
    /// Micro-batch index within the grad-accum window.
    pub micro_batch: u32,
    /// Serialized RNG seed (64-bit for mock; real CUDA uses curandState — stored as bytes).
    pub seed_bytes: Vec<u8>,
}
