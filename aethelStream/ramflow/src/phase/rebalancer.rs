// src/phase/rebalancer.rs — phase rebalancer stub
//
// The rebalancer is a synchronous operation that runs BETWEEN training phases
// (never mid-phase while the GPU has in-flight DMA transfers).
//
// Sequence:
//   1. Acquire "phase fence" — wait for all outstanding PoolSlot drops.
//   2. Resize each ring buffer to match the new PhaseMemoryProfile.
//   3. Signal completion; training resumes.

use crate::phase::classifier::PhaseMemoryProfile;

/// Moves tensors between tiers in response to memory-pressure events.
/// Also handles pool re-sizing at training-phase boundaries.
///
/// # Sprint 0 contract
/// Compiles; all methods `unimplemented!`.
pub struct PhaseRebalancer {
    _opaque: (),
}

impl PhaseRebalancer {
    /// Create a rebalancer wired to the given pool registry.
    pub fn new() -> Self {
        unimplemented!("PhaseRebalancer::new — Sprint 0 stub")
    }

    /// Resize all pools to match `profile`.
    ///
    /// MUST be called between phases — never during an active DMA transfer.
    /// Acquires the phase fence internally before resizing.
    #[allow(unused_variables)]
    pub fn rebalance_to_profile(&self, profile: &PhaseMemoryProfile) -> crate::Result<()> {
        unimplemented!("PhaseRebalancer::rebalance_to_profile — Sprint 0 stub")
    }

    /// Hot/warm/cold tier rebalance pass — moves individual tensors.
    pub fn rebalance(&self) -> crate::Result<()> {
        unimplemented!("PhaseRebalancer::rebalance — Sprint 0 stub")
    }
}

impl Default for PhaseRebalancer {
    fn default() -> Self {
        Self::new()
    }
}
