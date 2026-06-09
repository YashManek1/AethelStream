// src/phase/rebalancer.rs — phase-aware synchronous pool resizing

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use crate::phase::classifier::PhaseMemoryProfile;
use crate::pool::PoolRegistry;
use crate::{RamFlowError, Result};

/// Moves tensors between tiers and resizes pool rings at phase boundaries.
///
/// The rebalancer runs synchronously in the calling thread. It first acquires
/// a phase fence by waiting for all pool slots and old-slot CUDA copies to
/// drain, then resizes each ring to the target phase profile.
pub struct PhaseRebalancer {
    registry: Option<Arc<PoolRegistry>>,
    outstanding_cuda_copies: Arc<AtomicUsize>,
    fence_timeout: Duration,
}

impl PhaseRebalancer {
    /// Create an unattached rebalancer.
    ///
    /// Use [`Self::with_registry`] for production resizing. An unattached
    /// rebalancer keeps the Sprint 0 construction contract and makes tests for
    /// fence accounting possible without allocating pools.
    pub fn new() -> Self {
        PhaseRebalancer {
            registry: None,
            outstanding_cuda_copies: Arc::new(AtomicUsize::new(0)),
            fence_timeout: Duration::from_secs(30),
        }
    }

    /// Create a rebalancer wired to the given pool registry.
    pub fn with_registry(registry: Arc<PoolRegistry>) -> Self {
        PhaseRebalancer {
            registry: Some(registry),
            outstanding_cuda_copies: Arc::new(AtomicUsize::new(0)),
            fence_timeout: debug_fence_timeout_override().unwrap_or(Duration::from_secs(30)),
        }
    }

    /// Mark one old-slot CUDA copy as in-flight.
    ///
    /// Callers that launch `cudaMemcpyAsync` from a pool slot should pair this
    /// with [`Self::mark_cuda_copy_complete`] when the stream/event completes.
    pub fn mark_cuda_copy_started(&self) {
        self.outstanding_cuda_copies.fetch_add(1, Ordering::AcqRel);
    }

    /// Mark one old-slot CUDA copy as completed.
    pub fn mark_cuda_copy_complete(&self) {
        let mut observed = self.outstanding_cuda_copies.load(Ordering::Acquire);
        while observed > 0 {
            match self.outstanding_cuda_copies.compare_exchange(
                observed,
                observed - 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(next_observed) => observed = next_observed,
            }
        }
    }

    /// Number of outstanding CUDA copies tracked by the phase fence.
    pub fn outstanding_cuda_copies(&self) -> usize {
        self.outstanding_cuda_copies.load(Ordering::Acquire)
    }

    /// Resize all pools to match `profile`.
    ///
    /// # Errors
    /// Returns [`RamFlowError::PhaseTransitionError`] if no registry is wired,
    /// if the phase fence times out, or if any ring reports in-flight slots
    /// during resize. Allocation failures are propagated from the ring buffers.
    pub fn rebalance_to_profile(&self, profile: &PhaseMemoryProfile) -> Result<()> {
        let registry = self.registry.as_ref().ok_or_else(|| {
            RamFlowError::PhaseTransitionError(
                "PhaseRebalancer is not attached to a PoolRegistry".into(),
            )
        })?;
        self.acquire_phase_fence(registry)?;
        registry.resize_to_profile(profile)
    }

    /// Hot/warm/cold tier rebalance pass.
    ///
    /// Sprint 3B only resizes pool rings; tensor tier movement lands when the
    /// access classifier is wired into Module 3.
    pub fn rebalance(&self) -> Result<()> {
        Ok(())
    }

    fn acquire_phase_fence(&self, registry: &PoolRegistry) -> Result<()> {
        let deadline = Instant::now() + self.fence_timeout;
        loop {
            if registry.total_claimed_slots() == 0 && self.outstanding_cuda_copies() == 0 {
                return Ok(());
            }

            if Instant::now() >= deadline {
                return Err(RamFlowError::PhaseTransitionError(format!(
                    "phase fence timed out with {} pool slot(s) and {} CUDA copy/copies in-flight",
                    registry.total_claimed_slots(),
                    self.outstanding_cuda_copies()
                )));
            }

            thread::sleep(Duration::from_millis(1));
        }
    }
}

impl Default for PhaseRebalancer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(debug_assertions)]
fn debug_fence_timeout_override() -> Option<Duration> {
    std::env::var("RAMFLOW_PHASE_FENCE_TIMEOUT_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .map(Duration::from_millis)
}

#[cfg(not(debug_assertions))]
fn debug_fence_timeout_override() -> Option<Duration> {
    None
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::phase::classifier::TrainingPhase;
    use crate::pool::{LayerKind, TensorLocationDict};

    fn profile(attention_slots_needed: u32) -> PhaseMemoryProfile {
        PhaseMemoryProfile {
            phase: TrainingPhase::Recomputation {
                window_start: 0,
                window_end: attention_slots_needed.saturating_sub(1),
            },
            expected_peak_bytes: 0,
            attention_slots_needed,
            mlp_slots_needed: 1,
            norm_slots_needed: 1,
            optimizer_slots_needed: 1,
        }
    }

    #[test]
    fn forward_to_recomputation_resize_waits_for_in_flight_slot() {
        let forward_profile = PhaseMemoryProfile {
            phase: TrainingPhase::Forward {
                layers_in_flight: 1,
            },
            expected_peak_bytes: 0,
            attention_slots_needed: 1,
            mlp_slots_needed: 1,
            norm_slots_needed: 1,
            optimizer_slots_needed: 1,
        };
        let registry = Arc::new(
            PoolRegistry::new(&forward_profile, &TensorLocationDict::empty(), 1024)
                .expect("registry"),
        );
        let rebalancer = Arc::new(PhaseRebalancer::with_registry(Arc::clone(&registry)));
        let old_slot = registry
            .claim(LayerKind::Attention)
            .expect("claim old slot");

        let thread_rebalancer = Arc::clone(&rebalancer);
        let recomputation_profile = profile(3);
        let handle = thread::spawn(move || {
            thread_rebalancer
                .rebalance_to_profile(&recomputation_profile)
                .expect("rebalance")
        });

        thread::sleep(Duration::from_millis(10));
        assert_eq!(
            registry.capacity_for(LayerKind::Attention),
            1,
            "resize must wait while an old slot is in-flight"
        );

        drop(old_slot);
        handle.join().expect("rebalance thread panicked");

        assert_eq!(registry.claimed_slots_for(LayerKind::Attention), 0);
        assert_eq!(registry.capacity_for(LayerKind::Attention), 3);
        let mut slots = Vec::new();
        for _slot_index in 0..3 {
            slots.push(
                registry
                    .claim(LayerKind::Attention)
                    .expect("claim resized attention slot"),
            );
        }
        assert_eq!(registry.claimed_slots_for(LayerKind::Attention), 3);
    }
}
