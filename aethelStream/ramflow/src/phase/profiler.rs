// src/phase/profiler.rs — warm-up profiler and access-pattern profiler
//
// Sprint 0: all types declared, all methods unimplemented!.
//
// Two distinct profiling roles live here:
//   1. WarmupProfiler  — runs a 5-step mini training loop to measure per-phase
//      peak slot counts, writing hardware_profile.json.
//   2. AccessProfiler  — per-tensor frequency recorder used by TierClassifier.

use std::path::PathBuf;
use crate::phase::classifier::PhaseMemoryProfile;

// ---------------------------------------------------------------------------
// Warm-up profiler (Algorithm 2 — phase-aware pool sizing)
// ---------------------------------------------------------------------------

/// Runs instrumented training steps to measure peak pool usage per phase.
///
/// The measurements are written to `hardware_profile.json` and consumed by
/// the [`crate::phase::rebalancer::PhaseRebalancer`] on subsequent runs.
///
/// # Sprint 0 contract
/// Compiles; all methods `unimplemented!`.
pub struct WarmupProfiler {
    _opaque: (),
}

/// Configuration for the warm-up profiling pass.
#[derive(Debug, Clone)]
pub struct WarmupConfig {
    /// Number of mini-training steps to run (default: 5).
    pub steps: u32,
    /// Path to write the resulting hardware profile JSON.
    pub output_path: PathBuf,
    /// SHA-256 of `shard_index.json`; used to skip profiling on cache hit.
    pub model_sha256: [u8; 32],
}

impl WarmupProfiler {
    /// Create a new profiler.  Does NOT start profiling yet — call [`run`].
    #[allow(unused_variables)]
    pub fn new(config: WarmupConfig) -> crate::Result<Self> {
        unimplemented!("WarmupProfiler::new — Sprint 0 stub")
    }

    /// Check whether a valid `hardware_profile.json` exists for this model.
    ///
    /// If it does, [`run`] is a no-op and returns the cached profiles.
    pub fn is_cache_valid(&self) -> bool {
        unimplemented!("WarmupProfiler::is_cache_valid — Sprint 0 stub")
    }

    /// Execute the warm-up profiling pass.
    ///
    /// Returns three [`PhaseMemoryProfile`] records:
    /// `[forward_profile, backward_profile, recomputation_profile]`.
    pub fn run(&self) -> crate::Result<[PhaseMemoryProfile; 3]> {
        unimplemented!("WarmupProfiler::run — Sprint 0 stub")
    }

    /// Zero-copy threshold measurement: issue UVA reads and DMA copies for
    /// tensors of increasing sizes.  Returns the byte threshold at which DMA
    /// copy becomes faster than zero-copy on this machine.
    ///
    /// Result is stored in `hardware_profile.json` alongside the phase
    /// profiles and read back by the zero-copy router at init time.
    pub fn measure_zero_copy_crossover(&self) -> crate::Result<usize> {
        unimplemented!("WarmupProfiler::measure_zero_copy_crossover — Sprint 0 stub")
    }

    /// Measure optimal sampling interval (steps) for the pressure gauge so
    /// that pressure is re-sampled approximately every 10 seconds of wall time.
    pub fn measure_pressure_sample_interval(&self) -> crate::Result<u32> {
        unimplemented!("WarmupProfiler::measure_pressure_sample_interval — Sprint 0 stub")
    }
}

// ---------------------------------------------------------------------------
// Access-pattern profiler (per-tensor frequency tracking)
// ---------------------------------------------------------------------------

/// Records per-tensor access statistics used by the phase classifier.
///
/// # Sprint 0 contract
/// Compiles; all methods `unimplemented!`.
pub struct AccessProfiler {
    _opaque: (),
}

impl AccessProfiler {
    /// Create a new profiler.
    pub fn new() -> Self {
        unimplemented!("AccessProfiler::new — Sprint 0 stub")
    }

    /// Record an access to `tensor_id` at the current monotonic timestamp.
    #[allow(unused_variables)]
    pub fn record_access(&self, tensor_id: u64) {
        unimplemented!("AccessProfiler::record_access — Sprint 0 stub")
    }

    /// Return the access frequency (accesses/sec) for `tensor_id`.
    #[allow(unused_variables)]
    pub fn frequency(&self, tensor_id: u64) -> f64 {
        unimplemented!("AccessProfiler::frequency — Sprint 0 stub")
    }
}

impl Default for AccessProfiler {
    fn default() -> Self {
        Self::new()
    }
}
