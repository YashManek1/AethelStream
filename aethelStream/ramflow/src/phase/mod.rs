// src/phase/mod.rs — hot/warm/cold phase management module

/// `TrainingPhase` taxonomy, tier classifier, and `DefaultPhaseClassifier`.
pub mod classifier;
/// `WarmupProfiler`: measures peak pool usage per phase and caches results.
pub mod profiler;
/// `PhaseRebalancer`: resizes pool rings at phase boundaries.
pub mod rebalancer;

// re-export the most commonly imported phase types
pub use classifier::{
    DefaultPhaseClassifier, Direction, PhaseClassifier, PhaseMemoryProfile, Tier, TierClassifier,
    TrainingPhase,
};
pub use profiler::{AccessProfiler, ProfilePhase, WarmupConfig, WarmupProfiler};
pub use rebalancer::PhaseRebalancer;
