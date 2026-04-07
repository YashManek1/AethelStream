// src/phase/mod.rs — hot/warm/cold phase management module

pub mod classifier;
pub mod profiler;
pub mod rebalancer;

// re-export the most commonly imported phase types
pub use classifier::{
    DefaultPhaseClassifier,
    Direction,
    PhaseClassifier,
    PhaseMemoryProfile,
    Tier,
    TierClassifier,
    TrainingPhase,
};
pub use profiler::{AccessProfiler, WarmupProfiler, WarmupConfig};
pub use rebalancer::PhaseRebalancer;
