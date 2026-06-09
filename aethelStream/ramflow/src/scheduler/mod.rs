// src/scheduler/mod.rs — co-scheduler and memory-pressure gauge module

/// `CoScheduler` orchestrates tensor evictions and prefetch control.
/// Also owns `PerLayerScaleTable` (Algorithm 6).
pub mod coscheduler;

/// `MemoryPressureGauge`: real-time pool-fill sensor with callback bands.
pub mod pressure_gauge;

pub use coscheduler::{CoScheduler, PerLayerScaleTable};
pub use pressure_gauge::MemoryPressureGauge;
