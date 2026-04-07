// src/scheduler/pressure_gauge.rs — MemoryPressureGauge
//
// Sprint 0: correct field / API shape for all downstream sprint implementors.
//
// The gauge samples pool pressure every N training steps (N calibrated by
// the warm-up profiler so sampling happens ~every 10 seconds of wall time).
// It stores pressure as f32 bits in an AtomicU32 (Ordering::Relaxed) so
// background threads can read the latest estimate without a lock —
// this field is a best-effort hint, not a sequentially-consistent value.
//
// High-pressure callbacks: fired when pressure > high_threshold (default 0.80)
// Low-pressure callbacks:  fired when pressure < low_threshold  (default 0.40)
//
// The slow-path allocator also calls signal_stall() directly (without waiting
// for the next N-step sampling cycle) because a stall is an emergency.

use std::sync::RwLock;
use std::sync::atomic::AtomicU32;

/// Real-time memory pressure sensor for the RAM pool.
///
/// # Sprint 0 contract
/// Compiles; all methods `unimplemented!`.
pub struct MemoryPressureGauge {
    /// f32 pressure value stored as u32 bits (via f32::to_bits).
    /// Ordering::Relaxed — pressure is a best-effort hint.
    _pressure: AtomicU32,
    /// Callbacks fired when pressure > high_threshold.
    _high_callbacks: RwLock<Vec<Box<dyn Fn(f32) + Send + Sync>>>,
    /// Callbacks fired when pressure < low_threshold.
    _low_callbacks: RwLock<Vec<Box<dyn Fn(f32) + Send + Sync>>>,
    /// High threshold (default 0.80).
    _high_threshold: f32,
    /// Low threshold (default 0.40).
    _low_threshold: f32,
    /// Sampling interval in training steps.  Calibrated by WarmupProfiler.
    _sample_interval: u32,
}

impl MemoryPressureGauge {
    /// Create a gauge with default thresholds (0.80 high / 0.40 low).
    ///
    /// `sample_interval` — how many training steps between pressure samples.
    /// Use 30 as the default; the warm-up profiler will override this for
    /// machines where each step is very long or very short.
    #[allow(unused_variables)]
    pub fn new(sample_interval: u32) -> Self {
        unimplemented!("MemoryPressureGauge::new — Sprint 0 stub")
    }

    /// Register a callback to fire when pressure exceeds the high threshold.
    ///
    /// Module 3 registers: `prefetch_window -= 1; pause_signal.store(true)`.
    #[allow(unused_variables)]
    pub fn register_high_pressure(
        &self,
        cb: impl Fn(f32) + Send + Sync + 'static,
    ) {
        unimplemented!("MemoryPressureGauge::register_high_pressure — Sprint 0 stub")
    }

    /// Register a callback to fire when pressure drops below the low threshold.
    ///
    /// Module 3 registers: `prefetch_window += 1; pause_signal.store(false)`.
    #[allow(unused_variables)]
    pub fn register_low_pressure(
        &self,
        cb: impl Fn(f32) + Send + Sync + 'static,
    ) {
        unimplemented!("MemoryPressureGauge::register_low_pressure — Sprint 0 stub")
    }

    /// Sample pool state and fire any applicable callbacks.
    ///
    /// Called by the background sampling thread every `sample_interval` steps.
    #[allow(unused_variables)]
    pub fn sample_and_notify(&self, registry: &crate::pool::PoolRegistry) {
        unimplemented!("MemoryPressureGauge::sample_and_notify — Sprint 0 stub")
    }

    /// Emergency signal from the slow-path allocator: a pool stall has occurred.
    ///
    /// Unlike `sample_and_notify`, this fires the high-pressure callbacks
    /// immediately (not waiting for the next N-step sample) because a stall
    /// means the training pipeline is about to block.
    #[allow(unused_variables)]
    pub fn signal_stall(&self, layer_id: u32) {
        unimplemented!("MemoryPressureGauge::signal_stall — Sprint 0 stub")
    }

    /// Current pressure reading in `[0.0, 1.0]` (best-effort, may be stale).
    pub fn current_pressure(&self) -> f32 {
        unimplemented!("MemoryPressureGauge::current_pressure — Sprint 0 stub")
    }

    /// Start background sampling at the configured interval.
    ///
    /// Spawns a thread named `"ramflow-pressure-sampler"`.
    #[allow(unused_variables)]
    pub fn start(&self, registry: std::sync::Arc<crate::pool::PoolRegistry>) {
        unimplemented!("MemoryPressureGauge::start — Sprint 0 stub")
    }

    /// Current host-pinned memory pressure in `[0.0, 1.0]`.
    pub fn host_pressure(&self) -> f32 {
        unimplemented!("MemoryPressureGauge::host_pressure — Sprint 0 stub")
    }

    /// Current GPU VRAM pressure for `device` in `[0.0, 1.0]`.
    #[allow(unused_variables)]
    pub fn gpu_pressure(&self, device: u32) -> f32 {
        unimplemented!("MemoryPressureGauge::gpu_pressure — Sprint 0 stub")
    }
}

impl Default for MemoryPressureGauge {
    fn default() -> Self {
        Self::new(30)
    }
}
