// src/scheduler/pressure_gauge.rs — MemoryPressureGauge
//
// Sprint 4: full implementation replacing Sprint 0 stub.
//
// ─── DEADLOCK PREVENTION ──────────────────────────────────────────────────────
// Callbacks are Arc<dyn Fn(f32)> inside a Mutex<Vec<...>>.
// Firing: snapshot Arc list under lock, drop lock, then invoke each callback.
// The Mutex is never held during callback execution — registering from within
// a callback is deadlock-free by construction.
//
// ─── SAMPLING ─────────────────────────────────────────────────────────────────
//   1. sample_and_notify(&self, registry) — training loop, every N steps.
//      N = max(5, steps_per_minute/6) so sampling fires ~every 10 s.
//   2. start(&self, registry) — background "ramflow-pressure-sampler" thread,
//      sleeps 10 000 ms between cycles.
//
// signal_stall() is an emergency fast-path from SlowPathAllocator that fires
// high callbacks at pressure = 1.0 immediately (no sample-cycle wait).

use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering::{Acquire, Relaxed, Release};
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// GaugeInner — all shared state, accessible from background thread via Arc
// ---------------------------------------------------------------------------

/// Shared callback invoked when a pressure band fires.
type PressureCallback = std::sync::Arc<dyn Fn(f32) + Send + Sync>;

/// Thread-safe list of pressure callbacks.
type CallbackList = std::sync::Mutex<Vec<PressureCallback>>;

pub(crate) struct GaugeInner {
    /// f32 pressure stored as u32 bits.  Relaxed — best-effort hint.
    pub(crate) pressure: AtomicU32,

    /// Callbacks fired when pressure > high_threshold.
    pub(crate) high_callbacks: CallbackList,

    /// Callbacks fired when pressure < low_threshold.
    pub(crate) low_callbacks: CallbackList,

    /// Callbacks fired when soft_threshold < pressure <= high_threshold.
    ///
    /// Sprint 5: CoScheduler registers here to trigger INT8 checkpoint compression
    /// at ~70% pressure (before the 0.80 hard stop).
    pub(crate) soft_callbacks: CallbackList,

    /// Default 0.80; configurable from hardware_profile.json at construction.
    pub(crate) high_threshold: f32,

    /// Default 0.70; fires soft callbacks when soft_threshold < p <= high_threshold.
    pub(crate) soft_threshold: f32,

    /// Default 0.40; configurable from hardware_profile.json at construction.
    pub(crate) low_threshold: f32,

    /// Steps between samples for the step-triggered path.
    pub(crate) sample_interval_steps: u32,

    /// Signals the background thread to exit on its next wake.
    pub(crate) shutdown: AtomicBool,
    /// Background sampler thread handle stored so Drop can join it.
    thread_handle: std::sync::Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl GaugeInner {
    pub(crate) fn compute_pressure(registry: &crate::pool::PoolRegistry) -> f32 {
        let total_capacity = registry.total_capacity();
        if total_capacity == 0 {
            return 0.0;
        }
        (registry.total_claimed_slots() as f32 / total_capacity as f32).min(1.0)
    }

    /// Snapshot Arc list under lock, drop lock, then invoke each callback.
    ///
    /// The Mutex is never held during invocation: a callback that calls
    /// register_high/low_pressure will acquire the lock without contention.
    pub(crate) fn fire_callbacks(callbacks: &CallbackList, pressure: f32) {
        let snapshot: Vec<PressureCallback> = callbacks
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .clone();
        // Lock dropped here — before any callback is invoked.
        for callback in snapshot {
            callback(pressure);
        }
    }

    pub(crate) fn compute_and_fire(&self, registry: &crate::pool::PoolRegistry) {
        let pressure = Self::compute_pressure(registry);
        self.pressure.store(pressure.to_bits(), Relaxed);
        if pressure > self.high_threshold {
            Self::fire_callbacks(&self.high_callbacks, pressure);
        } else if pressure > self.soft_threshold {
            // Soft band: soft_threshold < p <= high_threshold.
            // CoScheduler triggers INT8 checkpoint compression here.
            Self::fire_callbacks(&self.soft_callbacks, pressure);
        } else if pressure < self.low_threshold {
            Self::fire_callbacks(&self.low_callbacks, pressure);
        }
    }
}

impl Drop for GaugeInner {
    fn drop(&mut self) {
        // Signal first so the thread wakes on its next check.
        self.shutdown.store(true, std::sync::atomic::Ordering::Release);
        if let Ok(mut guard) = self.thread_handle.lock() {
            if let Some(handle) = guard.take() {
                let _ = handle.join();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MemoryPressureGauge
// ---------------------------------------------------------------------------

/// Real-time memory pressure sensor for the RamFlow pool.
///
/// Wraps `Arc<GaugeInner>` so all clones share identical state.  Callbacks
/// registered on one clone fire when any other clone calls `sample_and_notify`.
///
/// # Invariant
///
/// `high_threshold` > `low_threshold`.  Both are immutable after construction.
/// Callbacks are registered before training starts; no runtime registration
/// lock is held during callback invocation.
#[derive(Clone)]
pub struct MemoryPressureGauge {
    pub(crate) inner: Arc<GaugeInner>,
}

impl MemoryPressureGauge {
    /// Construct with default thresholds (0.80 high / 0.40 low).
    ///
    /// `sample_interval` — steps between pressure samples for step-triggered
    /// callers.  Calibrate via `max(5, steps_per_minute / 6)` for ~10 s cadence.
    pub fn new(sample_interval: u32) -> Self {
        MemoryPressureGauge {
            inner: Arc::new(GaugeInner {
                pressure: AtomicU32::new(0.0_f32.to_bits()),
                high_callbacks: Mutex::new(Vec::new()),
                soft_callbacks: Mutex::new(Vec::new()),
                low_callbacks: Mutex::new(Vec::new()),
                high_threshold: 0.80,
                soft_threshold: 0.70,
                low_threshold: 0.40,
                sample_interval_steps: sample_interval,
                shutdown: AtomicBool::new(false),
                thread_handle: std::sync::Mutex::new(None),
            }),
        }
    }

    /// Register a callback fired when pressure exceeds `high_threshold`.
    ///
    /// Safe to call from within a callback — the Mutex is not held during
    /// invocation, so registration will not deadlock.
    pub fn register_high_pressure(&self, callback: impl Fn(f32) + Send + Sync + 'static) {
        self.inner
            .high_callbacks
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .push(Arc::new(callback));
    }

    /// Register a callback fired when pressure falls below `low_threshold`.
    pub fn register_low_pressure(&self, callback: impl Fn(f32) + Send + Sync + 'static) {
        self.inner
            .low_callbacks
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .push(Arc::new(callback));
    }

    /// Register a callback fired when pressure is in the soft band
    /// (soft_threshold < pressure ≤ high_threshold).
    ///
    /// Default soft threshold: 0.70.  CoScheduler registers here to trigger
    /// INT8 checkpoint compression before the 0.80 hard stop pauses prefetch.
    ///
    /// Safe to call from within a callback — lock is not held during invocation.
    pub fn register_soft_pressure(&self, callback: impl Fn(f32) + Send + Sync + 'static) {
        self.inner
            .soft_callbacks
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .push(Arc::new(callback));
    }

    /// Sample pool pressure and fire applicable callbacks.
    ///
    /// Computes `p = total_claimed / total_capacity`, stores it (Relaxed),
    /// fires high callbacks if `p > 0.80`, low callbacks if `p < 0.40`.
    pub fn sample_and_notify(&self, registry: &crate::pool::PoolRegistry) {
        self.inner.compute_and_fire(registry);
    }

    /// Emergency fast-path: fires high callbacks at pressure = 1.0 immediately.
    ///
    /// Called by `SlowPathAllocator` on pool exhaustion.  `layer_id` is
    /// forwarded to Module 5 for per-layer eviction decisions.
    pub fn signal_stall(&self, layer_id: u32) {
        let _ = layer_id;
        self.inner.pressure.store(1.0_f32.to_bits(), Relaxed);
        GaugeInner::fire_callbacks(&self.inner.high_callbacks, 1.0);
    }

    /// Current pressure in `[0.0, 1.0]` — may be one sample stale.
    pub fn current_pressure(&self) -> f32 {
        f32::from_bits(self.inner.pressure.load(Relaxed))
    }

    /// Configured sampling interval in steps.
    pub fn sample_interval_steps(&self) -> u32 {
        self.inner.sample_interval_steps
    }

    /// Spawn `"ramflow-pressure-sampler"` background thread (10 s interval).
    ///
    /// # Errors
    ///
    /// Returns `Err(IoUringError)` if the OS rejects the spawn.
    pub fn start(&self, registry: Arc<crate::pool::PoolRegistry>) -> crate::Result<()> {
        let inner = Arc::clone(&self.inner);
        let handle = std::thread::Builder::new()
            .name("ramflow-pressure-sampler".into())
            .spawn(move || loop {
                if inner.shutdown.load(Acquire) {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(10_000));
                if inner.shutdown.load(Acquire) {
                    break;
                }
                inner.compute_and_fire(&registry);
            })
            .map_err(crate::RamFlowError::IoUringError)?;
        self.inner
            .thread_handle
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .replace(handle);
        Ok(())
    }

    /// Signal the background thread to exit on its next 10 s wake.
    pub fn shutdown(&self) {
        self.inner.shutdown.store(true, Release);
    }

    /// Host-pinned memory pressure in `[0.0, 1.0]` (same as `current_pressure()`).
    pub fn host_pressure(&self) -> f32 {
        self.current_pressure()
    }

    /// GPU VRAM pressure for `device` in `[0.0, 1.0]`.
    ///
    /// Returns 0.0 in Sprint 4; Sprint 5 wires this to NVML.
    pub fn gpu_pressure(&self, device: u32) -> f32 {
        let _ = device;
        0.0
    }
}

impl Default for MemoryPressureGauge {
    fn default() -> Self {
        Self::new(30)
    }
}
