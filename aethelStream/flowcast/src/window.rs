//! A2: Adaptive T_iter window sizing.
//!
//! EWMA-smoothed (α ≈ 0.2) lookahead depth, clamped to `[1, W_max]`.
//! Pressure callbacks (registered on RamFlow's `MemoryPressureGauge`) cap the
//! window immediately on high pressure, growing it back on low pressure.
//! The pressure cap always wins over A2's grow heuristic.

use crate::Result;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Default EWMA alpha
// ---------------------------------------------------------------------------

/// Default EWMA smoothing factor α for T_iter window sizing.
pub const DEFAULT_ALPHA: f32 = 0.2;

// ---------------------------------------------------------------------------
// Shared pressure-cap state (written by callbacks, read by update)
// ---------------------------------------------------------------------------

/// Atomically shared window state so pressure callbacks (which may run on any
/// thread) can cap the window without acquiring a Mutex.
pub(crate) struct WindowAtomics {
    /// Current lookahead depth stored as f32 bits.
    pub(crate) t_iter_bits: AtomicU32,
    /// When true, high-pressure cap is in effect; A2 growth is suppressed.
    pub(crate) pressure_cap_active: AtomicBool,
    /// W_max stored as f32 bits (immutable after construction).
    w_max_bits: AtomicU32,
}

impl WindowAtomics {
    fn new(initial: f32, w_max: f32) -> Self {
        Self {
            t_iter_bits: AtomicU32::new(initial.to_bits()),
            pressure_cap_active: AtomicBool::new(false),
            w_max_bits: AtomicU32::new(w_max.to_bits()),
        }
    }

    fn t_iter(&self) -> f32 {
        f32::from_bits(self.t_iter_bits.load(Ordering::Acquire))
    }

    fn w_max(&self) -> f32 {
        f32::from_bits(self.w_max_bits.load(Ordering::Relaxed))
    }

    fn set_t_iter(&self, value: f32) {
        self.t_iter_bits.store(value.to_bits(), Ordering::Release);
    }
}

// ---------------------------------------------------------------------------
// AdaptiveWindow
// ---------------------------------------------------------------------------

/// Adaptive prefetch lookahead window (Algorithm A2).
///
/// # Pressure integration
/// Register this window on `MemoryPressureGauge` via `register_pressure_callbacks`
/// before starting the training loop. High-pressure callback immediately caps the
/// window; low-pressure callback re-enables A2 growth. The cap always wins.
pub struct AdaptiveWindow {
    /// Shared state accessible by pressure callbacks.
    atoms: Arc<WindowAtomics>,
    /// EWMA smoothing factor α ∈ (0, 1].
    alpha: f32,
    /// Smoothed GPU idle fraction.
    smoothed_idle: f32,
    /// Smoothed I/O latency (ms).
    smoothed_io_ms: f32,
}

impl AdaptiveWindow {
    /// Create a new window with `initial_t_iter` and EWMA `alpha`.
    ///
    /// `w_max` is the ceiling from the profiler (W_max = ⌈t_ssd/t_gpu⌉ + 2).
    pub fn new(initial_t_iter: f32, alpha: f32, w_max: f32) -> Self {
        let clamped = initial_t_iter.max(1.0).min(w_max.max(1.0));
        Self {
            atoms: Arc::new(WindowAtomics::new(clamped, w_max.max(1.0))),
            alpha,
            smoothed_idle: 0.0,
            smoothed_io_ms: 0.0,
        }
    }

    /// Register high/low pressure callbacks on `gauge`.
    ///
    /// Must be called before the first `update()`. The callbacks hold a weak
    /// `Arc` clone of the internal state, so the window can outlive the gauge.
    pub fn register_pressure_callbacks(&self, gauge: &ramflow::MemoryPressureGauge) {
        // High pressure: cap window to 1, set flag.
        let atoms_high = Arc::clone(&self.atoms);
        gauge.register_high_pressure(move |_pressure| {
            atoms_high.pressure_cap_active.store(true, Ordering::Release);
            atoms_high.set_t_iter(1.0);
        });

        // Low pressure: lift cap, allow growth.
        let atoms_low = Arc::clone(&self.atoms);
        gauge.register_low_pressure(move |_pressure| {
            atoms_low.pressure_cap_active.store(false, Ordering::Release);
        });
    }

    /// Current lookahead depth (integer layers = `t_iter().round() as u32`).
    pub fn t_iter(&self) -> f32 {
        self.atoms.t_iter()
    }

    /// Update window given observed `gpu_idle_fraction` and `io_latency_ms`.
    ///
    /// Algorithm:
    /// 1. EWMA-smooth both signals.
    /// 2. If high pressure cap is active → hold at 1 (cap always wins).
    /// 3. Else if GPU idle > 0.10 (I/O bound) → grow by 1.
    /// 4. Else if I/O latency increasing and idle < 0.05 → shrink by 1.
    /// 5. Clamp to `[1, W_max]`.
    ///
    /// # Errors
    /// Always `Ok(())`.
    pub fn update(&mut self, gpu_idle_fraction: f32, io_latency_ms: f32) -> Result<()> {
        self.smoothed_idle =
            self.alpha * gpu_idle_fraction + (1.0 - self.alpha) * self.smoothed_idle;
        let prev_io = self.smoothed_io_ms;
        self.smoothed_io_ms =
            self.alpha * io_latency_ms + (1.0 - self.alpha) * self.smoothed_io_ms;

        // Pressure cap always wins.
        if self.atoms.pressure_cap_active.load(Ordering::Acquire) {
            self.atoms.set_t_iter(1.0);
            return Ok(());
        }

        let mut next = self.atoms.t_iter();
        if self.smoothed_idle > 0.10 {
            // GPU idling waiting for I/O → grow window.
            next += 1.0;
        } else if self.smoothed_io_ms > prev_io * 1.05 && self.smoothed_idle < 0.05 {
            // I/O latency rising, GPU not stalling → shrink.
            next -= 1.0;
        }

        next = next.max(1.0).min(self.atoms.w_max());
        self.atoms.set_t_iter(next);
        Ok(())
    }

    /// Force lookahead up by 1, respecting W_max and pressure cap.
    ///
    /// # Errors
    /// Always `Ok(())`.
    pub fn increase_lookahead(&mut self) -> Result<()> {
        if self.atoms.pressure_cap_active.load(Ordering::Acquire) {
            self.atoms.set_t_iter(1.0);
            return Ok(());
        }
        let next = (self.atoms.t_iter() + 1.0).min(self.atoms.w_max());
        self.atoms.set_t_iter(next);
        Ok(())
    }

    /// Force lookahead down by 1, floor = 1.
    ///
    /// # Errors
    /// Always `Ok(())`.
    pub fn decrease_lookahead(&mut self) -> Result<()> {
        let next = (self.atoms.t_iter() - 1.0).max(1.0);
        self.atoms.set_t_iter(next);
        Ok(())
    }

    /// W_max ceiling for this window.
    pub fn w_max(&self) -> f32 {
        self.atoms.w_max()
    }

    /// Whether the high-pressure cap is currently active.
    pub fn pressure_cap_active(&self) -> bool {
        self.atoms.pressure_cap_active.load(Ordering::Acquire)
    }
}
