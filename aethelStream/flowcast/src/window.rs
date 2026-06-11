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
    /// EWMA-smoothed NVMe→RAM transfer time (milliseconds).
    smoothed_ssd_ms: f32,
    /// EWMA-smoothed GPU compute time per layer (milliseconds).
    smoothed_gpu_ms: f32,
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
            smoothed_ssd_ms: 0.0,
            smoothed_gpu_ms: 0.0,
        }
    }

    /// Register high/soft/low pressure callbacks on `gauge`.
    ///
    /// Must be called before the first `update()`. The callbacks hold an
    /// `Arc` clone of the internal state, so the window can outlive the gauge.
    ///
    /// Pass `Some(backend)` in production so high/low callbacks call
    /// `backend.set_pause(true/false)` directly.  Pass `None` in tests where
    /// no real backend is present (mock path does not need pause/resume control).
    pub fn register_pressure_callbacks(
        &self,
        gauge: &ramflow::MemoryPressureGauge,
        backend: Option<std::sync::Arc<dyn crate::backend::IoBackend>>,
    ) {
        // High pressure: cap window to 1, set flag, pause backend if present.
        let atoms_high = Arc::clone(&self.atoms);
        let backend_high = backend.clone();
        gauge.register_high_pressure(move |_pressure| {
            atoms_high.pressure_cap_active.store(true, Ordering::Release);
            atoms_high.set_t_iter(1.0);
            if let Some(ref be) = backend_high {
                be.set_pause(true);
            }
        });

        // Soft pressure: pre-emptive window shrink before the hard stop.
        // Does not pause the backend; just tightens the window by half.
        let atoms_soft = Arc::clone(&self.atoms);
        gauge.register_soft_pressure(move |_pressure| {
            let current = atoms_soft.t_iter();
            atoms_soft.set_t_iter((current * 0.5).max(1.0));
        });

        // Low pressure: lift cap, resume backend if present.
        let atoms_low = Arc::clone(&self.atoms);
        let backend_low = backend;
        gauge.register_low_pressure(move |_pressure| {
            atoms_low.pressure_cap_active.store(false, Ordering::Release);
            if let Some(ref be) = backend_low {
                be.set_pause(false);
            }
        });
    }

    /// Current lookahead depth (integer layers = `t_iter().round() as u32`).
    pub fn t_iter(&self) -> f32 {
        self.atoms.t_iter()
    }

    /// Update window given observed `ssd_transfer_ms` and `gpu_compute_ms`.
    ///
    /// Algorithm (A2-b/c/d spec):
    /// 1. EWMA-smooth both signals.
    /// 2. Compute `next` using grow/shrink predicates based on t_ssd/t_gpu ratio.
    ///    * Grow if `smoothed_ssd_ms > 0.8 × smoothed_gpu_ms` (SSD is the
    ///      bottleneck: GPU will stall without a larger prefetch window).
    ///    * Shrink if `smoothed_gpu_ms > 2.0 × smoothed_ssd_ms` (SSD is much
    ///      faster than GPU: window is oversized and wasting RAM).
    /// 3. Clamp `next` to `[1, W_max]`.
    /// 4. Apply pressure cap override **after** clamp (A2-d fix: EWMA always
    ///    runs; cap overrides the computed value rather than short-circuiting).
    ///
    /// # Errors
    /// Always `Ok(())`.
    pub fn update(&mut self, ssd_transfer_ms: f32, gpu_compute_ms: f32) -> Result<()> {
        // Step 1: EWMA smooth (always runs, even under pressure — A2-d).
        self.smoothed_ssd_ms =
            self.alpha * ssd_transfer_ms + (1.0 - self.alpha) * self.smoothed_ssd_ms;
        self.smoothed_gpu_ms =
            self.alpha * gpu_compute_ms + (1.0 - self.alpha) * self.smoothed_gpu_ms;

        // Step 2: grow/shrink decision (A2-b/c spec predicates).
        let mut next = self.atoms.t_iter();
        if self.smoothed_ssd_ms > 0.8 * self.smoothed_gpu_ms {
            // SSD is the bottleneck: widen the prefetch window.
            next += 1.0;
        } else if self.smoothed_gpu_ms > 2.0 * self.smoothed_ssd_ms {
            // SSD vastly faster than GPU: window is oversized, reclaim RAM.
            next -= 1.0;
        }

        // Step 3: clamp to [1, W_max].
        next = next.max(1.0).min(self.atoms.w_max());

        // Step 4: pressure cap overrides computed value (A2-d fix — cap checked
        // AFTER the clamp so EWMA is always updated regardless of pressure).
        if self.atoms.pressure_cap_active.load(Ordering::Acquire) {
            next = 1.0;
        }

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
