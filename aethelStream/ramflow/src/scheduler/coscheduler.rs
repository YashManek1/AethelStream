// src/scheduler/coscheduler.rs — CoScheduler + PerLayerScaleTable
//
// Sprint 4: full implementation replacing Sprint 0 stub.
//
// ─── CoScheduler ─────────────────────────────────────────────────────────────
// Registers two callbacks on MemoryPressureGauge at construction:
//   High (>0.80): prefetch_window -= 1; pause_signal.store(true, Release)
//   Low  (<0.40): prefetch_window += 1; pause_signal.store(false, Release)
//
// pause_signal is also readable by DirectNvmeEngine (PressurePause error path).
//
// ─── PerLayerScaleTable (Algorithm 6) ────────────────────────────────────────
// EWA overflow density per layer (alpha = 0.05, ~20-step window).
// Thresholds are configurable struct fields (not compile-time constants) so
// hardware_profile.json can override them:
//   density > overflow_high_threshold (default 0.001): halve scale, floor 1.0
//   density < overflow_low_threshold  (default 0.0001) AND scale < 65536:
//                                                        double scale, cap 65536
//
// BF16 short-circuit: if bf16_mode is true all scales are fixed at 1.0
// (Ampere+ hardware has native overflow immunity for BF16).

use std::collections::BTreeMap;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicI32;
use std::sync::atomic::Ordering::{AcqRel, Acquire, Release};
use std::sync::{Arc, Mutex};

use crate::scheduler::pressure_gauge::MemoryPressureGauge;

// ---------------------------------------------------------------------------
// PerLayerScaleTable — Algorithm 6
// ---------------------------------------------------------------------------

/// Per-layer EWA overflow density tracker and loss scaler.
///
/// For each transformer layer tracks:
/// - `density[i]`  EWA of overflow fraction (overflowed / total elements)
/// - `scale[i]`    current per-layer loss scale (initial: 65536.0)
/// - `gradient_variance[i]`  running mean of gradient magnitude (Idea 1 signal
///   for INT4 ↔ FP16 precision switching read by Module 3)
/// - `resident[i]`  hot-set caching flag (Module 3 reads for prefetch priority)
///
/// Alpha defaults to 0.05 (≈ 20-step smoothing window).
/// Thresholds are configurable struct fields read from hardware_profile.json.
pub struct PerLayerScaleTable {
    /// EWA of overflow fraction per layer.
    density: Vec<f32>,

    /// Current per-layer loss scale (init 65536.0, floor 1.0, cap 65536.0).
    scale: Vec<f32>,

    /// Windowed mean of gradient magnitude per layer (Idea 1 signal for Module 3).
    gradient_variance: Vec<GradientVarianceWindow>,

    /// Hot-set caching flags — true means Module 3 should keep this layer resident.
    resident: Vec<bool>,

    /// EWA decay constant (default 0.05 ≈ 20-step window).
    alpha: f32,

    /// Overflow fraction above which scale is halved.  Default 0.001 (0.1%).
    overflow_high_threshold: f32,

    /// Overflow fraction below which scale is doubled.  Default 0.0001 (0.01%).
    overflow_low_threshold: f32,

    /// When true (Ampere+ with BF16 mode active), skip overflow scaling entirely
    /// and fix all scales at 1.0 — BF16 has native NaN/Inf immunity.
    bf16_mode: bool,
}

#[derive(Clone)]
struct GradientVarianceWindow {
    samples: Vec<f32>,
    next: usize,
    len: usize,
    sum: f32,
}

impl GradientVarianceWindow {
    fn new(capacity: usize) -> Self {
        Self {
            samples: vec![0.0; capacity.max(1)],
            next: 0,
            len: 0,
            sum: 0.0,
        }
    }

    fn push(&mut self, value: f32) {
        if self.len < self.samples.len() {
            self.samples[self.next] = value;
            self.sum += value;
            self.len += 1;
        } else {
            let old_value = self.samples[self.next];
            self.samples[self.next] = value;
            self.sum += value - old_value;
        }
        self.next = (self.next + 1) % self.samples.len();
    }

    fn mean(&self) -> f32 {
        if self.len == 0 {
            return 0.0;
        }
        self.sum / self.len as f32
    }
}

impl PerLayerScaleTable {
    /// Create a table for a model with `num_layers` layers.
    ///
    /// All scales initialised to `65536.0`.  All densities and variances to `0.0`.
    /// Uses default thresholds (0.001 high / 0.0001 low).
    pub fn new(num_layers: usize, alpha: f32) -> Self {
        Self::with_thresholds(num_layers, alpha, 0.001, 0.0001)
    }

    /// Create with explicit overflow thresholds read from hardware_profile.json.
    ///
    /// `overflow_high_threshold` — fraction above which scale is halved.
    /// `overflow_low_threshold`  — fraction below which scale is doubled.
    pub fn with_thresholds(
        num_layers: usize,
        alpha: f32,
        overflow_high_threshold: f32,
        overflow_low_threshold: f32,
    ) -> Self {
        let gradient_window_len = configured_gradient_window_len();
        PerLayerScaleTable {
            density: vec![0.0; num_layers],
            scale: vec![65536.0; num_layers],
            gradient_variance: vec![GradientVarianceWindow::new(gradient_window_len); num_layers],
            resident: vec![false; num_layers],
            alpha,
            overflow_high_threshold,
            overflow_low_threshold,
            bf16_mode: false,
        }
    }

    /// Enable BF16 short-circuit: fixes all scales at 1.0, skips overflow math.
    ///
    /// Call this when the target GPU is Ampere+ and BF16 mode is active.
    pub fn enable_bf16_mode(&mut self) {
        self.bf16_mode = true;
        for scale_entry in &mut self.scale {
            *scale_entry = 1.0;
        }
    }

    /// Update EWA density and adjust scale for `layer_idx`.
    ///
    /// Only `layer_idx`'s density and scale change; all other layers are untouched.
    ///
    /// `n_total` — total FP16 elements in the gradient tensor.
    /// `n_overflow` — NaN/Inf elements counted by `count_overflow_fp16` kernel.
    ///
    /// # Errors
    ///
    /// Returns [`crate::RamFlowError::ConfigError`] if `layer_idx` is out of bounds for
    /// this table (i.e., >= the number of layers the table was constructed with).
    pub fn update(
        &mut self,
        layer_idx: usize,
        n_total: usize,
        n_overflow: u32,
    ) -> Result<(), crate::RamFlowError> {
        if layer_idx >= self.density.len() {
            return Err(crate::RamFlowError::ConfigError(format!(
                "PerLayerScaleTable::update: layer_idx {layer_idx} out of bounds (table has {} layers)",
                self.density.len()
            )));
        }
        if n_total == 0 {
            return Ok(());
        }
        if self.bf16_mode {
            return Ok(());
        }

        let fraction = n_overflow as f32 / n_total as f32;
        let new_density = self.alpha * fraction + (1.0 - self.alpha) * self.density[layer_idx];
        self.density[layer_idx] = new_density;

        if new_density > self.overflow_high_threshold {
            self.scale[layer_idx] = (self.scale[layer_idx] * 0.5).max(1.0);
        } else if new_density < self.overflow_low_threshold && self.scale[layer_idx] < 65536.0 {
            self.scale[layer_idx] = (self.scale[layer_idx] * 2.0).min(65536.0);
        }
        Ok(())
    }

    /// Current loss scale for `layer_idx`.
    ///
    /// Returns 1.0 for all layers when `bf16_mode` is active.
    ///
    /// # Errors
    ///
    /// Returns [`crate::RamFlowError::ConfigError`] if `layer_idx` is out of bounds for
    /// this table (i.e., >= the number of layers the table was constructed with).
    pub fn get_scale(&self, layer_idx: usize) -> Result<f32, crate::RamFlowError> {
        if self.bf16_mode {
            return Ok(1.0);
        }
        self.scale.get(layer_idx).copied().ok_or_else(|| {
            crate::RamFlowError::ConfigError(format!(
                "PerLayerScaleTable::get_scale: layer_idx {layer_idx} out of bounds (table has {} layers)",
                self.scale.len()
            ))
        })
    }

    /// Current EWA overflow density for `layer_idx`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::RamFlowError::ConfigError`] if `layer_idx` is out of bounds for
    /// this table (i.e., >= the number of layers the table was constructed with).
    pub fn get_density(&self, layer_idx: usize) -> Result<f32, crate::RamFlowError> {
        self.density.get(layer_idx).copied().ok_or_else(|| {
            crate::RamFlowError::ConfigError(format!(
                "PerLayerScaleTable::get_density: layer_idx {layer_idx} out of bounds (table has {} layers)",
                self.density.len()
            ))
        })
    }

    /// Update the gradient variance estimate for `layer_idx` (Idea 1 signal).
    ///
    /// Module 3 reads this to decide INT4 vs FP16 streaming precision per layer.
    pub fn update_gradient_variance(&mut self, layer_idx: usize, grad_mean_sq: f32) {
        if layer_idx < self.gradient_variance.len() {
            self.gradient_variance[layer_idx].push(grad_mean_sq);
        }
    }

    /// Gradient variance for `layer_idx` (for Module 3's precision scheduler).
    pub fn gradient_variance(&self, layer_idx: usize) -> f32 {
        self.gradient_variance
            .get(layer_idx)
            .map(GradientVarianceWindow::mean)
            .unwrap_or(0.0)
    }

    /// Mark or unmark `layer_idx` as hot-set resident.
    ///
    /// Module 3 reads this flag to decide prefetch priority: resident layers
    /// are kept in RAM between forward and backward passes.
    pub fn mark_resident(&mut self, layer_idx: usize, resident: bool) {
        if layer_idx < self.resident.len() {
            self.resident[layer_idx] = resident;
        }
    }

    /// Whether `layer_idx` is currently marked hot-set resident.
    pub fn is_resident(&self, layer_idx: usize) -> bool {
        self.resident.get(layer_idx).copied().unwrap_or(false)
    }

    /// Reset all scales to `65536.0`.
    ///
    /// Called by the parity guard every 500 steps to prevent drift accumulation.
    pub fn reset_all_scales(&mut self) {
        if self.bf16_mode {
            return;
        }
        for scale_entry in &mut self.scale {
            *scale_entry = 65536.0;
        }
    }
}

fn configured_gradient_window_len() -> usize {
    std::env::var("RAMFLOW_GRADIENT_VARIANCE_WINDOW")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(50)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn gradient_variance_uses_windowed_mean_instead_of_raw_last_spike() {
        let mut table = PerLayerScaleTable::new(2, 0.05);

        for _ in 0..49 {
            table.update_gradient_variance(1, 1.0);
        }
        table.update_gradient_variance(1, 101.0);

        let smoothed = table.gradient_variance(1);
        assert!(
            smoothed < 101.0,
            "windowed gradient variance should smooth the raw last spike"
        );
        assert!(
            (smoothed - 3.0).abs() < 0.001,
            "expected mean of 49 baseline samples and one spike, got {smoothed}"
        );
        assert_eq!(table.gradient_variance(0), 0.0);
    }
}

// ---------------------------------------------------------------------------
// CoScheduler
// ---------------------------------------------------------------------------

/// Orchestrates tensor evictions and prefetch control based on
/// `MemoryPressureGauge` signals.
///
/// At construction, registers two callbacks on the shared `GaugeInner`:
/// - **High** (> 0.80): decrement `prefetch_window`; set `pause_signal = true`.
/// - **Low**  (< 0.40): increment `prefetch_window`; set `pause_signal = false`.
///
/// `pause_signal` is read by `DirectNvmeEngine::prefetch` to emit
/// `Err(PressurePause)` when pressure is high.
pub struct CoScheduler {
    /// Gauge whose callbacks were registered at construction.
    gauge: MemoryPressureGauge,

    /// Current prefetch window size.  Shared with the registered callbacks via Arc.
    /// Initial value: 2.  Decremented on high pressure; incremented on low pressure.
    prefetch_window: Arc<AtomicI32>,

    /// Set true by the high-pressure callback; cleared by the low-pressure callback.
    /// Read by DirectNvmeEngine to emit PressurePause.
    pause_signal: Arc<AtomicBool>,

    /// Set true by the soft-pressure callback (0.70–0.80); cleared by low-pressure.
    ///
    /// Module 5 reads this via should_compress_checkpoints() to decide whether
    /// to compress activation checkpoints to INT8 before the next layer streams in.
    compress_trigger: Arc<AtomicBool>,

    /// Registered tensors: tensor_id → eviction priority (lower = evict first).
    tensor_registry: Mutex<BTreeMap<u64, u8>>,
}

impl CoScheduler {
    /// Construct a co-scheduler and register pressure callbacks on `gauge`.
    ///
    /// Both `prefetch_window` and `pause_signal` are `Arc`-shared with the
    /// closures so the callbacks mutate the same atomics that `is_paused()` and
    /// `prefetch_window()` read.
    ///
    /// # Errors
    ///
    /// Currently infallible.  Returns `Result` so callers can handle future
    /// resource-allocation failures (e.g., NVMe engine registration) without
    /// an API break.
    pub fn new(gauge: MemoryPressureGauge) -> crate::Result<Self> {
        let prefetch_window = Arc::new(AtomicI32::new(2));
        let pause_signal = Arc::new(AtomicBool::new(false));
        let compress_trigger = Arc::new(AtomicBool::new(false));

        // High-pressure callback (> 0.80) — shrink window, pause prefetch.
        // AcqRel on fetch_sub: makes prior writes in this thread visible to
        // the next thread that observes the decremented window.
        let window_high = Arc::clone(&prefetch_window);
        let pause_high = Arc::clone(&pause_signal);
        gauge.register_high_pressure(move |_pressure| {
            window_high.fetch_sub(1, AcqRel);
            pause_high.store(true, Release);
        });

        // Soft-pressure callback (0.70 < p <= 0.80) — trigger INT8 checkpoint
        // compression before the hard stop activates.  Module 5 reads
        // should_compress_checkpoints() on each backward step.
        let compress_soft = Arc::clone(&compress_trigger);
        gauge.register_soft_pressure(move |_pressure| {
            compress_soft.store(true, Release);
        });

        // Low-pressure callback (< 0.40) — resume prefetch and clear compression.
        let window_low = Arc::clone(&prefetch_window);
        let pause_low = Arc::clone(&pause_signal);
        let compress_low = Arc::clone(&compress_trigger);
        gauge.register_low_pressure(move |_pressure| {
            window_low.fetch_add(1, AcqRel);
            pause_low.store(false, Release);
            // Clear compression trigger once pressure fully recovers.
            compress_low.store(false, Release);
        });

        Ok(CoScheduler {
            gauge,
            prefetch_window,
            pause_signal,
            compress_trigger,
            tensor_registry: Mutex::new(BTreeMap::new()),
        })
    }

    /// Run one scheduling tick — evict cold tensors, allow prefetch to resume.
    ///
    /// In Sprint 4 the tick is driven entirely by pressure callbacks; this method
    /// is reserved for future proactive eviction based on age metadata.
    pub fn tick(&self) -> crate::Result<()> {
        Ok(())
    }

    /// Register a tensor for eviction eligibility.
    ///
    /// `priority` — lower is more evictable (0 = evict first, 255 = keep last).
    pub fn register_tensor(&self, tensor_id: u64, priority: u8) {
        self.tensor_registry
            .lock()
            // SAFETY: this mutex only protects priority metadata. If a prior
            // holder panicked, recovering the map is preferable to silently
            // dropping future eviction registrations; no unsafe memory state is
            // represented by this BTreeMap.
            .unwrap_or_else(|poison| poison.into_inner())
            .insert(tensor_id, priority);
    }

    /// Deregister a tensor (freed or permanently evicted).
    pub fn deregister_tensor(&self, tensor_id: u64) {
        self.tensor_registry
            .lock()
            // SAFETY: same rationale as register_tensor(): poison only means a
            // previous metadata update panicked, and continuing with the guarded
            // BTreeMap preserves scheduler correctness better than leaking stale
            // tensor priorities.
            .unwrap_or_else(|poison| poison.into_inner())
            .remove(&tensor_id);
    }

    /// Whether the NVMe prefetch engine is currently paused due to high pressure.
    pub fn is_paused(&self) -> bool {
        self.pause_signal.load(Acquire)
    }

    /// Current prefetch window size.
    ///
    /// Can go negative (fully stopped) when multiple high-pressure events fire
    /// before a low-pressure event restores it.  `DirectNvmeEngine` treats any
    /// value ≤ 0 as "do not prefetch" (redundantly with `pause_signal`).
    pub fn prefetch_window(&self) -> i32 {
        self.prefetch_window.load(Acquire)
    }

    /// Reference to the underlying pressure gauge.
    ///
    /// Module 3 uses this to call `sample_and_notify` from the training loop.
    pub fn gauge(&self) -> &MemoryPressureGauge {
        &self.gauge
    }

    /// Whether the compression subsystem should currently compress gradient checkpoints.
    ///
    /// Returns `true` when a soft-pressure event has fired and the pressure is between the
    /// soft threshold (0.70) and the high threshold (0.80). Module 5 calls this before
    /// each backward step to decide whether to compress activation checkpoints.
    ///
    /// Uses `Ordering::Acquire` to observe the `Release` store from the pressure callback.
    pub fn should_compress_checkpoints(&self) -> bool {
        self.compress_trigger.load(std::sync::atomic::Ordering::Acquire)
    }
}
