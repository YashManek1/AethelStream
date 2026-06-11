//! A8: Importance-aware priority queue and adaptive-precision selection.
//!
//! Reads `PerLayerScaleTable::gradient_variance` (windowed mean) to assign
//! priority and precision per layer:
//!
//! - High-variance layers → higher `importance` score → earlier ready slot,
//!   precision promoted to FP16 (or kept at configured default).
//! - Low-variance layers → lower score → precision demoted to INT4 sooner.
//!
//! Starvation prevention (A8-d fix): every request is stamped with an
//! `enqueue_step` counter.  `pop()` computes an *effective* importance of
//! `importance + age × AGING_WEIGHT`.  Once `age ≥ MAX_AGE_STEPS` the
//! effective importance becomes `f32::MAX`, force-promoting the request to the
//! front of the queue.

use crate::config::Precision;
use crate::Result;
use ramflow::PerLayerScaleTable;

/// Per-step importance bonus applied to aged-out requests.
pub const AGING_WEIGHT: f32 = 1.0;
/// Steps after which a request is force-promoted (effective importance = f32::MAX).
pub const MAX_AGE_STEPS: u64 = 50;

/// A queued prefetch request with importance score and selected precision.
#[derive(Debug, Clone)]
pub struct PrefetchRequest {
    /// Layer index.
    pub layer_idx: u32,
    /// Importance score (higher = higher priority, earlier pop).
    pub importance: f32,
    /// Precision selected for this layer based on gradient variance.
    pub precision: Precision,
    /// Push step counter at the time this request was enqueued.
    /// Used by `pop()` to compute age-adjusted effective importance (A8-d).
    pub enqueue_step: u64,
}

// ---------------------------------------------------------------------------
// Variance thresholds for precision selection
// ---------------------------------------------------------------------------

/// Gradient variance above which a layer is promoted to FP16.
pub const HIGH_VARIANCE_THRESHOLD: f32 = 0.1;
/// Gradient variance below which a layer is demoted to INT4.
pub const LOW_VARIANCE_THRESHOLD: f32 = 0.01;

/// Select per-layer precision based on gradient variance.
///
/// - variance > `HIGH_VARIANCE_THRESHOLD` → FP16 (high sensitivity)
/// - variance < `LOW_VARIANCE_THRESHOLD`  → INT4 (low sensitivity, compress)
/// - otherwise                            → `default_precision`
pub fn precision_for_variance(variance: f32, default_precision: Precision) -> Precision {
    if variance > HIGH_VARIANCE_THRESHOLD {
        Precision::FP16
    } else if variance < LOW_VARIANCE_THRESHOLD {
        Precision::INT4
    } else {
        default_precision
    }
}

/// Importance score for a layer given its gradient variance.
///
/// `score = max(1.0, variance × 1000)` so:
/// - high-variance layers get large scores → pop first
/// - low-variance layers get score = 1.0 → eventually served via aging
pub fn importance_for_variance(variance: f32) -> f32 {
    (variance * 1000.0).max(1.0)
}

/// Compute effective importance at `current_step` for a request enqueued at
/// `request.enqueue_step`.
///
/// `effective = importance + age × AGING_WEIGHT`, capped at `f32::MAX` when
/// `age ≥ MAX_AGE_STEPS` to force-pop long-waiting requests.
fn effective_importance(request: &PrefetchRequest, current_step: u64) -> f32 {
    let age = current_step.saturating_sub(request.enqueue_step);
    if age >= MAX_AGE_STEPS {
        return f32::MAX;
    }
    request.importance + age as f32 * AGING_WEIGHT
}

// ---------------------------------------------------------------------------
// PriorityQueue
// ---------------------------------------------------------------------------

/// Importance-aware prefetch queue with starvation prevention (A8-d).
///
/// Internally a `Vec`; `pop()` is O(n) but correctly applies the age-adjusted
/// effective-importance comparison, which a `BinaryHeap` cannot do (heap
/// ordering is fixed at insertion time).
pub struct PriorityQueue {
    inner: Vec<PrefetchRequest>,
    /// Global step counter, incremented on each `push`.  Used to stamp
    /// `enqueue_step` and to compute age in `pop`.
    current_step: u64,
}

impl PriorityQueue {
    /// Create an empty priority queue.
    pub fn new() -> Self {
        Self { inner: Vec::new(), current_step: 0 }
    }

    /// Enqueue a request, stamping it with the current step counter.
    ///
    /// # Errors
    /// Always `Ok(())`.
    pub fn push(&mut self, mut request: PrefetchRequest) -> Result<()> {
        request.enqueue_step = self.current_step;
        self.current_step += 1;
        self.inner.push(request);
        Ok(())
    }

    /// Pop the highest effective-importance request.
    ///
    /// Effective importance = `importance + age × AGING_WEIGHT`, or `f32::MAX`
    /// when `age ≥ MAX_AGE_STEPS`.  When multiple requests share the same
    /// effective importance (e.g. several are force-promoted simultaneously),
    /// the one with the smallest `enqueue_step` (longest wait) is preferred,
    /// preserving FIFO ordering among equally-important requests.
    pub fn pop(&mut self) -> Option<PrefetchRequest> {
        if self.inner.is_empty() {
            return None;
        }
        let step = self.current_step;
        let best_pos = self
            .inner
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let ea = effective_importance(a, step);
                let eb = effective_importance(b, step);
                match ea.partial_cmp(&eb).unwrap_or(std::cmp::Ordering::Equal) {
                    std::cmp::Ordering::Equal => {
                        // Tiebreak: oldest enqueue_step (smallest value) wins.
                        b.enqueue_step.cmp(&a.enqueue_step)
                    }
                    ord => ord,
                }
            })
            .map(|(pos, _)| pos)
            .unwrap_or(0);
        Some(self.inner.swap_remove(best_pos))
    }

    /// Peek at the highest effective-importance request without removing it.
    pub fn peek(&self) -> Option<&PrefetchRequest> {
        if self.inner.is_empty() {
            return None;
        }
        let step = self.current_step;
        self.inner.iter().max_by(|a, b| {
            let ea = effective_importance(a, step);
            let eb = effective_importance(b, step);
            match ea.partial_cmp(&eb).unwrap_or(std::cmp::Ordering::Equal) {
                std::cmp::Ordering::Equal => b.enqueue_step.cmp(&a.enqueue_step),
                ord => ord,
            }
        })
    }

    /// Rebuild the queue from a `PerLayerScaleTable` snapshot.
    ///
    /// For each layer index in `layer_indices`, reads gradient variance,
    /// computes importance and precision, and enqueues a `PrefetchRequest`.
    /// Existing entries are replaced.  All rebuilt entries are stamped with
    /// the current step (fresh enqueue time).
    ///
    /// # Errors
    /// Always `Ok(())`.
    pub fn rebuild_from_scale_table(
        &mut self,
        layer_indices: impl Iterator<Item = u32>,
        scale_table: &PerLayerScaleTable,
        default_precision: Precision,
    ) -> Result<()> {
        self.inner.clear();
        for layer_idx in layer_indices {
            let variance = scale_table.gradient_variance(layer_idx as usize);
            let importance = importance_for_variance(variance);
            let precision = precision_for_variance(variance, default_precision);
            self.inner.push(PrefetchRequest {
                layer_idx,
                importance,
                precision,
                enqueue_step: self.current_step,
            });
            self.current_step += 1;
        }
        Ok(())
    }

    /// Update importance scores from `(layer_idx, importance_score)` pairs.
    ///
    /// Layers not mentioned are kept unchanged.  Re-stamps updated entries
    /// with the current step so aging resets for promoted layers.
    ///
    /// # Errors
    /// Always `Ok(())`.
    pub fn update_importances(&mut self, scores: &[(u32, f32)]) -> Result<()> {
        let updates: std::collections::HashMap<u32, f32> = scores.iter().copied().collect();
        for entry in &mut self.inner {
            if let Some(&new_importance) = updates.get(&entry.layer_idx) {
                entry.importance = new_importance;
                entry.enqueue_step = self.current_step;
                self.current_step += 1;
            }
        }
        Ok(())
    }

    /// Number of pending requests.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl Default for PriorityQueue {
    fn default() -> Self {
        Self::new()
    }
}
