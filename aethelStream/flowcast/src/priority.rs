//! A8: Importance-aware priority queue and adaptive-precision selection.
//!
//! Reads `PerLayerScaleTable::gradient_variance` (windowed mean) to assign
//! priority and precision per layer:
//!
//! - High-variance layers → higher `importance` score → earlier ready slot,
//!   precision promoted to FP16 (or kept at configured default).
//! - Low-variance layers → lower score → precision demoted to INT4 sooner.
//!
//! Every layer is eventually emitted (no starvation): a layer's score is
//! bounded below by `1.0` so it can never be permanently blocked.

use crate::config::Precision;
use crate::Result;
use ramflow::PerLayerScaleTable;
use std::collections::BinaryHeap;

/// A queued prefetch request with importance score and selected precision.
#[derive(Debug, Clone)]
pub struct PrefetchRequest {
    /// Layer index.
    pub layer_idx: u32,
    /// Importance score (higher = higher priority, earlier ready slot).
    pub importance: f32,
    /// Precision selected for this layer based on gradient variance.
    pub precision: Precision,
}

impl PartialEq for PrefetchRequest {
    fn eq(&self, other: &Self) -> bool {
        self.layer_idx == other.layer_idx
    }
}
impl Eq for PrefetchRequest {}

impl PartialOrd for PrefetchRequest {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for PrefetchRequest {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.importance
            .partial_cmp(&other.importance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
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

/// Importance score for `layer_idx` given its gradient variance.
///
/// score = max(1.0, variance × 1000) so:
/// - high-variance layers get large scores → pop first from the max-heap
/// - low-variance layers get score = 1.0 → eventually served (no starvation)
pub fn importance_for_variance(variance: f32) -> f32 {
    (variance * 1000.0).max(1.0)
}

// ---------------------------------------------------------------------------
// PriorityQueue
// ---------------------------------------------------------------------------

/// Importance-aware prefetch queue.
pub struct PriorityQueue {
    inner: BinaryHeap<PrefetchRequest>,
}

impl PriorityQueue {
    /// Create an empty priority queue.
    pub fn new() -> Self {
        Self { inner: BinaryHeap::new() }
    }

    /// Enqueue a request.
    ///
    /// # Errors
    /// Always `Ok(())`.
    pub fn push(&mut self, request: PrefetchRequest) -> Result<()> {
        self.inner.push(request);
        Ok(())
    }

    /// Pop the highest-importance request.
    pub fn pop(&mut self) -> Option<PrefetchRequest> {
        self.inner.pop()
    }

    /// Peek at the highest-importance request without removing it.
    pub fn peek(&self) -> Option<&PrefetchRequest> {
        self.inner.peek()
    }

    /// Rebuild the queue from a `PerLayerScaleTable` snapshot.
    ///
    /// For each layer index in `layer_indices`, reads gradient variance,
    /// computes importance and precision, and enqueues a `PrefetchRequest`.
    /// Existing entries are replaced.
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
            self.inner.push(PrefetchRequest { layer_idx, importance, precision });
        }
        Ok(())
    }

    /// Update importance scores from `(layer_idx, importance_score)` pairs.
    ///
    /// Replaces all existing entries that match a given layer_idx with the
    /// new importance. Layers not mentioned are kept unchanged.
    ///
    /// # Errors
    /// Always `Ok(())`.
    pub fn update_importances(&mut self, scores: &[(u32, f32)]) -> Result<()> {
        let updates: std::collections::HashMap<u32, f32> = scores.iter().copied().collect();
        let mut entries: Vec<PrefetchRequest> = std::mem::take(&mut self.inner).into_vec();
        for entry in &mut entries {
            if let Some(&new_importance) = updates.get(&entry.layer_idx) {
                entry.importance = new_importance;
            }
        }
        self.inner = entries.into_iter().collect();
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
