//! A1/A2 layer ordering and prefetch sequence to FlowCast.
//!
//! Owns computing the order in which layers are processed during forward and
//! backward passes, emitting the prefetch sequence that M5 sends to FlowCast.

/// The ascending layer order for one segment's recompute-forward pass.
#[derive(Debug, Clone)]
pub struct SegmentRecomputeOrder {
    /// Index of this checkpoint segment (0 = first segment of the model).
    pub segment_index: u32,
    /// Layers in ascending order for this segment's recompute pass.
    pub layers_ascending: Vec<u32>,
}

/// Emits per-direction prefetch sequences that M5 sends to FlowCast.
///
/// A segment is a contiguous run of `checkpoint_freq` layers. Forward processes
/// segments 0, 1, 2, … ascending. Backward processes them in reverse (last first),
/// each with an ascending recompute then a descending backward.
pub struct LayerSchedule {
    /// Total number of transformer layers.
    num_layers: u32,
    /// Checkpoint frequency k: one checkpoint every k layers.
    checkpoint_freq: u32,
}

impl LayerSchedule {
    /// Create a layer schedule for `num_layers` transformer layers checkpointed
    /// every `checkpoint_freq` layers.
    pub fn new(num_layers: u32, checkpoint_freq: u32) -> Self {
        Self { num_layers, checkpoint_freq }
    }

    /// Ascending layer indices for the forward pass: `[0, 1, …, num_layers-1]`.
    pub fn forward_order(&self) -> Vec<u32> {
        (0..self.num_layers).collect()
    }

    /// Number of checkpoint segments: `ceil(num_layers / checkpoint_freq)`.
    pub fn num_segments(&self) -> u32 {
        self.num_layers.div_ceil(self.checkpoint_freq)
    }

    /// Segments in **reverse** order for the backward pass.
    ///
    /// Each segment's `layers_ascending` is the ascending layer sequence for the
    /// recompute-forward within that segment. The outer `Vec` is descending by
    /// segment index (last segment first), matching the backward data-flow direction.
    pub fn backward_segments(&self) -> Vec<SegmentRecomputeOrder> {
        let k = self.checkpoint_freq;
        let ns = self.num_segments();
        let mut segs = Vec::with_capacity(ns as usize);
        for s in (0..ns).rev() {
            let start = s * k;
            let end = ((s + 1) * k).min(self.num_layers);
            segs.push(SegmentRecomputeOrder {
                segment_index: s,
                layers_ascending: (start..end).collect(),
            });
        }
        segs
    }
}
