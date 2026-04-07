// src/phase/classifier.rs — training-phase types and phase classifier
//
// Sprint 0: all types declared, all methods unimplemented!.
// These are the canonical phase definitions used by every other module:
//   - Module 3 reads current_phase() to decide prefetch window size.
//   - Module 5 calls notify_* to drive phase transitions.
//   - The rebalancer subscribes to phase-transition callbacks.

use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Training-phase taxonomy (Algorithm 2 from the spec)
// ---------------------------------------------------------------------------

/// The three structurally distinct phases of the training loop.
///
/// Each phase has a different number of layer buffers simultaneously in RAM,
/// a different tensor mix, and a different peak RAM moment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrainingPhase {
    /// Streaming forward pass: 2 layer slots in RAM (current + prefetch).
    Forward { layers_in_flight: u32 },

    /// Streaming backward pass: 2 streaming slots + optimizer state fragments.
    Backward { checkpoint_interval: u32 },

    /// Mini-forward-pass recomputation window within backward.
    /// Peak RAM: `window_end - window_start + 2` layer slots simultaneously.
    Recomputation { window_start: u32, window_end: u32 },
}

/// Pass direction — used by Module 5 when notifying phase transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Layers processed 0 → L−1.
    Forward,
    /// Layers processed L−1 → 0.
    Backward,
}

// ---------------------------------------------------------------------------
// Phase profiling records (written to hardware_profile.json)
// ---------------------------------------------------------------------------

/// Per-phase peak memory requirements, measured by the warm-up profiler.
///
/// Written to `hardware_profile.json` after the 5-step profiling pass and
/// read back on subsequent runs (skipping the warm-up if the model hasn't
/// changed).
#[derive(Debug, Clone)]
pub struct PhaseMemoryProfile {
    pub phase: TrainingPhase,
    /// Estimated peak pinned RAM in bytes across the entire phase.
    pub expected_peak_bytes: usize,
    pub attention_slots_needed: u32,
    pub mlp_slots_needed: u32,
    pub norm_slots_needed: u32,
    pub optimizer_slots_needed: u32,
}

// ---------------------------------------------------------------------------
// Phase classifier trait (Module 5 calls into this; Module 2 implements it)
// ---------------------------------------------------------------------------

/// Observer interface for the training-loop driver (Module 5) to push phase
/// state into the memory manager without creating a circular dependency.
pub trait PhaseClassifier: Send + Sync {
    /// Returns the current training phase.
    fn current_phase(&self) -> TrainingPhase;

    /// Called by Module 5 at the start of each layer's computation.
    fn notify_layer_start(&self, layer_idx: u32, direction: Direction);

    /// Called by Module 5 when a recomputation window opens.
    fn notify_backward_recompute_start(&self, from_checkpoint: u32, to_layer: u32);
}

// ---------------------------------------------------------------------------
// Hot/warm/cold tier classification (per-tensor, used by pool routing)
// ---------------------------------------------------------------------------

/// Residence tier for a single tensor, assigned by access-frequency analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Frequently accessed; must stay in VRAM.
    Hot,
    /// Occasionally accessed; reside in pinned host memory.
    Warm,
    /// Rarely accessed; spilled to NVMe.
    Cold,
}

/// Classifies tensors into tiers using a decaying frequency counter.
///
/// # Sprint 0 contract
/// Compiles; all methods `unimplemented!`.
pub struct TierClassifier {
    _opaque: (),
}

impl TierClassifier {
    /// Create a new classifier.
    pub fn new() -> Self {
        unimplemented!("TierClassifier::new — Sprint 0 stub")
    }

    /// Classify `tensor_id` into a [`Tier`] based on recent access history.
    #[allow(unused_variables)]
    pub fn classify(&self, tensor_id: u64) -> Tier {
        unimplemented!("TierClassifier::classify — Sprint 0 stub")
    }
}

/// Sprint 0 concrete implementation of [`PhaseClassifier`] — always panics.
pub struct DefaultPhaseClassifier {
    _opaque: (),
}

impl DefaultPhaseClassifier {
    /// Construct a new classifier backed by the training schedule.
    ///
    /// `profile_path` is the path to `hardware_profile.json`; if the file
    /// exists and the model SHA256 matches, the warm-up profiler is skipped.
    #[allow(unused_variables)]
    pub fn new(profile_path: PathBuf) -> crate::Result<Self> {
        unimplemented!("DefaultPhaseClassifier::new — Sprint 0 stub")
    }
}

impl PhaseClassifier for DefaultPhaseClassifier {
    fn current_phase(&self) -> TrainingPhase {
        unimplemented!("DefaultPhaseClassifier::current_phase — Sprint 0 stub")
    }

    fn notify_layer_start(&self, _layer_idx: u32, _direction: Direction) {
        unimplemented!("DefaultPhaseClassifier::notify_layer_start — Sprint 0 stub")
    }

    fn notify_backward_recompute_start(&self, _from: u32, _to: u32) {
        unimplemented!("DefaultPhaseClassifier::notify_backward_recompute_start — Sprint 0 stub")
    }
}
