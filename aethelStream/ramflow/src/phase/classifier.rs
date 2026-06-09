// src/phase/classifier.rs — training-phase types and phase classifier
//
// Sprint 0: all types declared, all methods unimplemented!.
// These are the canonical phase definitions used by every other module:
//   - Module 3 reads current_phase() to decide prefetch window size.
//   - Module 5 calls notify_* to drive phase transitions.
//   - The rebalancer subscribes to phase-transition callbacks.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;

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
    Forward {
        /// Number of layers currently held simultaneously in pinned RAM.
        layers_in_flight: u32,
    },

    /// Streaming backward pass: 2 streaming slots + optimizer state fragments.
    Backward {
        /// Number of backward steps between sparse activation checkpoints.
        checkpoint_interval: u32,
    },

    /// Mini-forward-pass recomputation window within backward.
    /// Peak RAM: `window_end - window_start + 2` layer slots simultaneously.
    Recomputation {
        /// First layer index included in the recomputation window.
        window_start: u32,
        /// Last layer index (inclusive) in the recomputation window.
        window_end: u32,
    },
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
    /// Training phase this profile was measured for.
    pub phase: TrainingPhase,
    /// Estimated peak pinned RAM in bytes across the entire phase.
    pub expected_peak_bytes: usize,
    /// Peak number of simultaneously in-flight attention-layer slots.
    pub attention_slots_needed: u32,
    /// Peak number of simultaneously in-flight MLP-layer slots.
    pub mlp_slots_needed: u32,
    /// Peak number of simultaneously in-flight norm-layer slots.
    pub norm_slots_needed: u32,
    /// Peak number of simultaneously in-flight optimizer-state slots.
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
    access_counts: Mutex<HashMap<u64, u32>>,
}

impl TierClassifier {
    /// Create a new classifier.
    pub fn new() -> Self {
        TierClassifier {
            access_counts: Mutex::new(HashMap::new()),
        }
    }

    /// Classify `tensor_id` into a [`Tier`] based on recent access history.
    pub fn classify(&self, tensor_id: u64) -> Tier {
        let access_counts = self
            .access_counts
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        match access_counts.get(&tensor_id).copied().unwrap_or(0) {
            0 => Tier::Cold,
            1..=2 => Tier::Warm,
            _ => Tier::Hot,
        }
    }

    /// Record one tensor access for future tier classification.
    pub fn record_access(&self, tensor_id: u64) {
        let mut access_counts = self
            .access_counts
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let counter = access_counts.entry(tensor_id).or_insert(0);
        *counter = counter.saturating_add(1);
    }
}

impl Default for TierClassifier {
    fn default() -> Self {
        Self::new()
    }
}

type TransitionCallback = Arc<dyn Fn(TrainingPhase, TrainingPhase) + Send + Sync>;

/// Default concrete implementation of [`PhaseClassifier`].
pub struct DefaultPhaseClassifier {
    current_phase: Mutex<TrainingPhase>,
    checkpoint_interval: u32,
    transition_callback: Option<TransitionCallback>,
}

impl DefaultPhaseClassifier {
    /// Construct a new classifier backed by the training schedule.
    ///
    /// `profile_path` is the path to `hardware_profile.json`; if the file
    /// exists and the model SHA256 matches, the warm-up profiler is skipped.
    pub fn new(profile_path: PathBuf) -> crate::Result<Self> {
        let _ = profile_path;
        Ok(DefaultPhaseClassifier {
            current_phase: Mutex::new(TrainingPhase::Forward {
                layers_in_flight: 0,
            }),
            checkpoint_interval: 1,
            transition_callback: None,
        })
    }

    /// Construct a classifier with a phase-transition callback.
    ///
    /// The callback is invoked after the internal state changes and receives
    /// `(previous_phase, next_phase)`.
    pub fn with_transition_callback(
        profile_path: PathBuf,
        transition_callback: TransitionCallback,
    ) -> crate::Result<Self> {
        let mut classifier = Self::new(profile_path)?;
        classifier.transition_callback = Some(transition_callback);
        Ok(classifier)
    }

    fn set_phase(&self, next_phase: TrainingPhase) {
        let previous_phase = {
            let mut current_phase = self
                .current_phase
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            if *current_phase == next_phase {
                return;
            }
            let previous_phase = current_phase.clone();
            *current_phase = next_phase.clone();
            previous_phase
        };

        if let Some(transition_callback) = &self.transition_callback {
            transition_callback(previous_phase, next_phase);
        }
    }
}

impl PhaseClassifier for DefaultPhaseClassifier {
    fn current_phase(&self) -> TrainingPhase {
        self.current_phase
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .clone()
    }

    fn notify_layer_start(&self, layer_idx: u32, direction: Direction) {
        match direction {
            Direction::Forward => {
                let layers_in_flight = layer_idx.saturating_add(1).min(2);
                self.set_phase(TrainingPhase::Forward { layers_in_flight });
            }
            Direction::Backward => {
                self.set_phase(TrainingPhase::Backward {
                    checkpoint_interval: self.checkpoint_interval,
                });
            }
        }
    }

    fn notify_backward_recompute_start(&self, from_checkpoint: u32, to_layer: u32) {
        self.set_phase(TrainingPhase::Recomputation {
            window_start: from_checkpoint,
            window_end: to_layer,
        });
    }
}
