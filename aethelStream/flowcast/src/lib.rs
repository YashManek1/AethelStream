#![deny(
    clippy::unwrap_used,
    clippy::panic,
    clippy::expect_used,
    missing_docs
)]

//! **flowcast** -- Prefetch Engine & I/O Pipeline for AethelStream (Module 3).
//!
//! FlowCast owns *policy*: what to prefetch, when, at which precision, and
//! whether to skip a write-back.  It delegates all *mechanism* to RamFlow
//! (Module 2): pinned allocation, io_uring ring, pressure gauge, scale table.
//!
//! # Quick-start
//! ```ignore
//! use flowcast::{FlowCast, FlowCastConfig};
//! use flowcast::backend::mock::MockBackend;
//!
//! let config = FlowCastConfig { num_shards: 32, ..Default::default() };
//! let backend = Box::new(MockBackend::new());
//! let mut fc = FlowCast::new(config, backend)?;
//! fc.advance_step(Direction::Forward, 0)?;
//! let layer = fc.wait_for_layer(1)?;
//! fc.retire_layer(layer)?;
//! fc.shutdown()?;
//! ```

pub mod backend;
pub mod completion_router;
pub mod config;
pub mod decode;
pub mod error;
pub mod hotset;
pub mod peer;
pub mod priority;
pub mod profiler;
pub mod ready;
pub mod state_machine;
pub mod telemetry;
pub mod window;
pub mod writeback;

// --------------------------------------------------------------------------
// Primary re-exports
// --------------------------------------------------------------------------

pub use config::{FlowCastConfig, HardwareProfile, LayerTiming, Precision};
pub use error::{FlowCastError, Result};
pub use ready::ReadyLayer;

// Re-export Direction from RamFlow so Module 5 only imports from flowcast.
pub use ramflow::phase::Direction;

// --------------------------------------------------------------------------
// FlowCast facade
// --------------------------------------------------------------------------

use backend::IoBackend;
use completion_router::CompletionRouter;
use hotset::HotSet;
use priority::PriorityQueue;
use profiler::Profiler;
use state_machine::PrefetchStateMachine;
use std::sync::Arc;
use telemetry::Telemetry;
use window::AdaptiveWindow;
use writeback::WritebackScheduler;

/// The FlowCast pipeline facade.
///
/// Owns all A1-A10 stage instances and coordinates them into a single
/// training-step interface for Module 5.
///
/// # Invariants
/// * `advance_step` must be called before `wait_for_layer` on each step.
/// * `retire_layer` must be called for every `ReadyLayer` returned by
///   `wait_for_layer`; failing to do so leaks a pinned-RAM slot.
#[allow(dead_code)]
pub struct FlowCast {
    config: FlowCastConfig,
    backend: Arc<dyn IoBackend>,
    /// Pinned-RAM pool registry -- must outlive all PoolSlots in the state machine.
    pool: ramflow::PoolRegistry,
    state_machine: Arc<PrefetchStateMachine>,
    window: AdaptiveWindow,
    scheduler: WritebackScheduler,
    hotset: HotSet,
    priority_queue: PriorityQueue,
    decoder: decode::QuantizedDecoder,
    router: CompletionRouter,
    telemetry: Telemetry,
    profiler: Profiler,
}

impl FlowCast {
    /// Initialise a FlowCast pipeline.
    ///
    /// Validates `config`, probes the backend, and schedules warm-up profiling
    /// for the first N steps if no `hardware_profile` is supplied.
    ///
    /// # Errors
    /// * `FlowCastError::Config` -- invalid configuration field.
    /// * `FlowCastError::BackendIo` -- backend failed to start.
    pub fn new(_config: FlowCastConfig, _backend: Box<dyn IoBackend>) -> Result<Self> {
        unimplemented!("FlowCast::new")
    }

    /// Notify FlowCast that the GPU is now executing `current_layer` in
    /// `direction`, and trigger prefetch of the next layer(s).
    ///
    /// Must be called at the start of each layer kernel launch.
    ///
    /// # Errors
    /// * `FlowCastError::InvalidTransition` -- direction changed mid-pass.
    /// * `FlowCastError::BackendIo` -- prefetch SQE submission failed.
    pub fn advance_step(&mut self, _direction: Direction, _current_layer: u32) -> Result<()> {
        unimplemented!("FlowCast::advance_step")
    }

    /// Return the `ReadyLayer` for `layer_idx` if its prefetch has completed.
    ///
    /// Returns `Err(PrefetchMiss { layer_idx })` if the buffer is not yet
    /// resident.
    ///
    /// # Errors
    /// * `FlowCastError::PrefetchMiss` -- buffer not yet resident.
    pub fn wait_for_layer(&mut self, _layer_idx: u32) -> Result<ReadyLayer> {
        unimplemented!("FlowCast::wait_for_layer")
    }

    /// Signal that M5 is done with `layer`; return the buffer to the pool.
    ///
    /// # Errors
    /// * `FlowCastError::BackendIo` -- writeback submission failed.
    pub fn retire_layer(&mut self, _layer: ReadyLayer) -> Result<()> {
        unimplemented!("FlowCast::retire_layer")
    }

    /// Drain pending completions from the backend into the ready map.
    ///
    /// Called internally by `advance_step`; exposed for testing.
    ///
    /// # Errors
    /// * `FlowCastError::BackendIo` -- poll failed.
    pub fn poll_completions(&mut self) -> Result<u32> {
        unimplemented!("FlowCast::poll_completions")
    }

    /// Telemetry snapshot for the current pipeline state.
    pub fn telemetry(&self) -> telemetry::TelemetrySnapshot {
        self.telemetry.snapshot()
    }

    /// Graceful shutdown: flush pending writes and stop background threads.
    ///
    /// # Errors
    /// * `FlowCastError::BackendIo` -- flush or shutdown failed.
    pub fn shutdown(&mut self) -> Result<()> {
        unimplemented!("FlowCast::shutdown")
    }
}
