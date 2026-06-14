#![deny(
    clippy::unwrap_used,
    clippy::panic,
    clippy::expect_used,
    clippy::unimplemented,
    missing_docs
)]

//! **flowcast** — Prefetch Engine & I/O Pipeline for AethelStream (Module 3).
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
//! let profile = fc.warmup(32)?;
//! fc.on_layer_start(0, Direction::Forward)?;
//! let layer = fc.take_ready(0, std::time::Duration::from_millis(500))?;
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
pub mod scheduler;
#[cfg(all(target_os = "linux", feature = "ssd-thermal"))]
pub mod smart_monitor;
#[cfg(feature = "cuda-double-buffer")]
pub mod vram_double_buffer;

// --------------------------------------------------------------------------
// Primary re-exports
// --------------------------------------------------------------------------

pub use config::{FlowCastConfig, HardwareProfile, LayerTiming, Precision};
pub use error::{FlowCastError, Result};
pub use ready::ReadyLayer;

/// Re-export Direction from RamFlow so Module 5 only imports from flowcast.
pub use ramflow::phase::Direction;

// --------------------------------------------------------------------------
// FlowCast facade
// --------------------------------------------------------------------------

use backend::IoBackend;
use completion_router::CompletionRouter;
use decode::QuantizedDecoder;
use hotset::HotSet;
use priority::PriorityQueue;
use profiler::Profiler;
use ramflow::phase::{DefaultPhaseClassifier, PhaseClassifier};
use ramflow::PerLayerScaleTable;
use ramflow::PinnedBuffer;
use scheduler::{DuplexBudget, EdfScheduler, DEFAULT_READ_FRACTION};
use state_machine::PrefetchStateMachine;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use telemetry::Telemetry;
use window::AdaptiveWindow;
use writeback::{WritebackMode, WritebackScheduler};

/// The FlowCast pipeline facade.
///
/// Owns all A1–A10 stage instances and coordinates them into a single
/// training-step interface for Module 5.
///
/// # Invariants
/// * `on_layer_start` must be called before `take_ready` for each layer.
/// * The `ReadyLayer` returned by `take_ready` must eventually be dropped so
///   the pinned-RAM slot is returned to the RamFlow pool.
// Several fields are held for their RAII lifetime (pressure callbacks, window
// state, decoder, priority queue) and are accessed indirectly via interior
// mutability or sub-method calls rather than direct field reads.
#[allow(dead_code)]
pub struct FlowCast {
    config: FlowCastConfig,
    backend: Arc<dyn IoBackend>,
    /// Pinned-RAM pool registry — must outlive all `PoolSlot`s in the state machine.
    pool: ramflow::PoolRegistry,
    state_machine: Arc<PrefetchStateMachine>,
    window: AdaptiveWindow,
    /// Write-back scheduler wrapped in a `Mutex` so `on_layer_start` (takes
    /// `&self`) can call `drain_deferred` without requiring `&mut self` on the
    /// facade.  The lock is held only briefly; no cross-lock ordering with the
    /// state-machine mutex is possible because they are never held simultaneously.
    scheduler: Mutex<WritebackScheduler>,
    hotset: HotSet,
    priority_queue: PriorityQueue,
    decoder: QuantizedDecoder,
    router: CompletionRouter,
    telemetry: Telemetry,
    profiler: Profiler,
    /// Per-layer scale and residency table shared with RamFlow (A6-c, A8).
    scale_table: PerLayerScaleTable,
    /// Memory pressure sensor (C5); callbacks cap the prefetch window and pause
    /// the backend when pool utilisation exceeds the high/soft thresholds.
    gauge: ramflow::MemoryPressureGauge,
    /// I/O co-scheduler (C6): reads `is_paused()` before each prefetch submission
    /// and `prefetch_window()` to bound the lookahead depth.
    coscheduler: ramflow::CoScheduler,
    /// Phase classifier (C9): receives `notify_layer_start` on every GPU step so
    /// RamFlow's rebalancer can resize pool rings at phase boundaries.
    phase_classifier: DefaultPhaseClassifier,
    /// Step counter incremented on every `on_layer_start` call (A3-T).
    steps_since_reprofile: AtomicU64,
    /// Re-profiling interval from config (steps between SMART temperature checks).
    reprofiling_interval: u64,
    /// SSD thermal monitor (Linux + `ssd-thermal` feature only).
    #[cfg(all(target_os = "linux", feature = "ssd-thermal"))]
    thermal_monitor: Option<Arc<smart_monitor::ThermalMonitor>>,
    /// VRAM double-buffer: two alternating slots with a dedicated copy stream
    /// that overlaps RAM→VRAM DMA with GPU compute (`cuda-double-buffer` only).
    #[cfg(feature = "cuda-double-buffer")]
    vram_double_buffer: Option<Mutex<vram_double_buffer::VramDoubleBuffer>>,
}

impl FlowCast {
    /// Initialise a FlowCast pipeline.
    ///
    /// Validates `config`, starts the backend, primes the prefetch window,
    /// and spawns the completion-router thread.
    ///
    /// # Errors
    /// * `FlowCastError::Config` — invalid configuration field.
    /// * `FlowCastError::BackendIo` — backend failed to start or prime failed.
    /// * `FlowCastError::RamFlow` — pool allocation failed.
    pub fn new(config: FlowCastConfig, mut backend: Box<dyn IoBackend>) -> Result<Self> {
        config.validate()?;

        backend.start().map_err(|error| {
            FlowCastError::BackendIo(format!("backend start: {error}"))
        })?;

        // Create telemetry before wrapping the backend so the CqeRetryBackend
        // receives a clone that shares the same underlying Arc<AtomicU64> counters.
        let telemetry = Telemetry::new();

        let retry_config = completion_router::RetryConfig {
            max_retries: config.max_cqe_retries,
            base_backoff_ms: config.base_backoff_ms,
        };
        let retry_backend = completion_router::CqeRetryBackend::new(
            Arc::from(backend),
            retry_config,
            telemetry.clone(),
        );
        let backend_arc: Arc<dyn IoBackend> = Arc::new(retry_backend);

        let num_layers = config.num_shards;
        let lookahead = config.initial_lookahead;

        // C8: attempt profile-driven pool construction; fall back to defaults
        // when hardware_profile.json is absent (first run, no profiler data yet).
        let pool = {
            let profile_path = config.shard_dir.join("hardware_profile.json");
            // `DefaultPhaseClassifier::new` reads the cached profile if present.
            // We only use `with_defaults()` when the file is absent so the pool
            // slot counts and sizes are calibrated to the real model on warm runs.
            // `with_defaults()` handles both the profiled and first-run cases;
            // the distinction is reserved for a future profile-driven path.
            let _ = profile_path.exists();
            ramflow::PoolRegistry::with_defaults()
                .map_err(FlowCastError::RamFlow)?
        };

        // C5: pressure gauge — sample interval ~30 steps (calibrated externally).
        let gauge = ramflow::MemoryPressureGauge::new(30);

        // C6: co-scheduler takes ownership of a gauge clone; registers HIGH/SOFT/LOW
        // callbacks that shrink the prefetch window and pause the backend.
        let coscheduler = ramflow::CoScheduler::new(gauge.clone())
            .map_err(FlowCastError::RamFlow)?;

        // C9: phase classifier — reads hardware_profile.json when available.
        let phase_classifier = DefaultPhaseClassifier::new(
            config.shard_dir.join("hardware_profile.json"),
        ).map_err(FlowCastError::RamFlow)?;

        let state_machine = Arc::new(PrefetchStateMachine::new(
            num_layers,
            lookahead,
            config.default_precision,
        ));

        let router = CompletionRouter::spawn(
            Arc::clone(&backend_arc),
            Arc::clone(&state_machine),
        )?;

        let w_max = compute_initial_w_max(lookahead);
        let window = AdaptiveWindow::new(lookahead as f32, config.ewma_alpha, w_max as f32);

        // C5: register adaptive-window pressure callbacks (high/soft/low).
        // Must be called after `window` and `backend_arc` are both live.
        window.register_pressure_callbacks(&gauge, Some(Arc::clone(&backend_arc)));

        let mut scheduler = WritebackScheduler::with_config(
            WritebackMode::Immediate,
            writeback::WritebackConfig {
                skip_threshold: 1e-6,
                max_skip_rate: 0.5,
                max_inflight_writes: lookahead.max(4),
                shard_dir: config.shard_dir.clone(),
                write_budget_bytes: 100 * 1024 * 1024 * 1024,
            },
        );
        scheduler.set_telemetry(Arc::new(telemetry.clone()));

        let hotset = HotSet::new(8);
        let priority_queue = PriorityQueue::new();
        let decoder = QuantizedDecoder::new(Precision::FP16);
        let profiler = Profiler::new(config.shard_dir.clone());
        let scale_table = PerLayerScaleTable::new(num_layers as usize, 0.05);

        // A5-EDF: install deadline scheduler from profiler data when available.
        // On first run (no hardware_profile), EDF falls back to sequential window order.
        if let Some(ref profile) = config.hardware_profile {
            if !profile.layer_plan.is_empty() {
                let edf = Arc::new(EdfScheduler::new(
                    &profile.layer_plan,
                    profile.pcie_bandwidth_gbs,
                    num_layers,
                ));
                state_machine.set_edf_scheduler(edf);
            }
        }

        // A5-DB: DuplexBudget — split NVMe bandwidth 60 % reads / 40 % writes.
        // Use measured NVMe bandwidth when a hardware profile is available; fall
        // back to 3.0 GB/s (conservative estimate for a typical consumer NVMe).
        let nvme_bandwidth_gbs = config
            .hardware_profile
            .as_ref()
            .map(|profile| profile.nvme_bandwidth_gbs)
            .unwrap_or(3.0_f32);
        let duplex_budget = Arc::new(DuplexBudget::new(
            nvme_bandwidth_gbs,
            DEFAULT_READ_FRACTION,
            1.0,
        ));
        state_machine.set_duplex_budget(Arc::clone(&duplex_budget));
        scheduler.set_duplex_budget(duplex_budget);

        // A3-T: periodic SSD thermal re-profiling (Linux + ssd-thermal only).
        let reprofiling_interval = config.reprofiling_interval_steps;
        #[cfg(all(target_os = "linux", feature = "ssd-thermal"))]
        let thermal_monitor: Option<Arc<smart_monitor::ThermalMonitor>> = {
            let device_path = std::env::var("FLOWCAST_NVME_DEVICE")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|_| std::path::PathBuf::from("/dev/nvme0"));
            Some(Arc::new(smart_monitor::ThermalMonitor::new(
                device_path,
                config.shard_dir.clone(),
                num_layers,
            )))
        };

        // A11: VRAM double-buffer — two alternating device-memory slots with a
        // dedicated copy stream so RAM→VRAM DMA overlaps GPU compute.
        // Slot size is read from the hardware profile when available; falls back
        // to 512 MiB (conservative estimate for a 70B model layer).
        #[cfg(feature = "cuda-double-buffer")]
        let vram_double_buffer: Option<Mutex<vram_double_buffer::VramDoubleBuffer>> = {
            let slot_bytes = config
                .hardware_profile
                .as_ref()
                .and_then(|profile| profile.layer_plan.first())
                .map(|layer_timing| layer_timing.shard_bytes as usize)
                .unwrap_or(512 * 1024 * 1024);
            Some(Mutex::new(vram_double_buffer::VramDoubleBuffer::new(slot_bytes)))
        };

        state_machine
            .prime_window(Direction::Forward, &pool, &*backend_arc)
            .map_err(|error| {
                FlowCastError::BackendIo(format!("prime_window failed: {error}"))
            })?;

        Ok(Self {
            config,
            backend: backend_arc,
            pool,
            state_machine,
            window,
            scheduler: Mutex::new(scheduler),
            hotset,
            priority_queue,
            decoder,
            router,
            telemetry,
            profiler,
            scale_table,
            gauge,
            coscheduler,
            phase_classifier,
            steps_since_reprofile: AtomicU64::new(0),
            reprofiling_interval,
            #[cfg(all(target_os = "linux", feature = "ssd-thermal"))]
            thermal_monitor,
            #[cfg(feature = "cuda-double-buffer")]
            vram_double_buffer,
        })
    }

    // ------------------------------------------------------------------
    // API-b: warm-up profiler
    // ------------------------------------------------------------------

    /// Run the warm-up profiler over `num_layers` model layers.
    ///
    /// Profiles 5 representative layers, measures t_ssd / t_pcie / t_gpu, and
    /// returns a `HardwareProfile` with per-layer timing.  On a SHA-256 cache
    /// hit (shard_index.json unchanged), returns the cached profile immediately.
    ///
    /// Installs (or refreshes) both the EDF deadline scheduler and the DuplexBudget
    /// token bucket with the newly measured bandwidth figures.
    ///
    /// # Errors
    /// `FlowCastError::ProfileIo` on filesystem failure.
    pub fn warmup(&mut self, num_layers: u32) -> Result<HardwareProfile> {
        let profile = self.profiler.warmup(num_layers)?;

        // A5-EDF: install (or refresh) the EDF scheduler now that timing data is available.
        if !profile.layer_plan.is_empty() {
            let edf = Arc::new(EdfScheduler::new(
                &profile.layer_plan,
                profile.pcie_bandwidth_gbs,
                num_layers,
            ));
            self.state_machine.set_edf_scheduler(edf);
        }

        // A5-DB: refresh DuplexBudget with the measured NVMe bandwidth.
        let duplex_budget = Arc::new(DuplexBudget::new(
            profile.nvme_bandwidth_gbs,
            DEFAULT_READ_FRACTION,
            1.0,
        ));
        self.state_machine.set_duplex_budget(Arc::clone(&duplex_budget));
        match self.scheduler.lock() {
            Ok(mut sched) => sched.set_duplex_budget(duplex_budget),
            Err(poison) => poison.into_inner().set_duplex_budget(duplex_budget),
        }

        Ok(profile)
    }

    // ------------------------------------------------------------------
    // API-c: per-layer start notification
    // ------------------------------------------------------------------

    /// Notify FlowCast that the GPU is now executing `layer_idx` in `direction`.
    ///
    /// Submits prefetch requests for the next window of layers, skipping any
    /// layer already resident in the hot-set.  Skips new prefetch submissions
    /// while the co-scheduler is paused (C6) to avoid pool exhaustion.  Always
    /// notifies the phase classifier (C9) so pool rebalancing stays current.
    ///
    /// Also drains write-backs previously deferred by the DuplexBudget (A5-DB):
    /// the token bucket is refilled inside `on_layer_start_with_residency`, so
    /// any accumulated write tokens are available for drain immediately after.
    ///
    /// # Errors
    /// * `FlowCastError::RamFlow` — pool exhausted.
    /// * `FlowCastError::BackendIo` — SQE submission failed or mutex poisoned.
    pub fn on_layer_start(&self, layer_idx: u32, direction: Direction) -> Result<()> {
        // C9: update phase classifier regardless of pause state so the rebalancer
        // always knows which layer the GPU is on.
        self.phase_classifier.notify_layer_start(layer_idx, direction);

        // C6: honour the co-scheduler pause signal — do not submit new SQEs while
        // memory pressure is critical.  In-flight requests continue to drain.
        if self.coscheduler.is_paused() {
            return Ok(());
        }

        let state_machine = self.state_machine.clone();
        let backend = self.backend.clone();
        let hotset = &self.hotset;
        let scale_table = &self.scale_table;
        let pool = &self.pool;
        // The state machine refills the DuplexBudget at the top of this call.
        state_machine.on_layer_start_with_residency(
            layer_idx,
            direction,
            pool,
            &*backend,
            |idx| hotset.is_resident(idx, scale_table),
        )?;

        // A5-DB: drain write-backs deferred due to write-token exhaustion.
        // Must happen AFTER state_machine.on_layer_start_with_residency so the
        // token bucket has been refilled and write tokens are freshly available.
        match self.scheduler.lock() {
            Ok(mut sched) => sched.drain_deferred(&*backend)?,
            Err(poison) => poison.into_inner().drain_deferred(&*backend)?,
        }

        // A3-T: periodic SSD thermal check and re-profiling (ssd-thermal, Linux only).
        // Increment step counter; fire SMART read at each interval-th step.
        let _step = self.steps_since_reprofile.fetch_add(1, Ordering::Relaxed).wrapping_add(1);
        #[cfg(all(target_os = "linux", feature = "ssd-thermal"))]
        if let Some(ref monitor) = self.thermal_monitor {
            monitor.tick(_step, self.reprofiling_interval);
            // Update telemetry with the latest SMART reading (refreshed every interval steps).
            let temp = monitor.ssd_temp_celsius();
            if temp > 0.0 {
                self.telemetry.record_thermal_state(temp, monitor.thermal_state().as_u8());
                self.telemetry.set_reprofiling_events(monitor.reprofiling_events());
            }
            // Apply any completed background re-profile result to the adaptive window.
            if let Some(outcome) = monitor.poll_outcome() {
                self.window.apply_w_max_update(outcome.w_max);
            }
        }

        Ok(())
    }

    // ------------------------------------------------------------------
    // API-d: blocking take_ready
    // ------------------------------------------------------------------

    /// Block until `layer_idx` is resident in pinned RAM, then return it.
    ///
    /// When the `cuda-double-buffer` feature is enabled, also issues an
    /// asynchronous RAM→VRAM copy on the dedicated copy stream and records a
    /// CUDA event.  Module 5 must call
    /// `cuda_stream_wait_event(compute_stream, ready_layer.copy_event)` before
    /// dispatching the compute kernel on the returned VRAM slot.
    ///
    /// Waits up to `timeout`. Returns `FlowCastError::PrefetchMiss` if the
    /// buffer does not arrive in time.
    ///
    /// # Errors
    /// * `FlowCastError::PrefetchMiss` — buffer not yet resident.
    /// * `FlowCastError::BackendIo` — mutex or condvar poisoned, or VRAM
    ///   double-buffer slot capacity exceeded.
    pub fn take_ready(&self, layer_idx: u32, timeout: Duration) -> Result<ReadyLayer> {
        #[cfg(not(feature = "cuda-double-buffer"))]
        let ready = self.state_machine.take_ready(layer_idx, timeout)?;
        #[cfg(feature = "cuda-double-buffer")]
        let mut ready = self.state_machine.take_ready(layer_idx, timeout)?;

        #[cfg(feature = "cuda-double-buffer")]
        if let Some(ref vdb_mutex) = self.vram_double_buffer {
            // Copy the pinned-RAM bytes into a temporary Vec so we can release
            // the borrow on `ready` before mutating `ready.slab_device_ptrs`.
            let src = ready.as_slice().to_vec();
            let mut vdb = vdb_mutex.lock().map_err(|_| {
                FlowCastError::BackendIo(
                    "vram_double_buffer mutex poisoned".to_string(),
                )
            })?;
            let (vram_ptr, event) = vdb.advance(layer_idx, &src).map_err(|error| {
                FlowCastError::BackendIo(format!("vram double buffer advance: {error}"))
            })?;
            // Expose the VRAM device pointer via the existing slab-pointer field so
            // M5 does not need a new code path: slab 0 holds the double-buffer slot.
            ready.slab_device_ptrs = vec![(0, vram_ptr)];
            ready.copy_event = Some(event);
        }

        Ok(ready)
    }

    // ------------------------------------------------------------------
    // API-e: write-back notification
    // ------------------------------------------------------------------

    /// Notify FlowCast that the optimizer has updated `layer_idx` weights.
    ///
    /// Applies gradient-threshold write-skipping (A9) and submits an async
    /// write via the write-back scheduler (A4).  Uses the layer's gradient
    /// variance from `PerLayerScaleTable` as the `lr_grad_norm` proxy.
    ///
    /// When the DuplexBudget (A5-DB) write-token bucket is exhausted, the write
    /// is deferred to the internal queue and flushed at the next `on_layer_start`.
    ///
    /// # Errors
    /// `FlowCastError::BackendIo` on SQE submission failure.
    pub fn on_weights_updated(&mut self, layer_idx: u32, src: &PinnedBuffer) -> Result<()> {
        let lr_grad_norm = self.scale_table.gradient_variance(layer_idx as usize);
        let backend = self.backend.clone();
        match self.scheduler.lock() {
            Ok(mut sched) => sched.on_weights_updated(layer_idx, src, 0, lr_grad_norm, &*backend),
            Err(poison) => {
                poison.into_inner().on_weights_updated(layer_idx, src, 0, lr_grad_norm, &*backend)
            }
        }
    }

    // ------------------------------------------------------------------
    // Legacy API (kept for Module 5 compatibility)
    // ------------------------------------------------------------------

    /// Notify FlowCast that the GPU is now executing `current_layer` in
    /// `direction`, and trigger prefetch of the next layer(s).
    ///
    /// Delegates to `on_layer_start` (same semantics, kept for compat).
    ///
    /// # Errors
    /// See `on_layer_start`.
    pub fn advance_step(&mut self, direction: Direction, current_layer: u32) -> Result<()> {
        self.on_layer_start(current_layer, direction)
    }

    /// Return the `ReadyLayer` for `layer_idx` if its prefetch has completed.
    ///
    /// Delegates to `take_ready` with a 500 ms default timeout.
    ///
    /// # Errors
    /// * `FlowCastError::PrefetchMiss` — buffer not yet resident.
    pub fn wait_for_layer(&mut self, layer_idx: u32) -> Result<ReadyLayer> {
        self.take_ready(layer_idx, Duration::from_millis(500))
    }

    /// Signal that M5 is done with `layer`; returns the pinned-RAM slot to the pool.
    ///
    /// Dropping `layer` automatically returns the slot via `PoolSlot::drop`.
    ///
    /// # Errors
    /// Always `Ok(())`.
    pub fn retire_layer(&mut self, _layer: ReadyLayer) -> Result<()> {
        Ok(())
    }

    /// Drain pending completions from the backend into the ready map.
    ///
    /// In production the `CompletionRouter` thread is the sole drainer; exposing
    /// this in production creates a second drainer which can steal completions
    /// from the router thread and stall `take_ready`.  Only available in `#[cfg(test)]`.
    ///
    /// # Errors
    /// * `FlowCastError::BackendIo` — poll failed.
    #[cfg(test)]
    pub fn poll_completions(&mut self) -> Result<u32> {
        self.state_machine.poll_and_route(&*self.backend)
    }

    /// Telemetry snapshot for the current pipeline state.
    pub fn telemetry(&self) -> telemetry::TelemetrySnapshot {
        self.telemetry.snapshot()
    }

    /// Graceful shutdown: stop the completion-router thread and shut down the backend.
    ///
    /// # Errors
    /// * `FlowCastError::BackendIo` — router shutdown or backend shutdown failed.
    pub fn shutdown(&mut self) -> Result<()> {
        self.router.shutdown()?;
        // After join() the router thread has exited and dropped its Arc clone,
        // so Arc::get_mut succeeds here.
        if let Some(backend) = Arc::get_mut(&mut self.backend) {
            backend.shutdown()?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Initial W_max estimate before the warm-up profiler runs.
///
/// Uses `lookahead + 2` as a conservative ceiling until real timing is available.
fn compute_initial_w_max(lookahead: u32) -> u32 {
    lookahead.saturating_add(2).max(4)
}
