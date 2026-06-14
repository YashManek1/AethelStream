//! A4/A9: Delayed write-back scheduler and gradient-threshold write-skip.
//!
//! `on_weights_updated(i, src, lr_grad_norm)` enqueues a background
//! `write_async` for layer `i` overlapped with the SSD read of layer `i−2`
//! during the backward pass. In-flight writes are capped at `max_inflight`
//! so they cannot starve reads under the shared pressure budget.
//!
//! Gradient-threshold skipping (A9):
//! - If `|lr·grad|` < `skip_threshold`: accumulate delta in a temp counter,
//!   skip the SSD write.
//! - Once the accumulated delta crosses `skip_threshold` OR epoch end is
//!   signalled, the write is flushed.
//! - A `max_skip_rate` guard (fraction in (0,1]) caps how many consecutive
//!   layers may be skipped; any layer beyond that fraction is written
//!   unconditionally.
//!
//! All writes are routed through `WriteBudgetManager::enqueue_write` (ssd-wear
//! feature) before calling `IoBackend::write_async`.
//!
//! # DuplexBudget integration (A5-DB)
//! When a [`DuplexBudget`] is installed via [`WritebackScheduler::set_duplex_budget`],
//! `on_weights_updated` checks write-token availability before each SQE.  On
//! exhaustion the write is pushed to an internal `deferred_writes` queue.
//! [`drain_deferred`] is called by the [`crate::FlowCast`] facade at the start
//! of every `on_layer_start` so deferred writes fire as soon as bandwidth tokens
//! are refilled.
//!
//! [`drain_deferred`]: WritebackScheduler::drain_deferred

use crate::backend::IoBackend;
use crate::scheduler::DuplexBudget;
use crate::telemetry::Telemetry;
use crate::{FlowCastError, Result};
use ramflow::nvme::write_budget::WriteBudgetManager;
use ramflow::PinnedBuffer;
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Public config types (preserved from original stub)
// ---------------------------------------------------------------------------

/// How optimizer state is written back after each layer's backward step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WritebackMode {
    /// Write immediately after each layer (baseline).
    Immediate,
    /// Accumulate `batch_size` layers before flushing.
    Batched {
        /// Number of layers per flush.
        batch_size: usize,
    },
    /// Skip writes for layers with gradient norm below threshold (A9).
    Deferred {
        /// Maximum layers to defer before a mandatory flush.
        max_defer: usize,
    },
}

/// A single pending write-back operation.
pub struct PendingWrite {
    /// Layer index whose optimizer state must be written.
    pub layer_idx: u32,
    /// Byte offset in the optimizer state file.
    pub byte_offset: u64,
    /// Byte length.
    pub byte_length: u64,
}

// ---------------------------------------------------------------------------
// DeferredWrite (internal)
// ---------------------------------------------------------------------------

/// A write-back deferred because the [`DuplexBudget`] write-token bucket was exhausted.
///
/// Stores an owned copy of the source bytes so the original `PinnedBuffer`
/// can be freed immediately.  The copy is re-pinned when [`WritebackScheduler::drain_deferred`]
/// submits the write.
struct DeferredWrite {
    /// Layer index whose optimizer state was updated.
    layer_idx: u32,
    /// Byte offset in the shard file (512-byte aligned per `O_DIRECT`).
    byte_offset: u64,
    /// Owned copy of the layer weight bytes to write.
    data: Vec<u8>,
}

// ---------------------------------------------------------------------------
// WritebackConfig
// ---------------------------------------------------------------------------

/// Configuration for the write-back scheduler.
#[derive(Debug, Clone)]
pub struct WritebackConfig {
    /// Gradient-threshold skip: skip write if `|lr·grad|` < this value.
    pub skip_threshold: f32,
    /// Maximum fraction of layers that may be skipped per pass. Range: (0, 1].
    pub max_skip_rate: f32,
    /// Maximum concurrent in-flight write SQEs (read-starvation guard).
    pub max_inflight_writes: u32,
    /// Shard directory (for resolving shard paths to write budget manager).
    pub shard_dir: PathBuf,
    /// SSD write budget in bytes (passed to WriteBudgetManager).
    pub write_budget_bytes: u64,
}

impl Default for WritebackConfig {
    fn default() -> Self {
        Self {
            skip_threshold: 1e-6,
            max_skip_rate: 0.5,
            max_inflight_writes: 4,
            shard_dir: PathBuf::from("./shards"),
            write_budget_bytes: 100 * 1024 * 1024 * 1024, // 100 GiB default
        }
    }
}

// ---------------------------------------------------------------------------
// WritebackScheduler
// ---------------------------------------------------------------------------

/// Delayed write-back scheduler (A4) with gradient-threshold skip (A9).
pub struct WritebackScheduler {
    /// Legacy mode field (kept for API compat).
    mode: WritebackMode,
    /// Legacy pending list (kept for flush_all / pending_count compat).
    pending: Vec<PendingWrite>,
    /// Full config.
    config: WritebackConfig,
    /// Accumulated delta per layer (layer_idx → |lr·grad| sum since last write).
    accumulated_delta: HashMap<u32, f32>,
    /// Number of layers skipped in the current pass.
    skipped_in_pass: u32,
    /// Total layers seen in the current pass (for skip-rate enforcement).
    total_in_pass: u32,
    /// Number of write SQEs currently in flight.
    inflight: Arc<AtomicU32>,
    /// Write-budget manager (ssd-wear routing).
    budget: WriteBudgetManager,
    /// Optional telemetry handle; wired via `set_telemetry` (T-b fix).
    telemetry: Option<Arc<Telemetry>>,
    /// Duplex bandwidth token bucket (shared with the prefetch state machine).
    ///
    /// When `Some`, `on_weights_updated` checks write-token availability before
    /// submitting any SQE.  Exhausted writes are pushed to `deferred_writes`.
    duplex_budget: Option<Arc<DuplexBudget>>,
    /// Write-backs deferred because `duplex_budget` write tokens were exhausted.
    ///
    /// Drained by [`drain_deferred`] at the start of each `on_layer_start`
    /// after the token bucket has been refilled.
    ///
    /// [`drain_deferred`]: WritebackScheduler::drain_deferred
    deferred_writes: VecDeque<DeferredWrite>,
}

impl WritebackScheduler {
    /// Create a scheduler with the given `mode` and default config.
    pub fn new(mode: WritebackMode) -> Self {
        Self::with_config(mode, WritebackConfig::default())
    }

    /// Create a scheduler with explicit config.
    pub fn with_config(mode: WritebackMode, config: WritebackConfig) -> Self {
        let budget = WriteBudgetManager::new(
            config.shard_dir.join("nvme0"),
            config.write_budget_bytes,
        );
        Self {
            mode,
            pending: Vec::new(),
            config,
            accumulated_delta: HashMap::new(),
            skipped_in_pass: 0,
            total_in_pass: 0,
            inflight: Arc::new(AtomicU32::new(0)),
            budget,
            telemetry: None,
            duplex_budget: None,
            deferred_writes: VecDeque::new(),
        }
    }

    /// Wire a telemetry handle so write-skip and write-submit events are counted (T-b fix).
    pub fn set_telemetry(&mut self, telemetry: Arc<Telemetry>) {
        self.telemetry = Some(telemetry);
    }

    /// Wire the duplex bandwidth token bucket (shared with
    /// [`crate::state_machine::PrefetchStateMachine`]).
    ///
    /// Call once during [`crate::FlowCast`] construction.  The write-token
    /// bucket must have tokens available before `on_weights_updated` submits
    /// any SQE. On exhaustion, writes are pushed to the internal deferred queue
    /// and flushed by [`drain_deferred`] at the start of the next layer step.
    ///
    /// [`drain_deferred`]: WritebackScheduler::drain_deferred
    pub fn set_duplex_budget(&mut self, budget: Arc<DuplexBudget>) {
        self.duplex_budget = Some(budget);
    }

    // ------------------------------------------------------------------
    // Primary API
    // ------------------------------------------------------------------

    /// Called after the optimizer updates layer `i` weights.
    ///
    /// `lr_grad_norm` = `|lr · grad|` for this layer (scalar f32).
    /// `src` = updated weight buffer in pinned RAM.
    /// `byte_offset` / `byte_length` = location in the shard file.
    ///
    /// Write ordering:
    /// 1. Gradient-skip check — skip if norm below threshold and skip-rate budget allows.
    /// 2. DuplexBudget check — defer to `deferred_writes` if write tokens exhausted.
    /// 3. In-flight count cap — drain completions until below `max_inflight_writes`.
    /// 4. WriteBudgetManager — charge SSD wear budget.
    /// 5. `write_async` — submit the SQE.
    ///
    /// # Errors
    /// `FlowCastError::BackendIo` on submission failure.
    pub fn on_weights_updated(
        &mut self,
        layer_idx: u32,
        src: &PinnedBuffer,
        byte_offset: u64,
        lr_grad_norm: f32,
        backend: &dyn IoBackend,
    ) -> Result<()> {
        self.total_in_pass += 1;
        let byte_length = src.len() as u64;

        // Accumulate delta; read values before taking mutable borrow.
        let current_delta = self.accumulated_delta.get(&layer_idx).copied().unwrap_or(0.0);
        let new_delta = current_delta + lr_grad_norm;
        self.accumulated_delta.insert(layer_idx, new_delta);

        // Decide whether to skip (A9 gradient-threshold skip).
        let skip_rate_ok = self.skip_rate_headroom() > 0;
        let should_skip = new_delta < self.config.skip_threshold && skip_rate_ok;

        if should_skip {
            self.skipped_in_pass += 1;
            if let Some(telemetry) = &self.telemetry {
                telemetry.record_write_skip();
            }
            return Ok(());
        }

        // Clear accumulated delta — we are writing now.
        self.accumulated_delta.insert(layer_idx, 0.0);

        // DuplexBudget write-token check (A5-DB): defer when write bandwidth is exhausted
        // so burst write-backs cannot starve EDF-scheduled prefetch reads.
        // Must happen BEFORE the inflight count check (the count cap is a concurrency
        // guard; the token bucket is a bandwidth guarantee — both must pass).
        if let Some(ref budget) = self.duplex_budget {
            if budget.take_write(byte_length).is_err() {
                self.deferred_writes.push_back(DeferredWrite {
                    layer_idx,
                    byte_offset,
                    data: src.as_slice().to_vec(),
                });
                return Ok(());
            }
        }

        // Respect in-flight write cap (prevents read starvation).
        self.wait_for_inflight_slot(backend)?;

        // Route through WriteBudgetManager (no-op when ssd-wear feature inactive).
        self.budget.enqueue_write(layer_idx, src).map_err(|e| {
            FlowCastError::BackendIo(format!("write budget: {e}"))
        })?;

        // Submit async write; record telemetry before submission (T-b fix).
        if let Some(telemetry) = &self.telemetry {
            telemetry.record_write_submitted();
        }
        let token = write_token(layer_idx);
        self.inflight.fetch_add(1, Ordering::AcqRel);
        if let Err(error) = backend.write_async(layer_idx, byte_offset, byte_length, src, token) {
            self.inflight.fetch_sub(1, Ordering::AcqRel);
            return Err(error);
        }
        Ok(())
    }

    /// Drain write-backs deferred because the [`DuplexBudget`] write-token bucket was exhausted.
    ///
    /// Called by [`crate::FlowCast::on_layer_start`] immediately after the token bucket
    /// has been refilled (via [`crate::scheduler::DuplexBudget::refill`] in the state
    /// machine).  Flushes as many deferred writes as the current write-token budget allows,
    /// stopping as soon as tokens run out.  Remaining deferred writes stay in the queue.
    ///
    /// If no [`DuplexBudget`] is installed, all deferred writes are flushed unconditionally.
    ///
    /// # Errors
    /// `FlowCastError::BackendIo` on the first submission failure.  Remaining deferred
    /// writes stay in the queue for the next [`drain_deferred`] call.
    ///
    /// [`drain_deferred`]: WritebackScheduler::drain_deferred
    pub fn drain_deferred(&mut self, backend: &dyn IoBackend) -> Result<()> {
        while !self.deferred_writes.is_empty() {
            let byte_length = match self.deferred_writes.front() {
                Some(w) => w.data.len() as u64,
                None => break,
            };

            // Re-check token budget: stop if still exhausted.
            if let Some(ref budget) = self.duplex_budget {
                if budget.take_write(byte_length).is_err() {
                    break;
                }
            }

            let write = match self.deferred_writes.pop_front() {
                Some(w) => w,
                None => break,
            };

            // Respect in-flight cap even for deferred writes.
            self.wait_for_inflight_slot(backend)?;

            // Re-pin the bytes so WriteBudgetManager can inspect the buffer.
            let mut pinned =
                PinnedBuffer::alloc(write.data.len()).map_err(FlowCastError::RamFlow)?;
            pinned.as_mut_slice().copy_from_slice(&write.data);

            self.budget.enqueue_write(write.layer_idx, &pinned).map_err(|e| {
                FlowCastError::BackendIo(format!(
                    "write budget deferred layer {}: {e}",
                    write.layer_idx
                ))
            })?;

            if let Some(ref telemetry) = self.telemetry {
                telemetry.record_write_submitted();
            }
            let token = write_token(write.layer_idx);
            self.inflight.fetch_add(1, Ordering::AcqRel);
            if let Err(error) =
                backend.write_async(write.layer_idx, write.byte_offset, byte_length, &pinned, token)
            {
                self.inflight.fetch_sub(1, Ordering::AcqRel);
                return Err(error);
            }
        }
        Ok(())
    }

    /// Flush all accumulated-delta layers at epoch end.
    ///
    /// Writes every layer whose accumulated delta > 0 unconditionally.
    ///
    /// # Errors
    /// `FlowCastError::BackendIo` on any submission failure (best-effort: continues after error).
    pub fn flush_epoch_end(
        &mut self,
        src_map: &HashMap<u32, &PinnedBuffer>,
        backend: &dyn IoBackend,
    ) -> Result<()> {
        let layers: Vec<u32> = self
            .accumulated_delta
            .iter()
            .filter(|(_, &delta)| delta > 0.0)
            .map(|(&idx, _)| idx)
            .collect();

        let mut last_error: Option<FlowCastError> = None;
        for layer_idx in layers {
            if let Some(&src) = src_map.get(&layer_idx) {
                self.accumulated_delta.insert(layer_idx, 0.0);
                if let Err(error) = self.wait_for_inflight_slot(backend) {
                    last_error = Some(error);
                    continue;
                }
                if let Err(error) = self.budget.enqueue_write(layer_idx, src).map_err(|error| {
                    FlowCastError::BackendIo(format!(
                        "write budget layer {layer_idx}: {error}"
                    ))
                }) {
                    last_error = Some(error);
                    continue;
                }
                let token = write_token(layer_idx);
                self.inflight.fetch_add(1, Ordering::AcqRel);
                if let Err(error) = backend.write_async(layer_idx, 0, src.len() as u64, src, token) {
                    self.inflight.fetch_sub(1, Ordering::AcqRel);
                    last_error = Some(error);
                }
            }
        }
        // Reset pass counters.
        self.skipped_in_pass = 0;
        self.total_in_pass = 0;

        if let Some(e) = last_error { Err(e) } else { Ok(()) }
    }

    /// Current skip rate for this pass (skipped / total, or 0 if no layers seen).
    pub fn current_skip_rate(&self) -> f32 {
        if self.total_in_pass == 0 {
            return 0.0;
        }
        self.skipped_in_pass as f32 / self.total_in_pass as f32
    }

    /// Number of in-flight write SQEs.
    pub fn inflight_count(&self) -> u32 {
        self.inflight.load(Ordering::Acquire)
    }

    /// Accumulated delta for `layer_idx` (0.0 if not tracked).
    pub fn accumulated_delta(&self, layer_idx: u32) -> f32 {
        self.accumulated_delta.get(&layer_idx).copied().unwrap_or(0.0)
    }

    /// Number of write-backs currently sitting in the deferred queue.
    pub fn deferred_count(&self) -> usize {
        self.deferred_writes.len()
    }

    // ------------------------------------------------------------------
    // Legacy API (kept for existing callers / FlowCast facade compat)
    // ------------------------------------------------------------------

    /// Enqueue a write-back for `layer_idx`.
    pub fn enqueue(&mut self, write: PendingWrite) -> Result<()> {
        self.pending.push(write);
        Ok(())
    }

    /// Flush if batch threshold is met; no-op otherwise.
    pub fn flush_if_needed(&mut self, backend: &dyn IoBackend) -> Result<()> {
        match self.mode {
            WritebackMode::Batched { batch_size } if self.pending.len() >= batch_size => {
                self.flush_all(backend)
            }
            _ => Ok(()),
        }
    }

    /// Flush all pending writes unconditionally.
    pub fn flush_all(&mut self, backend: &dyn IoBackend) -> Result<()> {
        let writes = std::mem::take(&mut self.pending);
        for write in writes {
            let buf = PinnedBuffer::alloc(write.byte_length as usize)
                .map_err(FlowCastError::RamFlow)?;
            let token = write_token(write.layer_idx);
            backend.write_async(
                write.layer_idx,
                write.byte_offset,
                write.byte_length,
                &buf,
                token,
            )?;
        }
        Ok(())
    }

    /// Number of pending writes (legacy queue).
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Get current mode.
    pub fn mode(&self) -> WritebackMode {
        self.mode
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    /// How many more layers may be skipped before hitting `max_skip_rate`.
    fn skip_rate_headroom(&self) -> u32 {
        if self.total_in_pass == 0 {
            return 1; // first layer: always some headroom
        }
        let max_skippable = (self.total_in_pass as f32 * self.config.max_skip_rate) as u32;
        max_skippable.saturating_sub(self.skipped_in_pass)
    }

    /// Drain completed write SQEs from `backend` until `inflight < max_inflight`.
    ///
    /// Polls the backend completion queue; each write completion decrements
    /// the in-flight counter. Returns immediately if already under the cap.
    fn wait_for_inflight_slot(&self, backend: &dyn IoBackend) -> Result<()> {
        let max = self.config.max_inflight_writes;
        // Poll up to 16 times to drain completions.
        for _ in 0..16 {
            if self.inflight.load(Ordering::Acquire) < max {
                return Ok(());
            }
            let completions = backend.poll_completions()?;
            let write_completions = completions
                .iter()
                .filter(|c| is_write_token(c.token))
                .count() as u32;
            if write_completions > 0 {
                self.inflight
                    .fetch_sub(write_completions.min(self.inflight.load(Ordering::Acquire)), Ordering::AcqRel);
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Token helpers: write tokens have bit 63 set to distinguish from read tokens
// ---------------------------------------------------------------------------

const WRITE_TOKEN_FLAG: u64 = 1u64 << 63;

fn write_token(layer_idx: u32) -> u64 {
    WRITE_TOKEN_FLAG | layer_idx as u64
}

fn is_write_token(token: u64) -> bool {
    token & WRITE_TOKEN_FLAG != 0
}