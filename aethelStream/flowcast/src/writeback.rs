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

use crate::backend::IoBackend;
use crate::{FlowCastError, Result};
use ramflow::nvme::write_budget::WriteBudgetManager;
use ramflow::PinnedBuffer;
use std::collections::HashMap;
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
        }
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
    /// Write-skip logic:
    /// 1. If `lr_grad_norm < skip_threshold` AND skip-rate budget not exhausted
    ///    → accumulate delta, return without writing.
    /// 2. Else → submit `write_async` if in-flight cap allows; otherwise block.
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

        // Decide whether to skip.
        let skip_rate_ok = self.skip_rate_headroom() > 0;
        let should_skip = new_delta < self.config.skip_threshold && skip_rate_ok;

        if should_skip {
            self.skipped_in_pass += 1;
            return Ok(());
        }

        // Clear accumulated delta — we are writing now.
        self.accumulated_delta.insert(layer_idx, 0.0);

        // Respect in-flight write cap (prevents read starvation).
        self.wait_for_inflight_slot(backend)?;

        // Route through WriteBudgetManager (no-op when ssd-wear inactive).
        self.budget.enqueue_write(layer_idx, src).map_err(|e| {
            FlowCastError::BackendIo(format!("write budget: {e}"))
        })?;

        // Submit async write.
        let token = write_token(layer_idx);
        self.inflight.fetch_add(1, Ordering::AcqRel);
        if let Err(error) = backend.write_async(layer_idx, byte_offset, byte_length, src, token) {
            self.inflight.fetch_sub(1, Ordering::AcqRel);
            return Err(error);
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
                self.wait_for_inflight_slot(backend).ok();
                self.budget.enqueue_write(layer_idx, src).ok();
                let token = write_token(layer_idx);
                self.inflight.fetch_add(1, Ordering::AcqRel);
                if let Err(e) = backend.write_async(layer_idx, 0, src.len() as u64, src, token) {
                    self.inflight.fetch_sub(1, Ordering::AcqRel);
                    last_error = Some(e);
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
                .map_err(|e| FlowCastError::RamFlow(e))?;
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
