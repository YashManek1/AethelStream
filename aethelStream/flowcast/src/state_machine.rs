//! A1: Bidirectional prefetch state machine.
//!
//! Tracks which pass is active and maintains two maps shared with the
//! completion-router thread via `Arc<(Mutex<MachineInner>, Condvar)>`:
//! * `in_flight`: token -> InFlightEntry for submitted, not-yet-done reads.
//! * `ready`: layer_idx -> PoolSlot for completed, not-yet-consumed reads.

use crate::backend::{Completion, IoBackend};
use crate::config::Precision;
use crate::ready::ReadyLayer;
use crate::{FlowCastError, Result};
use ramflow::phase::Direction;
use ramflow::pool::{LayerKind, PoolSlot};
use ramflow::PoolRegistry;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// Current training pass of the prefetch pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Pipeline has not been started.
    Idle,
    /// Forward pass active (layers 0 to L-1).
    Forward,
    /// Backward pass active (layers L-1 to 0).
    Backward,
    /// Mini-forward recomputation within backward.
    Recompute {
        /// Last layer (inclusive) of the recomputation window.
        window_end: u32,
    },
}

struct InFlightEntry {
    layer_idx: u32,
    slot: PoolSlot,
}

/// Shared mutable core of the state machine.
///
/// Paired with a `Condvar` for efficient waiting in
/// [`PrefetchStateMachine::take_ready`].
pub(crate) struct MachineInner {
    /// Current training phase.
    pub(crate) phase: Phase,
    /// Current pass direction.
    pub(crate) direction: Direction,
    /// token -> InFlightEntry for submitted, not-yet-completed reads.
    in_flight: HashMap<u64, InFlightEntry>,
    /// layer_idx -> PoolSlot for completed, not-yet-consumed reads.
    ready: HashMap<u32, PoolSlot>,
}

/// Bidirectional prefetch state machine (Algorithm A1).
///
/// The `shared` field is an `Arc` cloned into the `CompletionRouter` thread
/// so both sides can access `in_flight` and `ready` under the same mutex.
pub struct PrefetchStateMachine {
    /// Interior-mutable shared state cloned into CompletionRouter.
    pub(crate) shared: Arc<(Mutex<MachineInner>, Condvar)>,
    /// Monotonically increasing token source (Relaxed: uniqueness is all
    /// that matters; no ordering relationship with other ops is required).
    next_token: AtomicU64,
    /// Total layers in the model.
    total_layers: u32,
    /// Fixed lookahead window (Sprint 2; A2 makes this adaptive in Sprint 3).
    lookahead: u32,
    /// Precision tag on returned [`ReadyLayer`]s.
    default_precision: Precision,
}

impl PrefetchStateMachine {
    /// Create a state machine for a model with `total_layers` layers.
    ///
    /// * `lookahead` -- layers to prefetch ahead of the GPU (fixed for S2).
    /// * `default_precision` -- precision tag on returned [`ReadyLayer`]s.
    pub fn new(total_layers: u32, lookahead: u32, default_precision: Precision) -> Self {
        let inner = MachineInner {
            phase: Phase::Idle,
            direction: Direction::Forward,
            in_flight: HashMap::new(),
            ready: HashMap::new(),
        };
        Self {
            shared: Arc::new((Mutex::new(inner), Condvar::new())),
            next_token: AtomicU64::new(1),
            total_layers,
            lookahead,
            default_precision,
        }
    }

    /// Submit prefetch requests for the initial window before the training loop.
    ///
    /// * Forward: submits layers `0..=lookahead-1`.
    /// * Backward: submits layers `total-lookahead..=total-1`.
    ///
    /// Call exactly once before the first [`on_layer_start`].
    ///
    /// # Errors
    /// * [`FlowCastError::RamFlow`] -- pool exhausted.
    /// * [`FlowCastError::BackendIo`] -- I/O submission rejected.
    ///
    /// [`on_layer_start`]: Self::on_layer_start
    pub fn prime_window(
        &self,
        direction: Direction,
        pool: &PoolRegistry,
        backend: &dyn IoBackend,
    ) -> Result<()> {
        let targets: Vec<u32> = match direction {
            Direction::Forward => (0..self.lookahead.min(self.total_layers)).collect(),
            Direction::Backward => {
                let end = self.total_layers;
                let start = end.saturating_sub(self.lookahead);
                (start..end).rev().collect()
            }
        };
        for target in targets {
            self.submit_prefetch_for(target, pool, backend)?;
        }
        Ok(())
    }

    /// Notify that the GPU has begun executing `layer_idx`; submit next window.
    ///
    /// * Forward: submits `layer_idx+1..=layer_idx+W`.
    /// * Backward: submits `layer_idx-W..=layer_idx-1`.
    ///
    /// Already in-flight or ready layers are skipped.
    ///
    /// # Errors
    /// * [`FlowCastError::RamFlow`] -- pool exhausted.
    /// * [`FlowCastError::BackendIo`] -- I/O submission rejected.
    pub fn on_layer_start(
        &self,
        layer_idx: u32,
        direction: Direction,
        pool: &PoolRegistry,
        backend: &dyn IoBackend,
    ) -> Result<()> {
        {
            let (mutex, _) = self.shared.as_ref();
            let mut inner = mutex.lock().map_err(|_| {
                FlowCastError::BackendIo("state machine mutex poisoned".to_string())
            })?;
            inner.phase = match direction {
                Direction::Forward => Phase::Forward,
                Direction::Backward => Phase::Backward,
            };
            inner.direction = direction;
        }
        for target in self.prefetch_targets(layer_idx, direction) {
            self.submit_prefetch_for(target, pool, backend)?;
        }
        Ok(())
    }

    /// Block until `layer_idx` is resident in pinned RAM, then return it.
    ///
    /// Waits up to `timeout`. Returns [`FlowCastError::PrefetchMiss`] if the
    /// buffer does not arrive in time.
    ///
    /// # Errors
    /// * [`FlowCastError::PrefetchMiss`] -- buffer not resident before timeout.
    /// * [`FlowCastError::BackendIo`] -- mutex or condvar poisoned.
    pub fn take_ready(&self, layer_idx: u32, timeout: Duration) -> Result<ReadyLayer> {
        let (mutex, condvar) = self.shared.as_ref();
        let start = Instant::now();
        loop {
            let elapsed = start.elapsed();
            if elapsed >= timeout {
                return Err(FlowCastError::PrefetchMiss { layer_idx });
            }
            let remaining = timeout - elapsed;
            let guard = mutex.lock().map_err(|_| {
                FlowCastError::BackendIo("state machine mutex poisoned".to_string())
            })?;
            if guard.ready.contains_key(&layer_idx) {
                let mut guard = guard;
                let slot = guard
                    .ready
                    .remove(&layer_idx)
                    .ok_or(FlowCastError::PrefetchMiss { layer_idx })?;
                return Ok(ReadyLayer {
                    layer_idx,
                    precision: self.default_precision,
                    slot,
                });
            }
            match condvar.wait_timeout(guard, remaining) {
                Ok((_, wait_result)) if wait_result.timed_out() => {
                    return Err(FlowCastError::PrefetchMiss { layer_idx });
                }
                Ok(_) => {}
                Err(_) => {
                    return Err(FlowCastError::BackendIo(
                        "condvar poisoned during wait_timeout".to_string(),
                    ));
                }
            }
        }
    }

    /// Drain completions from `backend` and route them (single-threaded mode).
    ///
    /// In production, prefer [`crate::completion_router::CompletionRouter`].
    ///
    /// # Errors
    /// * [`FlowCastError::BackendIo`] -- poll failed.
    pub fn poll_and_route(&self, backend: &dyn IoBackend) -> Result<u32> {
        let completions = backend.poll_completions()?;
        let count = completions.len() as u32;
        self.route_completions(completions);
        Ok(count)
    }

    /// Route a batch of backend completions into the ready map.
    ///
    /// Called by the `CompletionRouter` thread; also reachable via
    /// [`poll_and_route`] for single-threaded test flows.
    ///
    /// [`poll_and_route`]: Self::poll_and_route
    pub(crate) fn route_completions(&self, completions: Vec<Completion>) {
        let (mutex, condvar) = self.shared.as_ref();
        let mut inner = match mutex.lock() {
            Ok(guard) => guard,
            Err(poison) => poison.into_inner(),
        };
        let mut any_ready = false;
        for c in completions {
            if let Some(entry) = inner.in_flight.remove(&c.token) {
                if c.result >= 0 {
                    inner.ready.insert(entry.layer_idx, entry.slot);
                    any_ready = true;
                }
                // Negative result: slot dropped here -> returned to pool ring.
            }
        }
        drop(inner);
        if any_ready {
            condvar.notify_all();
        }
    }

    /// Current training phase.
    pub fn phase(&self) -> Phase {
        let (mutex, _) = self.shared.as_ref();
        match mutex.lock() {
            Ok(guard) => guard.phase,
            Err(poison) => poison.into_inner().phase,
        }
    }

    fn submit_prefetch_for(
        &self,
        target_layer: u32,
        pool: &PoolRegistry,
        backend: &dyn IoBackend,
    ) -> Result<()> {
        // Claim the pool slot outside the lock: pool.claim can block on the
        // slow path waiting for a slot to be freed, and holding the state
        // machine mutex during that wait would prevent the completion router
        // from ever calling route_completions (deadlock).
        let kind = layer_kind_for(target_layer);
        let slot = pool.claim(kind).map_err(FlowCastError::RamFlow)?;
        let token = self.next_token.fetch_add(1, Ordering::Relaxed);
        let slot_len = slot.buffer().len() as u64;

        let (mutex, _) = self.shared.as_ref();
        let mut inner = mutex.lock().map_err(|_| {
            FlowCastError::BackendIo("state machine mutex poisoned".to_string())
        })?;

        // Re-check under the lock: a previous on_layer_start iteration may
        // have submitted this layer while we were claiming the slot outside.
        let already_tracked = inner
            .in_flight
            .values()
            .any(|e| e.layer_idx == target_layer)
            || inner.ready.contains_key(&target_layer);
        if already_tracked {
            // slot returned to pool via Drop.
            return Ok(());
        }

        // Insert into in_flight BEFORE calling backend.prefetch.
        //
        // FileReadBackend completes synchronously: it reads the file and pushes
        // the completion into its pending queue inside prefetch(). The
        // CompletionRouter drains that queue every ~50 us on a background
        // thread. If prefetch() were called first and in_flight.insert came
        // second, the router could drain the completion, fail to find the token
        // in in_flight, and silently discard it -- leaving the layer stuck
        // forever and causing take_ready to time out.
        //
        // Lock-ordering safety: the router acquires the FileReadBackend lock in
        // poll_completions, releases it fully, and only then acquires the state
        // machine lock in route_completions. It never holds both locks at once,
        // so calling backend.prefetch (which briefly locks the backend) while
        // holding the state machine lock cannot produce an AB-BA deadlock.
        inner.in_flight.insert(token, InFlightEntry { layer_idx: target_layer, slot });

        // Submit while holding the state machine lock. The dst borrow is
        // scoped to this block so it ends before the error-path remove below.
        let prefetch_result = {
            let dst = inner.in_flight[&token].slot.buffer();
            backend.prefetch(target_layer, 0, slot_len, dst, token)
        };

        if let Err(error) = prefetch_result {
            // Submission failed: remove the entry so the slot is returned to
            // the pool ring via PoolSlot::drop.
            inner.in_flight.remove(&token);
            return Err(error);
        }

        Ok(())
    }

    /// Submit prefetch requests, skipping layers that are already resident.
    ///
    /// `resident_fn` is a closure that returns `true` for layers that are in
    /// the hot-set (already in pinned RAM); those layers skip I/O entirely.
    pub fn on_layer_start_with_residency(
        &self,
        layer_idx: u32,
        direction: Direction,
        pool: &PoolRegistry,
        backend: &dyn IoBackend,
        resident_fn: impl Fn(u32) -> bool,
    ) -> Result<()> {
        {
            let (mutex, _) = self.shared.as_ref();
            let mut inner = mutex.lock().map_err(|_| {
                FlowCastError::BackendIo("state machine mutex poisoned".to_string())
            })?;
            inner.phase = match direction {
                Direction::Forward => Phase::Forward,
                Direction::Backward => Phase::Backward,
            };
            inner.direction = direction;
        }
        for target in self.prefetch_targets(layer_idx, direction) {
            if resident_fn(target) {
                // Layer already resident in pinned RAM — no I/O needed.
                continue;
            }
            self.submit_prefetch_for(target, pool, backend)?;
        }
        Ok(())
    }

    fn prefetch_targets(&self, current_layer: u32, direction: Direction) -> Vec<u32> {
        match direction {
            Direction::Forward => {
                let start = current_layer.saturating_add(1);
                let end = current_layer.saturating_add(self.lookahead);
                (start..=end).filter(|&j| j < self.total_layers).collect()
            }
            Direction::Backward => {
                let start = current_layer.saturating_sub(self.lookahead);
                (start..current_layer).rev().collect()
            }
        }
    }
}

/// Sprint 2 heuristic: map layer index to the appropriate pool ring.
///
/// Real routing requires `TensorLocationDict` (Sprint 3).
fn layer_kind_for(layer_idx: u32) -> LayerKind {
    match layer_idx % 4 {
        0 => LayerKind::Attention,
        1 => LayerKind::Mlp,
        2 => LayerKind::Norm,
        _ => LayerKind::Attention,
    }
}