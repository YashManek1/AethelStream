//! A1: Bidirectional prefetch state machine.
//!
//! Tracks which pass is active and maintains two direction-keyed maps shared
//! with the completion-router thread via `Arc<(Mutex<MachineInner>, Condvar)>`:
//! * `in_flight`: token → `InFlightEntry` for submitted, not-yet-done reads.
//! * `ready_forward` / `ready_backward`: layer_idx → `PoolSlot` for completed,
//!   direction-specific reads (API-j fix: separate maps prevent overwrite when
//!   the same layer_idx appears in both forward and backward windows).

use crate::backend::{Completion, IoBackend};
use crate::config::Precision;
use crate::decode::QuantizedDecoder;
use crate::ready::ReadyLayer;
use crate::{FlowCastError, Result};
use ramflow::phase::Direction;
use ramflow::pool::{LayerKind, PoolSlot};
use ramflow::PoolRegistry;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
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
        /// First layer (inclusive) of the recomputation window (A1-b fix).
        window_start: u32,
        /// Last layer (inclusive) of the recomputation window.
        window_end: u32,
    },
}

struct InFlightEntry {
    layer_idx: u32,
    /// Direction this read was submitted for, used by `route_completions` to
    /// insert into the correct ready map (API-j fix).
    direction: Direction,
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
    /// token → `InFlightEntry` for submitted, not-yet-completed reads.
    in_flight: HashMap<u64, InFlightEntry>,
    /// Forward-pass completions: layer_idx → PoolSlot.
    ///
    /// Kept separate from `ready_backward` to prevent silent overwrite when the
    /// same layer_idx appears in both windows.  `Direction` does not implement
    /// `Hash`, so a combined map is not possible without a wrapper.
    ready_forward: HashMap<u32, PoolSlot>,
    /// Backward-pass completions: layer_idx → PoolSlot.
    ready_backward: HashMap<u32, PoolSlot>,
}

impl MachineInner {
    /// Returns `true` if `layer_idx` already has a ready slot for `direction`.
    fn is_ready(&self, layer_idx: u32, direction: Direction) -> bool {
        match direction {
            Direction::Forward => self.ready_forward.contains_key(&layer_idx),
            Direction::Backward => self.ready_backward.contains_key(&layer_idx),
        }
    }

    /// Remove and return the ready slot for `layer_idx` and `direction`.
    fn take_slot(&mut self, layer_idx: u32, direction: Direction) -> Option<PoolSlot> {
        match direction {
            Direction::Forward => self.ready_forward.remove(&layer_idx),
            Direction::Backward => self.ready_backward.remove(&layer_idx),
        }
    }

    /// Insert a completed slot into the direction-appropriate map.
    fn insert_ready(&mut self, layer_idx: u32, direction: Direction, slot: PoolSlot) {
        match direction {
            Direction::Forward => { self.ready_forward.insert(layer_idx, slot); }
            Direction::Backward => { self.ready_backward.insert(layer_idx, slot); }
        }
    }
}

/// Per-layer precision, byte offset, and compressed byte-length from `TensorLocationDict`.
///
/// Populated from M1's shard index when available.  Key: `layer_idx →
/// (Precision, byte_offset, compressed_byte_length)`.
///
/// `byte_offset` is the 512-byte-aligned file offset required by `O_DIRECT`
/// (enforced by M1 at shard creation time).
type ShardIndex = HashMap<u32, (Precision, u64, u64)>;

/// Bidirectional prefetch state machine (Algorithm A1).
///
/// The `shared` field is an `Arc` cloned into the `CompletionRouter` thread
/// so both sides can access `in_flight` and `ready_*` maps under the same mutex.
pub struct PrefetchStateMachine {
    /// Interior-mutable shared state cloned into CompletionRouter.
    pub(crate) shared: Arc<(Mutex<MachineInner>, Condvar)>,
    /// Monotonically increasing token source (Relaxed: uniqueness is all that
    /// matters; no ordering relationship with other operations is required).
    next_token: AtomicU64,
    /// Total layers in the model.
    total_layers: u32,
    /// Fixed lookahead window (A2 makes this adaptive at runtime).
    lookahead: u32,
    /// Precision tag applied to returned [`ReadyLayer`]s when shard_index has
    /// no per-layer entry for the requested layer.
    default_precision: Precision,
    /// Optional per-layer precision and compressed byte-length loaded from M1.
    ///
    /// When present, `submit_prefetch_for` reads `compressed_byte_length`
    /// instead of the full pool-slot size (A7-b fix) and tags the resulting
    /// `ReadyLayer` with the per-layer precision (A7-a fix).
    shard_index: RwLock<ShardIndex>,
}

impl PrefetchStateMachine {
    /// Create a state machine for a model with `total_layers` layers.
    ///
    /// * `lookahead` — layers to prefetch ahead of the GPU.
    /// * `default_precision` — precision tag applied to `ReadyLayer`s when
    ///   no per-layer entry exists in `shard_index`.
    pub fn new(total_layers: u32, lookahead: u32, default_precision: Precision) -> Self {
        let inner = MachineInner {
            phase: Phase::Idle,
            direction: Direction::Forward,
            in_flight: HashMap::new(),
            ready_forward: HashMap::new(),
            ready_backward: HashMap::new(),
        };
        Self {
            shared: Arc::new((Mutex::new(inner), Condvar::new())),
            next_token: AtomicU64::new(1),
            total_layers,
            lookahead,
            default_precision,
            shard_index: RwLock::new(HashMap::new()),
        }
    }

    /// Populate per-layer precision, byte offset, and compressed byte-length from M1.
    ///
    /// Must be called before the first `prime_window` call when mixed-precision
    /// or INT4/INT8 layers are present.
    ///
    /// The tuple is `(precision, byte_offset, compressed_byte_length)`.
    /// `byte_offset` must be 512-byte aligned (enforced by M1).
    pub fn set_shard_index(&self, index: HashMap<u32, (Precision, u64, u64)>) {
        match self.shard_index.write() {
            Ok(mut guard) => *guard = index,
            Err(poison) => *poison.into_inner() = index,
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
    /// * [`FlowCastError::RamFlow`] — pool exhausted.
    /// * [`FlowCastError::BackendIo`] — I/O submission rejected.
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
            self.submit_prefetch_for(target, direction, pool, backend)?;
        }
        Ok(())
    }

    /// Notify that the GPU has begun executing `layer_idx`; submit next window.
    ///
    /// Delegates to [`on_layer_start_with_residency`] with a no-op residency
    /// function (A1-c2 fix: eliminates the duplicate code path).
    ///
    /// # Errors
    /// * [`FlowCastError::RamFlow`] — pool exhausted.
    /// * [`FlowCastError::BackendIo`] — I/O submission rejected.
    ///
    /// [`on_layer_start_with_residency`]: Self::on_layer_start_with_residency
    pub fn on_layer_start(
        &self,
        layer_idx: u32,
        direction: Direction,
        pool: &PoolRegistry,
        backend: &dyn IoBackend,
    ) -> Result<()> {
        self.on_layer_start_with_residency(layer_idx, direction, pool, backend, |_| false)
    }

    /// Block until `layer_idx` is resident in pinned RAM, then return it.
    ///
    /// Waits up to `timeout`. Returns [`FlowCastError::PrefetchMiss`] if the
    /// buffer does not arrive in time.
    ///
    /// # Errors
    /// * [`FlowCastError::PrefetchMiss`] — buffer not resident before timeout.
    /// * [`FlowCastError::BackendIo`] — mutex or condvar poisoned.
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

            let direction = guard.direction;
            if guard.is_ready(layer_idx, direction) {
                let mut guard = guard;
                let slot = guard
                    .take_slot(layer_idx, direction)
                    .ok_or(FlowCastError::PrefetchMiss { layer_idx })?;

                // Determine per-layer precision and decode requirement (A7-a/c).
                let (layer_precision, layer_needs_decode) = {
                    let index = self.shard_index.read().map_err(|_| {
                        FlowCastError::BackendIo("shard_index rwlock poisoned".to_string())
                    })?;
                    let precision = index
                        .get(&layer_idx)
                        .map(|(precision, _, _)| *precision)
                        .unwrap_or(self.default_precision);
                    let needs_decode = QuantizedDecoder::needs_decode(precision);
                    (precision, needs_decode)
                };

                return Ok(ReadyLayer {
                    layer_idx,
                    precision: layer_precision,
                    weight: slot,
                    slab_device_ptrs: Vec::new(),
                    needs_decode: layer_needs_decode,
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
    /// * [`FlowCastError::BackendIo`] — poll failed.
    pub fn poll_and_route(&self, backend: &dyn IoBackend) -> Result<u32> {
        let completions = backend.poll_completions()?;
        let count = completions.len() as u32;
        self.route_completions(completions);
        Ok(count)
    }

    /// Route a batch of backend completions into the direction-appropriate ready map.
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
        for completion in completions {
            if let Some(entry) = inner.in_flight.remove(&completion.token) {
                if completion.result >= 0 {
                    inner.insert_ready(entry.layer_idx, entry.direction, entry.slot);
                    any_ready = true;
                }
                // Negative result: slot returned to pool ring via Drop.
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

    /// Submit prefetch requests, skipping layers that are already resident.
    ///
    /// `resident_fn` returns `true` for layers resident in the hot-set (already
    /// in pinned RAM); those layers skip I/O entirely.
    ///
    /// This is the canonical implementation; [`on_layer_start`] delegates here
    /// with `resident_fn = |_| false` (A1-c2 fix).
    ///
    /// [`on_layer_start`]: Self::on_layer_start
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
                continue;
            }
            self.submit_prefetch_for(target, direction, pool, backend)?;
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

    fn submit_prefetch_for(
        &self,
        target_layer: u32,
        direction: Direction,
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

        // Determine the byte offset and compressed byte-length for this layer (C2 fix).
        // When shard_index provides offset + compressed size, use them so:
        //   - byte_offset points to the correct file position (O_DIRECT, 512-aligned).
        //   - read_len transfers only compressed bytes, not the full pool-slot size.
        let slot_len = slot.buffer().len() as u64;
        let (byte_offset, read_len) = {
            let index = self.shard_index.read().map_err(|_| {
                FlowCastError::BackendIo("shard_index rwlock poisoned".to_string())
            })?;
            index
                .get(&target_layer)
                .map(|(_, byte_offset, byte_len)| (*byte_offset, *byte_len))
                .unwrap_or_else(|| {
                    let len = match self.default_precision {
                        Precision::INT4 => slot_len / 4,
                        Precision::INT8 => slot_len / 2,
                        _ => slot_len,
                    };
                    (0u64, len)
                })
        };

        let (mutex, _) = self.shared.as_ref();
        let mut inner = mutex.lock().map_err(|_| {
            FlowCastError::BackendIo("state machine mutex poisoned".to_string())
        })?;

        // Re-check under the lock: a previous on_layer_start iteration may have
        // submitted this layer while we were claiming the slot outside.
        // Check BOTH ready maps and in_flight (API-j: direction-aware check).
        let already_tracked = inner
            .in_flight
            .values()
            .any(|entry| entry.layer_idx == target_layer)
            || inner.is_ready(target_layer, direction);
        if already_tracked {
            // slot returned to pool via Drop.
            return Ok(());
        }

        // Insert into in_flight BEFORE calling backend.prefetch.
        //
        // FileReadBackend completes synchronously: it reads the file and pushes
        // the completion into its pending queue inside prefetch(). The
        // CompletionRouter drains that queue every ~50 µs on a background
        // thread. If prefetch() were called first and in_flight.insert came
        // second, the router could drain the completion, fail to find the token
        // in in_flight, and silently discard it — leaving the layer stuck
        // forever and causing take_ready to time out.
        //
        // Lock-ordering safety: the router acquires the backend lock in
        // poll_completions, releases it fully, then acquires the state machine
        // lock in route_completions. It never holds both at once, so calling
        // backend.prefetch (briefly locks the backend) while holding the state
        // machine lock cannot produce an AB-BA deadlock.
        inner.in_flight.insert(
            token,
            InFlightEntry { layer_idx: target_layer, direction, slot },
        );

        let prefetch_result = {
            let dst = inner.in_flight[&token].slot.buffer();
            backend.prefetch(target_layer, byte_offset, read_len, dst, token)
        };

        if let Err(error) = prefetch_result {
            inner.in_flight.remove(&token);
            return Err(error);
        }

        Ok(())
    }
}

/// Map a layer index to the appropriate pool ring.
///
/// Heuristic: layer 0 is the embedding table (32 MiB ring); subsequent layers
/// alternate Attention / Mlp (64 MiB rings each).  Norm layers (tiny 1 MiB)
/// are not guessed here because we have no total_layers context — the caller
/// should pass a `TensorLocationDict`-derived shard_index when Norm shards are
/// present so `submit_prefetch_for` can select the correct ring via metadata.
fn layer_kind_for(layer_idx: u32) -> LayerKind {
    if layer_idx == 0 {
        LayerKind::Embedding
    } else if layer_idx % 2 == 1 {
        LayerKind::Attention
    } else {
        LayerKind::Mlp
    }
}
