//! Completion router: drains the backend completion queue on a dedicated thread
//! and routes finished buffers into the state machine ready map.
//!
//! The router polls [`IoBackend::poll_completions`] every ~50 µs, forwarding
//! each batch to `PrefetchStateMachine::route_completions`.  The condvar
//! inside the state machine wakes any caller blocked in `take_ready`.

use crate::backend::{BackendCapabilities, Completion, IoBackend};
use crate::state_machine::PrefetchStateMachine;
use crate::telemetry::Telemetry;
use crate::{FlowCastError, Result};
use ramflow::{CqeErrorKind, PinnedBuffer, classify_cqe_error};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Dedicated thread that routes I/O completions to the state machine.
///
/// Drop or call [`shutdown`] to stop the thread cleanly.
///
/// [`shutdown`]: CompletionRouter::shutdown
pub struct CompletionRouter {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    /// Cumulative count of transient poll errors observed by the router thread.
    ///
    /// Non-zero values indicate backend instability; exposed via [`error_count`].
    ///
    /// [`error_count`]: Self::error_count
    poll_errors: Arc<AtomicU64>,
}

impl CompletionRouter {
    /// Spawn the completion-router thread.
    ///
    /// The thread polls `backend.poll_completions()` every ~50 µs and routes
    /// each [`crate::backend::Completion`] to `state_machine`.
    ///
    /// # Errors
    /// Returns [`FlowCastError::BackendIo`] if the OS rejects the thread spawn.
    pub fn spawn(
        backend: Arc<dyn IoBackend>,
        state_machine: Arc<PrefetchStateMachine>,
    ) -> Result<Self> {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = stop.clone();
        let poll_errors = Arc::new(AtomicU64::new(0));
        let poll_errors_clone = Arc::clone(&poll_errors);
        let handle = thread::Builder::new()
            .name("flowcast-cqe-router".to_string())
            .spawn(move || router_loop(backend, state_machine, stop_clone, poll_errors_clone))
            .map_err(|error| {
                FlowCastError::BackendIo(format!(
                    "failed to spawn completion-router thread: {error}"
                ))
            })?;
        Ok(Self {
            stop,
            thread: Some(handle),
            poll_errors,
        })
    }

    /// Cumulative count of transient poll errors seen by the router thread.
    ///
    /// A non-zero value signals repeated backend failures; the training loop
    /// should surface this as a warning rather than ignoring it silently.
    pub fn error_count(&self) -> u64 {
        self.poll_errors.load(Ordering::Relaxed)
    }

    /// Stop the router thread and wait for it to exit.
    ///
    /// # Errors
    /// Returns [`FlowCastError::BackendIo`] if the thread panicked.
    pub fn shutdown(&mut self) -> Result<()> {
        self.stop.store(true, Ordering::Release);
        if let Some(handle) = self.thread.take() {
            handle.join().map_err(|_| {
                FlowCastError::BackendIo("completion-router thread panicked".to_string())
            })?;
        }
        Ok(())
    }
}

impl Drop for CompletionRouter {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

fn router_loop(
    backend: Arc<dyn IoBackend>,
    state_machine: Arc<PrefetchStateMachine>,
    stop: Arc<AtomicBool>,
    poll_errors: Arc<AtomicU64>,
) {
    while !stop.load(Ordering::Acquire) {
        match backend.poll_completions() {
            Ok(completions) if !completions.is_empty() => {
                state_machine.route_completions(completions);
            }
            Ok(_) => {}
            Err(_error) => {
                // Transient poll error — count for observability, then continue.
                // Panicking here would kill the router thread mid-training; the
                // counter lets the training loop detect sustained failure patterns.
                poll_errors.fetch_add(1, Ordering::Relaxed);
            }
        }
        thread::sleep(Duration::from_micros(50));
    }
}

// ===========================================================================
// CQE retry decorator
// ===========================================================================

/// I/O parameters needed to re-submit a failed prefetch read.
///
/// One entry is stored per in-flight token.  The `dst` pointer mirrors the
/// address inside the `InFlightEntry`'s `PoolSlot` in the state machine; it is
/// valid as long as the slot has not been freed (i.e. we have not yet forwarded
/// a negative completion for this token to `route_completions`).
pub struct PendingRead {
    /// Token originally supplied to `prefetch`.
    pub token: u64,
    /// Shard identifier (index into the model's shard directory).
    pub shard_id: u32,
    /// Byte offset within the shard file.
    pub byte_offset: u64,
    /// Number of bytes to read.
    pub length: u64,
    /// Raw destination pointer into the PinnedBuffer slot owned by the state
    /// machine.  Valid for the entire retry lifetime (see struct-level docs).
    pub dst: *mut u8,
    /// How many times this entry has already been re-submitted.
    pub retry_count: u8,
    /// Earliest wall-clock time at which a re-submission is allowed.
    ///
    /// `None` means "immediately" (first attempt or no backoff required).
    pub next_retry_at: Option<Instant>,
}

// SAFETY: `PendingRead` is accessed only while a `Mutex` is held.
// The raw pointer `dst` points into a `PoolSlot` that remains alive until we
// forward a terminal negative completion; until then no other thread frees it.
unsafe impl Send for PendingRead {}

/// Retry policy parameters sourced from [`crate::config::FlowCastConfig`].
pub struct RetryConfig {
    /// Maximum re-submission attempts before a Transient error is promoted to terminal.
    pub max_retries: u8,
    /// Base backoff quantum; `backoff(n) = 2^n × base_backoff_ms` milliseconds.
    pub base_backoff_ms: u64,
}

/// Exponential backoff: `2^retry_n × base_ms` milliseconds, saturating on overflow.
fn backoff(retry_n: u8, base_ms: u64) -> Duration {
    // Cap the shift to 63 to stay within u64 width before multiplying.
    let shift = u32::from(retry_n).min(63);
    Duration::from_millis(base_ms.saturating_mul(1u64 << shift))
}

/// [`IoBackend`] decorator that intercepts failed CQEs and retries transient errors.
///
/// Wraps any `Arc<dyn IoBackend>`.  Every `prefetch` call registers the I/O
/// parameters (shard_id, byte_offset, length, dst pointer) so they can be
/// re-submitted on transient failure.  `poll_completions` classifies negative
/// results via [`classify_cqe_error`]:
///
/// - **Transient** (`EAGAIN`/`EINTR`/`EBUSY`): queued for exponential-backoff
///   re-submission; the state-machine slot stays alive.
/// - **MediaError** (`EIO`/`ENODEV`) or **Unknown**: forwarded immediately
///   (negative `result`) so the state machine can free the slot.
pub struct CqeRetryBackend {
    inner: Arc<dyn IoBackend>,
    /// Map from token → pending I/O params for all in-flight reads.
    pending_reads: Mutex<HashMap<u64, PendingRead>>,
    /// Reads waiting for their backoff deadline before re-submission.
    retry_queue: Mutex<Vec<PendingRead>>,
    /// Retry policy from `FlowCastConfig`.
    config: RetryConfig,
    /// Shared telemetry counters (cloned from the `FlowCast` instance).
    telemetry: Telemetry,
}

impl CqeRetryBackend {
    /// Wrap `inner` with CQE retry logic using `config` and `telemetry`.
    pub fn new(inner: Arc<dyn IoBackend>, config: RetryConfig, telemetry: Telemetry) -> Self {
        Self {
            inner,
            pending_reads: Mutex::new(HashMap::new()),
            retry_queue: Mutex::new(Vec::new()),
            config,
            telemetry,
        }
    }

    /// Re-submit all retry-queue entries whose backoff deadline has passed.
    ///
    /// Lock ordering: always acquires `retry_queue` first, releases it, then
    /// acquires `pending_reads`.  This ordering is the inverse of
    /// `poll_completions`; the two locks are never held simultaneously.
    fn drain_retry_queue(&self) {
        let now = Instant::now();
        // Drain the queue while holding the lock; split into due / not-due.
        let due: Vec<PendingRead> = {
            let mut queue = self.retry_queue.lock().unwrap_or_else(|p| p.into_inner());
            let mut not_due = Vec::with_capacity(queue.len());
            let mut ready = Vec::new();
            for pending in queue.drain(..) {
                if pending.next_retry_at.is_none_or(|deadline| now >= deadline) {
                    ready.push(pending);
                } else {
                    not_due.push(pending);
                }
            }
            *queue = not_due;
            ready
        };
        // `retry_queue` lock is now released.  Submit each due entry.
        let mut requeued = Vec::new();
        for pending in due {
            // SAFETY: `pending.dst` is valid because the `InFlightEntry`'s
            // `PoolSlot` in the state machine is still alive — we have not
            // forwarded a terminal negative completion for this token, so
            // `route_completions` has not freed the slot.
            // `AllocKind::External` prevents `Drop` from deallocating the memory.
            let view = unsafe { PinnedBuffer::external_view(pending.dst, pending.length as usize) };
            if self.inner
                .prefetch(pending.shard_id, pending.byte_offset, pending.length, &view, pending.token)
                .is_ok()
            {
                let mut reads = self.pending_reads.lock().unwrap_or_else(|p| p.into_inner());
                reads.insert(pending.token, pending);
            } else {
                requeued.push(pending);
            }
        }
        if !requeued.is_empty() {
            let mut queue = self.retry_queue.lock().unwrap_or_else(|p| p.into_inner());
            queue.extend(requeued);
        }
    }

    /// Number of entries currently waiting in the retry queue.
    ///
    /// Used by integration tests in `test_cqe_retry.rs` to assert retry-queue
    /// state without inspecting private fields.  Not intended for production use.
    #[doc(hidden)]
    pub fn retry_queue_len(&self) -> usize {
        self.retry_queue
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .len()
    }

    /// Insert a [`PendingRead`] directly (bypasses `prefetch`).
    ///
    /// Allows integration tests to simulate in-flight reads without a real
    /// io_uring ring.  Not intended for production use.
    #[doc(hidden)]
    pub fn insert_pending_for_test(&self, pending: PendingRead) {
        self.pending_reads
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(pending.token, pending);
    }
}

impl IoBackend for CqeRetryBackend {
    fn start(&mut self) -> Result<()> {
        // The inner backend was started by the caller before wrapping.
        Ok(())
    }

    fn prefetch(
        &self,
        shard_id: u32,
        byte_offset: u64,
        length: u64,
        dst: &PinnedBuffer,
        token: u64,
    ) -> Result<()> {
        {
            let mut reads = self.pending_reads.lock().unwrap_or_else(|p| p.into_inner());
            reads.insert(
                token,
                PendingRead {
                    token,
                    shard_id,
                    byte_offset,
                    length,
                    // SAFETY: The memory behind `dst` is mutable (posix_memalign /
                    // mmap) and will remain valid until the PoolSlot is freed.
                    // We cast *const u8 → *mut u8 because as_ptr() takes &self;
                    // the underlying allocation has write permission throughout.
                    dst: dst.as_ptr() as *mut u8,
                    retry_count: 0,
                    next_retry_at: None,
                },
            );
        }
        self.inner.prefetch(shard_id, byte_offset, length, dst, token)
    }

    fn write_async(
        &self,
        shard_id: u32,
        byte_offset: u64,
        length: u64,
        src: &PinnedBuffer,
        token: u64,
    ) -> Result<()> {
        self.inner.write_async(shard_id, byte_offset, length, src, token)
    }

    /// Drain retries, poll the inner backend, then classify and route completions.
    ///
    /// Lock ordering: `pending_reads` is acquired and **released** before
    /// `retry_queue` is acquired for the retry-push at the end, preventing
    /// deadlock with `drain_retry_queue` (which acquires in the opposite order).
    fn poll_completions(&self) -> Result<Vec<Completion>> {
        self.drain_retry_queue();
        let completions = self.inner.poll_completions()?;
        let mut ok: Vec<Completion> = Vec::with_capacity(completions.len());
        let mut to_retry: Vec<PendingRead> = Vec::new();
        {
            let mut reads = self.pending_reads.lock().unwrap_or_else(|p| p.into_inner());
            for comp in completions {
                if comp.result >= 0 {
                    reads.remove(&comp.token);
                    ok.push(comp);
                } else {
                    match classify_cqe_error(comp.result) {
                        CqeErrorKind::Transient => {
                            if let Some(mut pending) = reads.remove(&comp.token) {
                                if pending.retry_count < self.config.max_retries {
                                    pending.retry_count += 1;
                                    pending.next_retry_at = Some(
                                        Instant::now()
                                            + backoff(
                                                pending.retry_count,
                                                self.config.base_backoff_ms,
                                            ),
                                    );
                                    self.telemetry.record_cqe_retry();
                                    to_retry.push(pending);
                                    // Slot stays alive; do NOT push to `ok`.
                                    continue;
                                }
                            }
                            // Budget exhausted or no pending entry found.
                            self.telemetry.record_media_error();
                            ok.push(comp);
                        }
                        CqeErrorKind::MediaError | CqeErrorKind::Unknown(_) => {
                            reads.remove(&comp.token);
                            self.telemetry.record_media_error();
                            ok.push(comp);
                        }
                    }
                }
            }
        }
        // `pending_reads` lock is released.  Now push retries without risk of
        // lock-order inversion with `drain_retry_queue`.
        if !to_retry.is_empty() {
            let mut queue = self.retry_queue.lock().unwrap_or_else(|p| p.into_inner());
            queue.extend(to_retry);
        }
        Ok(ok)
    }

    fn is_paused(&self) -> bool {
        self.inner.is_paused()
    }

    fn set_pause(&self, paused: bool) {
        self.inner.set_pause(paused);
    }

    fn capabilities(&self) -> BackendCapabilities {
        self.inner.capabilities()
    }

    /// Shut down the inner backend.
    ///
    /// Requires that all other `Arc` clones of the inner backend have been
    /// dropped (the router thread must have exited first).
    fn shutdown(&mut self) -> Result<()> {
        if let Some(inner) = Arc::get_mut(&mut self.inner) {
            inner.shutdown()?;
        }
        Ok(())
    }
}
