//! Completion router: drains the backend completion queue on a dedicated thread
//! and routes finished buffers into the state machine ready map.
//!
//! The router polls [`IoBackend::poll_completions`] every ~50 µs, forwarding
//! each batch to `PrefetchStateMachine::route_completions`.  The condvar
//! inside the state machine wakes any caller blocked in `take_ready`.

use crate::backend::IoBackend;
use crate::state_machine::PrefetchStateMachine;
use crate::{FlowCastError, Result};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

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
