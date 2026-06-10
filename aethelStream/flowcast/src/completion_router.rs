//! Completion router: drains the backend completion queue on a dedicated thread
//! and routes finished buffers into the state machine ready map.
//!
//! The router polls [`IoBackend::poll_completions`] every ~50 µs, forwarding
//! each batch to [`PrefetchStateMachine::route_completions`].  The condvar
//! inside the state machine wakes any caller blocked in `take_ready`.

use crate::backend::IoBackend;
use crate::state_machine::PrefetchStateMachine;
use crate::{FlowCastError, Result};
use std::sync::atomic::{AtomicBool, Ordering};
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
        let handle = thread::Builder::new()
            .name("flowcast-cqe-router".to_string())
            .spawn(move || router_loop(backend, state_machine, stop_clone))
            .map_err(|error| {
                FlowCastError::BackendIo(format!(
                    "failed to spawn completion-router thread: {error}"
                ))
            })?;
        Ok(Self {
            stop,
            thread: Some(handle),
        })
    }

    /// Stop the router thread and wait for it to exit.
    ///
    /// # Errors
    /// Returns [`FlowCastError::BackendIo`] if the thread panicked.
    pub fn shutdown(&mut self) -> Result<()> {
        self.stop.store(true, Ordering::Relaxed);
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
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

fn router_loop(
    backend: Arc<dyn IoBackend>,
    state_machine: Arc<PrefetchStateMachine>,
    stop: Arc<AtomicBool>,
) {
    while !stop.load(Ordering::Relaxed) {
        match backend.poll_completions() {
            Ok(completions) if !completions.is_empty() => {
                state_machine.route_completions(completions);
            }
            Ok(_) => {}
            Err(_) => {
                // Transient poll error -- continue without crashing the thread.
            }
        }
        thread::sleep(Duration::from_micros(50));
    }
}
