// src/emergency.rs — opt-in emergency checkpoint signal hook.

use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::Duration;

use crate::{RamFlowError, Result};

type FlushCallback = dyn Fn() -> Result<()> + Send + Sync + 'static;

static REGISTERED: AtomicBool = AtomicBool::new(false);
static PENDING_SIGNAL: AtomicI32 = AtomicI32::new(0);
static FLUSH_CALLBACK: OnceLock<Mutex<Option<Arc<FlushCallback>>>> = OnceLock::new();

/// Guard returned by [`register_emergency_checkpoint`].
///
/// Dropping the guard disables the registered flush closure. On Unix, signal
/// handlers remain installed for the process lifetime but become inert once the
/// guard is dropped.
pub struct EmergencyCheckpointGuard {
    active: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}

impl Drop for EmergencyCheckpointGuard {
    fn drop(&mut self) {
        self.active.store(false, Ordering::Release);
        let _ = take_registered_flush();
        REGISTERED.store(false, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Register an opt-in emergency checkpoint flush hook.
///
/// The caller owns the actual checkpoint policy inside `flush`: it should submit
/// `DirectNvmeEngine::write_async` calls for the live layer buffers and wait
/// briefly for their completions. On Unix, SIGTERM/SIGINT trigger the closure
/// from a helper thread and then re-raise the original signal with the default
/// handler restored. On non-Unix targets this installs no OS signal handler,
/// but still records the closure for test-only simulated dispatch.
///
/// # Errors
///
/// Returns [`RamFlowError::ConfigError`] if a handler is already registered.
pub fn register_emergency_checkpoint<F>(flush: F) -> Result<EmergencyCheckpointGuard>
where
    F: Fn() -> Result<()> + Send + Sync + 'static,
{
    if REGISTERED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return Err(RamFlowError::ConfigError(
            "emergency checkpoint handler already registered".into(),
        ));
    }

    store_registered_flush(Arc::new(flush))?;

    let active = Arc::new(AtomicBool::new(true));

    #[cfg(unix)]
    install_signal_handlers();

    let worker = start_signal_worker(Arc::clone(&active))?;

    Ok(EmergencyCheckpointGuard {
        active,
        worker: Some(worker),
    })
}

fn callback_slot() -> &'static Mutex<Option<Arc<FlushCallback>>> {
    FLUSH_CALLBACK.get_or_init(|| Mutex::new(None))
}

fn store_registered_flush(callback: Arc<FlushCallback>) -> Result<()> {
    let mut slot = callback_slot()
        .lock()
        .map_err(|_| RamFlowError::ConfigError("emergency checkpoint mutex poisoned".into()))?;
    *slot = Some(callback);
    Ok(())
}

fn take_registered_flush() -> Option<Arc<FlushCallback>> {
    callback_slot()
        .lock()
        .map(|mut slot| slot.take())
        .unwrap_or(None)
}

fn load_registered_flush() -> Option<Arc<FlushCallback>> {
    callback_slot()
        .lock()
        .map(|slot| slot.as_ref().map(Arc::clone))
        .unwrap_or(None)
}

fn run_registered_flush() -> Result<bool> {
    let Some(callback) = load_registered_flush() else {
        return Ok(false);
    };
    callback()?;
    Ok(true)
}

fn start_signal_worker(active: Arc<AtomicBool>) -> crate::Result<thread::JoinHandle<()>> {
    thread::Builder::new()
        .name("ramflow-emergency-checkpoint".into())
        .spawn(move || {
            while active.load(Ordering::Acquire) {
                let signal = PENDING_SIGNAL.swap(0, Ordering::AcqRel);
                if signal == 0 {
                    thread::sleep(Duration::from_millis(10));
                    continue;
                }

                let _ = run_registered_flush();
                active.store(false, Ordering::Release);

                #[cfg(unix)]
                reraise_signal(signal);
            }
        })
        .map_err(crate::RamFlowError::IoUringError)
}

#[cfg(unix)]
fn install_signal_handlers() {
    // Safety: the handler only writes an atomic integer, which is the narrow
    // async-signal-safe handoff to the helper thread.
    unsafe {
        let handler = emergency_signal_handler as libc::sighandler_t;
        libc::signal(libc::SIGTERM, handler);
        libc::signal(libc::SIGINT, handler);
    }
}

#[cfg(unix)]
extern "C" fn emergency_signal_handler(signal: i32) {
    PENDING_SIGNAL.store(signal, Ordering::Release);
}

#[cfg(unix)]
fn reraise_signal(signal: i32) {
    // Safety: after the flush attempt, restore default disposition and re-raise
    // so the process exits with the original signal semantics.
    unsafe {
        libc::signal(signal, libc::SIG_DFL);
        libc::raise(signal);
    }
}

#[cfg(test)]
fn simulate_emergency_signal_for_test() -> Result<bool> {
    run_registered_flush()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn simulated_signal_path_invokes_registered_flush() {
        let flush_count = Arc::new(AtomicUsize::new(0));
        let flush_count_for_closure = Arc::clone(&flush_count);
        let guard = register_emergency_checkpoint(move || {
            flush_count_for_closure.fetch_add(1, Ordering::AcqRel);
            Ok(())
        })
        .expect("register emergency checkpoint");

        let invoked = simulate_emergency_signal_for_test().expect("simulate signal");

        assert!(invoked, "registered flush should run on simulated signal");
        assert_eq!(flush_count.load(Ordering::Acquire), 1);
        drop(guard);
    }
}
