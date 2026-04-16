// src/nvme/io_uring_setup.rs — io_uring ring initialisation (Sprint 2 Day 2)
//
// Wraps the io-uring crate's ring type and exposes the two things the rest of
// ramflow needs:
//   1. An initialized ring with SQPOLL if the kernel supports it (≥ 5.11).
//   2. The ability to submit SQEs and wait for CQEs.
//
// The io-uring crate (version 0.7, in Cargo.toml) maps directly to the
// io_uring(2) kernel interface. We use it rather than shelling out to liburing
// because:
//   a) No C dependency in the build graph.
//   b) Ownership and lifetimes are tracked by Rust.
//   c) The crate is already in Cargo.lock.
//
// SQPOLL mode explanation:
//   Without SQPOLL: submitting an SQE requires io_uring_enter() syscall.
//     At 80 layers × ~10 tensors each × 2 directions = 1600 syscalls/step.
//   With SQPOLL: a kernel thread polls the SQ. Userspace writes to shared
//     memory and the kernel thread picks it up. Zero syscalls per submission.
//     The kernel thread sleeps after IORING_SQ_NEED_WAKEUP ms of idle time.
//
//   SQPOLL requires Linux 5.11+ for stable behavior (CAP_SYS_NICE or
//   IORING_SETUP_SQ_AFF). We probe availability at runtime and fall back
//   to the standard mode gracefully.

use crate::{RamFlowError, Result};
use std::io;
#[cfg(all(target_os = "linux", feature = "io-uring-use-split"))]
use std::sync::Mutex;

// io-uring crate imports — Linux only.
// On non-Linux, this module provides stubs so the crate compiles.
#[cfg(target_os = "linux")]
use io_uring::IoUring;

// ---------------------------------------------------------------------------
// IoUringParams
// ---------------------------------------------------------------------------

/// Parameters for initializing an io_uring ring.
pub struct IoUringParams {
    /// Number of submission-queue entries.
    /// 128 is the default for AethelStream: at most ~80 in-flight reads
    /// + headroom for writes and metadata ops.
    pub sq_entries: u32,
    /// Number of completion-queue entries.
    /// Should be ≥ sq_entries. Typically 2× to avoid CQ overflow when
    /// completions arrive faster than the poller drains them.
    pub cq_entries: u32,
    /// Enable SQPOLL mode if the kernel supports it.
    /// Requires CAP_SYS_NICE or root on kernels < 5.11.
    pub try_sqpoll: bool,
}

impl Default for IoUringParams {
    fn default() -> Self {
        Self {
            sq_entries: 128,
            cq_entries: 256, // 2× sq_entries
            try_sqpoll: true,
        }
    }
}

// ---------------------------------------------------------------------------
// IoUringInstance
// ---------------------------------------------------------------------------

/// Owns a configured io_uring instance.
///
/// On Linux with the real io-uring crate, this wraps `io_uring::IoUring`.
/// On non-Linux, it is a no-op stub so the crate compiles cross-platform.
pub struct IoUringInstance {
    #[cfg(all(target_os = "linux", not(feature = "io-uring-use-split")))]
    inner: IoUring,

    #[cfg(all(target_os = "linux", feature = "io-uring-use-split"))]
    inner: Mutex<IoUring>,

    /// Whether SQPOLL was successfully enabled at construction.
    /// Callers can log this for diagnostics.
    pub sqpoll_active: bool,
}

impl IoUringInstance {
    /// Initialize an io_uring ring with the given parameters.
    ///
    /// Attempts SQPOLL if `params.try_sqpoll` is true and falls back to
    /// standard mode if the kernel rejects the flag (e.g., older kernel,
    /// insufficient privileges).
    pub fn setup(params: IoUringParams) -> Result<Self> {
        #[cfg(target_os = "linux")]
        {
            // --- Attempt SQPOLL first ---
            if params.try_sqpoll {
                let sqpoll_result = IoUring::builder()
                    .setup_sqpoll(2000) // kernel thread idles after 2000ms
                    .build(params.sq_entries);

                if let Ok(ring) = sqpoll_result {
                    return Ok(IoUringInstance {
                        #[cfg(not(feature = "io-uring-use-split"))]
                        inner: ring,
                        #[cfg(feature = "io-uring-use-split")]
                        inner: Mutex::new(ring),
                        sqpoll_active: true,
                    });
                }
                // SQPOLL failed (insufficient privilege, old kernel, etc.)
                // Fall through to standard mode.
                eprintln!(
                    "ramflow: SQPOLL io_uring unavailable (needs CAP_SYS_NICE \
                     or Linux ≥ 5.11). Falling back to standard mode."
                );
            }

            // --- Standard mode ---
            let ring = IoUring::builder()
                .build(params.sq_entries)
                .map_err(|e| RamFlowError::IoUringError(e))?;

            Ok(IoUringInstance {
                #[cfg(not(feature = "io-uring-use-split"))]
                inner: ring,
                #[cfg(feature = "io-uring-use-split")]
                inner: Mutex::new(ring),
                sqpoll_active: false,
            })
        }

        #[cfg(not(target_os = "linux"))]
        {
            // io_uring is Linux-only. On other platforms we return a stub.
            // DirectNvmeEngine uses cfg! guards to route to sync fallbacks.
            Ok(IoUringInstance {
                sqpoll_active: false,
            })
        }
    }

    /// Submit all pending SQEs to the kernel.
    ///
    /// In SQPOLL mode, the kernel thread picks up SQEs from the ring
    /// automatically — submit() is still called but may be a no-op.
    /// In standard mode, this issues the io_uring_enter() syscall.
    ///
    /// Returns the number of SQEs submitted.
    #[cfg(target_os = "linux")]
    pub fn submit(&self) -> Result<usize> {
        #[cfg(feature = "io-uring-use-split")]
        {
            let guard = self
                .inner
                .lock()
                .map_err(|_| RamFlowError::ConfigError("io_uring mutex poisoned".into()))?;
            return guard.submit().map_err(RamFlowError::IoUringError);
        }

        #[cfg(not(feature = "io-uring-use-split"))]
        self.inner.submit().map_err(RamFlowError::IoUringError)
    }

    #[cfg(not(target_os = "linux"))]
    pub fn submit(&self) -> Result<usize> {
        Ok(0)
    }

    /// Run a closure with mutable access to the submission queue.
    ///
    /// Default path uses `submission_shared` for lowest overhead.
    /// Fallback path (`io-uring-use-split`) uses `split()` under a mutex.
    #[cfg(target_os = "linux")]
    pub(crate) fn with_submission<T, F>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&mut io_uring::SubmissionQueue<'_>) -> Result<T>,
    {
        #[cfg(feature = "io-uring-use-split")]
        {
            let mut guard = self
                .inner
                .lock()
                .map_err(|_| RamFlowError::ConfigError("io_uring mutex poisoned".into()))?;
            let (_submitter, mut sq, _cq) = guard.split();
            let out = f(&mut sq)?;
            sq.sync();
            return Ok(out);
        }

        #[cfg(not(feature = "io-uring-use-split"))]
        {
            // SAFETY: caller serializes access through engine-level invariants.
            let mut sq = unsafe { self.inner.submission_shared() };
            f(&mut sq)
        }
    }

    /// Run a closure with mutable access to the completion queue.
    ///
    /// Default path uses `completion_shared`.
    /// Fallback path (`io-uring-use-split`) uses `split()` under a mutex.
    #[cfg(target_os = "linux")]
    pub(crate) fn with_completion<T, F>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&mut io_uring::CompletionQueue<'_>) -> Result<T>,
    {
        #[cfg(feature = "io-uring-use-split")]
        {
            let mut guard = self
                .inner
                .lock()
                .map_err(|_| RamFlowError::ConfigError("io_uring mutex poisoned".into()))?;
            let (_submitter, _sq, mut cq) = guard.split();
            return f(&mut cq);
        }

        #[cfg(not(feature = "io-uring-use-split"))]
        {
            // SAFETY: caller serializes access through engine-level invariants.
            let mut cq = unsafe { self.inner.completion_shared() };
            f(&mut cq)
        }
    }

    /// Wait for at least one CQE with a timeout.
    ///
    /// Used by the CQE poller thread. Returns immediately if a CQE is already
    /// available; blocks for at most `timeout_ms` milliseconds otherwise.
    #[cfg(target_os = "linux")]
    pub fn wait_for_cqe_timeout(&self, timeout_ms: u64) -> Result<()> {
        let ts = io_uring::types::Timespec::new()
            .sec(0)
            .nsec((timeout_ms * 1_000_000) as u32);

        #[cfg(feature = "io-uring-use-split")]
        {
            let guard = self
                .inner
                .lock()
                .map_err(|_| RamFlowError::ConfigError("io_uring mutex poisoned".into()))?;
            return guard
                .submitter()
                .wait_with_timeout(&ts)
                .map_err(RamFlowError::IoUringError);
        }

        #[cfg(not(feature = "io-uring-use-split"))]
        {
            self.inner
                .submitter()
                .wait_with_timeout(&ts)
                .map_err(RamFlowError::IoUringError)
        }
    }

    #[cfg(not(target_os = "linux"))]
    pub fn wait_for_cqe_timeout(&self, _timeout_ms: u64) -> Result<()> {
        Ok(())
    }
}
