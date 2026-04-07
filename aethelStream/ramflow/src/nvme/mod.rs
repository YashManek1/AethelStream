// src/nvme/mod.rs — Direct NVMe I/O engine (Algorithm 7 / io_uring)
//
// Sprint 0: all types declared, all logic unimplemented!.
//
// Key design points for Sprint 2 implementors:
//   - pause_signal is an AtomicBool set by the co-scheduler via callback.
//     prefetch() checks it BEFORE submitting each SQE.  A true value causes
//     an immediate Err(PressurePause) so the prefetcher enters a wait loop.
//   - The CQE poller thread is named "ramflow-cqe-poller" and pinned to a
//     dedicated CPU core (pthread_setaffinity_np).
//   - O_DIRECT requires: (a) buffer address aligned to 512 bytes — satisfied
//     by posix_memalign(64); (b) byte_offset multiple of 512 — must be
//     guaranteed by Module 1's sharding step.

pub mod fd_table;
pub mod io_uring_setup;
pub mod prefetch;
pub mod write_budget;

pub use engine::DirectNvmeEngine;

mod engine {
    use std::sync::atomic::{AtomicBool, Ordering};
    use crate::Result;
    use crate::error::RamFlowError;

    /// Zero-syscall NVMe engine built on io_uring.
    ///
    /// Owns the submission/completion ring pair, the open file-descriptor
    /// table, and the `pause_signal` atomic used by the co-scheduler.
    ///
    /// # Sprint 0 contract
    /// Compiles; every method `unimplemented!`.
    pub struct DirectNvmeEngine {
        /// Set to `true` by the co-scheduler's high-pressure callback.
        /// Checked before every SQE submission.
        _pause_signal: AtomicBool,
    }

    impl DirectNvmeEngine {
        /// Initialise the engine against the NVMe device at `path`.
        ///
        /// Opens all shard files with `O_RDONLY | O_DIRECT | O_CLOEXEC`.
        /// Sets up the io_uring ring (SQPOLL if kernel ≥ 5.11).
        /// Spawns and pins the CQE poller thread.
        #[allow(unused_variables)]
        pub fn open(path: &std::path::Path) -> Result<Self> {
            unimplemented!("DirectNvmeEngine::open — Sprint 0 stub")
        }

        /// Submit an async `io_uring_prep_read` for `shard_id`.
        ///
        /// Returns `Err(PressurePause)` immediately if `pause_signal` is set.
        /// The `token` is echoed back via the CQE channel so the prefetcher
        /// can match completions to pending requests.
        #[allow(unused_variables)]
        pub fn prefetch(
            &self,
            shard_id: u32,
            byte_offset: u64,
            length: u64,
            dst: &crate::allocator::PinnedBuffer,
            token: u64,
        ) -> Result<()> {
            if self._pause_signal.load(Ordering::Relaxed) {
                return Err(RamFlowError::IoUringError(
                    std::io::Error::new(
                        std::io::ErrorKind::WouldBlock,
                        "co-scheduler pressure pause"
                    )
                ));
            }
            unimplemented!("DirectNvmeEngine::prefetch — Sprint 0 stub")
        }

        /// Drain the completion queue and invoke registered callbacks.
        ///
        /// Returns the number of CQEs processed in this call.
        pub fn poll_completions(&self) -> Result<u32> {
            unimplemented!("DirectNvmeEngine::poll_completions — Sprint 0 stub")
        }

        /// Pre-warm the pool by issuing background reads for the first `n`
        /// shards. (Idea 9 — Shard Pre-Warming on Resume)
        ///
        /// Called when `hardware_profile.json` exists, concurrently with
        /// checkpoint loading so it costs zero wall-clock time.
        /// Uses the same io_uring ring as regular prefetch, so it takes
        /// whatever ring capacity is available without blocking training.
        #[allow(unused_variables)]
        pub fn prewarm_first_n(&self, n: u32) -> Result<()> {
            unimplemented!("DirectNvmeEngine::prewarm_first_n — Sprint 0 stub")
        }

        /// Submit an async write for `buf` to `shard_id` at `byte_offset`.
        ///
        /// The write-budget manager intercepts this call when the `ssd-wear`
        /// feature is active.  Without `ssd-wear`, the write proceeds directly.
        #[allow(unused_variables)]
        pub fn write_async(
            &self,
            shard_id: u32,
            byte_offset: u64,
            buf: &crate::allocator::PinnedBuffer,
        ) -> Result<()> {
            unimplemented!("DirectNvmeEngine::write_async — Sprint 0 stub")
        }

        /// Set the pause signal (called by the co-scheduler's high-pressure
        /// callback, or directly by the slow-path allocator on stall).
        pub fn set_pause(&self, paused: bool) {
            self._pause_signal.store(paused, Ordering::Release);
        }

        /// Current pause state.
        pub fn is_paused(&self) -> bool {
            self._pause_signal.load(Ordering::Acquire)
        }
    }
}
