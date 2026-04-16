// src/nvme/mod.rs — Direct NVMe I/O engine (Sprint 2)
//
// This file wires together IoUringInstance, FdTable, and PrefetchEngine into
// a single `DirectNvmeEngine` struct that the rest of ramflow uses.
//
// Compared to Sprint 0:
//   - DirectNvmeEngine::open() is fully implemented (was unimplemented!).
//   - prefetch() submits a real SQE and checks pause_signal.
//   - poll_completions() drains the CQE ring.
//   - start_cqe_poller() spawns the background CQE poller thread.
//   - prewarm_first_n() and write_async() remain stubbed (Sprint 3+).
//
// The pause_signal is an AtomicBool shared with:
//   - CoScheduler (sets it on high-pressure callback)
//   - SlowPathAllocator (sets it on pool stall)
//   - PrefetchEngine (reads it before every SQE submission)

pub mod fd_table;
pub mod io_uring_setup;
pub mod prefetch;
pub mod write_budget;

pub use engine::DirectNvmeEngine;

mod engine {
    use std::path::Path;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::mpsc::{self, Receiver, SyncSender};
    use std::sync::Arc;

    use crate::allocator::PinnedBuffer;
    use crate::error::RamFlowError;
    use crate::nvme::fd_table::FdTable;
    use crate::nvme::io_uring_setup::{IoUringInstance, IoUringParams};
    use crate::nvme::prefetch::{spawn_cqe_poller, CqeResult, PrefetchEngine, PrefetchToken};
    use crate::Result;

    // -----------------------------------------------------------------------
    // DirectNvmeEngine
    // -----------------------------------------------------------------------

    /// Zero-syscall NVMe engine built on io_uring.
    ///
    /// Owns the io_uring ring, file descriptor table, prefetch engine,
    /// CQE poller thread, and the completion channel.
    ///
    /// # Usage
    /// ```ignore
    /// let engine = DirectNvmeEngine::open(Path::new("/mnt/shards"), n_shards)?;
    /// engine.start_cqe_poller(cpu_core=2)?;
    ///
    /// // In the training loop:
    /// engine.prefetch(shard_id, byte_offset, length, &dst_buf, token)?;
    ///
    /// // Receive completions:
    /// let result = engine.completion_rx().recv().unwrap();
    /// assert!(result.result > 0);
    /// ```
    pub struct DirectNvmeEngine {
        /// Shared io_uring ring (shared with the CQE poller thread).
        ring: Arc<IoUringInstance>,

        /// Open file descriptors for each shard.
        fd_table: Arc<FdTable>,

        /// Core prefetch submission logic.
        prefetch_engine: Arc<PrefetchEngine>,

        /// Set to `true` by the co-scheduler's high-pressure callback.
        /// Checked before every SQE submission.
        pause_signal: Arc<AtomicBool>,

        /// Count of reads submitted to io_uring and not yet seen as completions.
        outstanding_reads: Arc<AtomicUsize>,

        /// Number of currently claimed pool slots tracked by scheduler.
        claimed_slots: Arc<AtomicUsize>,

        /// Pressure cutoff. New reads are paused when
        /// outstanding_reads + claimed_slots exceeds this value.
        pressure_threshold: Arc<AtomicUsize>,

        /// Sender side of the completion channel (held for poller thread spawn).
        completion_tx: SyncSender<CqeResult>,

        /// Receiver side — callers read from this to get completion tokens.
        completion_rx: Receiver<CqeResult>,

        /// Handle for the CQE poller thread (None until start_cqe_poller() is called).
        poller_handle: Option<std::thread::JoinHandle<()>>,

        /// Signal to stop the poller thread on shutdown.
        stop_signal: Arc<AtomicBool>,
    }

    impl DirectNvmeEngine {
        /// Initialize the engine for a directory of shard files.
        ///
        /// Opens `n_shards` files named `shard_0000.bin` through
        /// `shard_{n_shards-1:04}.bin` inside `shard_dir`.
        ///
        /// # Sprint 2 implementation notes
        /// - Files opened with O_RDONLY | O_DIRECT | O_CLOEXEC on Linux.
        /// - io_uring ring sized to 128 SQEs / 256 CQEs.
        /// - SQPOLL attempted first; falls back to standard mode if unavailable.
        /// - The CQE poller thread is NOT started here — call start_cqe_poller()
        ///   after construction once you know which CPU core to pin it to.
        ///
        /// # Alternative: use open_with_paths() if shard paths don't follow
        /// the default naming scheme.
        pub fn open(shard_dir: &Path, n_shards: u32) -> Result<Self> {
            // --- Build the shard file paths ---
            let mut shard_paths: Vec<std::path::PathBuf> = Vec::new();
            for i in 0..n_shards {
                let filename = format!("shard_{i:04}.bin");
                shard_paths.push(shard_dir.join(filename));
            }

            let path_refs: Vec<&Path> = shard_paths.iter().map(|p| p.as_path()).collect();
            Self::open_with_paths(&path_refs)
        }

        /// Initialize the engine with an explicit list of shard file paths.
        ///
        /// Shard IDs are assigned sequentially (index in `paths` = shard_id).
        pub fn open_with_paths(paths: &[&Path]) -> Result<Self> {
            // --- Initialize io_uring ring ---
            let ring = Arc::new(IoUringInstance::setup(IoUringParams {
                sq_entries: 128,
                cq_entries: 256,
                try_sqpoll: true,
            })?);

            // --- Open all shard files ---
            let mut fd_table = FdTable::new()?;
            fd_table.register_all(paths)?;
            let fd_table = Arc::new(fd_table);

            // --- Completion channel ---
            // Bounded to ring size: prevents the poller from racing far ahead
            // of the consumer. If full, the poller blocks (back-pressure).
            let (completion_tx, completion_rx) = mpsc::sync_channel::<CqeResult>(128);

            // --- Pause signal ---
            let pause_signal = Arc::new(AtomicBool::new(false));

            // --- Pressure accounting ---
            let outstanding_reads = Arc::new(AtomicUsize::new(0));
            let claimed_slots = Arc::new(AtomicUsize::new(0));
            let pressure_threshold = Arc::new(AtomicUsize::new(usize::MAX));

            // --- Prefetch engine ---
            let prefetch_engine = Arc::new(PrefetchEngine::new(
                ring.clone(),
                fd_table.clone(),
                pause_signal.clone(),
                outstanding_reads.clone(),
                claimed_slots.clone(),
                pressure_threshold.clone(),
            )?);

            Ok(DirectNvmeEngine {
                ring,
                fd_table,
                prefetch_engine,
                pause_signal,
                outstanding_reads,
                claimed_slots,
                pressure_threshold,
                completion_tx,
                completion_rx,
                poller_handle: None,
                stop_signal: Arc::new(AtomicBool::new(false)),
            })
        }

        /// Start the CQE poller thread pinned to `cpu_core`.
        ///
        /// Must be called before the engine receives any prefetch results.
        /// Can only be called once — subsequent calls return Ok(()) immediately.
        pub fn start_cqe_poller(&mut self, cpu_core: usize) -> Result<()> {
            if self.poller_handle.is_some() {
                return Ok(());
            }

            let handle = spawn_cqe_poller(
                self.ring.clone(),
                self.completion_tx.clone(),
                self.stop_signal.clone(),
                self.outstanding_reads.clone(),
                cpu_core,
            );

            self.poller_handle = Some(handle);
            Ok(())
        }

        /// Submit an async `io_uring_prep_read` for `shard_id`.
        ///
        /// Returns `Err(PressurePause)` immediately if `pause_signal` is set.
        /// The `token` is echoed back via the completion channel when the
        /// read completes.
        ///
        /// # O_DIRECT alignment requirement
        /// `byte_offset` must be a multiple of 512. This is enforced by
        /// Module 1 when sharding — the engine does not re-check it here
        /// (checking would add a branch to the critical path).
        pub fn prefetch(
            &self,
            shard_id: u32,
            byte_offset: u64,
            length: u64,
            dst: &PinnedBuffer,
            token: PrefetchToken,
        ) -> Result<()> {
            // pause_signal check is inside PrefetchEngine::schedule().
            // We delegate entirely to the prefetch engine.
            self.prefetch_engine
                .schedule(shard_id, byte_offset, length, dst, token)
        }

        /// Drain the completion channel and return all available results.
        ///
        /// Non-blocking: returns immediately with whatever completions are
        /// already available. The CQE poller thread continuously populates
        /// the channel in the background.
        ///
        /// Returns the number of CQEs processed.
        pub fn poll_completions(&self) -> Result<u32> {
            let mut count = 0u32;
            // try_recv() is non-blocking — returns Err(Empty) when no CQEs available.
            while let Ok(cqe_result) = self.completion_rx.try_recv() {
                // In Sprint 2, we just count completions.
                // Sprint 3 adds routing to the layer readiness state machine.
                if cqe_result.result < 0 {
                    // Negative result = errno. Convert to IoUringError.
                    return Err(RamFlowError::IoUringError(
                        std::io::Error::from_raw_os_error(-cqe_result.result),
                    ));
                }
                count += 1;
            }
            Ok(count)
        }

        /// Get a reference to the completion receiver for blocking receives.
        ///
        /// Module 3 can use this to block on a specific token:
        /// ```ignore
        /// loop {
        ///     let result = engine.completion_rx().recv().unwrap();
        ///     if result.token == expected_token { break; }
        ///     // else re-queue or dispatch to the right handler
        /// }
        /// ```
        pub fn completion_rx(&self) -> &Receiver<CqeResult> {
            &self.completion_rx
        }

        /// Pre-warm the pool by issuing background reads for the first `n` shards.
        ///
        /// Called when `hardware_profile.json` exists (Idea 9 — Shard Pre-Warming).
        /// Uses the same io_uring ring as regular prefetch.
        /// Sprint 3 implementation.
        #[allow(unused_variables)]
        pub fn prewarm_first_n(&self, n: u32) -> Result<()> {
            unimplemented!("DirectNvmeEngine::prewarm_first_n — Sprint 3 stub")
        }

        /// Submit an async write for `buf` to `shard_id` at `byte_offset`.
        ///
        /// The write-budget manager intercepts this call when the `ssd-wear`
        /// feature is active. Sprint 3 implementation.
        #[allow(unused_variables)]
        pub fn write_async(
            &self,
            shard_id: u32,
            byte_offset: u64,
            buf: &PinnedBuffer,
        ) -> Result<()> {
            unimplemented!("DirectNvmeEngine::write_async — Sprint 3 stub")
        }

        /// Set the pause signal (called by the co-scheduler's high-pressure
        /// callback, or directly by the slow-path allocator on stall).
        pub fn set_pause(&self, paused: bool) {
            self.pause_signal.store(paused, Ordering::Release);
        }

        /// Current pause state.
        pub fn is_paused(&self) -> bool {
            self.pause_signal.load(Ordering::Acquire)
        }

        /// Update scheduler-tracked claimed slot count used in pressure gating.
        pub fn set_claimed_slots(&self, count: usize) {
            self.claimed_slots.store(count, Ordering::Release);
            self.prefetch_engine.set_claimed_slots(count);
        }

        /// Configure pressure threshold used for prefetch admission control.
        pub fn set_pressure_threshold(&self, threshold: usize) {
            self.pressure_threshold.store(threshold, Ordering::Release);
            self.prefetch_engine.set_pressure_threshold(threshold);
        }

        /// Expose current in-flight ring reads for co-scheduler decisions.
        pub fn outstanding_reads(&self) -> usize {
            self.outstanding_reads.load(Ordering::Acquire)
        }

        /// Number of registered shard files.
        pub fn shard_count(&self) -> usize {
            self.fd_table.len()
        }
    }

    impl Drop for DirectNvmeEngine {
        fn drop(&mut self) {
            // Signal the poller thread to stop.
            self.stop_signal.store(true, Ordering::Release);

            // Join the poller thread (wait for clean exit).
            if let Some(handle) = self.poller_handle.take() {
                // Ignore join errors — if the thread panicked, we're already
                // in a degraded state and there is nothing useful to do here.
                let _ = handle.join();
            }

            // FdTable's Drop will close all file descriptors.
            // IoUringInstance's Drop will clean up the ring.
            // These happen automatically when the Arc reference counts reach 0.
        }
    }

    // =======================================================================
    // Tests
    // =======================================================================

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn set_pause_roundtrip() {
            // We can test pause_signal without a real disk or GPU.
            let pause = Arc::new(AtomicBool::new(false));
            assert!(!pause.load(Ordering::Relaxed));
            pause.store(true, Ordering::Release);
            assert!(pause.load(Ordering::Acquire));
            pause.store(false, Ordering::Release);
            assert!(!pause.load(Ordering::Acquire));
        }

        #[test]
        fn completion_channel_send_recv() {
            // Verify the channel mechanics work without any io_uring.
            let (tx, rx) = mpsc::sync_channel::<CqeResult>(4);
            tx.send(CqeResult {
                token: 42,
                result: 1024,
            })
            .unwrap();
            tx.send(CqeResult {
                token: 99,
                result: 2048,
            })
            .unwrap();

            let a = rx.recv().unwrap();
            assert_eq!(a.token, 42);
            assert_eq!(a.result, 1024);

            let b = rx.recv().unwrap();
            assert_eq!(b.token, 99);
        }
    }
}
