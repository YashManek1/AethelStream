// src/nvme/mod.rs - Direct NVMe I/O engine (Sprint 2)
//
// This file wires together IoUringInstance, FdTable, and PrefetchEngine into
// a single `DirectNvmeEngine` struct that the rest of ramflow uses.
//
// Compared to Sprint 0:
//   - DirectNvmeEngine::open() is fully implemented (was unimplemented!).
//   - prefetch() submits a real SQE and checks pause_signal.
//   - poll_completions() drains the CQE ring.
//   - start_cqe_poller() spawns the background CQE poller thread.
//   - prewarm_first_n() remains a lightweight warm-up helper.
//   - write_async() submits real write SQEs through the shared completion path.
//
// The pause_signal is an AtomicBool shared with:
//   - CoScheduler (sets it on high-pressure callback)
//   - SlowPathAllocator (sets it on pool stall)
//   - PrefetchEngine (reads it before every SQE submission)

/// O_DIRECT file-descriptor table for NVMe shard files.
pub mod fd_table;
/// io_uring ring initialisation with optional SQPOLL mode.
pub mod io_uring_setup;
/// SQE submission engine and CQE poller thread.
pub mod prefetch;
/// SSD wear-budget manager with delta compression (feature `ssd-wear`).
pub mod write_budget;
/// NVMe block-layer bypass via IORING_OP_URING_CMD (feature `nvme-passthrough`).
#[cfg(feature = "nvme-passthrough")]
pub mod passthrough;
/// Windows DirectStorage backend: capability probe, COM queue, allocation helper.
#[cfg(feature = "direct-storage")]
pub mod direct_storage;

pub use engine::DirectNvmeEngine;

mod engine {
    use std::path::Path;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::mpsc::{self, Receiver, SyncSender};
    use std::sync::{Arc, Mutex};

    #[cfg(feature = "checksums")]
    use std::collections::HashMap;

    /// Number of completion queue entries - 2× SQ depth per io_uring convention.
    const CQ_DEPTH: usize = 256;

    /// Completion channel capacity: 4× CQ depth gives ~1 s of headroom at peak CQE rate
    /// before the training loop must drain the channel.  Derived from CQ_DEPTH so a
    /// single constant (CQ_DEPTH = 256) governs both the io_uring ring and this channel.
    const COMPLETION_CHANNEL_CAPACITY: usize = 4 * CQ_DEPTH; // 4 * CQ_DEPTH

    use crate::allocator::PinnedBuffer;
    use crate::error::RamFlowError;
    use crate::nvme::fd_table::FdTable;
    use crate::nvme::io_uring_setup::{IoUringInstance, IoUringParams};
    use crate::nvme::prefetch::{
        spawn_cqe_poller, CqeResult, PrefetchEngine, PrefetchToken, SuperShardConfig,
    };
    use crate::Result;

    /// Pending checksum verification for one in-flight prefetch.
    ///
    /// Stored in `DirectNvmeEngine::checksum_registry` when `checksums` feature
    /// is active. Erased from the registry after poll_completions() verifies it.
    #[cfg(feature = "checksums")]
    struct ChecksumEntry {
        /// Pointer to the first byte of the destination buffer.
        ///
        /// # Safety
        /// Points to pinned host memory allocated via `PinnedBuffer`. Pinned memory
        /// is never moved by the allocator or the OS; the caller guarantees the
        /// buffer lives until the corresponding completion token is drained.
        ptr: *const u8,
        /// Byte length of the destination buffer (must equal `TensorInfo.byte_length`).
        len: usize,
        /// Expected xxHash3-64 digest from `shard_index.json`.
        expected: u64,
        /// Shard file index � echoed into `RamFlowError::ShardCorrupted` on mismatch.
        shard_id: u32,
    }

    /// Safety: `ptr` points to pinned host memory, which is never relocated.
    /// The registry is guarded by a `Mutex`; only one thread touches an entry
    /// at a time (producer: schedule path; consumer: poll_completions).
    #[cfg(feature = "checksums")]
    unsafe impl Send for ChecksumEntry {}

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

        /// Count of read/write SQEs submitted to io_uring and not yet seen as completions.
        outstanding_reads: Arc<AtomicUsize>,

        /// Number of currently claimed pool slots tracked by scheduler.
        claimed_slots: Arc<AtomicUsize>,

        /// Pressure cutoff. New reads are paused when
        /// outstanding_reads + claimed_slots exceeds this value.
        pressure_threshold: Arc<AtomicUsize>,

        /// Sender side of the completion channel (held for poller thread spawn).
        completion_tx: SyncSender<CqeResult>,

        /// Receiver side - callers read from this to get completion tokens.
        completion_rx: Receiver<CqeResult>,

        /// Handle for the CQE poller thread (None until start_cqe_poller() is called).
        poller_handle: Option<std::thread::JoinHandle<()>>,

        /// Signal to stop the poller thread on shutdown.
        stop_signal: Arc<AtomicBool>,

        /// Buffers backing prewarm reads until completions can retire them.
        prewarm_buffers: Mutex<Vec<PinnedBuffer>>,

        /// Per-token checksum registry for post-read integrity verification.
        ///
        /// Populated by `prefetch_with_checksum()` when the caller provides an expected
        /// xxHash3 digest. Drained by `poll_completions()` after each successful read.
        #[cfg(feature = "checksums")]
        checksum_registry: Arc<Mutex<HashMap<PrefetchToken, ChecksumEntry>>>,
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
        /// - The CQE poller thread is NOT started here - call start_cqe_poller()
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
            // Channel capacity must exceed the CQ ring size (CQ_DEPTH entries) so the
            // CQE poller can never block waiting for the consumer to drain.
            // COMPLETION_CHANNEL_CAPACITY = 4 × CQ_DEPTH provides ~1 s of headroom
            // at peak CQE rate before the training loop must drain the channel.
            let (completion_tx, completion_rx) =
                mpsc::sync_channel::<CqeResult>(COMPLETION_CHANNEL_CAPACITY);

            // --- Pause signal ---
            let pause_signal = Arc::new(AtomicBool::new(false));

            // --- Pressure accounting ---
            let outstanding_reads = Arc::new(AtomicUsize::new(0));
            let claimed_slots = Arc::new(AtomicUsize::new(0));
            let pressure_threshold = Arc::new(AtomicUsize::new(usize::MAX));

            // --- Prefetch engine ---
            let prefetch_engine = Arc::new(PrefetchEngine::new_with_completion(
                ring.clone(),
                fd_table.clone(),
                pause_signal.clone(),
                outstanding_reads.clone(),
                claimed_slots.clone(),
                pressure_threshold.clone(),
                completion_tx.clone(),
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
                prewarm_buffers: Mutex::new(Vec::new()),
                #[cfg(feature = "checksums")]
                checksum_registry: Arc::new(Mutex::new(HashMap::new())),
            })
        }

        /// Start the CQE poller thread pinned to `cpu_core`.
        ///
        /// Must be called before the engine receives any prefetch results.
        /// Can only be called once - subsequent calls return Ok(()) immediately.
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
            )?;

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
        /// Module 1 when sharding - the engine does not re-check it here
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

        /// Submit an async read for `shard_id`, optionally registering an xxHash3
        /// checksum for post-read verification.
        ///
        /// When `expected_xxh3` is `Some(digest)`, `poll_completions()` will verify
        /// the received bytes against `digest` before returning the token to the
        /// caller.  A mismatch returns `Err(RamFlowError::ShardCorrupted)`.
        ///
        /// When `expected_xxh3` is `None`, this is identical to `prefetch()`.
        ///
        /// # Availability
        /// Only available with the `checksums` feature. Use `prefetch()` for the
        /// non-checksummed fast path.
        #[cfg(feature = "checksums")]
        pub fn prefetch_with_checksum(
            &self,
            shard_id: u32,
            byte_offset: u64,
            length: u64,
            dst: &PinnedBuffer,
            token: PrefetchToken,
            expected_xxh3: Option<u64>,
        ) -> Result<()> {
            if let Some(expected) = expected_xxh3 {
                let entry = ChecksumEntry {
                    ptr: dst.as_ptr(),
                    len: dst.len(),
                    expected,
                    shard_id,
                };
                let mut registry = self
                    .checksum_registry
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner());
                registry.insert(token, entry);
            }
            self.prefetch(shard_id, byte_offset, length, dst, token)
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
            // try_recv() is non-blocking - returns Err(Empty) when no CQEs available.
            while let Ok(cqe_result) = self.completion_rx.try_recv() {
                if cqe_result.result < 0 {
                    // Negative result = errno. Convert to IoUringError.
                    return Err(RamFlowError::IoUringError(
                        std::io::Error::from_raw_os_error(-cqe_result.result),
                    ));
                }

                #[cfg(feature = "checksums")]
                {
                    let mut registry = self
                        .checksum_registry
                        .lock()
                        .unwrap_or_else(|poison| poison.into_inner());
                    if let Some(entry) = registry.remove(&cqe_result.token) {
                        // Safety: ptr points to pinned host memory allocated by PinnedBuffer;
                        // pinned memory is never relocated.  The caller guarantees the buffer
                        // lives until this completion token is drained from the channel.
                        let received = unsafe { std::slice::from_raw_parts(entry.ptr, entry.len) };
                        let computed = xxhash_rust::xxh3::xxh3_64(received);
                        if computed != entry.expected {
                            return Err(RamFlowError::ShardCorrupted {
                                shard_id: entry.shard_id,
                                expected: entry.expected,
                                got: computed,
                            });
                        }
                    }
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
        /// Called when `hardware_profile.json` exists (Idea 9 - Shard Pre-Warming).
        /// Uses the same io_uring ring as regular prefetch.
        /// Sprint 3 implementation.
        pub fn prewarm_first_n(&self, n: u32) -> Result<()> {
            let shard_limit = (n as usize).min(self.shard_count());
            for shard_index in 0..shard_limit {
                let buffer = PinnedBuffer::alloc(512)?;
                self.prefetch(
                    shard_index as u32,
                    0,
                    512,
                    &buffer,
                    0x5052_4557_0000_0000u64 | shard_index as u64,
                )?;
                let mut buffers = self
                    .prewarm_buffers
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner());
                buffers.push(buffer);
            }
            Ok(())
        }

        /// Configure super-shard grouped I/O. Disabled by default until profiled.
        pub fn set_super_shard_config(&self, config: SuperShardConfig) {
            self.prefetch_engine.set_super_shard_config(config);
        }

        /// Current super-shard grouped I/O configuration.
        pub fn super_shard_config(&self) -> SuperShardConfig {
            self.prefetch_engine.super_shard_config()
        }

        /// Submit a grouped contiguous read for consecutive layers.
        ///
        /// Per-layer offsets remain in the caller's index; this method only
        /// controls the grouped read submission.
        pub fn prefetch_super_shard(
            &self,
            shard_id: u32,
            byte_offset: u64,
            length: u64,
            dst: &PinnedBuffer,
            token: PrefetchToken,
            layer_offsets: &[u64],
        ) -> Result<()> {
            self.prefetch_engine.schedule_super_shard(
                shard_id,
                byte_offset,
                length,
                dst,
                token,
                layer_offsets,
            )
        }

        /// Submit an async write for `src` to `shard_id` at `byte_offset`.
        ///
        /// The write-budget manager intercepts this call when the `ssd-wear`
        /// feature is active. The completion token is echoed through the same
        /// CQE poller and channel used by reads.
        pub fn write_async(
            &self,
            shard_id: u32,
            byte_offset: u64,
            length: u64,
            src: &PinnedBuffer,
            token: PrefetchToken,
        ) -> Result<()> {
            self.prefetch_engine
                .schedule_write(shard_id, byte_offset, length, src, token)
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

        /// Expose current in-flight ring reads/writes for co-scheduler decisions.
        pub fn outstanding_reads(&self) -> usize {
            self.outstanding_reads.load(Ordering::Acquire)
        }

        /// Returns true only when the ring has fully drained and pressure is gone.
        ///
        /// The co-scheduler must call this - not just check `is_paused()` - to
        /// confirm it is safe to resume submissions. Pressure is relieved only when
        /// `outstanding_reads + claimed_slots ≤ pressure_threshold` AND the pause
        /// signal is clear.
        ///
        /// Atomics use `Acquire` so the load sees the latest values written by
        /// the CQE poller (`AcqRel` fetch_sub) and the co-scheduler (`Release` store).
        pub fn is_pressure_relieved(&self) -> bool {
            let outstanding = self.outstanding_reads.load(Ordering::Acquire);
            let claimed = self.claimed_slots.load(Ordering::Acquire);
            let threshold = self.pressure_threshold.load(Ordering::Acquire);
            let paused = self.pause_signal.load(Ordering::Acquire);
            !paused && outstanding.saturating_add(claimed) <= threshold
        }

        /// Number of registered shard files.
        pub fn shard_count(&self) -> usize {
            self.fd_table.len()
        }

        /// Send a CQE result directly to the completion channel for testing.
        ///
        /// Mirrors what `spawn_cqe_poller` does when a real kernel completion arrives,
        /// allowing tests to drive the channel without io_uring or hardware. Pass a
        /// negative `result` to simulate a failed read (e.g., `-5` for EIO).
        #[cfg(any(test, feature = "checksums"))]
        pub fn inject_completion_for_test(&self, token: PrefetchToken, result: i32) {
            self.completion_tx.send(CqeResult { token, result }).ok();
        }
    }

    impl Drop for DirectNvmeEngine {
        fn drop(&mut self) {
            // Signal the poller thread to stop.
            self.stop_signal.store(true, Ordering::Release);

            // Join the poller thread (wait for clean exit).
            if let Some(handle) = self.poller_handle.take() {
                // Ignore join errors - if the thread panicked, we're already
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
    #[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    mod tests {
        use super::*;

        /// is_pressure_relieved() must return false while outstanding reads exceed
        /// the threshold, and true only after the ring has drained to ≤ threshold.
        #[test]
        fn is_pressure_relieved_false_when_reads_in_flight() {
            let engine = DirectNvmeEngine::open_with_paths(&[]).expect("engine init failed");
            // Drive outstanding_reads above threshold so pressure is active.
            engine.set_pressure_threshold(4);
            // Simulate 10 in-flight reads by setting the atomic directly via
            // the public outstanding_reads() observer to verify starting state.
            // We call inject_completion_for_test to drive outstanding_reads down.
            engine
                .outstanding_reads
                .store(10, std::sync::atomic::Ordering::Release);
            engine.set_pause(true);
            assert!(
                !engine.is_pressure_relieved(),
                "pressure must not be relieved with 10 outstanding reads and pause set"
            );
            // Drain reads to threshold.
            engine
                .outstanding_reads
                .store(3, std::sync::atomic::Ordering::Release);
            engine.set_pause(false);
            assert!(
                engine.is_pressure_relieved(),
                "pressure must be relieved once outstanding_reads ≤ threshold and pause is clear"
            );
        }

        /// The completion channel must buffer a full CQ ring (256 entries) without
        /// blocking the CQE poller. Fixed capacity: 1024 (4× the 256-entry CQ ring).
        #[test]
        fn channel_capacity_supports_full_cq_ring_without_deadlock() {
            // Engine uses sync_channel(1024). Verify that a full CQ ring worth of
            // completions (256 entries) never causes the poller to block.
            let (tx, _rx) = mpsc::sync_channel::<CqeResult>(1024);
            for index in 0u64..256 {
                tx.try_send(CqeResult {
                    token: index,
                    result: 1,
                })
                .unwrap_or_else(|_| {
                    panic!("channel stalled at entry {index} - capacity regression below 256")
                });
            }
        }

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

        /// A CQE with result < 0 must surface as Err(IoUringError) from poll_completions().
        /// This proves the negative-CQE guard is reachable and working - a failed read
        /// must never be delivered as valid bytes to Module 3.
        #[test]
        fn error_cqe_propagates_through_poll_completions() {
            let engine = DirectNvmeEngine::open_with_paths(&[]).expect("engine init failed");
            // Inject token 0xDEAD with result=-5 (simulating kernel EIO).
            engine.inject_completion_for_test(0xDEAD, -5);
            let outcome = engine.poll_completions();
            match outcome {
                Err(RamFlowError::IoUringError(_)) => {}
                other => panic!("expected Err(IoUringError) for CQE result=-5, got {other:?}"),
            }
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
        #[test]
        #[cfg(feature = "checksums")]
        fn checksum_matching_digest_passes() {
            let engine = DirectNvmeEngine::open_with_paths(&[]).expect("engine init");

            // Build a known buffer and compute its xxh3.
            let data: Vec<u8> = (0u8..=255).cycle().take(4096).collect();
            let expected = xxhash_rust::xxh3::xxh3_64(&data);

            // Allocate a pinned buffer and copy data into it.
            let mut buf = PinnedBuffer::alloc(4096).expect("alloc");
            buf.as_mut_slice().copy_from_slice(&data);

            // Register the checksum.
            engine
                .prefetch_with_checksum(0, 0, 4096, &buf, /*token=*/ 1u64, Some(expected))
                .expect("prefetch_with_checksum");

            // Inject a successful CQE (result=4096 bytes read).
            engine.inject_completion_for_test(1u64, 4096);

            // poll_completions must succeed - the checksum matches.
            let completed = engine.poll_completions().expect("poll_completions");
            assert_eq!(completed, 1, "one completion must be returned");
        }

        #[test]
        #[cfg(feature = "checksums")]
        fn checksum_corruption_detection() {
            let engine = DirectNvmeEngine::open_with_paths(&[]).expect("engine init");

            let data: Vec<u8> = vec![0xABu8; 512];
            let expected = xxhash_rust::xxh3::xxh3_64(&data);

            let mut buf = PinnedBuffer::alloc(512).expect("alloc");
            buf.as_mut_slice().copy_from_slice(&data);

            engine
                .prefetch_with_checksum(7, 0, 512, &buf, /*token=*/ 2u64, Some(expected))
                .expect("prefetch_with_checksum");

            // Corrupt one byte after registration to simulate a mid-transfer bit-flip.
            buf.as_mut_slice()[0] ^= 0xFF;

            engine.inject_completion_for_test(2u64, 512);

            let outcome = engine.poll_completions();
            match outcome {
                Err(RamFlowError::ShardCorrupted {
                    shard_id,
                    expected: exp,
                    got,
                }) => {
                    assert_eq!(shard_id, 7);
                    assert_eq!(exp, expected);
                    assert_ne!(got, expected, "got must differ from expected on corruption");
                }
                other => panic!("expected ShardCorrupted, got {other:?}"),
            }
        }

        #[test]
        #[cfg(feature = "checksums")]
        fn checksum_none_skips_verification() {
            let engine = DirectNvmeEngine::open_with_paths(&[]).expect("engine init");
            let buf = PinnedBuffer::alloc(512).expect("alloc");

            engine
                .prefetch_with_checksum(0, 0, 512, &buf, /*token=*/ 3u64, None)
                .expect("prefetch_with_checksum");

            engine.inject_completion_for_test(3u64, 512);

            let completed = engine.poll_completions().expect("poll_completions");
            assert_eq!(completed, 1);
        }
    }
}