// src/nvme/prefetch.rs — NVMe prefetch engine (Sprint 2 Day 3)
//
// Implements:
//   1. SQE submission via io_uring (one SQE per prefetch call).
//   2. A background CQE poller thread pinned to a dedicated CPU core.
//   3. A crossbeam channel that delivers completion tokens to the caller.
//
// Architecture:
//
//   Training loop (Module 3)
//       │
//       ▼
//   PrefetchEngine::schedule(tensor_id, token)
//       │  checks pause_signal
//       │  looks up shard_id + byte_offset in TensorLocationDict
//       │  claims a PinnedBuffer from PoolRegistry
//       │  writes SQE to io_uring ring
//       │
//       ▼
//   io_uring ring (kernel-managed)
//       │  NVMe DMA → PinnedBuffer
//       │
//       ▼
//   CQE poller thread ("ramflow-cqe-poller", pinned to CPU core N)
//       │  reads CQE.user_data = token
//       │  sends token on completion_tx channel
//       │
//       ▼
//   Module 3 receives token on completion_rx
//       │  knows the buffer for this tensor is ready
//       │  can now schedule GPU transfer
//
// Token system:
//   Every prefetch call assigns a u64 token to the SQE's user_data field.
//   The kernel echoes user_data back in the CQE. This is the standard io_uring
//   mechanism for correlating completions with submissions in a lock-free way.

use crate::allocator::PinnedBuffer;
use crate::nvme::fd_table::FdTable;
use crate::nvme::io_uring_setup::IoUringInstance;
use crate::{RamFlowError, Result};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

// crossbeam_channel is already pulled in transitively; we use bounded channels.
// A bounded channel with capacity = ring size prevents the sender from racing
// ahead of the receiver — if the receiver falls behind, back-pressure kicks in
// and schedule() blocks rather than silently losing completions.
// ---------------------------------------------------------------------------
// Completion token type
// ---------------------------------------------------------------------------

/// A token that uniquely identifies one in-flight prefetch operation.
///
/// The same value is written to `SQE.user_data` by the submitter and read from
/// `CQE.user_data` by the poller. Callers define what the token means —
/// Module 3 uses it as a tensor_id or a (layer_idx, tensor_name) hash.
pub type PrefetchToken = u64;

// ---------------------------------------------------------------------------
// CqeResult — what the poller thread sends back
// ---------------------------------------------------------------------------

/// The result of a completed io_uring CQE.
#[derive(Debug)]
pub struct CqeResult {
    /// The token that was passed to `schedule()`.
    pub token: PrefetchToken,
    /// Bytes read (≥ 0) on success, or negative errno on failure.
    /// Module 3 should check `result >= 0` before using the buffer.
    pub result: i32,
}

// ---------------------------------------------------------------------------
// PrefetchEngine
// ---------------------------------------------------------------------------

/// Issues `io_uring_prep_read` SQEs for tensor prefetches and delivers
/// completion tokens via a channel.
///
/// # Thread model
/// - The engine itself is `Send` and can be shared via `Arc<PrefetchEngine>`.
/// - One background thread (the "cqe-poller") reads completions and sends
///   tokens. The caller receives tokens on `completion_rx`.
/// - Multiple training threads can call `schedule()` concurrently — the SQ
///   lock in the io_uring crate serializes submissions.
pub struct PrefetchEngine {
    /// The io_uring ring. Arc so the CQE poller thread can share it.
    ring: Arc<IoUringInstance>,

    /// Open file descriptors for each shard file.
    fd_table: Arc<FdTable>,

    /// Set to true by the co-scheduler when memory pressure is high.
    /// Checked before every SQE submission.
    pause_signal: Arc<AtomicBool>,

    /// Number of in-flight reads already submitted to the ring but not yet
    /// observed in CQE completion processing.
    outstanding_reads: Arc<AtomicUsize>,

    /// Number of currently claimed slots in pool management.
    /// The co-scheduler updates this value to account for pinned-memory pressure.
    claimed_slots: Arc<AtomicUsize>,

    /// Maximum safe pressure budget before new prefetch reads are paused.
    /// Condition: outstanding_reads + claimed_slots > pressure_threshold.
    pressure_threshold: Arc<AtomicUsize>,
}

// Manual Send impl: all fields are Send individually.
// PrefetchEngine crosses thread boundaries when passed to Arc::new().
unsafe impl Send for PrefetchEngine {}
unsafe impl Sync for PrefetchEngine {}

impl PrefetchEngine {
    /// Create a new PrefetchEngine.
    ///
    /// `pause_signal` — shared with `DirectNvmeEngine::set_pause()`.
    /// `fd_table` — must already have all shard files registered.
    pub fn new(
        ring: Arc<IoUringInstance>,
        fd_table: Arc<FdTable>,
        pause_signal: Arc<AtomicBool>,
        outstanding_reads: Arc<AtomicUsize>,
        claimed_slots: Arc<AtomicUsize>,
        pressure_threshold: Arc<AtomicUsize>,
    ) -> Result<Self> {
        Ok(PrefetchEngine {
            ring,
            fd_table,
            pause_signal,
            outstanding_reads,
            claimed_slots,
            pressure_threshold,
        })
    }

    /// Check whether a new prefetch may be submitted.
    ///
    /// Returns Err(PressurePause) if the co-scheduler has raised the
    /// pause signal. The caller should sleep for one layer's compute time
    /// and retry.
    fn check_pause(&self) -> Result<()> {
        if self.pause_signal.load(Ordering::Relaxed) {
            // layer_id=0 here; the full engine passes the real layer_id
            return Err(RamFlowError::PressurePause(0));
        }

        let outstanding = self.outstanding_reads.load(Ordering::Acquire);
        let claimed = self.claimed_slots.load(Ordering::Acquire);
        let threshold = self.pressure_threshold.load(Ordering::Acquire);

        if outstanding.saturating_add(claimed) > threshold {
            return Err(RamFlowError::PressurePause(0));
        }

        Ok(())
    }

    /// Update claimed slots count used by pressure gating.
    pub fn set_claimed_slots(&self, count: usize) {
        self.claimed_slots.store(count, Ordering::Release);
    }

    /// Update threshold used by pressure gating.
    pub fn set_pressure_threshold(&self, threshold: usize) {
        self.pressure_threshold.store(threshold, Ordering::Release);
    }

    /// Current in-flight read count tracked at SQE submit/CQE completion boundaries.
    pub fn outstanding_reads(&self) -> usize {
        self.outstanding_reads.load(Ordering::Acquire)
    }

    /// Returns true if a new submission is currently allowed by pressure gating.
    ///
    /// This evaluates both the explicit pause flag and the
    /// `outstanding_reads + claimed_slots > pressure_threshold` rule.
    pub fn submission_allowed(&self) -> bool {
        self.check_pause().is_ok()
    }

    /// Schedule a prefetch of `shard_id` from `byte_offset` for `length` bytes
    /// into `dst`. Associates `token` with the operation for completion matching.
    ///
    /// # O_DIRECT alignment
    /// Both `dst.as_ptr()` and `byte_offset` must be multiples of 512.
    /// PinnedBuffer satisfies the pointer requirement (posix_memalign(64),
    /// and 64 divides 512). Byte offsets must be verified by Module 1.
    ///
    /// # Returns
    /// Ok(()) if the SQE was submitted.
    /// Err(PressurePause) if the co-scheduler has paused prefetching.
    /// Err(IoUringError) if the SQE submission fails.
    #[cfg(target_os = "linux")]
    pub fn schedule(
        &self,
        shard_id: u32,
        byte_offset: u64,
        length: u64,
        dst: &PinnedBuffer,
        token: PrefetchToken,
    ) -> Result<()> {
        self.check_pause()?;

        use io_uring::{opcode, types};
        use std::os::unix::io::AsRawFd;

        let raw_fd = self.fd_table.get_raw_fd(shard_id).ok_or_else(|| {
            RamFlowError::ConfigError(format!("shard_id {shard_id} not registered in FdTable"))
        })?;

        // Prepare a read SQE:
        //   fd          — the O_DIRECT file descriptor for this shard
        //   buf         — the pinned destination buffer
        //   len         — bytes to read
        //   offset      — byte position within the shard file
        //   user_data   — echoed back in the CQE for token matching
        let read_op = opcode::Read::new(types::Fd(raw_fd), dst.as_ptr() as *mut u8, length as u32)
            .offset(byte_offset)
            .build()
            .user_data(token);

        // Push the SQE onto the submission ring.
        // SAFETY: The SQE references `dst` — the caller must keep `dst` alive
        // until the corresponding CQE is received. Module 3 enforces this by
        // holding the PoolSlot until the completion token arrives.
        self.ring.with_submission(|sq| {
            unsafe {
                sq.push(&read_op).map_err(|_| {
                    RamFlowError::IoUringError(std::io::Error::new(
                        std::io::ErrorKind::WouldBlock,
                        "io_uring submission queue full",
                    ))
                })?;
            }
            Ok(())
        })?;

        // Submit to the kernel (no-op in SQPOLL mode; issues enter() otherwise).
        self.ring.submit()?;
        self.outstanding_reads.fetch_add(1, Ordering::AcqRel);

        Ok(())
    }

    /// Non-Linux stub: io_uring not available.
    #[cfg(not(target_os = "linux"))]
    pub fn schedule(
        &self,
        _shard_id: u32,
        _byte_offset: u64,
        _length: u64,
        _dst: &PinnedBuffer,
        _token: PrefetchToken,
    ) -> Result<()> {
        self.check_pause()?;
        // On non-Linux, use a synchronous fallback read (Sprint 2 spec).
        // Full implementation deferred to Sprint 3's cross-platform path.
        todo!("Synchronous fallback read for non-Linux — Sprint 3")
    }
}

// ---------------------------------------------------------------------------
// CQE poller thread
// ---------------------------------------------------------------------------

/// Start the CQE poller thread for `engine`.
///
/// The thread:
///   1. Calls `ring.wait_for_cqe_timeout(1ms)` — blocks until a CQE arrives
///      or the timeout expires.
///   2. Drains all available CQEs, sending each token on `tx`.
///   3. Repeats until `stop_signal` is set.
///
/// Thread pinning: `pthread_setaffinity_np` forces the thread to a dedicated
/// CPU core. This prevents OS migration (which adds 10–100µs of latency when
/// the thread's cache state is lost).
///
/// # Parameters
/// - `engine`: shared reference to the engine owning the ring.
/// - `tx`: sender side of the completion channel.
/// - `stop_signal`: set to true to request graceful shutdown.
/// - `cpu_core`: logical CPU index to pin the thread to (0-indexed).
///
/// Returns the `JoinHandle` — the caller should join on shutdown.
pub fn spawn_cqe_poller(
    ring: Arc<IoUringInstance>,
    tx: std::sync::mpsc::SyncSender<CqeResult>,
    stop_signal: Arc<AtomicBool>,
    outstanding_reads: Arc<AtomicUsize>,
    cpu_core: usize,
) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name("ramflow-cqe-poller".into())
        // Stack size: 256 KB. The poller does no deep recursion.
        .stack_size(256 * 1024)
        .spawn(move || {
            // Pin this thread to `cpu_core` so the OS never migrates it.
            pin_thread_to_core(cpu_core);

            loop {
                // Exit cleanly when requested.
                if stop_signal.load(Ordering::Relaxed) {
                    break;
                }

                // Block for up to 1ms waiting for a CQE.
                // 1ms is short enough to check the stop_signal frequently
                // but long enough that busy-polling doesn't burn CPU.
                let _ = ring.wait_for_cqe_timeout(1);

                // Drain all available CQEs from the completion queue.
                #[cfg(target_os = "linux")]
                {
                    let drain_result = ring.with_completion(|cq| {
                        for cqe in cq {
                            let result = CqeResult {
                                token: cqe.user_data(),
                                result: cqe.result(),
                            };

                            // Completion means one in-flight ring entry has retired.
                            // Keep the counter saturating at zero if mismatch occurs.
                            let prior = outstanding_reads.load(Ordering::Acquire);
                            if prior > 0 {
                                outstanding_reads.fetch_sub(1, Ordering::AcqRel);
                            }

                            // send() blocks if the channel is full (back-pressure).
                            // This is intentional — the training loop should not
                            // submit more SQEs than the receiver can process.
                            if tx.send(result).is_err() {
                                // Receiver dropped: training is shutting down.
                                return Ok(());
                            }
                        }
                        Ok(())
                    });

                    if drain_result.is_err() {
                        // Ring error: continue loop and re-check stop signal.
                        continue;
                    }
                }
            }
        })
        .expect("failed to spawn ramflow-cqe-poller thread")
}

// ---------------------------------------------------------------------------
// Thread CPU pinning
// ---------------------------------------------------------------------------

/// Pin the calling thread to `cpu_core` using `pthread_setaffinity_np`.
///
/// On non-Linux platforms, this is a no-op (the OS will schedule the thread
/// wherever it wants, which is acceptable for development).
///
/// Why this matters:
///   When a thread migrates between CPU cores, all data in the previous
///   core's L1/L2 cache must be refetched. For the CQE poller, which spends
///   most of its time touching the io_uring completion ring, this causes a
///   cache miss storm every time migration occurs — typically adding 10–100µs
///   to the next completion detection.
fn pin_thread_to_core(cpu_core: usize) {
    #[cfg(target_os = "linux")]
    unsafe {
        // libc::cpu_set_t is a bitmask of CPUs. We set exactly one bit.
        let mut cpu_set = std::mem::zeroed::<libc::cpu_set_t>();
        libc::CPU_SET(cpu_core, &mut cpu_set);

        let rc = libc::pthread_setaffinity_np(
            libc::pthread_self(),
            std::mem::size_of::<libc::cpu_set_t>(),
            &cpu_set,
        );

        if rc != 0 {
            // Non-fatal: log and continue without pinning.
            // This can happen if cpu_core >= number of logical CPUs,
            // or if the process doesn't have CAP_SYS_NICE.
            eprintln!(
                "ramflow-cqe-poller: pthread_setaffinity_np(core={cpu_core}) \
                 failed with errno {rc}. Thread will not be pinned."
            );
        }
    }

    // On non-Linux: no-op.
    #[cfg(not(target_os = "linux"))]
    let _ = cpu_core;
}

// ---------------------------------------------------------------------------
// Integration test — Sprint 2 Test 4
// ---------------------------------------------------------------------------
//
// Creates a temp file with known pseudorandom bytes.
// Submits an io_uring read for a sub-range.
// Compares with std::fs::read (equivalent of fread).
//
// Run with: cargo test --features mock-cuda (Linux only for io_uring path)
// or:        cargo test --no-default-features --features mock-cuda

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nvme::io_uring_setup::IoUringParams;
    use std::io::Write;

    #[test]
    fn pressure_gate_blocks_when_outstanding_plus_claimed_exceeds_threshold() {
        let ring = Arc::new(IoUringInstance::setup(IoUringParams::default()).unwrap());
        let fd_table = Arc::new(FdTable::new().unwrap());
        let pause = Arc::new(AtomicBool::new(false));
        let outstanding = Arc::new(AtomicUsize::new(8));
        let claimed = Arc::new(AtomicUsize::new(9));
        let threshold = Arc::new(AtomicUsize::new(16));

        let engine =
            PrefetchEngine::new(ring, fd_table, pause, outstanding, claimed, threshold).unwrap();

        let decision = engine.check_pause();
        assert!(
            decision.is_err(),
            "pressure gate should block this submission"
        );
    }

    #[test]
    fn pressure_gate_allows_when_under_threshold() {
        let ring = Arc::new(IoUringInstance::setup(IoUringParams::default()).unwrap());
        let fd_table = Arc::new(FdTable::new().unwrap());
        let pause = Arc::new(AtomicBool::new(false));
        let outstanding = Arc::new(AtomicUsize::new(4));
        let claimed = Arc::new(AtomicUsize::new(5));
        let threshold = Arc::new(AtomicUsize::new(16));

        let engine =
            PrefetchEngine::new(ring, fd_table, pause, outstanding, claimed, threshold).unwrap();

        let decision = engine.check_pause();
        assert!(
            decision.is_ok(),
            "pressure gate should allow this submission"
        );
    }

    // Generate deterministic pseudorandom bytes (no rand crate dependency).
    fn pseudo_random_bytes(len: usize, seed: u64) -> Vec<u8> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                // xorshift64
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state & 0xFF) as u8
            })
            .collect()
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn integration_uring_read_matches_fread() {
        use crate::nvme::io_uring_setup::{IoUringInstance, IoUringParams};
        use std::fs::OpenOptions;
        use std::io::{Read, Seek, SeekFrom};
        use std::sync::mpsc;

        // --- Create a temp file with 4 MB of pseudorandom bytes ---
        const FILE_SIZE: usize = 4 * 1024 * 1024; // 4 MB
        let content = pseudo_random_bytes(FILE_SIZE, 0xDEAD_BEEF_1337_CAFE);

        let tmp_path =
            std::path::PathBuf::from(format!("/tmp/ramflow_uring_test_{}", std::process::id()));
        {
            let mut f = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&tmp_path)
                .expect("could not create temp file");
            f.write_all(&content).expect("write failed");
            f.sync_all().expect("fsync failed");
        }

        // --- Read range: bytes 512..2048 (512-byte aligned, as O_DIRECT requires) ---
        let read_offset: u64 = 512;
        let read_length: u64 = 1536; // 2048 - 512

        // --- Reference read via std::fs ---
        let expected = {
            let mut f = OpenOptions::new().read(true).open(&tmp_path).unwrap();
            f.seek(SeekFrom::Start(read_offset)).unwrap();
            let mut buf = vec![0u8; read_length as usize];
            f.read_exact(&mut buf).unwrap();
            buf
        };

        // --- io_uring read ---
        // Note: O_DIRECT requires the destination buffer to be 512-byte aligned.
        // PinnedBuffer uses posix_memalign(64). 64 divides 512, so the buffer
        // start is 64-byte aligned, which may not be 512-byte aligned.
        // For this test, we use a non-O_DIRECT open (the fd_table fallback on
        // Linux when not running as root) to keep the test hermetic.
        // Production use of O_DIRECT requires the NVMe device and root/CAP_SYS_RAWIO.
        let mut fd_table = FdTable::new().unwrap();
        // Register without O_DIRECT for test hermiticity.
        {
            use std::os::unix::io::{FromRawFd, IntoRawFd, OwnedFd};
            let f = OpenOptions::new().read(true).open(&tmp_path).unwrap();
            let raw = f.into_raw_fd();
            // We have to manually insert since register() uses O_DIRECT.
            // In production, register() is used; here we directly insert.
            // This is acceptable for the test: we're testing the io_uring
            // SQE/CQE path, not the O_DIRECT open.
            fd_table.fds.insert(0, unsafe { OwnedFd::from_raw_fd(raw) });
            fd_table.count = 1;
        }

        let ring = Arc::new(
            IoUringInstance::setup(IoUringParams {
                sq_entries: 16,
                cq_entries: 32,
                try_sqpoll: false, // don't require root for test
            })
            .expect("io_uring setup failed"),
        );

        let pause = Arc::new(AtomicBool::new(false));
        let engine = PrefetchEngine::new(
            ring.clone(),
            Arc::new(fd_table),
            pause,
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(usize::MAX)),
        )
        .unwrap();

        // Allocate destination buffer.
        // mock-cuda: PinnedBuffer::alloc still runs posix_memalign, so alignment is real.
        let dst = PinnedBuffer::alloc(read_length as usize).expect("PinnedBuffer::alloc failed");

        let token: PrefetchToken = 0xABCD_1234;

        engine
            .schedule(0, read_offset, read_length, &dst, token)
            .expect("schedule() failed");

        // Poll for completion manually (poller thread not started in this test).
        {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            loop {
                let mut completed = false;
                ring.with_completion(|cq| {
                    if let Some(cqe) = cq.next() {
                        assert_eq!(cqe.user_data(), token, "token mismatch");
                        assert!(cqe.result() > 0, "read failed: errno {}", -cqe.result());
                        completed = true;
                    }
                    Ok(())
                })
                .unwrap();
                if completed {
                    break;
                }
                if std::time::Instant::now() > deadline {
                    panic!("CQE not received within 5 seconds");
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }

        // Compare the read data with the reference.
        let actual = dst.as_slice();
        assert_eq!(
            actual,
            expected.as_slice(),
            "io_uring read data does not match fread() reference"
        );

        let _ = std::fs::remove_file(&tmp_path);
    }
}
