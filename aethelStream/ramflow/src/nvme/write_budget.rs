// src/nvme/write_budget.rs — SSD write-budget manager (Idea 4)
//
// Feature-gated: #[cfg(feature = "ssd-wear")]
//
// When `ssd-wear` is INACTIVE: WriteBudgetManager is a transparent no-op —
//   consume() always returns Ok(()), remaining() always returns u64::MAX,
//   enqueue_write() passes through immediately, force_flush() does nothing.
//
// When `ssd-wear` is ACTIVE:
//   - Reads SMART "Data Units Written" at startup via ioctl on /dev/nvme0
//     (Linux) or falls back to zero on other platforms.
//   - Tracks cumulative SMART units written throughout the run.
//   - Switches WriteStrategy automatically as the budget is consumed:
//       Full          → while remaining > 50% of budget_units
//       DeltaCompress → while remaining ≤ 50% and > 10%
//       Deferred{4}   → while remaining ≤ 10%
//   - Batches Deferred writes: flushes once batch_size entries accumulate.
//   - Delta-compression: stores layer_{idx:04d}.delta.zstd alongside the
//     original shard; Module 1's ShardLoader applies it on load.
//
// ─── NVMe SMART Unit Convention ───────────────────────────────────────────
//
//   NVMe SMART/Health Information Log page (Log ID 0x02) reports
//   "Data Units Written" at byte offset 48 as a 128-bit LE integer.
//   Per the NVMe specification 2.0, Section 6.1.4:
//     1 data unit = 1 000 × 512-byte LBA sectors = 512 000 bytes.
//   We track budgets and consumption in these SMART units throughout.
//   SMART_UNIT_BYTES = 512_000.

#[cfg(feature = "ssd-wear")]
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
#[cfg(feature = "ssd-wear")]
use std::sync::atomic::Ordering;
#[cfg(feature = "ssd-wear")]
use std::sync::Mutex;

/// NVMe SMART "Data Units Written" unit size in bytes.
///
/// Per NVMe Base Specification 2.0, Section 6.1.4:
///   1 data unit = 1 000 × 512-byte LBA sectors = 512 000 bytes.
pub const SMART_UNIT_BYTES: u64 = 512_000;

// ---------------------------------------------------------------------------
// WriteStrategy
// ---------------------------------------------------------------------------

/// Controls how updated layer weights are persisted to NVMe.
///
/// Selected automatically by [`WriteBudgetManager`] based on remaining budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteStrategy {
    /// Write the full updated tensor shard.  Highest write amplification.
    Full,
    /// Compute `delta = updated - original`, compress with zstd, write delta.
    ///
    /// Module 1's ShardLoader detects `layer_{idx:04d}.delta.zstd` and applies
    /// it on load.  Typical compression ratio: 8–15× at standard learning rates.
    DeltaCompress,
    /// Accumulate `batch_size` layers before submitting a write batch.
    ///
    /// Allows the NVMe controller to reorder writes for wear-level optimisation.
    Deferred {
        /// Number of layers to accumulate before flushing the write batch.
        batch_size: u32,
    },
}

// ---------------------------------------------------------------------------
// SmartSource trait — abstraction over SMART TBW reads
// ---------------------------------------------------------------------------

/// Reads the cumulative "Data Units Written" counter from NVMe SMART telemetry.
///
/// Abstracted via a trait so tests can inject a `MockSmartSource` instead
/// of requiring a real `/dev/nvmeN` device node.
pub trait SmartSource: Send + Sync {
    /// Return the current "Data Units Written" value (1 unit = 512 000 bytes).
    ///
    /// On Linux this reads NVMe Admin Command 0x02 (Get Log Page, ID 0x02) via
    /// `NVME_IOCTL_ADMIN_CMD`.  Other platforms return 0 (ZeroSmartSource).
    fn read_units_written(&self) -> crate::Result<u64>;
}

// ---------------------------------------------------------------------------
// MockSmartSource — test double
// ---------------------------------------------------------------------------

/// Simulates a SMART counter for unit tests — avoids the /dev/nvmeN ioctl.
///
/// `initial_units` is returned on every call to `read_units_written`,
/// representing the baseline TBW already on the device before this run starts.
#[cfg(feature = "ssd-wear")]
pub struct MockSmartSource {
    initial_units: u64,
}

#[cfg(feature = "ssd-wear")]
impl MockSmartSource {
    /// Create a mock source that reports `initial_units` as the starting value.
    pub fn new(initial_units: u64) -> Self {
        MockSmartSource { initial_units }
    }
}

#[cfg(feature = "ssd-wear")]
impl SmartSource for MockSmartSource {
    fn read_units_written(&self) -> crate::Result<u64> {
        Ok(self.initial_units)
    }
}

// ---------------------------------------------------------------------------
// ZeroSmartSource — zero-cost non-Linux fallback
// ---------------------------------------------------------------------------

/// Returns 0 units on every read.  Used on Windows and macOS where the NVMe
/// Admin Command ioctl is unavailable.
#[cfg(feature = "ssd-wear")]
struct ZeroSmartSource;

#[cfg(feature = "ssd-wear")]
impl SmartSource for ZeroSmartSource {
    fn read_units_written(&self) -> crate::Result<u64> {
        Ok(0)
    }
}

// ---------------------------------------------------------------------------
// NvmeSmartReader — live SMART via ioctl (Linux only)
// ---------------------------------------------------------------------------

/// Reads "Data Units Written" from the NVMe SMART/Health Information log via
/// the Linux kernel `NVME_IOCTL_ADMIN_CMD` ioctl.
#[cfg(all(feature = "ssd-wear", target_os = "linux"))]
struct NvmeSmartReader {
    /// Path to the NVMe device node, e.g. `/dev/nvme0`.
    device_path: PathBuf,
}

#[cfg(all(feature = "ssd-wear", target_os = "linux"))]
impl SmartSource for NvmeSmartReader {
    /// Issue NVMe Admin Command (Get Log Page, Log ID 0x02) and return the
    /// lower 64 bits of the 128-bit "Data Units Written" field.
    ///
    /// # Safety invariants
    /// - `log_buf` is a 512-byte stack buffer whose address is passed to the kernel.
    ///   The ioctl copies at most `data_len` (512) bytes into it — within bounds.
    /// - The ioctl is synchronous: all writes to `log_buf` complete before the
    ///   function returns.
    fn read_units_written(&self) -> crate::Result<u64> {
        use std::fs::OpenOptions;
        use std::os::unix::io::AsRawFd;

        // NVMe Admin Command: Get Log Page, Log Page ID = 0x02 (SMART/Health).
        // Reference: NVM Express Base Specification 2.0, Section 5.14.
        //
        // SMART/Health Information Log layout (512 bytes):
        //   bytes 48–63: Data Units Written (128-bit LE, 1 unit = 512 000 B)
        //
        // NVME_IOCTL_ADMIN_CMD = 0xC0484E41 on x86_64 Linux (64-byte struct).
        const NVME_IOCTL_ADMIN_CMD: u64 = 0xC048_4E41;
        const SMART_LOG_SIZE: u32 = 512;
        const GET_LOG_PAGE_OPCODE: u8 = 0x02;
        const SMART_LOG_PAGE_ID: u32 = 0x02;
        // cdw10[15:0]  = Log Page ID
        // cdw10[27:16] = NUMDL = (DWORDS to return) - 1 = (512/4 - 1) = 127
        let numdl: u32 = (SMART_LOG_SIZE / 4) - 1;
        let cdw10: u32 = SMART_LOG_PAGE_ID | (numdl << 16);

        #[repr(C)]
        struct NvmeAdminCmd {
            opcode: u8,
            flags: u8,
            rsvd1: u16,
            nsid: u32,
            cdw2: u32,
            cdw3: u32,
            metadata: u64,
            addr: u64,
            metadata_len: u32,
            data_len: u32,
            cdw10: u32,
            cdw11: u32,
            cdw12: u32,
            cdw13: u32,
            cdw14: u32,
            cdw15: u32,
            timeout_ms: u32,
            result: u32,
        }

        let mut log_buf = [0u8; 512];

        let mut cmd = NvmeAdminCmd {
            opcode: GET_LOG_PAGE_OPCODE,
            flags: 0,
            rsvd1: 0,
            nsid: 0xFFFF_FFFF, // global namespace
            cdw2: 0,
            cdw3: 0,
            metadata: 0,
            addr: log_buf.as_mut_ptr() as u64,
            metadata_len: 0,
            data_len: SMART_LOG_SIZE,
            cdw10,
            cdw11: 0,
            cdw12: 0,
            cdw13: 0,
            cdw14: 0,
            cdw15: 0,
            timeout_ms: 5_000,
            result: 0,
        };

        let file = OpenOptions::new()
            .read(true)
            .open(&self.device_path)
            .map_err(|error| {
                crate::RamFlowError::IoUringError(std::io::Error::new(
                    error.kind(),
                    format!("open {:?}: {error}", self.device_path),
                ))
            })?;

        // SAFETY:
        // - `cmd` is a fully-initialised `NvmeAdminCmd` on the stack.
        // - `cmd.addr` holds the address of `log_buf`, a 512-byte stack array.
        // - The kernel writes at most `cmd.data_len` (512) bytes into `log_buf`.
        // - The ioctl returns synchronously; `log_buf` is valid for its full lifetime.
        let ioctl_result = unsafe { libc::ioctl(file.as_raw_fd(), NVME_IOCTL_ADMIN_CMD, &mut cmd) };
        if ioctl_result < 0 {
            return Err(crate::RamFlowError::IoUringError(
                std::io::Error::last_os_error(),
            ));
        }

        // Data Units Written: bytes 48–55 (lower 64 bits of the 128-bit LE field).
        // Truncation to u64 covers ~9 EB of writes — adequate for any consumer SSD.
        let units_written = u64::from_le_bytes(
            log_buf[48..56]
                .try_into()
                .expect("slice is exactly 8 bytes — infallible"),
        );
        Ok(units_written)
    }
}

// ---------------------------------------------------------------------------
// DeferredEntry — pending write waiting for the next batch flush
// ---------------------------------------------------------------------------

#[cfg(feature = "ssd-wear")]
struct DeferredEntry {
    /// Shard index that owns this write (logged for debugging).
    shard_id: u32,
    /// Bytes to flush — may be a full shard or a delta-compressed payload.
    payload: Vec<u8>,
}

#[cfg(feature = "ssd-wear")]
type BudgetCallback = Box<dyn Fn(u64, u64) + Send + Sync>;

// ---------------------------------------------------------------------------
// WriteBudgetManager — ssd-wear ACTIVE path
// ---------------------------------------------------------------------------

/// Tracks cumulative NVMe bytes written and enforces a user-configured TBW cap.
///
/// # Feature gate
/// Meaningful only when the `ssd-wear` Cargo feature is active.  When disabled,
/// all methods are no-ops and `remaining()` always returns `u64::MAX`.
///
/// # Automatic strategy selection
/// `consume()` auto-switches between `Full`, `DeltaCompress`, and `Deferred{4}`
/// based on how much of `budget_units` has been consumed:
///
/// | Remaining fraction | Strategy              |
/// |--------------------|-----------------------|
/// | > 50%              | `Full`                |
/// | 10–50%             | `DeltaCompress`       |
/// | ≤ 10%              | `Deferred { 4 }`      |
///
/// A budget-warning callback fires when remaining < 20%.
#[cfg(feature = "ssd-wear")]
pub struct WriteBudgetManager {
    /// Path to the NVMe device node (informational; used in error messages).
    device_path: PathBuf,
    /// SMART "Data Units Written" at the time of construction (baseline).
    units_at_start: u64,
    /// Total allowed SMART units to write in this run.
    budget_units: u64,
    /// Cumulative SMART units written since construction.
    units_written: AtomicU64,
    /// Current write strategy, updated by every `consume()` call.
    strategy: Mutex<WriteStrategy>,
    /// Source for SMART telemetry reads (injectable for tests).
    smart_source: Box<dyn SmartSource>,
    /// Optional callback: `fn(remaining_units, budget_units)` — fired below 20%.
    budget_callback: Mutex<Option<BudgetCallback>>,
    /// Deferred write queue.  Flushed once `batch_size` entries accumulate.
    deferred_queue: Mutex<Vec<DeferredEntry>>,
}

/// No-op stub used when the `ssd-wear` feature is inactive.
///
/// All methods compile to nothing — zero overhead on the critical training path.
#[cfg(not(feature = "ssd-wear"))]
pub struct WriteBudgetManager {
    _device_path: PathBuf,
    _units_at_start: u64,
    _budget_units: u64,
    _units_written: AtomicU64,
    _strategy: WriteStrategy,
}

// ---------------------------------------------------------------------------
// impl WriteBudgetManager — public API (both feature paths)
// ---------------------------------------------------------------------------

impl WriteBudgetManager {
    /// Create a manager with a lifetime TBW budget of `budget_bytes` bytes.
    ///
    /// When `ssd-wear` is active, reads the current SMART "Data Units Written"
    /// baseline from `device_path`.  On non-Linux targets or ioctl failure,
    /// the baseline defaults to 0 (safe conservative estimate).
    ///
    /// When `ssd-wear` is inactive, all arguments are ignored.
    #[allow(unused_variables)]
    pub fn new(device_path: PathBuf, budget_bytes: u64) -> Self {
        #[cfg(feature = "ssd-wear")]
        {
            #[cfg(target_os = "linux")]
            let source: Box<dyn SmartSource> = Box::new(NvmeSmartReader {
                device_path: device_path.clone(),
            });
            #[cfg(not(target_os = "linux"))]
            let source: Box<dyn SmartSource> = Box::new(ZeroSmartSource);
            WriteBudgetManager::new_with_source(device_path, budget_bytes, source)
        }
        #[cfg(not(feature = "ssd-wear"))]
        {
            WriteBudgetManager {
                _device_path: device_path,
                _units_at_start: 0,
                _budget_units: u64::MAX,
                _units_written: AtomicU64::new(0),
                _strategy: WriteStrategy::Full,
            }
        }
    }

    /// Account for a write of `bytes` against the budget.
    ///
    /// Automatically switches write strategy at the 50% and 10% thresholds.
    /// Fires the budget-warning callback when remaining drops below 20%.
    ///
    /// Returns [`crate::RamFlowError::WearBudgetExceeded`] once
    /// `units_written` exceeds `budget_units`.  The current training step
    /// **always completes first** — this error never interrupts mid-step I/O.
    #[allow(unused_variables)]
    pub fn consume(&self, bytes: u64) -> crate::Result<()> {
        #[cfg(feature = "ssd-wear")]
        {
            // Convert bytes → SMART units (round up).
            let units = bytes.saturating_add(SMART_UNIT_BYTES - 1) / SMART_UNIT_BYTES;
            let prior_written = self.units_written.fetch_add(units, Ordering::AcqRel);
            let total_written = prior_written + units;

            // Auto-switch strategy based on remaining fraction.
            let remaining = self.budget_units.saturating_sub(total_written);
            let new_strategy = if remaining > self.budget_units / 2 {
                WriteStrategy::Full
            } else if remaining > self.budget_units / 10 {
                WriteStrategy::DeltaCompress
            } else {
                WriteStrategy::Deferred { batch_size: 4 }
            };

            *self
                .strategy
                .lock()
                .unwrap_or_else(|poison| poison.into_inner()) = new_strategy;

            // Fire budget-warning callback when remaining < 20%.
            let warning_threshold = self.budget_units / 5;
            if remaining < warning_threshold {
                let guard = self
                    .budget_callback
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner());
                if let Some(ref callback) = *guard {
                    callback(remaining, self.budget_units);
                }
            }

            if total_written > self.budget_units {
                return Err(crate::RamFlowError::WearBudgetExceeded(format!(
                    "written {total_written} SMART units exceeds budget {0} units",
                    self.budget_units
                )));
            }
            Ok(())
        }
        #[cfg(not(feature = "ssd-wear"))]
        {
            Ok(())
        }
    }

    /// Enqueue a layer write and decide whether to flush based on strategy.
    ///
    /// Under `Full` or `DeltaCompress`: registers the byte cost immediately
    /// via `consume()` (the I/O itself is handled by `DirectNvmeEngine::write_async`,
    /// which consults the budget on every write).
    ///
    /// Under `Deferred{batch_size}`: buffers the payload; flushes and accounts
    /// for all bytes when `batch_size` entries have accumulated.
    #[allow(unused_variables)]
    pub fn enqueue_write(
        &self,
        shard_id: u32,
        buf: &crate::allocator::PinnedBuffer,
    ) -> crate::Result<()> {
        #[cfg(feature = "ssd-wear")]
        {
            let strategy = *self
                .strategy
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            match strategy {
                WriteStrategy::Full | WriteStrategy::DeltaCompress => {
                    // Immediate accounting — the actual write is DirectNvmeEngine's job.
                    self.consume(buf.len() as u64)?;
                }
                WriteStrategy::Deferred { batch_size } => {
                    // Snapshot buffer bytes for deferred accounting.
                    let payload: Vec<u8> = buf.as_slice().to_vec();
                    let should_flush = {
                        let mut queue = self
                            .deferred_queue
                            .lock()
                            .unwrap_or_else(|poison| poison.into_inner());
                        queue.push(DeferredEntry { shard_id, payload });
                        queue.len() >= batch_size as usize
                    };
                    if should_flush {
                        self.force_flush()?;
                    }
                }
            }
            Ok(())
        }
        #[cfg(not(feature = "ssd-wear"))]
        {
            Ok(())
        }
    }

    /// Current write strategy.  Changes automatically as the budget is consumed.
    pub fn strategy(&self) -> WriteStrategy {
        #[cfg(feature = "ssd-wear")]
        {
            *self
                .strategy
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
        }
        #[cfg(not(feature = "ssd-wear"))]
        {
            self._strategy
        }
    }

    /// SMART units remaining in the write budget.
    ///
    /// Returns `u64::MAX` when `ssd-wear` is inactive.
    pub fn remaining(&self) -> u64 {
        #[cfg(feature = "ssd-wear")]
        {
            let written = self.units_written.load(Ordering::Acquire);
            self.budget_units.saturating_sub(written)
        }
        #[cfg(not(feature = "ssd-wear"))]
        {
            u64::MAX
        }
    }

    /// Flush all deferred writes, accounting for their bytes.
    ///
    /// Call at training end (or explicit teardown) to ensure no deferred writes
    /// are silently dropped if the batch threshold was never reached.
    ///
    /// No-op when `ssd-wear` is inactive or the deferred queue is empty.
    pub fn force_flush(&self) -> crate::Result<()> {
        #[cfg(feature = "ssd-wear")]
        {
            let batch: Vec<DeferredEntry> = {
                let mut queue = self
                    .deferred_queue
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner());
                std::mem::take(&mut *queue)
            };
            // Lock is released before consume() runs — prevents recursive lock.
            for entry in &batch {
                self.consume(entry.payload.len() as u64)?;
            }
        }
        Ok(())
    }

    /// Register a callback fired when remaining budget falls below 20%.
    ///
    /// Signature: `fn(remaining_units: u64, budget_units: u64)`.
    /// Replaces any previously registered callback.
    ///
    /// Default behaviour when no callback is set: `WearBudgetExceeded` is
    /// returned by `consume()` once budget is exhausted — no other signalling.
    #[allow(unused_variables)]
    pub fn set_budget_warning_callback(&self, callback: impl Fn(u64, u64) + Send + Sync + 'static) {
        #[cfg(feature = "ssd-wear")]
        {
            *self
                .budget_callback
                .lock()
                .unwrap_or_else(|poison| poison.into_inner()) = Some(Box::new(callback));
        }
    }
}

// ---------------------------------------------------------------------------
// ssd-wear-only constructors
// ---------------------------------------------------------------------------

#[cfg(feature = "ssd-wear")]
impl WriteBudgetManager {
    /// Construct with a custom [`SmartSource`].
    ///
    /// Use [`MockSmartSource`] in tests to avoid requiring a real NVMe device.
    /// `budget_bytes` is converted to SMART units (`/ 512_000`, clamped to ≥1).
    ///
    /// # Panics
    ///
    /// Does not panic — if the SMART read fails, `units_at_start` defaults to 0.
    pub fn new_with_source(
        device_path: PathBuf,
        budget_bytes: u64,
        smart_source: Box<dyn SmartSource>,
    ) -> Self {
        let units_at_start = smart_source.read_units_written().unwrap_or(0);
        let budget_units = (budget_bytes / SMART_UNIT_BYTES).max(1);
        WriteBudgetManager {
            device_path,
            units_at_start,
            budget_units,
            units_written: AtomicU64::new(0),
            strategy: Mutex::new(WriteStrategy::Full),
            smart_source,
            budget_callback: Mutex::new(None),
            deferred_queue: Mutex::new(Vec::new()),
        }
    }

    /// Cumulative SMART units written since construction (Relaxed — may be one
    /// `consume()` stale; use for monitoring, not budget enforcement).
    pub fn units_written_snapshot(&self) -> u64 {
        self.units_written.load(Ordering::Relaxed)
    }

    /// SMART units value read at construction from the live device.
    ///
    /// Add this to `units_written_snapshot()` to get the device's absolute TBW.
    pub fn units_at_start(&self) -> u64 {
        self.units_at_start
    }
}

// ---------------------------------------------------------------------------
// Delta compression helpers (ssd-wear only — require zstd)
// ---------------------------------------------------------------------------

/// Compute `delta = updated - original` (LE i16 wrapping arithmetic),
/// compress with zstd level 3, and write to `layer_{idx:04}.delta.zstd`
/// inside `output_dir`.
///
/// Returns the compressed size in bytes on success.
///
/// # Round-trip contract (Module 1 ShardLoader read contract)
///
/// ```text
/// Detect:  layer_{idx:04}.delta.zstd exists alongside layer_{idx:04}.safetensor
/// Load:
///   1. Read original safetensor → Vec<u8>
///   2. zstd::decode_all(delta_file) → delta_bytes
///   3. assert_eq!(delta_bytes.len(), original_bytes.len())
///   4. for i in 0..len/2:
///        let orig  = i16::from_le_bytes(original[2i..2i+2]);
///        let delta = i16::from_le_bytes(delta[2i..2i+2]);
///        result[2i..2i+2] = orig.wrapping_add(delta).to_le_bytes();
/// Fallthrough: if no .delta.zstd exists, load .safetensor normally
/// ```
///
/// # Errors
///
/// Returns `RamFlowError::IoUringError` wrapping any I/O or zstd compression failure.
#[cfg(feature = "ssd-wear")]
pub fn compress_delta(
    layer_idx: u32,
    updated: &[u8],
    original: &[u8],
    output_dir: &Path,
) -> crate::Result<usize> {
    if updated.len() != original.len() {
        return Err(crate::RamFlowError::IoUringError(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "compress_delta layer {layer_idx:04}: updated len {} != original len {}",
                updated.len(),
                original.len()
            ),
        )));
    }

    // Compute LE i16 delta: delta[i] = updated_i16[i] - original_i16[i] (wrapping).
    // Wrapping arithmetic guarantees bit-exact round-trip:
    //   original_i16[i] + delta_i16[i] (wrapping) == updated_i16[i].
    let mut delta_bytes: Vec<u8> = Vec::with_capacity(updated.len());
    for chunk_index in 0..(updated.len() / 2) {
        let byte_index = chunk_index * 2;
        let updated_val = i16::from_le_bytes([updated[byte_index], updated[byte_index + 1]]);
        let original_val = i16::from_le_bytes([original[byte_index], original[byte_index + 1]]);
        let delta_val = updated_val.wrapping_sub(original_val);
        let delta_le = delta_val.to_le_bytes();
        delta_bytes.push(delta_le[0]);
        delta_bytes.push(delta_le[1]);
    }
    // Handle any trailing odd byte (should not occur for FP16, guarded defensively).
    if !updated.len().is_multiple_of(2) {
        let last = updated.len() - 1;
        let trailing_delta = (updated[last] as i8).wrapping_sub(original[last] as i8) as u8;
        delta_bytes.push(trailing_delta);
    }

    // Compress with zstd level 3: fast enough for end-of-step use; ~8× ratio
    // on near-zero deltas typical at standard learning rates.
    let compressed = zstd::encode_all(delta_bytes.as_slice(), 3).map_err(|error| {
        crate::RamFlowError::IoUringError(std::io::Error::new(
            error.kind(),
            format!("zstd encode layer {layer_idx:04}: {error}"),
        ))
    })?;

    let compressed_size = compressed.len();
    let output_path = output_dir.join(format!("layer_{layer_idx:04}.delta.zstd"));

    std::fs::write(&output_path, &compressed).map_err(|error| {
        crate::RamFlowError::IoUringError(std::io::Error::new(
            error.kind(),
            format!("write delta {:?}: {error}", output_path),
        ))
    })?;

    Ok(compressed_size)
}

/// Decompress `layer_{idx:04}.delta.zstd` and apply it to `original`,
/// returning the reconstructed updated tensor.
///
/// Inverse of [`compress_delta`].  Uses wrapping addition to recover updated
/// FP16 weights bit-exactly.
///
/// # Errors
///
/// Returns `RamFlowError::IoUringError` wrapping any I/O or decompression failure.
#[cfg(feature = "ssd-wear")]
pub fn decompress_and_apply_delta(
    layer_idx: u32,
    original: &[u8],
    delta_dir: &Path,
) -> crate::Result<Vec<u8>> {
    let delta_path = delta_dir.join(format!("layer_{layer_idx:04}.delta.zstd"));

    let compressed = std::fs::read(&delta_path).map_err(|error| {
        crate::RamFlowError::IoUringError(std::io::Error::new(
            error.kind(),
            format!("read delta {:?}: {error}", delta_path),
        ))
    })?;

    let delta_bytes = zstd::decode_all(compressed.as_slice()).map_err(|error| {
        crate::RamFlowError::IoUringError(std::io::Error::new(
            error.kind(),
            format!("zstd decode layer {layer_idx:04}: {error}"),
        ))
    })?;

    if delta_bytes.len() != original.len() {
        return Err(crate::RamFlowError::IoUringError(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "delta len {} != original len {} for layer {layer_idx:04}",
                delta_bytes.len(),
                original.len()
            ),
        )));
    }

    // Apply: updated[i] = original[i] + delta[i] (LE i16 wrapping addition).
    let mut updated = vec![0u8; original.len()];
    for chunk_index in 0..(original.len() / 2) {
        let byte_index = chunk_index * 2;
        let orig_val = i16::from_le_bytes([original[byte_index], original[byte_index + 1]]);
        let delta_val = i16::from_le_bytes([delta_bytes[byte_index], delta_bytes[byte_index + 1]]);
        let updated_val = orig_val.wrapping_add(delta_val);
        let updated_le = updated_val.to_le_bytes();
        updated[byte_index] = updated_le[0];
        updated[byte_index + 1] = updated_le[1];
    }
    // Handle any trailing odd byte.
    if !original.len().is_multiple_of(2) {
        let last = original.len() - 1;
        updated[last] = (original[last] as i8).wrapping_add(delta_bytes[last] as i8) as u8;
    }

    Ok(updated)
}
