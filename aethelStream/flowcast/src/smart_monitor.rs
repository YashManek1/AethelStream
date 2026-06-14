//! SSD thermal monitoring for AethelStream''s FlowCast prefetch engine (A3-T).
//!
//! Reads NVMe SMART/Health Information Log temperature (Log ID 0x02, bytes 1-2
//! = Composite Temperature in Kelvin) and triggers periodic re-profiling when
//! the SSD temperature exceeds configurable thresholds.
//!
//! # Why this matters
//! Consumer NVMe SSDs throttle hard under sustained sequential load.
//! AethelStream produces exactly this workload (continuous per-layer reads at
//! layer-streaming bandwidth). No prior art (TERAIO, LoHan, DeepNVMe) handles
//! SSD thermal throttling; this module is a genuine paper contribution.
//!
//! # Design note: MISSING item for future consolidation
//! The NVMe Admin Command ioctl setup duplicates the implementation in
//! `ramflow::nvme::write_budget::NvmeSmartReader`. Both readers issue the same
//! Get Log Page command (opcode 0x02, Log ID 0x02, 512 bytes) but extract
//! different fields — bytes 1-2 (temperature) here vs bytes 48-55 (data units
//! written) there. A future consolidation should expose a single shared
//! `ramflow::nvme::read_smart_log_page(&Path) -> Result<[u8; 512]>` helper.
//!
//! # Linux + `ssd-thermal` feature
//! This entire module is compiled only when `target_os = "linux"` **and** the
//! `ssd-thermal` Cargo feature is active (controlled by the `#[cfg(…)]` on the
//! `pub mod smart_monitor;` declaration in `lib.rs`).

use crate::profiler;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// SmartNotAvailable
// ---------------------------------------------------------------------------

/// Error returned when the NVMe SMART temperature cannot be read.
///
/// Returned by [`TemperatureSource::read_celsius`] when the device node cannot
/// be opened or the ioctl fails (device absent, permission denied, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmartNotAvailable;

impl std::fmt::Display for SmartNotAvailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NVMe SMART temperature not available (ioctl failed or no device)")
    }
}

// ---------------------------------------------------------------------------
// ThermalState
// ---------------------------------------------------------------------------

/// Thermal operating state of the NVMe SSD.
///
/// Derived by comparing the Composite Temperature against two configurable
/// thresholds (see [`ThermalConfig`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThermalState {
    /// Temperature below `warn_celsius` (default 65 C). Full-speed operation.
    Normal,
    /// Temperature in `[warn_celsius, throttle_celsius)`. Single-layer probe.
    Warm,
    /// Temperature at or above `throttle_celsius` (default 75 C). Full 5-layer re-profile.
    Throttling,
}

impl ThermalState {
    /// Classify a temperature reading into a thermal state.
    pub fn classify(temp_celsius: f32, warn: f32, throttle: f32) -> Self {
        if temp_celsius >= throttle {
            ThermalState::Throttling
        } else if temp_celsius >= warn {
            ThermalState::Warm
        } else {
            ThermalState::Normal
        }
    }

    /// Encode as u8 for atomic storage: 0 = Normal, 1 = Warm, 2 = Throttling.
    pub fn as_u8(self) -> u8 {
        match self {
            ThermalState::Normal => 0,
            ThermalState::Warm => 1,
            ThermalState::Throttling => 2,
        }
    }

    /// Decode from u8 (any unknown value decodes as Normal).
    pub fn from_u8(value: u8) -> Self {
        match value {
            1 => ThermalState::Warm,
            2 => ThermalState::Throttling,
            _ => ThermalState::Normal,
        }
    }
}

// ---------------------------------------------------------------------------
// ThermalConfig
// ---------------------------------------------------------------------------

/// Temperature thresholds for [`ThermalState`] classification.
///
/// Both thresholds are read from environment variables on construction so they
/// can be tuned per-drive without recompilation.
#[derive(Debug, Clone)]
pub struct ThermalConfig {
    /// Temperature at or above which the SSD is considered `Warm` (default 65 C).
    pub warn_celsius: f32,
    /// Temperature at or above which the SSD is considered `Throttling` (default 75 C).
    pub throttle_celsius: f32,
}

impl ThermalConfig {
    /// Read thresholds from `FLOWCAST_SSD_WARN_CELSIUS` and
    /// `FLOWCAST_SSD_THROTTLE_CELSIUS` environment variables with safe defaults.
    pub fn from_env() -> Self {
        let warn = std::env::var("FLOWCAST_SSD_WARN_CELSIUS")
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .unwrap_or(65.0);
        let throttle = std::env::var("FLOWCAST_SSD_THROTTLE_CELSIUS")
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .unwrap_or(75.0);
        Self { warn_celsius: warn, throttle_celsius: throttle }
    }
}

impl Default for ThermalConfig {
    fn default() -> Self {
        Self { warn_celsius: 65.0, throttle_celsius: 75.0 }
    }
}

// ---------------------------------------------------------------------------
// TemperatureSource trait
// ---------------------------------------------------------------------------

/// Reads the NVMe Composite Temperature from SMART/Health telemetry.
///
/// Abstracted to allow test doubles without a real NVMe device.
pub trait TemperatureSource: Send + Sync {
    /// Read the current Composite Temperature in degrees Celsius.
    ///
    /// Returns [`SmartNotAvailable`] when the ioctl fails or the device is absent.
    fn read_celsius(&self) -> Result<f32, SmartNotAvailable>;
}

// ---------------------------------------------------------------------------
// FixedTempSource — test double
// ---------------------------------------------------------------------------

/// A [`TemperatureSource`] that always returns a fixed temperature.
///
/// Intended for unit and integration tests that need to drive thermal-state
/// transitions without access to a real NVMe device.
pub struct FixedTempSource {
    celsius: f32,
}

impl FixedTempSource {
    /// Create a source that always reports `celsius` degrees C.
    pub fn new(celsius: f32) -> Self {
        Self { celsius }
    }
}

impl TemperatureSource for FixedTempSource {
    fn read_celsius(&self) -> Result<f32, SmartNotAvailable> {
        Ok(self.celsius)
    }
}

// ---------------------------------------------------------------------------
// UnavailableTempSource — test double for failure paths
// ---------------------------------------------------------------------------

/// A [`TemperatureSource`] that always returns [`SmartNotAvailable`].
///
/// Use in tests to verify that `ThermalMonitor::tick` is a no-op when the
/// SMART ioctl is unavailable (e.g., no `/dev/nvme0`, permission denied).
pub struct UnavailableTempSource;

impl TemperatureSource for UnavailableTempSource {
    fn read_celsius(&self) -> Result<f32, SmartNotAvailable> {
        Err(SmartNotAvailable)
    }
}

// ---------------------------------------------------------------------------
// SMART log parsing
// ---------------------------------------------------------------------------

/// Parse the Composite Temperature from a raw 512-byte NVMe SMART log page.
///
/// Bytes 1-2 of Log ID 0x02 contain the Composite Temperature as a u16 LE
/// value in Kelvin (0 = not reported by the device).
///
/// Returns `Err(SmartNotAvailable)` if `raw_kelvin == 0`.
pub fn parse_temperature_from_log(log: &[u8; 512]) -> Result<f32, SmartNotAvailable> {
    let raw_kelvin = u16::from_le_bytes([log[1], log[2]]);
    if raw_kelvin == 0 {
        return Err(SmartNotAvailable);
    }
    Ok(raw_kelvin as f32 - 273.0)
}

// ---------------------------------------------------------------------------
// SmartTempReader — live NVMe SMART ioctl
// ---------------------------------------------------------------------------

/// Reads NVMe SMART Composite Temperature from a `/dev/nvmeN` device node.
///
/// Issues an NVMe Admin Command (Get Log Page, Log ID 0x02) via the Linux
/// `NVME_IOCTL_ADMIN_CMD` ioctl to read 512 bytes of SMART/Health log data.
pub struct SmartTempReader {
    device_path: PathBuf,
}

impl SmartTempReader {
    /// Create a reader for the NVMe device at `device_path` (e.g. `/dev/nvme0`).
    pub fn new(device_path: PathBuf) -> Self {
        Self { device_path }
    }
}

impl TemperatureSource for SmartTempReader {
    fn read_celsius(&self) -> Result<f32, SmartNotAvailable> {
        read_celsius_from_device(&self.device_path)
    }
}

// MISSING: consolidation with ramflow::nvme::write_budget::NvmeSmartReader.
// Both issue NVME_IOCTL_ADMIN_CMD for SMART Log Page 0x02 but extract
// different byte offsets (1-2 = temperature vs 48-55 = data units written).
// Future work: expose ramflow::nvme::read_smart_log_page as a shared helper.
fn read_celsius_from_device(device_path: &Path) -> Result<f32, SmartNotAvailable> {
    use std::fs::OpenOptions;
    use std::os::unix::io::AsRawFd;

    // Linux NVMe Admin passthrough ioctl (same constant as write_budget.rs).
    const NVME_IOCTL_ADMIN_CMD: libc::c_ulong = 0xC048_4E41;
    const GET_LOG_PAGE_OPCODE: u8 = 0x02;
    const SMART_LOG_SIZE: u32 = 512;
    const SMART_LOG_PAGE_ID: u32 = 0x02;
    let numdl: u32 = (SMART_LOG_SIZE / 4).saturating_sub(1);
    let cdw10: u32 = SMART_LOG_PAGE_ID | (numdl << 16);

    // NVMe Admin Command structure (Linux uapi nvme_ioctl.h).
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
        nsid: 0xFFFF_FFFF,
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
        .open(device_path)
        .map_err(|_| SmartNotAvailable)?;

    // SAFETY:
    // - `cmd` is a fully-initialised `NvmeAdminCmd` on the stack.
    // - `cmd.addr` points to `log_buf`, a 512-byte stack buffer valid for the
    //   duration of the call.
    // - The kernel writes at most `cmd.data_len` (512) bytes into `log_buf`.
    // - The ioctl is synchronous; `log_buf` is valid until we read it below.
    let ret = unsafe {
        libc::ioctl(file.as_raw_fd(), NVME_IOCTL_ADMIN_CMD, &mut cmd)
    };
    if ret < 0 {
        return Err(SmartNotAvailable);
    }

    parse_temperature_from_log(&log_buf)
}

// ---------------------------------------------------------------------------
// ReprofileOutcome
// ---------------------------------------------------------------------------

/// Result of a completed background re-profiling job.
///
/// Written by the background thread; polled via [`ThermalMonitor::poll_outcome`].
pub struct ReprofileOutcome {
    /// Mean SSD transfer time measured during this probe (milliseconds).
    pub mean_t_ssd_ms: f32,
    /// New W_max computed as ceil(t_ssd / t_gpu) + 2 from this probe.
    pub w_max: u32,
    /// SSD temperature at the time the background probe was launched (Celsius).
    pub ssd_temp_celsius: f32,
}

// ---------------------------------------------------------------------------
// ThermalMonitor
// ---------------------------------------------------------------------------

/// Periodic SSD thermal monitor and re-profiling trigger.
///
/// Wired into [`crate::FlowCast::on_layer_start`]: every `reprofiling_interval`
/// steps, reads the NVMe Composite Temperature (fast inline ioctl) and
/// optionally spawns a background thread to re-run the A3 timing probe.
///
/// The background thread never blocks the training loop — the training loop
/// polls for the outcome via [`poll_outcome`] at the start of each step.
///
/// [`poll_outcome`]: ThermalMonitor::poll_outcome
pub struct ThermalMonitor {
    source: Box<dyn TemperatureSource>,
    config: ThermalConfig,
    /// Result of the most recently completed re-profiling job.
    last_outcome: Arc<Mutex<Option<ReprofileOutcome>>>,
    /// True while a background re-profiling thread is running.
    running: Arc<AtomicBool>,
    /// Total re-profiling background threads spawned (monotonic).
    events: AtomicU64,
    /// Latest Composite Temperature as f32 bits (Relaxed — best-effort gauge).
    ssd_temp_celsius_bits: AtomicU32,
    /// Latest ThermalState as u8 (Relaxed — best-effort gauge).
    thermal_state_u8: AtomicU8,
    /// Shard directory passed to the background probe thread.
    shard_dir: PathBuf,
    /// Total layer count (for selecting representative probe layer indices).
    num_layers: u32,
}

impl ThermalMonitor {
    /// Create a monitor that reads live SMART temperature from `device_path`.
    ///
    /// `device_path` is typically `/dev/nvme0`; override via
    /// `FLOWCAST_NVME_DEVICE` env var (see `FlowCast::new`).
    pub fn new(device_path: PathBuf, shard_dir: PathBuf, num_layers: u32) -> Self {
        Self::with_source(
            Box::new(SmartTempReader::new(device_path)),
            shard_dir,
            num_layers,
        )
    }

    /// Create a monitor with a custom [`TemperatureSource`] (for testing).
    pub fn with_source(
        source: Box<dyn TemperatureSource>,
        shard_dir: PathBuf,
        num_layers: u32,
    ) -> Self {
        Self {
            source,
            config: ThermalConfig::from_env(),
            last_outcome: Arc::new(Mutex::new(None)),
            running: Arc::new(AtomicBool::new(false)),
            events: AtomicU64::new(0),
            ssd_temp_celsius_bits: AtomicU32::new(0),
            thermal_state_u8: AtomicU8::new(0),
            shard_dir,
            num_layers,
        }
    }

    /// Tick the thermal monitor at step `step` with re-profiling interval `interval`.
    ///
    /// Returns immediately (`Ok` no-op) if:
    /// - `interval == 0` (re-profiling disabled), or
    /// - `step % interval != 0` (not a check step), or
    /// - SMART ioctl fails (device absent / no permission).
    ///
    /// At each interval-th step:
    /// 1. Reads the SMART Composite Temperature (fast inline ioctl).
    /// 2. Classifies thermal state.
    /// 3. Spawns a background timing probe when state is `Warm` or `Throttling`,
    ///    or when `Normal` and `step % (interval * 10) == 0` (sparse check).
    ///    Only one background thread runs at a time (guarded by `running` flag).
    pub fn tick(&self, step: u64, interval: u64) {
        if interval == 0 || step % interval != 0 {
            return;
        }

        let temp_celsius = match self.source.read_celsius() {
            Ok(temp) => temp,
            Err(_) => return,
        };

        self.ssd_temp_celsius_bits.store(temp_celsius.to_bits(), Ordering::Relaxed);
        let state = ThermalState::classify(
            temp_celsius,
            self.config.warn_celsius,
            self.config.throttle_celsius,
        );
        self.thermal_state_u8.store(state.as_u8(), Ordering::Relaxed);

        let is_sparse = step % interval.saturating_mul(10) == 0;
        let needs_probe = match state {
            ThermalState::Throttling | ThermalState::Warm => true,
            ThermalState::Normal => is_sparse,
        };

        if !needs_probe {
            return;
        }

        // Guard: don't spawn a second thread if one is already running.
        if self.running.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_err() {
            return;
        }

        let probe_indices = match state {
            ThermalState::Throttling => representative_indices(self.num_layers),
            ThermalState::Warm | ThermalState::Normal => vec![self.num_layers / 2],
        };

        self.events.fetch_add(1, Ordering::Relaxed);

        let last_outcome = Arc::clone(&self.last_outcome);
        let running = Arc::clone(&self.running);
        let shard_dir = self.shard_dir.clone();

        std::thread::spawn(move || {
            let (mean_t_ssd_ms, w_max) = profiler::probe_layers(&shard_dir, &probe_indices);
            let outcome = ReprofileOutcome { mean_t_ssd_ms, w_max, ssd_temp_celsius: temp_celsius };
            *last_outcome.lock().unwrap_or_else(|poison| poison.into_inner()) = Some(outcome);
            // Release ordering: the training-loop reader of `running` sees the
            // outcome write before it sees running = false.
            running.store(false, Ordering::Release);
        });
    }

    /// Poll for the most recently completed re-profiling outcome.
    ///
    /// Returns `Some(outcome)` and clears the internal slot when a new result is
    /// ready; returns `None` when no result has arrived since the last call.
    ///
    /// Non-blocking: uses `try_lock` so the training loop is never stalled.
    pub fn poll_outcome(&self) -> Option<ReprofileOutcome> {
        self.last_outcome
            .try_lock()
            .ok()
            .and_then(|mut guard| guard.take())
    }

    /// Latest Composite Temperature read from SMART telemetry (degrees Celsius).
    ///
    /// Returns `0.0` if no successful SMART read has occurred yet.
    pub fn ssd_temp_celsius(&self) -> f32 {
        f32::from_bits(self.ssd_temp_celsius_bits.load(Ordering::Relaxed))
    }

    /// Latest thermal state.
    pub fn thermal_state(&self) -> ThermalState {
        ThermalState::from_u8(self.thermal_state_u8.load(Ordering::Relaxed))
    }

    /// Total number of background re-profiling threads spawned (monotonic).
    pub fn reprofiling_events(&self) -> u64 {
        self.events.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn representative_indices(num_layers: u32) -> Vec<u32> {
    let last = num_layers.saturating_sub(1);
    let mut indices = vec![0, last / 4, last / 2, last * 3 / 4, last];
    indices.dedup();
    indices
}