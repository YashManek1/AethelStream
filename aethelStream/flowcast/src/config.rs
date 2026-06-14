//! FlowCast configuration types.
//!
//! `FlowCastConfig` is the single input to `FlowCast::new`.
//! `HardwareProfile` holds measured timing; it is written to disk after the
//! warm-up profiler completes and loaded on subsequent runs.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Streaming precision mode for a layer slot.
///
/// FlowCast selects the mode per-layer based on importance scores (A8).
/// INT4/INT8 apply to weights in the middle of the network; FP16 is used
/// for edge layers where numerical sensitivity is high.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Precision {
    /// Full precision (FP32) — only used during warm-up profiling.
    FP32,
    /// Half precision (FP16) — edge layers and activations.
    FP16,
    /// Brain floating-point (BF16) — preferred on Ampere/Ada.
    BF16,
    /// 8-bit integer — mid-network weights after calibration.
    INT8,
    /// 4-bit integer — highest compression, mid-network only.
    INT4,
}

/// Opaque CUDA device pointer (raw 64-bit address in GPU address space).
///
/// Stored as `u64` so the type is usable in non-CUDA builds without conditional
/// compilation on every struct that carries device pointers.
pub type DevicePointer = u64;

/// Per-layer timing record produced by the warm-up profiler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerTiming {
    /// Layer index (matches `shard_index.json`).
    pub layer_idx: u32,

    /// Measured forward-pass GPU kernel time (milliseconds).
    pub forward_ms: f32,

    /// Measured backward-pass GPU kernel time (milliseconds).
    pub backward_ms: f32,

    /// Byte size of the layer shard (weights + optional adapter).
    pub shard_bytes: u64,

    /// Estimated NVMe → RAM transfer time at measured bandwidth (milliseconds).
    pub transfer_ms: f32,

    /// Per-layer PCIe host→device DMA transfer time (milliseconds), measured
    /// individually per sample layer in the warm-up profiler (A3-b).
    #[serde(default)]
    pub pcie_transfer_ms: f32,
}

impl Default for LayerTiming {
    fn default() -> Self {
        Self {
            layer_idx: 0,
            forward_ms: 0.0,
            backward_ms: 0.0,
            shard_bytes: 0,
            transfer_ms: 0.0,
            pcie_transfer_ms: 0.0,
        }
    }
}

/// Hardware-profile produced by the warm-up profiler and cached to disk.
///
/// Written to `<shard_dir>/hardware_profile.json` after the first warm-up.
/// All time values are exponentially weighted averages over
/// `HardwareProfile::sample_count` steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    // ---- bandwidth section ----
    /// Measured NVMe → system-RAM bandwidth (GB/s).
    pub nvme_bandwidth_gbs: f32,

    /// Measured PCIe host → device bandwidth (GB/s).
    pub pcie_bandwidth_gbs: f32,

    /// Measured GPU global-memory bandwidth (GB/s).
    pub gpu_bandwidth_gbs: f32,

    // ---- timing section ----
    /// Mean forward-pass GPU time averaged across all layers (milliseconds).
    pub mean_forward_ms: f32,

    /// Mean backward-pass GPU time averaged across all layers (milliseconds).
    pub mean_backward_ms: f32,

    /// Number of warm-up steps used to produce this profile.
    pub sample_count: u32,

    // ---- layer_plan section ----
    /// Per-layer timing breakdown, ordered by `layer_idx`.
    ///
    /// The prefetch engine uses this to size the T_iter lookahead window:
    /// a layer must be fully transferred before its GPU kernel begins.
    pub layer_plan: Vec<LayerTiming>,

    // ---- super-shard section ----
    /// Optimal super-shard transfer size in bytes, measured by the A3
    /// latency-vs-size curve probe.
    ///
    /// `0` means not yet measured; `SuperShardBackend` falls back to its
    /// configured `group_size` in that case.  On PCIe 4 NVMe this is
    /// typically 4–16 MiB.
    #[serde(default)]
    pub optimal_super_shard_bytes: u64,
}

/// Configuration for a FlowCast pipeline instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowCastConfig {
    /// Root directory that contains `shard_NNNN.bin` files and `shard_index.json`.
    pub shard_dir: PathBuf,

    /// Number of shard files (must match the model's layer count).
    pub num_shards: u32,

    /// Prefetch lookahead depth in layers (initial value; A2 adapts it).
    pub initial_lookahead: u32,

    /// EWMA alpha for the adaptive window (A2). Range: (0, 1].
    pub ewma_alpha: f32,

    /// Memory pressure fraction at which prefetch is paused. Range: (0, 1].
    ///
    /// Passed to `MemoryPressureGauge::register_high_pressure` via RamFlow.
    pub pressure_threshold: f32,

    /// Default streaming precision. A8 may override per-layer.
    pub default_precision: Precision,

    /// Previously measured hardware profile; `None` triggers warm-up profiling.
    pub hardware_profile: Option<HardwareProfile>,

    /// CPU core for the io_uring CQE poller thread.
    pub io_poller_cpu_core: usize,

    /// CPU core for the completion-router thread.
    pub completion_router_cpu_core: usize,

    /// Target GPU utilisation fraction (used by A2 to judge under-prefetching).
    pub target_gpu_utilisation: f32,

    /// Steps between periodic SSD temperature checks and re-profiling (`ssd-thermal` feature).
    ///
    /// Set to 0 to disable periodic re-profiling. Default: 5 000 steps.
    #[serde(default = "default_reprofiling_interval_steps")]
    pub reprofiling_interval_steps: u64,

    /// Maximum times a transient CQE error (EAGAIN, EINTR, EBUSY) is retried
    /// before escalating to `RamFlowError::MediaError`.
    ///
    /// Default: 3. On consumer SSDs under thermal throttling, EAGAIN is common;
    /// three retries with exponential backoff resolve most transient stalls.
    #[serde(default = "default_max_cqe_retries")]
    pub max_cqe_retries: u8,

    /// Base backoff interval for CQE retries in milliseconds.
    ///
    /// Retry attempt `n` (1-indexed) waits `2^n × base_backoff_ms` before
    /// re-submitting the SQE. Default: 5 ms.
    #[serde(default = "default_base_backoff_ms")]
    pub base_backoff_ms: u64,
}

fn default_reprofiling_interval_steps() -> u64 {
    5_000
}

fn default_max_cqe_retries() -> u8 {
    3
}

fn default_base_backoff_ms() -> u64 {
    5
}

impl Default for FlowCastConfig {
    fn default() -> Self {
        Self {
            shard_dir: PathBuf::from("./shards"),
            num_shards: 0,
            initial_lookahead: 2,
            ewma_alpha: 0.3,
            pressure_threshold: 0.80,
            default_precision: Precision::FP16,
            hardware_profile: None,
            io_poller_cpu_core: 0,
            completion_router_cpu_core: 1,
            target_gpu_utilisation: 0.95,
            reprofiling_interval_steps: 5_000,
            max_cqe_retries: default_max_cqe_retries(),
            base_backoff_ms: default_base_backoff_ms(),
        }
    }
}

impl FlowCastConfig {
    /// Validate all fields.  Called inside `FlowCast::new`.
    pub fn validate(&self) -> crate::Result<()> {
        if !(0.0 < self.ewma_alpha && self.ewma_alpha <= 1.0) {
            return Err(crate::FlowCastError::Config(
                "ewma_alpha must be in (0, 1]".to_string(),
            ));
        }
        if !(0.0 < self.pressure_threshold && self.pressure_threshold <= 1.0) {
            return Err(crate::FlowCastError::Config(
                "pressure_threshold must be in (0, 1]".to_string(),
            ));
        }
        if !(0.0 < self.target_gpu_utilisation && self.target_gpu_utilisation <= 1.0) {
            return Err(crate::FlowCastError::Config(
                "target_gpu_utilisation must be in (0, 1]".to_string(),
            ));
        }
        Ok(())
    }
}
