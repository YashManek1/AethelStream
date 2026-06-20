//! Training plan types consumed from M9 (SARP) and user configuration.

/// Action assigned to a checkpoint segment by the M9 SARP planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ActivationAction {
    /// Recompute the segment forward during backward (idle compute is free in I/O-bound regime).
    Recompute,
    /// Keep the segment activations resident in VRAM from the forward pass.
    RetainVram,
    /// Fetch activations from the M2 LZ4 compressed-RAM tier (no recompute, no SSD).
    PageCompressedRam,
    /// Fetch activations via M3 write-back from NVMe tier.
    PageNvme,
}

/// Per-segment schedule emitted by M9's SARP DP.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SegmentPlan {
    /// Which checkpoint segment this plan applies to.
    pub segment_index: u32,
    /// Action to take during backward for this segment.
    pub action: ActivationAction,
    /// Per-op selective-recompute mask (A2′). `true` = recompute; `false` = retain.
    /// Indexed by op position within the segment (attention interior = true by default).
    pub recompute_ops: Vec<bool>,
}

/// Training mode / tier (read from M9 TrainingPlan.tier).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TrainingTier {
    /// All layers trainable; full GaLore low-rank states; full-param write-back.
    FullGaLore,
    /// Base weights frozen (read-only stream, no write-back); only LoRA A/B trainable.
    LoraOnly,
    /// Freeze all but the K most-sensitive layers; frozen layers skip the hook entirely.
    TopKFreeze(u32),
    /// INT4 weights everywhere; checkpoint compression forced.
    Int4Everywhere,
}

/// Read-only view of the M9 TrainingPlan fields consumed by M5.
///
/// In production this will be deserialized from the M9 ElasticScale output.
/// For the S0 scaffold it is constructed directly in tests.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrainingPlan {
    /// Checkpoint frequency `k`: one sparse checkpoint every `k` layers.
    pub checkpoint_freq: u32,
    /// Micro-batch token count `s` (tokens per micro-batch per gradient-accumulation step).
    pub micro_batch: u32,
    /// Gradient-accumulation depth `G`.
    pub grad_accum: u32,
    /// Per-layer precision schedule (index = layer_idx). Uses flowcast::Precision.
    pub precision_schedule: Vec<crate::Precision>,
    /// Low-rank dimension for GaLore projection.
    pub optimizer_rank: u32,
    /// Training mode.
    pub tier: TrainingTier,
    /// Prefetch window hint `W` (number of layers to keep in-flight simultaneously).
    pub w_max_hint: u32,
    /// Per-segment activation materialization schedule from the M9 SARP DP.
    /// Empty when M9 has not yet run; M5 falls back to the A2/A9 heuristic.
    pub activation_schedule: Vec<SegmentPlan>,
    /// Interval between parity diagnostic checks (A7). 0 = disabled.
    pub parity_check_interval: u64,
    /// Interval between M4 projection refreshes (forwarded to optimizer; M5 does not own this).
    pub projection_refresh_interval: u64,
    /// Maximum gradient norm for global clipping (A6).
    pub max_grad_norm: f32,
}

impl Default for TrainingPlan {
    fn default() -> Self {
        Self {
            checkpoint_freq: 4,
            micro_batch: 2048,
            grad_accum: 2,
            precision_schedule: Vec::new(),
            optimizer_rank: 64,
            tier: TrainingTier::LoraOnly,
            w_max_hint: 4,
            activation_schedule: Vec::new(),
            parity_check_interval: 500,
            projection_refresh_interval: 200,
            max_grad_norm: 1.0,
        }
    }
}

/// A partial plan update (e.g., window-size or precision change from M9 mid-run).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PlanDelta {
    /// Override the prefetch window hint.
    pub w_max_hint: Option<u32>,
    /// Override the checkpoint frequency.
    pub checkpoint_freq: Option<u32>,
    /// Override per-layer precision (only layers listed here are changed).
    pub precision_overrides: Vec<(u32, crate::Precision)>,
}
