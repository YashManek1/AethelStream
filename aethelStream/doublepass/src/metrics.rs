//! Per-step performance and correctness telemetry consumed by M8.
//!
//! All numbers computed on the mock CPU f32 path (`mock-cuda` feature) unless
//! explicitly tagged `[GPU]`. Do **not** cite `[MOCK]` numbers as measured GPU
//! throughput; real hardware measurements require `--features cuda`.

/// Per-segment activation-handling log entry produced by the SARP executor (A2').
///
/// One entry is appended to [`StepMetrics::recompute_mode_per_segment`] per segment
/// per backward pass via [`StepMetrics::record_segment`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SegmentMetrics {
    /// Segment index (0-based, ascending from the last layer toward layer 0).
    pub segment_index: u32,
    /// Activation action applied for this segment this step.
    ///
    /// One of `"Recompute"`, `"RetainVram"`, `"PageCompressedRam"`, `"PageNvme"`.
    pub action: String,
    /// Weight bytes streamed for this segment's recompute forward pass.
    ///
    /// Zero for `"RetainVram"` (weights already resident in VRAM).
    pub recompute_weight_bytes: u64,
    /// PCIe bytes transferred for RAM-side activation paging.
    ///
    /// Non-zero only for `"PageCompressedRam"`.
    pub pcie_bytes: u64,
    /// NVMe bytes read/written for SSD-side activation paging.
    ///
    /// Non-zero only for `"PageNvme"`.
    pub ssd_bytes: u64,
    /// Wall time of this segment's activation path in milliseconds.
    ///
    /// [MOCK]: always `0.0` on the CPU f32 path.
    pub latency_ms: f32,
}

/// Snapshot of one layer's FP16 dynamic loss-scale state at step end.
///
/// Entries are pushed into [`StepMetrics::scale_table_snapshot`] by the A5
/// precision module via [`StepMetrics::push_scale_entry`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScaleEntry {
    /// Layer index (0-based).
    pub layer_idx: u32,
    /// Current FP16 dynamic loss scale for this layer.
    ///
    /// Fixed at `1.0` in BF16 mode (`PerLayerScaleTable::enable_bf16_mode`).
    pub scale: f32,
    /// EWA overflow density: exponentially weighted fraction of FP16 values that
    /// overflowed in recent steps. Zero in BF16 mode.
    pub overflow_density: f64,
}

/// Complete per-step telemetry snapshot returned by the M5 training loop.
///
/// Returned by [`crate::DoublePass::step`] and serialised to JSON via
/// [`StepMetrics::to_json`] for M8 (throughput monitor) consumption.
///
/// # Mock-vs-measured separation
///
/// Fields tagged `[MOCK]` are derived from the CPU f32 simulation path
/// (`mock-cuda` feature). They reflect code-path correctness but **not** real
/// GPU performance. Fields measured even on the mock path (byte counts derived
/// from model config, parity errors, prefetch counts) are not tagged.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StepMetrics {
    // Identity
    /// Training step index (0-based), matching the outer training loop counter.
    pub step_index: u64,

    // Throughput
    /// Total model-weight bytes streamed from NVMe/RAM during this step.
    ///
    /// Equals forward + backward + recompute bytes combined.
    pub weight_bytes_streamed: u64,

    /// Weight bytes attributed to the forward pass alone.
    pub forward_weight_bytes: u64,

    /// Weight bytes attributed to the backward pass (excluding recompute overhead).
    pub backward_weight_bytes: u64,

    /// Number of micro-batches gradient-accumulated this step.
    pub grad_accum_steps: u32,

    /// Total tokens processed (`batch x seq_len x G`).
    pub tokens_processed: u64,

    /// Tokens per wall-clock second. [MOCK]: derived from CPU f32 timing.
    pub tokens_per_sec: f32,

    // Latency breakdown
    /// Total step wall time in milliseconds. [MOCK]: CPU measurement.
    pub step_wall_ms: f32,

    /// Sequence of GPU idle gap durations (ms) between I/O completion and compute start.
    ///
    /// [MOCK]: always empty — no real GPU timeline available on CPU path.
    pub gpu_idle_gaps_ms: Vec<f32>,

    /// Fraction of step wall time the GPU was idle (0.0-1.0). [MOCK]: always `0.0`.
    pub gpu_idle_fraction: f32,

    // I/O health
    /// Number of `FlowCastError::PrefetchMiss` events this step.
    ///
    /// Target is **0** for a well-tuned prefetch window. Non-zero values indicate
    /// the window is too narrow for the current NVMe layer latency.
    pub prefetch_misses: u32,

    // SARP segment log
    /// Per-segment breakdown of activation action, paging bytes, and latency.
    ///
    /// Populated by the SARP executor (A2') during `full_backward_sarp`.
    pub recompute_mode_per_segment: Vec<SegmentMetrics>,

    /// Weighted sum of FLOP fractions recomputed across all segments this step.
    ///
    /// `0.0` = all activations retained. `1.0` = every op recomputed from scratch.
    pub recompute_flop_fraction: f64,

    // Parity / gradient health
    /// Relative parity error from the most recent A7 check.
    ///
    /// Defined as `max|stream_grad - ref_grad| / (max|ref_grad| + eps)`.
    /// `f64::NAN` when no parity check was scheduled this step.
    pub parity_rel_error: f64,

    /// Sliding history of parity relative errors (newest last, capped at 50 entries).
    pub parity_rel_history: Vec<f64>,

    /// Number of layers currently in the parity escalation set (FP32 recompute forced).
    pub escalated_layer_count: u32,

    // Precision / scale table
    /// Per-layer FP16 dynamic loss-scale snapshot at step end.
    ///
    /// Empty in BF16 mode (all scales fixed at `1.0` — no overflow possible).
    pub scale_table_snapshot: Vec<ScaleEntry>,
}

impl Default for StepMetrics {
    fn default() -> Self {
        Self {
            step_index: 0,
            weight_bytes_streamed: 0,
            forward_weight_bytes: 0,
            backward_weight_bytes: 0,
            grad_accum_steps: 0,
            tokens_processed: 0,
            tokens_per_sec: 0.0,
            step_wall_ms: 0.0,
            gpu_idle_gaps_ms: Vec::new(),
            gpu_idle_fraction: 0.0,
            prefetch_misses: 0,
            recompute_mode_per_segment: Vec::new(),
            recompute_flop_fraction: 0.0,
            parity_rel_error: f64::NAN,
            parity_rel_history: Vec::new(),
            escalated_layer_count: 0,
            scale_table_snapshot: Vec::new(),
        }
    }
}

impl StepMetrics {
    /// Serialise this telemetry snapshot to a pretty-printed JSON string for M8.
    ///
    /// Returns `{"error":"<msg>"}` on the (theoretically impossible) case where
    /// serialisation fails for a struct that derives `serde::Serialize`.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self)
            .unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"))
    }

    /// Append a parity check result and update the sliding history (retains last 50).
    ///
    /// Also sets `parity_rel_error` to `rel`.
    pub fn record_parity(&mut self, rel: f64) {
        self.parity_rel_error = rel;
        self.parity_rel_history.push(rel);
        const MAX_HISTORY: usize = 50;
        if self.parity_rel_history.len() > MAX_HISTORY {
            self.parity_rel_history.remove(0);
        }
    }

    /// Append a segment activation record from the SARP executor.
    ///
    /// Called once per segment during `full_backward_sarp`.
    pub fn record_segment(
        &mut self,
        segment_index: u32,
        action: &str,
        recompute_weight_bytes: u64,
        pcie_bytes: u64,
        ssd_bytes: u64,
    ) {
        self.recompute_mode_per_segment.push(SegmentMetrics {
            segment_index,
            action: action.to_string(),
            recompute_weight_bytes,
            pcie_bytes,
            ssd_bytes,
            latency_ms: 0.0,
        });
    }

    /// Append a GPU idle gap observation.
    ///
    /// Pass `gap_ms = 0.0` on the mock (CPU f32) path.
    pub fn record_gpu_idle_gap(&mut self, gap_ms: f32) {
        self.gpu_idle_gaps_ms.push(gap_ms);
        let total: f32 = self.gpu_idle_gaps_ms.iter().sum();
        if self.step_wall_ms > 0.0 {
            self.gpu_idle_fraction = (total / self.step_wall_ms).min(1.0);
        }
    }

    /// Push a per-layer scale-table snapshot entry.
    ///
    /// Called by the A5 precision module after each step.
    pub fn push_scale_entry(&mut self, layer_idx: u32, scale: f32, overflow_density: f64) {
        self.scale_table_snapshot.push(ScaleEntry {
            layer_idx,
            scale,
            overflow_density,
        });
    }
}
