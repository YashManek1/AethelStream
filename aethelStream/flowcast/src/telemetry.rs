//! Cross-cutting metrics counters — sampled lock-light by the completion router.
//!
//! All counters are `AtomicU64` (Relaxed ordering — best-effort gauges).
//! `snapshot()` produces a `TelemetrySnapshot` for the paper's JSON dump via
//! `to_json()`.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Snapshot of all telemetry counters at one point in time.
///
/// Produced by `Telemetry::snapshot()` and serialisable to JSON for the paper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetrySnapshot {
    /// Total `prefetch` SQEs submitted.
    pub prefetch_submitted: u64,
    /// Total `prefetch` CQEs received (success + failure).
    pub prefetch_completed: u64,
    /// Prefetch CQEs with `result < 0` (errno).
    pub prefetch_errors: u64,
    /// `PrefetchMiss` errors returned to M5.
    pub miss_count: u64,
    /// Total GPU idle microseconds (gap between `on_layer_start` calls).
    pub gpu_idle_us: u64,
    /// Hot-set cache hits (layer found resident, no I/O issued).
    pub hotset_hits: u64,
    /// Hot-set cache misses (layer not resident, I/O issued).
    pub hotset_misses: u64,
    // --- new fields ---
    /// Bytes read from NVMe (sum of all completed prefetch lengths).
    pub nvme_bytes_read: u64,
    /// Current io_uring submission-queue depth (snapshot).
    pub queue_depth: u64,
    /// Current ready-queue depth (layers completed but not yet consumed by M5).
    pub ready_queue_depth: u64,
    /// Number of times the adaptive window was grown.
    pub window_grow_events: u64,
    /// Number of times the adaptive window was shrunk.
    pub window_shrink_events: u64,
    /// Total nanoseconds spent in decode kernels.
    pub decode_ns: u64,
    /// Layers whose write was skipped due to the gradient threshold (A9).
    pub write_skip_count: u64,
    /// Total write SQEs submitted.
    pub write_submitted: u64,
    /// Instantaneous NVMe read throughput (MB/s), populated by
    /// `Telemetry::record_nvme_throughput_mbps` (T-a fix).
    pub nvme_throughput_mbps: f32,
}

impl TelemetrySnapshot {
    /// Prefetch success rate: `completed / submitted`. Returns 0 if no data.
    pub fn prefetch_hit_rate(&self) -> f32 {
        if self.prefetch_submitted == 0 {
            return 0.0;
        }
        let ok = self.prefetch_completed.saturating_sub(self.prefetch_errors);
        ok as f32 / self.prefetch_submitted as f32
    }

    /// Hot-set hit rate: `hits / (hits + misses)`. Returns 0 if no data.
    pub fn hotset_hit_rate(&self) -> f32 {
        let total = self.hotset_hits + self.hotset_misses;
        if total == 0 { 0.0 } else { self.hotset_hits as f32 / total as f32 }
    }

    /// GPU idle fraction: `idle_us / total_elapsed_us`. Returns 0 if zero.
    pub fn gpu_idle_fraction(&self, total_elapsed_us: u64) -> f32 {
        if total_elapsed_us == 0 { 0.0 } else {
            self.gpu_idle_us as f32 / total_elapsed_us as f32
        }
    }

    /// Write-skip rate: `write_skip / (write_skip + write_submitted)`.
    pub fn write_skip_rate(&self) -> f32 {
        let total = self.write_skip_count + self.write_submitted;
        if total == 0 { 0.0 } else { self.write_skip_count as f32 / total as f32 }
    }

    /// Serialise to pretty JSON (paper dump).
    ///
    /// # Errors
    /// Returns `Err` if `serde_json` serialisation fails (never in practice).
    pub fn to_json(&self) -> crate::Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| {
            crate::FlowCastError::ProfileIo(format!("telemetry JSON serialise: {e}"))
        })
    }
}

/// Live telemetry counters shared across FlowCast threads.
///
/// Cloning is cheap (`Arc` bumps only).
#[derive(Clone)]
pub struct Telemetry {
    prefetch_submitted: Arc<AtomicU64>,
    prefetch_completed: Arc<AtomicU64>,
    prefetch_errors: Arc<AtomicU64>,
    miss_count: Arc<AtomicU64>,
    gpu_idle_us: Arc<AtomicU64>,
    hotset_hits: Arc<AtomicU64>,
    hotset_misses: Arc<AtomicU64>,
    nvme_bytes_read: Arc<AtomicU64>,
    queue_depth: Arc<AtomicU64>,
    ready_queue_depth: Arc<AtomicU64>,
    window_grow_events: Arc<AtomicU64>,
    window_shrink_events: Arc<AtomicU64>,
    decode_ns: Arc<AtomicU64>,
    write_skip_count: Arc<AtomicU64>,
    write_submitted: Arc<AtomicU64>,
    /// Instantaneous NVMe throughput stored as f32 bits (T-a fix).
    nvme_throughput_mbps_bits: Arc<std::sync::atomic::AtomicU32>,
    /// Timestamp of the last `on_layer_start` call (for idle-gap measurement).
    last_layer_start: Arc<std::sync::Mutex<Option<Instant>>>,
}

impl Telemetry {
    /// Create a zeroed telemetry instance.
    pub fn new() -> Self {
        Self {
            prefetch_submitted: Arc::new(AtomicU64::new(0)),
            prefetch_completed: Arc::new(AtomicU64::new(0)),
            prefetch_errors: Arc::new(AtomicU64::new(0)),
            miss_count: Arc::new(AtomicU64::new(0)),
            gpu_idle_us: Arc::new(AtomicU64::new(0)),
            hotset_hits: Arc::new(AtomicU64::new(0)),
            hotset_misses: Arc::new(AtomicU64::new(0)),
            nvme_bytes_read: Arc::new(AtomicU64::new(0)),
            queue_depth: Arc::new(AtomicU64::new(0)),
            ready_queue_depth: Arc::new(AtomicU64::new(0)),
            window_grow_events: Arc::new(AtomicU64::new(0)),
            window_shrink_events: Arc::new(AtomicU64::new(0)),
            decode_ns: Arc::new(AtomicU64::new(0)),
            write_skip_count: Arc::new(AtomicU64::new(0)),
            write_submitted: Arc::new(AtomicU64::new(0)),
            nvme_throughput_mbps_bits: Arc::new(std::sync::atomic::AtomicU32::new(0)),
            last_layer_start: Arc::new(std::sync::Mutex::new(None)),
        }
    }

    /// Record the current instantaneous NVMe read throughput (T-a fix).
    ///
    /// Called by the completion router after each batch to update the gauge.
    pub fn record_nvme_throughput_mbps(&self, mbps: f32) {
        self.nvme_throughput_mbps_bits
            .store(mbps.to_bits(), Ordering::Relaxed);
    }

    /// Record a submitted prefetch SQE of `byte_length` bytes.
    pub fn record_prefetch_submitted(&self, byte_length: u64) {
        self.prefetch_submitted.fetch_add(1, Ordering::Relaxed);
        self.nvme_bytes_read.fetch_add(byte_length, Ordering::Relaxed);
    }

    /// Record a completed prefetch CQE.
    pub fn record_prefetch_completed(&self, success: bool) {
        self.prefetch_completed.fetch_add(1, Ordering::Relaxed);
        if !success {
            self.prefetch_errors.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record a `PrefetchMiss` returned to M5.
    pub fn record_miss(&self) {
        self.miss_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record GPU idle gap: called at `on_layer_start`; measures time since
    /// the previous call (i.e., time the GPU was waiting for I/O).
    pub fn record_layer_start(&self) {
        let now = Instant::now();
        let mut guard = self.last_layer_start.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(prev) = guard.take() {
            let gap_us = prev.elapsed().as_micros() as u64;
            self.gpu_idle_us.fetch_add(gap_us, Ordering::Relaxed);
        }
        *guard = Some(now);
    }

    /// Add an explicit GPU idle duration (for callers that measure it directly).
    pub fn record_gpu_idle_us(&self, micros: u64) {
        self.gpu_idle_us.fetch_add(micros, Ordering::Relaxed);
    }

    /// Record a hot-set hit or miss.
    pub fn record_hotset(&self, hit: bool) {
        if hit {
            self.hotset_hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.hotset_misses.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Update the current io_uring SQ depth gauge.
    pub fn set_queue_depth(&self, depth: u64) {
        self.queue_depth.store(depth, Ordering::Relaxed);
    }

    /// Update the current ready-queue depth gauge.
    pub fn set_ready_queue_depth(&self, depth: u64) {
        self.ready_queue_depth.store(depth, Ordering::Relaxed);
    }

    /// Record an adaptive-window grow event.
    pub fn record_window_grow(&self) {
        self.window_grow_events.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an adaptive-window shrink event.
    pub fn record_window_shrink(&self) {
        self.window_shrink_events.fetch_add(1, Ordering::Relaxed);
    }

    /// Add nanoseconds spent in a decode kernel.
    pub fn record_decode_ns(&self, nanos: u64) {
        self.decode_ns.fetch_add(nanos, Ordering::Relaxed);
    }

    /// Record a write-skip event (gradient below threshold, no SSD write).
    pub fn record_write_skip(&self) {
        self.write_skip_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a write SQE submitted to NVMe.
    pub fn record_write_submitted(&self) {
        self.write_submitted.fetch_add(1, Ordering::Relaxed);
    }

    /// Snapshot all counters atomically (best-effort — no global lock).
    pub fn snapshot(&self) -> TelemetrySnapshot {
        TelemetrySnapshot {
            prefetch_submitted: self.prefetch_submitted.load(Ordering::Relaxed),
            prefetch_completed: self.prefetch_completed.load(Ordering::Relaxed),
            prefetch_errors: self.prefetch_errors.load(Ordering::Relaxed),
            miss_count: self.miss_count.load(Ordering::Relaxed),
            gpu_idle_us: self.gpu_idle_us.load(Ordering::Relaxed),
            hotset_hits: self.hotset_hits.load(Ordering::Relaxed),
            hotset_misses: self.hotset_misses.load(Ordering::Relaxed),
            nvme_bytes_read: self.nvme_bytes_read.load(Ordering::Relaxed),
            queue_depth: self.queue_depth.load(Ordering::Relaxed),
            ready_queue_depth: self.ready_queue_depth.load(Ordering::Relaxed),
            window_grow_events: self.window_grow_events.load(Ordering::Relaxed),
            window_shrink_events: self.window_shrink_events.load(Ordering::Relaxed),
            decode_ns: self.decode_ns.load(Ordering::Relaxed),
            write_skip_count: self.write_skip_count.load(Ordering::Relaxed),
            write_submitted: self.write_submitted.load(Ordering::Relaxed),
            nvme_throughput_mbps: f32::from_bits(
                self.nvme_throughput_mbps_bits.load(Ordering::Relaxed),
            ),
        }
    }
}

impl Default for Telemetry {
    fn default() -> Self {
        Self::new()
    }
}
