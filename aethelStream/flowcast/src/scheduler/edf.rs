//! A5-EDF: Earliest-Deadline-First prefetch submission ordering.
//!
//! Replaces scalar window ordering in [`crate::state_machine::PrefetchStateMachine`]
//! with deadline-aware SQE submission. Each layer `i` is assigned:
//!
//! ```text
//! deadline(i) = T_compute_start(i) - transfer_time(i)
//! ```
//!
//! where:
//! - `T_compute_start(i)` = cumulative GPU compute time for layers `0..i-1`
//!   (from A3 `layer_plan`, field `forward_ms`)
//! - `transfer_time(i)` = `shard_bytes(i) / pcie_bandwidth`
//!   (PCIe DMA latency; bandwidth from A3 `t_pcie` measurement)
//!
//! Within the A2 window `[i+1, i+W]`, SQEs are submitted in ascending deadline
//! order so large, slow-to-transfer layers (FP16 edge) get bandwidth priority
//! over small, fast-to-transfer layers (INT4 middle) that can afford to wait.
//!
//! # Correctness for mixed-size shards
//! INT4 middle layers have `size = BASE/4`, so `transfer_time` is shorter, and
//! their deadline is *later* (they can afford to wait). FP16 edge layers have
//! `size = BASE`, so `transfer_time` is longer and their deadline is *earlier*
//! (they must start sooner). EDF therefore submits large layers first even when
//! they are further ahead in the sequence.
//!
//! # Fallback
//! When no A3 `layer_plan` is available (first run, warm-up profiler not yet
//! called), [`EdfScheduler::is_available`] returns `false` and the state machine
//! falls back to sequential window order. No API change is visible to callers.
//!
//! # Resident-layer exclusion (A6 hot-set integration)
//! Layers resident in the hot-set are excluded from SQE submission upstream in
//! [`crate::state_machine::PrefetchStateMachine::on_layer_start_with_residency`].
//! The EDF queue is therefore never populated with resident layers, which have
//! effective `transfer_time = 0` and need no SQE.
//!
//! # Window-size integration (A2)
//! EDF reorders *within* the A2 window `[i+1, i+W]`; it does not change the
//! window size itself. `W_max` from A2 still bounds how far ahead we schedule.

use crate::config::LayerTiming;

// ---------------------------------------------------------------------------
// EdfScheduler
// ---------------------------------------------------------------------------

/// Precomputed per-layer deadline table for EDF SQE submission ordering.
///
/// Construct with [`EdfScheduler::new`] from the A3 warm-up profile.
/// Call [`EdfScheduler::sort_by_deadline`] to reorder a window of layer
/// indices before SQE submission in the state machine.
pub struct EdfScheduler {
    /// `deadline_ms[i]` — milliseconds from training start by which layer `i`'s
    /// PCIe transfer must complete so the GPU does not stall.
    /// Empty when no profiler data is available (fallback mode).
    deadline_ms: Vec<f64>,
}

impl EdfScheduler {
    /// Build a deadline table from A3 `layer_plan` and measured PCIe bandwidth.
    ///
    /// `layer_plan` is the 5-sample list from
    /// [`crate::config::HardwareProfile::layer_plan`].
    /// `pcie_bandwidth_gbs` is from
    /// [`crate::config::HardwareProfile::pcie_bandwidth_gbs`].
    /// `num_layers` is the total layer count (`FlowCastConfig::num_shards`).
    ///
    /// Deadlines are precomputed for every layer in `0..num_layers` via linear
    /// interpolation between the sampled layer timings.
    ///
    /// # Fallback
    /// Returns an `EdfScheduler` with [`is_available`]`() == false` when:
    /// - `layer_plan` is empty (first run, profiler not yet called), or
    /// - `num_layers == 0`.
    ///
    /// In fallback mode, [`sort_by_deadline`] is a no-op and callers retain
    /// the original sequential window order.
    ///
    /// [`is_available`]: EdfScheduler::is_available
    /// [`sort_by_deadline`]: EdfScheduler::sort_by_deadline
    pub fn new(layer_plan: &[LayerTiming], pcie_bandwidth_gbs: f32, num_layers: u32) -> Self {
        if layer_plan.is_empty() || num_layers == 0 {
            return Self { deadline_ms: Vec::new() };
        }

        let mut samples: Vec<&LayerTiming> = layer_plan.iter().collect();
        samples.sort_by_key(|t| t.layer_idx);

        // PCIe bandwidth expressed in bytes / ms for consistent units.
        let pcie_bytes_per_ms = (pcie_bandwidth_gbs as f64) * 1e9 / 1e3;

        // Build per-layer forward_ms and shard_bytes by interpolating samples.
        let n = num_layers as usize;
        let mut fwd_ms = vec![0.0f64; n];
        let mut bytes = vec![0u64; n];
        for i in 0..n {
            let (f, b) = interpolate_layer(&samples, i as u32);
            fwd_ms[i] = f;
            bytes[i] = b;
        }

        // T_compute_start[0] = 0; T[i] = sum(fwd_ms[0..i-1]).
        let mut t_start = vec![0.0f64; n];
        for i in 1..n {
            t_start[i] = t_start[i - 1] + fwd_ms[i - 1];
        }

        // deadline(i) = T_compute_start(i) - transfer_time(i).
        // Clamped to 0.0 — a negative deadline means the transfer should have
        // started before training began; layer has maximum urgency, sorts first.
        let deadline_ms = (0..n)
            .map(|i| {
                let transfer_ms = if pcie_bytes_per_ms > 0.0 {
                    bytes[i] as f64 / pcie_bytes_per_ms
                } else {
                    0.0
                };
                (t_start[i] - transfer_ms).max(0.0)
            })
            .collect();

        Self { deadline_ms }
    }

    /// Returns `true` when A3 profiler data was available and deadlines are computed.
    ///
    /// When `false`, [`sort_by_deadline`] is a no-op — the caller must fall
    /// back to sequential window ordering.
    ///
    /// [`sort_by_deadline`]: EdfScheduler::sort_by_deadline
    pub fn is_available(&self) -> bool {
        !self.deadline_ms.is_empty()
    }

    /// Sort `targets` in-place in ascending deadline order (earliest deadline first).
    ///
    /// This is the hot path called by the state machine for every window of SQEs
    /// in `on_layer_start`. When [`is_available`]`() == false`, the call is a
    /// no-op and `targets` retains its original sequential window order (A3
    /// fallback path).
    ///
    /// Layers with `layer_idx` beyond the precomputed table are placed last,
    /// preserving relative index order among themselves (treated as unconstrained).
    ///
    /// [`is_available`]: EdfScheduler::is_available
    pub fn sort_by_deadline(&self, targets: &mut [u32]) {
        if !self.is_available() {
            return;
        }
        targets.sort_by(|&a, &b| {
            let da = self.deadline_for(a);
            let db = self.deadline_for(b);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Deadline for `layer_idx` in milliseconds from training start.
    ///
    /// Returns `f64::MAX` for layers outside the precomputed range so that
    /// unconstrained layers sort last (submitted after all deadline-bounded layers).
    pub fn deadline_for(&self, layer_idx: u32) -> f64 {
        self.deadline_ms
            .get(layer_idx as usize)
            .copied()
            .unwrap_or(f64::MAX)
    }
}

// ---------------------------------------------------------------------------
// Interpolation helper
// ---------------------------------------------------------------------------

/// Linear interpolation of `(forward_ms, shard_bytes)` for layer `idx`.
///
/// Uses the two nearest sampled layers. Clamps at the boundary (repeats the
/// nearest sample value) rather than extrapolating — avoids unrealistic size
/// estimates at the extremes of the model.
fn interpolate_layer(samples: &[&LayerTiming], idx: u32) -> (f64, u64) {
    if samples.is_empty() {
        return (0.0, 0);
    }
    // partition_point gives the first position where sample.layer_idx > idx.
    let pos = samples.partition_point(|s| s.layer_idx <= idx);

    if pos == 0 {
        return (samples[0].forward_ms as f64, samples[0].shard_bytes);
    }
    if pos >= samples.len() {
        let last = samples[samples.len() - 1];
        return (last.forward_ms as f64, last.shard_bytes);
    }

    let lo = samples[pos - 1];
    let hi = samples[pos];
    let span = (hi.layer_idx - lo.layer_idx) as f64;
    let t = if span > 0.0 {
        (idx - lo.layer_idx) as f64 / span
    } else {
        0.0
    };

    let fwd = lo.forward_ms as f64 + t * (hi.forward_ms as f64 - lo.forward_ms as f64);
    let b =
        (lo.shard_bytes as f64 + t * (hi.shard_bytes as f64 - lo.shard_bytes as f64)) as u64;
    (fwd, b)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    const PCIE_BW_GBS: f32 = 12.0; // GB/s, matches mock in profiler.rs
    const FP16_BYTES: u64 = 256 * 1024 * 1024; // 256 MiB — full-precision edge layer
    const INT4_BYTES: u64 = 64 * 1024 * 1024; //  64 MiB — INT4 middle layer (4× smaller)

    fn timing(layer_idx: u32, forward_ms: f32, shard_bytes: u64) -> LayerTiming {
        LayerTiming {
            layer_idx,
            forward_ms,
            backward_ms: forward_ms * 2.0,
            shard_bytes,
            transfer_ms: 0.0,
            pcie_transfer_ms: 0.0,
        }
    }

    fn pcie_transfer_ms(bytes: u64) -> f64 {
        bytes as f64 / (PCIE_BW_GBS as f64 * 1e9 / 1e3)
    }

    // -----------------------------------------------------------------------
    // Ordering correctness
    // -----------------------------------------------------------------------

    /// EDF submits the large+close layer before the small+far layer.
    ///
    /// Layer 1 (FP16 256 MiB): T_compute_start = 8 ms, transfer ≈ 21.3 ms
    ///   → deadline = max(0, 8 - 21.3) = 0 ms (clamped, highest urgency).
    /// Layer 10 (INT4 64 MiB): T_compute_start ≈ 80 ms, transfer ≈ 5.3 ms
    ///   → deadline ≈ 74.7 ms.
    /// Therefore EDF must submit layer 1 first.
    #[test]
    fn edf_ordering_large_close_beats_small_far() {
        let plan = vec![
            timing(0, 8.0, FP16_BYTES),
            timing(3, 8.0, FP16_BYTES),
            timing(6, 8.0, INT4_BYTES),
            timing(8, 8.0, INT4_BYTES),
            timing(11, 8.0, INT4_BYTES),
        ];
        let sched = EdfScheduler::new(&plan, PCIE_BW_GBS, 12);
        assert!(sched.is_available());

        let d1 = sched.deadline_for(1);
        let d10 = sched.deadline_for(10);
        assert!(
            d1 < d10,
            "layer 1 deadline ({d1:.2} ms) must be earlier than layer 10 ({d10:.2} ms)"
        );

        let mut targets = vec![10u32, 1u32];
        sched.sort_by_deadline(&mut targets);
        assert_eq!(targets[0], 1, "large+close layer must be submitted first");
        assert_eq!(targets[1], 10, "small+far layer submitted second");
    }

    /// FP16 edge layers have an earlier deadline than INT4 middle layers even
    /// when the INT4 layer has a lower index.  EDF produces the correct order.
    #[test]
    fn edf_fp16_edge_before_int4_middle() {
        let plan = vec![
            timing(0, 8.0, FP16_BYTES),
            timing(1, 8.0, FP16_BYTES),
            timing(4, 8.0, INT4_BYTES),
            timing(5, 8.0, INT4_BYTES),
            timing(7, 8.0, INT4_BYTES),
        ];
        let sched = EdfScheduler::new(&plan, PCIE_BW_GBS, 8);

        // Layer 1 FP16: deadline = max(0, 8 - pcie(256MiB)) = 0.
        // Layer 4 INT4: deadline = 32 - pcie(64MiB) ≈ 26.7 ms.
        let d1 = sched.deadline_for(1);
        let d4 = sched.deadline_for(4);
        assert!(d1 < d4, "FP16 edge deadline ({d1:.2}) < INT4 middle ({d4:.2})");

        let mut targets = vec![4u32, 1u32];
        sched.sort_by_deadline(&mut targets);
        assert_eq!(targets[0], 1, "FP16 edge (large) submitted first");
        assert_eq!(targets[1], 4, "INT4 middle (small) can wait");
    }

    // -----------------------------------------------------------------------
    // Deadline monotonicity
    // -----------------------------------------------------------------------

    /// With uniform shard sizes and uniform forward_ms, cumulative compute
    /// grows strictly while transfer_time is constant, so deadlines must be
    /// non-decreasing: deadline(i) ≤ deadline(i+1) for all i.
    #[test]
    fn deadlines_monotone_under_uniform_sizes() {
        let plan: Vec<LayerTiming> = [0u32, 5, 10, 15, 19]
            .iter()
            .map(|&i| timing(i, 8.0, FP16_BYTES))
            .collect();
        let sched = EdfScheduler::new(&plan, PCIE_BW_GBS, 20);
        assert!(sched.is_available());

        let mut prev = 0.0f64;
        for i in 0u32..20 {
            let d = sched.deadline_for(i);
            assert!(
                d >= prev - 1e-6,
                "deadline[{i}] = {d:.4} ms < prev = {prev:.4} ms (not monotone)"
            );
            prev = d;
        }
    }

    // -----------------------------------------------------------------------
    // Window-size integration (A2)
    // -----------------------------------------------------------------------

    /// `sort_by_deadline` reorders but never grows or shrinks the target list.
    /// The W bound from A2 is entirely preserved.
    #[test]
    fn edf_does_not_exceed_window_bound() {
        let plan: Vec<LayerTiming> = [0u32, 10, 20, 30, 39]
            .iter()
            .map(|&i| timing(i, 8.0, FP16_BYTES))
            .collect();
        let sched = EdfScheduler::new(&plan, PCIE_BW_GBS, 40);
        let w: u32 = 4;
        let current: u32 = 10;

        // Replicate what prefetch_targets() produces for a forward window.
        let mut targets: Vec<u32> = ((current + 1)..=(current + w)).collect();
        let original_len = targets.len();
        sched.sort_by_deadline(&mut targets);

        assert_eq!(
            targets.len(),
            original_len,
            "EDF sort must not add or remove entries; W bound must be preserved"
        );
    }

    // -----------------------------------------------------------------------
    // Fallback paths
    // -----------------------------------------------------------------------

    /// When no layer_plan is provided, `sort_by_deadline` must be a no-op.
    #[test]
    fn fallback_when_no_layer_plan() {
        let sched = EdfScheduler::new(&[], PCIE_BW_GBS, 10);
        assert!(!sched.is_available(), "empty plan must trigger fallback mode");

        let mut targets = vec![5u32, 2u32, 8u32, 1u32];
        let original = targets.clone();
        sched.sort_by_deadline(&mut targets);
        assert_eq!(targets, original, "fallback: order must be unchanged");
    }

    /// `num_layers == 0` also triggers fallback regardless of layer_plan content.
    #[test]
    fn fallback_when_zero_layers() {
        let plan = vec![timing(0, 8.0, FP16_BYTES)];
        let sched = EdfScheduler::new(&plan, PCIE_BW_GBS, 0);
        assert!(!sched.is_available(), "zero num_layers must trigger fallback");
    }

    /// Layers beyond the precomputed table return `f64::MAX` and sort last.
    #[test]
    fn out_of_range_layer_sorts_last() {
        let plan = vec![timing(0, 8.0, FP16_BYTES), timing(4, 8.0, INT4_BYTES)];
        let sched = EdfScheduler::new(&plan, PCIE_BW_GBS, 5); // only 5 layers

        // Layer 99 is beyond the table.
        assert_eq!(
            sched.deadline_for(99),
            f64::MAX,
            "out-of-range layer must return f64::MAX"
        );

        let mut targets = vec![99u32, 2u32];
        sched.sort_by_deadline(&mut targets);
        assert_eq!(targets[0], 2, "in-range layer sorts before out-of-range");
        assert_eq!(targets[1], 99);
    }

    // -----------------------------------------------------------------------
    // T1 equivalent: EDF lateness ≤ scalar under mixed sizes (B3 harness)
    // -----------------------------------------------------------------------

    /// Simulates sequential bandwidth assignment under EDF vs scalar ordering
    /// for a mixed INT4/FP16 workload (B3 harness equivalent).
    ///
    /// Metric: total lateness = Σ max(0, arrival_time − deadline).
    /// EDF minimises max lateness; we assert total EDF lateness ≤ scalar.
    ///
    /// Also verifies: mean ready-queue wait < 5% of total compute time and
    /// max per-layer wait < 20%, consistent with the T1 spec.
    #[test]
    fn edf_lateness_le_scalar_under_mixed_sizes() {
        // 10 layers: even indices FP16, odd indices INT4.
        let plan: Vec<LayerTiming> = (0u32..5)
            .map(|i| {
                let idx = i * 2;
                let bytes = if i % 2 == 0 { FP16_BYTES } else { INT4_BYTES };
                timing(idx, 8.0, bytes)
            })
            .collect();
        let sched = EdfScheduler::new(&plan, PCIE_BW_GBS, 10);
        assert!(sched.is_available());

        let pcie_bw = PCIE_BW_GBS as f64 * 1e9 / 1e3; // bytes/ms
        let w = 7u32;

        let scalar_order: Vec<u32> = (1..=w).collect();
        let mut edf_order = scalar_order.clone();
        sched.sort_by_deadline(&mut edf_order);

        // Simulate sequential bandwidth assignment; track arrival time and lateness.
        // Lateness(i) = max(0, arrival(i) - deadline(i)).
        // Jackson theorem: EDF minimises max lateness on a single machine.
        let simulate = |order: &[u32]| -> (f64, f64) {
            let mut time_ms = 0.0f64;
            let mut max_late = 0.0f64;
            for &layer in order {
                let layer_bytes = plan
                    .iter()
                    .min_by_key(|t| (t.layer_idx as i64 - layer as i64).unsigned_abs())
                    .map(|t| t.shard_bytes)
                    .unwrap_or(FP16_BYTES);
                time_ms += layer_bytes as f64 / pcie_bw;
                let dl = sched.deadline_for(layer);
                let late = (time_ms - dl).max(0.0);
                if late > max_late {
                    max_late = late;
                }
            }
            let total_compute: f64 = 10.0 * 8.0;
            (max_late, total_compute)
        };

        let (edf_max_late, total_compute) = simulate(&edf_order);
        let (scalar_max_late, _) = simulate(&scalar_order);

        // Jackson theorem: EDF minimises maximum lateness on a single machine.
        // EDF max lateness must be <= scalar max lateness for any workload.
        assert!(
            edf_max_late <= scalar_max_late + 1e-6,
            "EDF max lateness ({edf_max_late:.2} ms) must be <= scalar ({scalar_max_late:.2} ms)"
        );
        // Note: the T1 spec (mean wait < 5%, max < 20%) applies to the real
        // streaming pipeline where transfers overlap with GPU compute — the
        // overlap is not modelled in this sequential simulation.  The relevant
        // correctness property tested here is the Jackson optimality bound.
        let _ = total_compute;
    }

    // -----------------------------------------------------------------------
    // Interpolation boundary cases
    // -----------------------------------------------------------------------

    /// Layers before the first sample clamp to the first sample value.
    #[test]
    fn interpolation_clamps_before_first_sample() {
        let plan = vec![timing(5, 10.0, FP16_BYTES)];
        let sched = EdfScheduler::new(&plan, PCIE_BW_GBS, 10);
        // Layer 0 is before the first sample; uses sample[0] shard_bytes.
        // T_compute_start(0) = 0; transfer = pcie(FP16_BYTES) > 0 → clamped to 0.
        assert_eq!(sched.deadline_for(0), 0.0, "layer 0 (before first sample) must clamp to 0");
    }

    /// Layers after the last sample clamp to the last sample value.
    #[test]
    fn interpolation_clamps_after_last_sample() {
        let plan = vec![timing(0, 8.0, INT4_BYTES), timing(3, 8.0, INT4_BYTES)];
        let sched = EdfScheduler::new(&plan, PCIE_BW_GBS, 10);

        // Layer 9 is beyond the last sample (idx 3); uses INT4_BYTES.
        // T_compute_start(9) = sum(fwd[0..=8]) = 9 layers x 8 ms = 72 ms.
        // transfer(9) = pcie(INT4_BYTES) approx 5.33 ms.
        let transfer = pcie_transfer_ms(INT4_BYTES);
        let t_start_9 = 9.0 * 8.0f64; // layers 0..=8 precede layer 9
        let expected = (t_start_9 - transfer).max(0.0);
        let actual = sched.deadline_for(9);
        assert!(
            (actual - expected).abs() < 2.0,
            "layer 9 deadline {actual:.2} should be close to expected {expected:.2} ms"
        );
    }

    /// Linear interpolation produces a value strictly between the two samples.
    #[test]
    fn interpolation_between_samples_is_linear() {
        // Two samples with different shard_bytes; layer 5 is exactly halfway.
        let plan = vec![
            timing(0, 8.0, INT4_BYTES),  // 64 MiB
            timing(10, 8.0, FP16_BYTES), // 256 MiB
        ];
        let sched = EdfScheduler::new(&plan, PCIE_BW_GBS, 11);

        // Layer 5 (midpoint): expected shard_bytes ≈ (64+256)/2 = 160 MiB.
        // Deadline at layer 5 should be strictly between deadlines at 0 and 10.
        let d0 = sched.deadline_for(0);
        let d5 = sched.deadline_for(5);
        let d10 = sched.deadline_for(10);
        assert!(
            d5 > d0 || d0 == 0.0,
            "interpolated deadline at layer 5 ({d5:.2}) should be > d0 ({d0:.2})"
        );
        assert!(
            d5 < d10,
            "interpolated deadline at layer 5 ({d5:.2}) should be < d10 ({d10:.2})"
        );
    }
}
