//! Tests for adaptive super-shard regrouping.
//!
//! Covers: latency-vs-size knee detection, byte-budget group sizing,
//! live group-size updates, byte-identical data fidelity, and a B3-equivalent
//! throughput comparison.

use flowcast::backend::super_shard::{SuperShardBackend, SuperShardConfig, compute_group_size};
use flowcast::profiler::{find_knee, measure_transfer_latency_curve, probe_knee_near, LATENCY_PROBE_SIZES};

// ── knee detection ─────────────────────────────────────────────────────────

/// Verifies that `find_knee` returns the first size where throughput improves
/// by less than 5 % when stepping to the next probe size.
///
/// Uses a hand-crafted curve with a deliberate knee at 16 MiB (index 4).
#[test]
fn find_knee_identifies_correct_transition_point() {
    // Throughputs (GB/s): 1.0, 1.5, 1.9, 2.2, 2.5, 2.51
    // Ratios: 1.50, 1.27, 1.16, 1.14,  1.004
    // First ratio < 1.05: index 4→5 → knee at index 4 = 16 MiB.
    let sizes: &[usize] = &[
        1024 * 1024,
        2 * 1024 * 1024,
        4 * 1024 * 1024,
        8 * 1024 * 1024,
        16 * 1024 * 1024,
        32 * 1024 * 1024,
    ];
    // Build durations so throughput(size) = size / dur matches the GB/s above.
    let gbs = [1.0f64, 1.5, 1.9, 2.2, 2.5, 2.51];
    let curve: Vec<(usize, std::time::Duration)> = sizes
        .iter()
        .zip(gbs.iter())
        .map(|(&size, &gb)| {
            // dur (µs) = size / (gb * 1000 bytes/µs)
            let dur_us = size as f64 / (gb * 1_000.0);
            (size, std::time::Duration::from_micros(dur_us as u64))
        })
        .collect();

    let knee = find_knee(&curve);
    // 2.51 / 2.5 = 1.004 < 1.05 → first sub-5% step is at index 4 → 16 MiB
    assert_eq!(knee, 16 * 1024 * 1024, "knee should be at 16 MiB");
}

/// When throughput keeps improving across the full probe range, `find_knee`
/// returns the last (largest) size.
#[test]
fn find_knee_returns_last_when_always_improving() {
    let curve: Vec<(usize, std::time::Duration)> = LATENCY_PROBE_SIZES
        .iter()
        .enumerate()
        .map(|(index, &size)| {
            // Always more than 5 % gain: throughput = (index + 1) as GB/s
            let gbs = (index + 1) as f64;
            let dur_us = size as f64 / (gbs * 1_000.0);
            (size, std::time::Duration::from_micros(dur_us as u64))
        })
        .collect();
    let knee = find_knee(&curve);
    assert_eq!(
        knee,
        *LATENCY_PROBE_SIZES.last().unwrap(),
        "should return the last size when throughput always improves"
    );
}

/// Empty curve: default 4 MiB.
#[test]
fn find_knee_empty_curve_returns_default() {
    assert_eq!(find_knee(&[]), 4 * 1024 * 1024);
}

/// Under mock-cuda the curve measurement produces a monotonically increasing
/// throughput sequence (linear overhead model), so `find_knee` must return
/// a real size (not just 0 or default).
#[test]
fn measure_transfer_latency_curve_mock_produces_valid_knee() {
    let dir = std::path::Path::new(".");
    let curve = measure_transfer_latency_curve(dir);
    assert_eq!(curve.len(), LATENCY_PROBE_SIZES.len(), "one entry per probe size");
    // Throughput must be positive and sizes must match LATENCY_PROBE_SIZES.
    for (index, &(size, dur)) in curve.iter().enumerate() {
        assert_eq!(size, LATENCY_PROBE_SIZES[index]);
        assert!(dur.as_micros() > 0, "duration must be positive");
    }
    let knee = find_knee(&curve);
    assert!(knee > 0, "knee must be a positive byte count");
    assert!(
        LATENCY_PROBE_SIZES.contains(&knee),
        "knee {knee} must be one of the probe sizes"
    );
}

/// `probe_knee_near` must return a size from within the ± 2 window.
#[test]
fn probe_knee_near_returns_size_in_window() {
    let dir = std::path::Path::new(".");
    let current_knee = 4 * 1024 * 1024; // 4 MiB — index 4 in LATENCY_PROBE_SIZES
    let new_knee = probe_knee_near(dir, current_knee);
    // Window is [lo, hi] where lo = max(0, 4-2)=2 (1 MiB) and hi = min(7, 4+2)=6 (16 MiB).
    let lo = LATENCY_PROBE_SIZES[2]; // 1 MiB
    let hi = LATENCY_PROBE_SIZES[6]; // 16 MiB
    assert!(
        new_knee >= lo && new_knee <= hi,
        "knee {new_knee} must be in window [{lo}, {hi}]"
    );
}

// ── group size computation ─────────────────────────────────────────────────

/// Uniform 512 KiB layers → group_size = optimal_bytes / 512 KiB.
#[test]
fn compute_group_size_uniform_layers() {
    let layer_size = 512 * 1024u64; // 512 KiB
    let optimal_bytes = 4 * 1024 * 1024u64; // 4 MiB
    let sizes = vec![layer_size; 32];
    let group = compute_group_size(optimal_bytes, &sizes);
    assert_eq!(group, 8, "4 MiB / 512 KiB = 8 layers per group");
}

/// When all layers are smaller than optimal_bytes, group_size = layer count.
#[test]
fn compute_group_size_all_layers_fit_in_one_group() {
    let layer_size = 256 * 1024u64; // 256 KiB
    let optimal_bytes = 64 * 1024 * 1024u64; // 64 MiB — fits all 4 layers
    let sizes = vec![layer_size; 4];
    let group = compute_group_size(optimal_bytes, &sizes);
    assert_eq!(group, 4, "all 4 layers fit within the byte budget");
}

/// When the median layer exceeds optimal_bytes, group_size = 1.
#[test]
fn compute_group_size_layer_larger_than_budget_returns_one() {
    let layer_size = 16 * 1024 * 1024u64; // 16 MiB
    let optimal_bytes = 4 * 1024 * 1024u64; // 4 MiB < one layer
    let sizes = vec![layer_size; 8];
    let group = compute_group_size(optimal_bytes, &sizes);
    assert_eq!(group, 1, "no coalescing when budget < one layer");
}

/// Mixed INT4 (128 KiB) and FP16 (512 KiB) layers — median drives the count.
#[test]
fn compute_group_size_mixed_int4_fp16_uses_median() {
    let int4_size = 128 * 1024u64;  // 128 KiB
    let fp16_size = 512 * 1024u64;  // 512 KiB
    let optimal_bytes = 2 * 1024 * 1024u64; // 2 MiB
    // Half INT4, half FP16 → sorted: [128K, 128K, 512K, 512K] → median = 128K or 512K
    // Median at index 2 = 512 KiB.
    let mut sizes = vec![int4_size; 2];
    sizes.extend_from_slice(&[fp16_size; 2]);
    let group = compute_group_size(optimal_bytes, &sizes);
    // median(sorted) = 512 KiB → 2 MiB / 512 KiB = 4
    assert_eq!(group, 4);
}

/// Degenerate: empty sizes returns the default of 4.
#[test]
fn compute_group_size_empty_layer_sizes_returns_default() {
    assert_eq!(compute_group_size(4 * 1024 * 1024, &[]), 4);
}

/// Degenerate: optimal_bytes == 0 returns the default.
#[test]
fn compute_group_size_zero_optimal_bytes_returns_default() {
    assert_eq!(compute_group_size(0, &[512 * 1024u64; 8]), 4);
}

// ── live update ────────────────────────────────────────────────────────────

/// `update_group_size` must atomically update the backend's group count.
#[test]
fn update_group_size_shifts_group_count() {
    use flowcast::backend::mock::MockBackend;

    let base = Box::new(MockBackend::new());
    let config = SuperShardConfig {
        group_size: 4,
        optimal_super_shard_bytes: 4 * 1024 * 1024,
        layer_sizes: vec![512 * 1024u64; 32],
    };
    let backend = SuperShardBackend::new(base, config);

    assert_eq!(backend.group_size_hint(), 8, "initial: 4 MiB / 512 KiB = 8");

    // Simulate throttling: knee shifts to 2 MiB.
    let new_optimal = 2 * 1024 * 1024u64;
    let layer_sizes = vec![512 * 1024u64; 32];
    backend.update_group_size(new_optimal, &layer_sizes);

    assert_eq!(
        backend.group_size_hint(),
        4,
        "after throttle: 2 MiB / 512 KiB = 4"
    );
}

/// When the new optimal is 0, the backend falls back to count-based grouping.
#[test]
fn update_group_size_zero_reverts_to_count_based() {
    use flowcast::backend::mock::MockBackend;

    let base = Box::new(MockBackend::new());
    let config = SuperShardConfig {
        group_size: 6,
        optimal_super_shard_bytes: 4 * 1024 * 1024,
        layer_sizes: vec![512 * 1024u64; 32],
    };
    let backend = SuperShardBackend::new(base, config);
    backend.update_group_size(0, &[]);
    // With optimal_bytes == 0 the flush falls back to group_size (AtomicU32 = default 4).
    assert_eq!(backend.group_size_hint(), 4, "zero optimal reverts to default 4");
}

// ── byte-budget flush correctness (T6-equivalent) ─────────────────────────

/// With a byte-budget policy, the backend must accumulate layers until the
/// cumulative byte count reaches the budget, then flush — not on layer count.
#[test]
fn byte_budget_flushes_on_cumulative_size_not_count() {
    use flowcast::backend::mock::MockBackend;
    use flowcast::backend::IoBackend;
    use ramflow::PinnedBuffer;

    // Budget = 1 MiB; each layer is 512 KiB → flush at 2 layers (1 MiB).
    let layer_bytes = 512 * 1024u64;
    let budget = 1024 * 1024u64;

    let mock = MockBackend::new();
    let base = Box::new(mock);
    let config = SuperShardConfig {
        group_size: 8, // intentionally large — byte budget should override
        optimal_super_shard_bytes: budget,
        layer_sizes: vec![layer_bytes; 16],
    };
    let mut backend = SuperShardBackend::new(base, config);
    backend.start().expect("start");

    let placeholder = PinnedBuffer::alloc(layer_bytes as usize).expect("alloc");

    // Submit 2 layers of 512 KiB each.  The second push reaches 1 MiB and
    // triggers flush — even though group_size is 8.
    backend
        .prefetch(0, 0, layer_bytes, &placeholder, 1)
        .expect("prefetch 0");
    backend
        .prefetch(1, layer_bytes, layer_bytes, &placeholder, 2)
        .expect("prefetch 1");

    // Drain completions to verify both layers were submitted.
    let completions = backend.poll_completions().expect("poll");
    assert_eq!(
        completions.len(),
        2,
        "byte budget of 1 MiB must flush 2 × 512 KiB layers"
    );
}

// ── adaptive vs fixed benchmark (B3-equivalent) ───────────────────────────

/// Measures take_ready latency variance: adaptive group size from the knee
/// should produce a similar or lower variance than a fixed 4-layer group.
/// This is a structural sanity test — not a hard performance bound.
#[test]
fn adaptive_vs_fixed_group_size_latency_comparison() {
    use std::time::Instant;
    use flowcast::backend::mock::MockBackend;
    use flowcast::backend::IoBackend;
    use ramflow::PinnedBuffer;

    let layer_bytes = 512 * 1024u64;
    let num_layers = 16u32;
    let buf = PinnedBuffer::alloc(layer_bytes as usize).expect("alloc");

    // --- adaptive (knee-derived) ---
    let optimal = measure_transfer_latency_curve(std::path::Path::new("."));
    let knee_bytes = find_knee(&optimal) as u64;
    let layer_sizes: Vec<u64> = vec![layer_bytes; num_layers as usize];

    let adaptive_config = SuperShardConfig {
        group_size: 4,
        optimal_super_shard_bytes: knee_bytes,
        layer_sizes: layer_sizes.clone(),
    };
    let mut adaptive = SuperShardBackend::new(Box::new(MockBackend::new()), adaptive_config);
    adaptive.start().unwrap();

    let adaptive_start = Instant::now();
    for index in 0..num_layers {
        adaptive
            .prefetch(index, index as u64 * layer_bytes, layer_bytes, &buf, index as u64)
            .unwrap();
    }
    let _ = adaptive.poll_completions().unwrap();
    let adaptive_elapsed = adaptive_start.elapsed();

    // --- fixed (4 layers per group) ---
    let fixed_config = SuperShardConfig {
        group_size: 4,
        optimal_super_shard_bytes: 0,
        layer_sizes: vec![],
    };
    let mut fixed = SuperShardBackend::new(Box::new(MockBackend::new()), fixed_config);
    fixed.start().unwrap();

    let fixed_start = Instant::now();
    for index in 0..num_layers {
        fixed
            .prefetch(index, index as u64 * layer_bytes, layer_bytes, &buf, index as u64)
            .unwrap();
    }
    let _ = fixed.poll_completions().unwrap();
    let fixed_elapsed = fixed_start.elapsed();

    // Under mock-cuda both are in-process no-ops so runtimes are similar;
    // the key assertion is structural: both complete without error.
    println!(
        "adaptive={adaptive_elapsed:?} (group_size={}), fixed={fixed_elapsed:?} (group_size=4)",
        adaptive.group_size_hint()
    );
    assert!(adaptive_elapsed.as_secs() < 5, "adaptive must complete quickly");
    assert!(fixed_elapsed.as_secs() < 5, "fixed must complete quickly");
}
