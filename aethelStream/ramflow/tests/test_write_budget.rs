// tests/test_write_budget.rs — Sprint 5 Module 2 Tests 10, 11, 12
//
// Run Tests 10 and 11 (ssd-wear required):
//   cargo test --no-default-features --features "mock-cuda,ssd-wear" --test test_write_budget
// Run Test 12 (no ssd-wear needed):
//   cargo test --no-default-features --features mock-cuda --test test_write_budget test_12
//
// Test 10: WriteBudgetManager tracks NVMe wear and auto-switches WriteStrategy.
//   MockSmartSource simulates a device at 1 000 units already written.
//   Budget = 100 units.  Consuming 51 units switches to DeltaCompress;
//   consuming 91 switches to Deferred{4}.
//
// Test 11: compress_delta / decompress_and_apply_delta round-trip is bit-exact.
//   Generate realistic FP16-shaped "original" and "updated" weight buffers.
//   Compress delta → decompress+apply → assert byte-for-byte equal to "updated".
//
// Test 12: INT8 checkpoint quantization round-trip stays within the theoretical
//   error bound (no GPU required — pure Rust simulation of the CUDA kernel math).
//   Assert max absolute error < 1/127.0 per element (one quantisation step).

#[cfg(feature = "ssd-wear")]
use ramflow::nvme::write_budget::{
    compress_delta, decompress_and_apply_delta, MockSmartSource, WriteBudgetManager, WriteStrategy,
    SMART_UNIT_BYTES,
};
#[cfg(feature = "ssd-wear")]
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Test 10 — wear budget tracking + WriteStrategy auto-switch
// ---------------------------------------------------------------------------

/// Verify WriteBudgetManager tracks SMART units and switches strategy at the
/// correct consumption thresholds: Full → DeltaCompress at 50%, Deferred{4} at 90%.
#[cfg(feature = "ssd-wear")]
#[test]
fn test_10_wear_tracking_consumes_budget_and_auto_switches_strategy() {
    // MockSmartSource reports 1 000 units already on the device baseline.
    let source = Box::new(MockSmartSource::new(1_000));
    // Budget: 100 SMART units = 100 × 512 000 = 51 200 000 bytes.
    let budget_bytes: u64 = 100 * SMART_UNIT_BYTES;
    let manager =
        WriteBudgetManager::new_with_source(PathBuf::from("/dev/nvme0"), budget_bytes, source);

    assert_eq!(
        manager.units_at_start(),
        1_000,
        "units_at_start must reflect MockSmartSource baseline"
    );
    assert_eq!(
        manager.remaining(),
        100,
        "remaining must equal budget_units (100) before any consumption"
    );
    assert_eq!(
        manager.strategy(),
        WriteStrategy::Full,
        "strategy must be Full at start — 100% remaining"
    );

    // ── Consume 10 units (10% consumed, 90% remaining) → still Full ──────────
    manager
        .consume(10 * SMART_UNIT_BYTES)
        .expect("consume 10 units");
    assert_eq!(manager.remaining(), 90, "remaining after 10 units");
    assert_eq!(
        manager.strategy(),
        WriteStrategy::Full,
        "strategy must remain Full while >50% remaining"
    );

    // ── Consume 41 more → 51 total → 49 remaining (<50%) → DeltaCompress ─────
    manager
        .consume(41 * SMART_UNIT_BYTES)
        .expect("consume 41 units");
    assert_eq!(manager.remaining(), 49, "remaining after 51 units consumed");
    assert_eq!(
        manager.strategy(),
        WriteStrategy::DeltaCompress,
        "strategy must switch to DeltaCompress when remaining ≤50%"
    );

    // ── Consume 40 more → 91 total → 9 remaining (<10%) → Deferred{4} ───────
    manager
        .consume(40 * SMART_UNIT_BYTES)
        .expect("consume 40 units");
    assert_eq!(manager.remaining(), 9, "remaining after 91 units consumed");
    assert_eq!(
        manager.strategy(),
        WriteStrategy::Deferred { batch_size: 4 },
        "strategy must switch to Deferred{{4}} when remaining ≤10%"
    );

    // Cumulative snapshot must match our arithmetic.
    assert_eq!(
        manager.units_written_snapshot(),
        91,
        "units_written_snapshot must equal total units consumed (10+41+40=91)"
    );
}

// ---------------------------------------------------------------------------
// Test 11 — delta round-trip: bit-exact FP16 weight reconstruction
// ---------------------------------------------------------------------------

/// Verify compress_delta / decompress_and_apply_delta is a lossless round-trip.
/// Delta = updated - original (LE i16 wrapping); application = original + delta.
/// Result must match updated exactly (every byte equal).
#[cfg(feature = "ssd-wear")]
#[test]
fn test_11_delta_round_trip_reconstructs_fp16_weights_exactly() {
    use std::time::{SystemTime, UNIX_EPOCH};

    // Unique temp dir per test run to avoid interference from parallel runs.
    let tmp_dir = std::env::temp_dir().join(format!(
        "ramflow_delta_test_{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    std::fs::create_dir_all(&tmp_dir).expect("create temp dir for delta test");

    const N_ELEMENTS: usize = 4_096; // 4 096 FP16 values = 8 192 bytes
    const LAYER_IDX: u32 = 7;

    // Generate deterministic FP16-shaped bytes: mimic a realistic weight tensor.
    // Low byte = (elem * 3) & 0xFF, high byte = (elem >> 5) & 0xFF.
    let original: Vec<u8> = (0..N_ELEMENTS * 2)
        .map(|byte_idx| {
            let element = byte_idx / 2;
            if byte_idx % 2 == 0 {
                ((element * 3) & 0xFF) as u8
            } else {
                ((element >> 5) & 0xFF) as u8
            }
        })
        .collect();

    // Updated: apply a small cyclic delta (-1, 0, +1) to each LE i16 value.
    let updated: Vec<u8> = (0..N_ELEMENTS)
        .flat_map(|element_idx| {
            let orig_val =
                i16::from_le_bytes([original[element_idx * 2], original[element_idx * 2 + 1]]);
            let delta: i16 = (element_idx % 3) as i16 - 1; // -1, 0, +1 cycling
            orig_val.wrapping_add(delta).to_le_bytes()
        })
        .collect();

    // Compress delta to the temp directory.
    let compressed_size = compress_delta(LAYER_IDX, &updated, &original, &tmp_dir)
        .expect("compress_delta must succeed");
    assert!(
        compressed_size > 0,
        "compressed delta must produce a non-zero .delta.zstd file"
    );

    // Decompress and apply to recover the updated tensor.
    let reconstructed = decompress_and_apply_delta(LAYER_IDX, &original, &tmp_dir)
        .expect("decompress_and_apply_delta must succeed");

    assert_eq!(
        reconstructed.len(),
        updated.len(),
        "reconstructed length {0} must equal updated length {1}",
        reconstructed.len(),
        updated.len()
    );
    assert_eq!(
        reconstructed, updated,
        "delta round-trip must be bit-exact: every byte must match"
    );

    // Best-effort cleanup — test result is already recorded.
    let _ = std::fs::remove_dir_all(&tmp_dir);
}

// ---------------------------------------------------------------------------
// Test 11b — delta round-trip: wrapping semantics audit (M2-8c)
// ---------------------------------------------------------------------------

/// Verify that wrapping_sub(compress) followed by wrapping_add(decompress)
/// reconstructs the updated weights exactly — for ALL i16 values including
/// wrap-around cases that would fail with saturating or checked arithmetic.
///
/// Audit finding M2-8c: No in-file round-trip test for delta compression.
/// This test executes the contract documented algebraically in the compress_delta
/// and decompress_and_apply_delta implementations.
///
/// If wrapping semantics were accidentally changed to saturating_sub, the
/// round-trip would fail for pairs that cross the i16 min/max boundary.
#[cfg(feature = "ssd-wear")]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#[test]
fn test_11b_delta_round_trip_wrapping_semantics_bit_exact() {
    use std::time::{SystemTime, UNIX_EPOCH};

    // Unique temp dir per test run to avoid interference from parallel runs.
    let tmp_dir = std::env::temp_dir().join(format!(
        "ramflow_delta_wrap_test_{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    std::fs::create_dir_all(&tmp_dir).expect("create temp dir for delta test");

    const LAYER_IDX: u32 = 42;

    // ── Case 1: Normal values (no wrap) ────────────────────────────────────
    let original_case1: Vec<u8> = [0i16, 100, -100, 1000, -1000]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let updated_case1: Vec<u8> = [10i16, 200, -50, 1500, -500]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();

    let _compressed_size = compress_delta(LAYER_IDX, &updated_case1, &original_case1, &tmp_dir)
        .expect("compress_delta case 1 failed");
    let restored = decompress_and_apply_delta(LAYER_IDX, &original_case1, &tmp_dir)
        .expect("decompress_and_apply_delta case 1 failed");
    assert_eq!(
        restored, updated_case1,
        "normal-values round-trip failed: restored != updated"
    );

    // ── Case 2: Wrap-around values — delta crosses i16::MAX boundary ───────
    // original = 30000, updated = -30000
    // delta = (-30000i16).wrapping_sub(30000i16) = -30000 - 30000 = -60000 (mod 2^16)
    //       = -60000 + 65536 = 5536
    // restored = 30000i16.wrapping_add(5536) = 35536 (mod 2^16)
    //          = 35536 - 65536 = -30000 ✓
    let orig_wrap: Vec<u8> = [30000i16].iter().flat_map(|v| v.to_le_bytes()).collect();
    let upd_wrap: Vec<u8> = [(-30000i16)].iter().flat_map(|v| v.to_le_bytes()).collect();

    let _compressed_size_wrap = compress_delta(LAYER_IDX, &upd_wrap, &orig_wrap, &tmp_dir)
        .expect("compress_delta wrap case failed");
    let restored_wrap = decompress_and_apply_delta(LAYER_IDX, &orig_wrap, &tmp_dir)
        .expect("decompress_and_apply_delta wrap case failed");
    assert_eq!(
        restored_wrap, upd_wrap,
        "wrap-around round-trip failed: delta cross-boundary case broke"
    );

    // ── Case 3: Zero delta (original == updated) — round-trip is identity ──
    let same: Vec<u8> = [42i16, -1, i16::MAX, i16::MIN]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();

    let _compressed_size_zero = compress_delta(LAYER_IDX, &same, &same, &tmp_dir)
        .expect("compress_delta zero case failed");
    let restored_zero = decompress_and_apply_delta(LAYER_IDX, &same, &tmp_dir)
        .expect("decompress_and_apply_delta zero case failed");
    assert_eq!(
        restored_zero, same,
        "zero-delta identity failed: same buffer should round-trip to itself"
    );

    // Best-effort cleanup — test result is already recorded.
    let _ = std::fs::remove_dir_all(&tmp_dir);
}

// ---------------------------------------------------------------------------
// Test 12 — INT8 checkpoint quantisation round-trip (pure Rust simulation)
// ---------------------------------------------------------------------------

/// Verify that per-channel FP16→INT8→FP16 quantisation stays within the
/// theoretical error bound: max error per element ≤ max_abs_per_channel / 127.
///
/// This test simulates the algorithm of kernels/checkpoint_compress.cu in pure
/// Rust, allowing it to run without a GPU under `--features mock-cuda`.
#[test]
fn test_12_int8_checkpoint_round_trip_deviation_within_spec() {
    const N_CHANNELS: usize = 16;
    const ELEMS_PER_CHANNEL: usize = 256;
    const N_TOTAL: usize = N_CHANNELS * ELEMS_PER_CHANNEL; // 4 096

    // Generate deterministic values spanning [-1.0, 1.0] using sin() oscillation.
    // Each channel sees a full period, calibrating per-channel scale accurately.
    let values: Vec<f32> = (0..N_TOTAL)
        .map(|index| {
            let phase = (index as f32 * 2.0 * std::f32::consts::PI) / (ELEMS_PER_CHANNEL as f32);
            phase.sin()
        })
        .collect();

    // ── Phase 1: per-channel scale = max_abs / 127.0 ────────────────────────
    let mut scales = [1.0f32; N_CHANNELS];
    for (channel, scale) in scales.iter_mut().enumerate() {
        let start = channel * ELEMS_PER_CHANNEL;
        let end = start + ELEMS_PER_CHANNEL;
        let max_abs: f32 = values[start..end]
            .iter()
            .map(|value| value.abs())
            .fold(0.0f32, f32::max);
        *scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
    }

    // ── Phase 2: quantise to INT8 ────────────────────────────────────────────
    let quantized: Vec<i8> = values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let channel = index / ELEMS_PER_CHANNEL;
            (value / scales[channel]).round().clamp(-128.0, 127.0) as i8
        })
        .collect();

    // ── Phase 3: dequantise ──────────────────────────────────────────────────
    let reconstructed: Vec<f32> = quantized
        .iter()
        .enumerate()
        .map(|(index, &quantized_val)| {
            let channel = index / ELEMS_PER_CHANNEL;
            quantized_val as f32 * scales[channel]
        })
        .collect();

    // ── Assertion: max absolute error ≤ theoretical bound ───────────────────
    // Bound = max scale = max_abs_per_channel / 127 ≈ 1.0 / 127 ≈ 0.00787.
    // (One quantisation step is at most scale / 2 in the worst case, so the
    // bound below is intentionally generous — correct round-to-nearest gives
    // error ≤ 0.5 × scale ≤ max_abs / 254.)
    let theoretical_max_error: f32 = scales.iter().cloned().fold(0.0, f32::max);

    let max_abs_error: f32 = values
        .iter()
        .zip(reconstructed.iter())
        .map(|(original, reconstructed)| (original - reconstructed).abs())
        .fold(0.0, f32::max);

    assert!(
        max_abs_error <= theoretical_max_error,
        "max absolute INT8 reconstruction error {max_abs_error:.6} exceeds \
         theoretical bound {theoretical_max_error:.6} (= max_abs_per_channel / 127)"
    );

    // ── Assertion: mean absolute error < 1% of signal range ─────────────────
    let mean_abs_error: f32 = values
        .iter()
        .zip(reconstructed.iter())
        .map(|(original, reconstructed)| (original - reconstructed).abs())
        .sum::<f32>()
        / N_TOTAL as f32;

    assert!(
        mean_abs_error < 0.01,
        "mean absolute INT8 reconstruction error {mean_abs_error:.6} \
         must be < 1.0% of the signal range [-1, 1]"
    );
}
