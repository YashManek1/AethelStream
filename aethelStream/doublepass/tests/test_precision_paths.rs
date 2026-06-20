// tests/test_precision_paths.rs — T-precision: A5 BF16 and FP16 paths.
//
// BF16 tests:
//   - effective_precision returns BF16 when hardware supports it.
//   - BF16 degrades to FP16 when hardware lacks support.
//   - PerLayerScaleTable in bf16_mode: all scales fixed at 1.0 (no overflow branch).
//
// FP16 tests:
//   - check_and_update_scale detects injected NaN overflow, returns true.
//   - Scale backed off (halved) after overflow.
//   - Scale grows back after sufficient clean steps.
//   - Clean data does not trigger skip.
//   - Empty gradient slice is handled safely.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use doublepass::precision::{check_and_update_scale, effective_precision};
use doublepass::Precision;
use ramflow::PerLayerScaleTable;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// FP16 NaN: exponent all-ones, mantissa non-zero → 0x7C01
const FP16_NAN: u16 = 0x7C01;
/// FP16 1.0: 0x3C00
const FP16_ONE: u16 = 0x3C00;

fn make_table(num_layers: usize) -> PerLayerScaleTable {
    PerLayerScaleTable::new(num_layers, 0.05)
}

// ---------------------------------------------------------------------------
// BF16 path tests
// ---------------------------------------------------------------------------

#[test]
fn bf16_effective_precision_when_hw_supports() {
    let schedule = vec![Precision::BF16; 4];
    for layer in 0..4_u32 {
        let p = effective_precision(layer, &schedule, true);
        assert_eq!(p, Precision::BF16, "layer {layer} should be BF16");
    }
}

#[test]
fn bf16_degrades_to_fp16_without_hw_support() {
    let schedule = vec![Precision::BF16; 4];
    for layer in 0..4_u32 {
        let p = effective_precision(layer, &schedule, false);
        assert_eq!(p, Precision::FP16, "layer {layer} should degrade to FP16");
    }
}

#[test]
fn bf16_mode_table_fixes_scale_at_one() {
    let mut table = make_table(4);
    table.enable_bf16_mode();
    for layer in 0..4_usize {
        assert_eq!(
            table.get_scale(layer).unwrap(),
            1.0_f32,
            "BF16 mode: scale[{layer}] must be 1.0"
        );
    }
}

#[test]
fn bf16_mode_table_update_is_noop() {
    // When in bf16_mode, update() must not change the scale.
    let mut table = make_table(2);
    table.enable_bf16_mode();
    // Inject 100 % overflow — in bf16_mode the table should ignore it.
    table.update(0, 1000, 1000).unwrap();
    assert_eq!(table.get_scale(0).unwrap(), 1.0_f32);
    assert_eq!(table.get_density(0).unwrap(), 0.0_f32, "density must stay 0 in bf16_mode");
}

#[test]
fn schedule_non_bf16_honoured_verbatim() {
    let schedule = vec![Precision::FP32, Precision::FP16, Precision::INT8];
    assert_eq!(effective_precision(0, &schedule, true), Precision::FP32);
    assert_eq!(effective_precision(1, &schedule, true), Precision::FP16);
    assert_eq!(effective_precision(2, &schedule, true), Precision::INT8);
}

#[test]
fn out_of_schedule_defaults_to_bf16_when_hw_supports() {
    let schedule: Vec<Precision> = vec![];
    assert_eq!(effective_precision(0, &schedule, true), Precision::BF16);
    assert_eq!(effective_precision(99, &schedule, true), Precision::BF16);
}

// ---------------------------------------------------------------------------
// FP16 path tests
// ---------------------------------------------------------------------------

#[test]
fn fp16_overflow_detected_returns_true() {
    let mut table = make_table(2);
    let n = 1000_usize;
    // All elements are NaN → 100 % overflow.
    let overflow_data: Vec<u16> = vec![FP16_NAN; n];

    let overflow =
        check_and_update_scale(&mut table, 0, overflow_data.as_ptr(), n).unwrap();
    assert!(overflow, "100 % NaN input must return overflow=true");
}

#[test]
fn fp16_overflow_backs_off_scale() {
    let mut table = make_table(2);
    let initial_scale = table.get_scale(0).unwrap();
    assert_eq!(initial_scale, 65536.0_f32, "initial scale must be 65536");

    let n = 1000_usize;
    let overflow_data: Vec<u16> = vec![FP16_NAN; n];
    let _ = check_and_update_scale(&mut table, 0, overflow_data.as_ptr(), n).unwrap();

    let new_scale = table.get_scale(0).unwrap();
    assert!(
        new_scale < initial_scale,
        "scale should back off after overflow; got {new_scale}"
    );
    // EWA density after 1 step of 100 % overflow: 0.05 >> high_threshold 0.001 → halved.
    assert_eq!(new_scale, 32768.0_f32, "scale should be halved to 32768");
}

#[test]
fn fp16_scale_growth_fires_on_clean_interval() {
    let mut table = make_table(2);
    let n = 1000_usize;

    // One overflow step to raise density.
    let overflow_data: Vec<u16> = vec![FP16_NAN; n];
    let _ = check_and_update_scale(&mut table, 0, overflow_data.as_ptr(), n).unwrap();
    let scale_after_overflow = table.get_scale(0).unwrap();
    assert_eq!(scale_after_overflow, 32768.0_f32);

    // Run clean steps until EWA density drops below low_threshold (0.0001).
    // Starting density ≈ 0.05; decays to < 0.0001 in ~125 steps (0.95^125 × 0.05 < 0.0001).
    let clean_data: Vec<u16> = vec![FP16_ONE; n];
    for _ in 0..200 {
        let _ = check_and_update_scale(&mut table, 0, clean_data.as_ptr(), n).unwrap();
    }

    let final_scale = table.get_scale(0).unwrap();
    assert!(
        final_scale > scale_after_overflow,
        "scale should grow after clean interval; got {final_scale}"
    );
}

#[test]
fn fp16_clean_data_no_skip() {
    let mut table = make_table(2);
    let clean_data: Vec<u16> = vec![FP16_ONE; 500];

    let overflow = check_and_update_scale(&mut table, 0, clean_data.as_ptr(), clean_data.len())
        .unwrap();
    assert!(!overflow, "clean data must not trigger skip");
}

#[test]
fn fp16_empty_slice_returns_false_no_panic() {
    let mut table = make_table(1);
    // n_elements=0 must be handled without touching the pointer.
    let overflow = check_and_update_scale(&mut table, 0, std::ptr::null(), 0).unwrap();
    assert!(!overflow, "empty slice must return false");
}

#[test]
fn fp16_other_layers_unaffected_by_single_layer_overflow() {
    let mut table = make_table(3);
    let scale_l1_before = table.get_scale(1).unwrap();

    let n = 500_usize;
    let overflow_data: Vec<u16> = vec![FP16_NAN; n];
    let _ = check_and_update_scale(&mut table, 0, overflow_data.as_ptr(), n).unwrap();

    // Layer 1 and 2 should be untouched.
    assert_eq!(
        table.get_scale(1).unwrap(),
        scale_l1_before,
        "layer 1 scale must not change when layer 0 overflows"
    );
    assert_eq!(
        table.get_scale(2).unwrap(),
        65536.0_f32,
        "layer 2 scale must remain at initial 65536"
    );
}
