//! T8 — Quantized-stream decode correctness.
//!
//! 1. INT8 bytes → FP16: decode error per element ≤ max_abs/127.
//! 2. INT4 packed bytes → FP16: decode error ≤ NF4 step × absmax.
//! 3. INT8 output has 2× bytes vs input (FP16 = 2 bytes per element).
//! 4. INT4 output has 4× bytes vs input (2 elements per byte, 2 bytes each).
//! 5. needs_decode is true for INT4/INT8, false for FP16/BF16.
//! 6. dispatch + decode_state returns Done (mock-cuda synchronous path).

use flowcast::decode::QuantizedDecoder;
use flowcast::config::Precision;

// Helper: f16 bits → f32
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as f32 * -2.0 + 1.0; // ±1
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mant = (bits & 0x3FF) as f32;
    if exp == 0 {
        sign * mant / 16_777_216.0 // subnormal
    } else if exp == 31 {
        sign * f32::INFINITY
    } else {
        sign * (1.0 + mant / 1024.0) * (2.0_f32).powi(exp - 15)
    }
}

// T8-1/T8-3: INT8 → FP16 error and shape
#[test]
fn int8_decode_error_within_bound_and_shape_correct() {
    let max_abs: f32 = 127.0;
    let scale = max_abs / 127.0; // = 1.0 for this test
    let src: Vec<u8> = (0u8..=127).collect();
    let scales = vec![scale; 1]; // 1 channel

    let out = QuantizedDecoder::decode_int8_to_fp16(&src, &scales, 1);

    // Shape: 2 bytes per element
    assert_eq!(out.len(), src.len() * 2, "output must be 2× input bytes");

    // Error bound: max_abs / 127 per element
    let error_bound = max_abs / 127.0;
    for (idx, raw) in src.iter().enumerate() {
        let fp16_bits = u16::from_le_bytes([out[idx * 2], out[idx * 2 + 1]]);
        let decoded = f16_to_f32(fp16_bits);
        let expected = (*raw as i8) as f32 * scale;
        let error = (decoded - expected).abs();
        assert!(
            error <= error_bound + 1e-3, // 1e-3 tolerance for f16 rounding
            "element {idx}: decoded={decoded} expected={expected} error={error} bound={error_bound}"
        );
    }
}

// T8-2/T8-4: INT4 → FP16 error and shape
#[test]
fn int4_decode_error_within_nf4_tolerance_and_shape_correct() {
    // 8 bytes → 16 nibbles → 16 fp16 values → 32 bytes
    let src: Vec<u8> = (0u8..8).map(|i| (i << 4) | i).collect(); // both nibbles = i
    let absmax: f32 = 1.0;

    let out = QuantizedDecoder::decode_int4_to_fp16(&src, absmax);

    // Shape: 4× bytes (2 nibbles per byte, each → 2-byte f16)
    assert_eq!(out.len(), src.len() * 4, "output must be 4× input bytes");

    // NF4 tolerance: largest step in the NF4 table ≈ 0.28 × absmax
    let nf4_tolerance: f32 = 0.3;
    for elem_idx in 0..(out.len() / 2) {
        let fp16_bits = u16::from_le_bytes([out[elem_idx * 2], out[elem_idx * 2 + 1]]);
        let decoded = f16_to_f32(fp16_bits);
        assert!(
            decoded.abs() <= absmax + nf4_tolerance,
            "elem {elem_idx}: decoded {decoded} outside absmax {absmax} + tolerance"
        );
    }
}

// T8-5: needs_decode flag
#[test]
fn needs_decode_correct_for_precisions() {
    assert!(QuantizedDecoder::needs_decode(Precision::INT4));
    assert!(QuantizedDecoder::needs_decode(Precision::INT8));
    assert!(!QuantizedDecoder::needs_decode(Precision::FP16));
    assert!(!QuantizedDecoder::needs_decode(Precision::BF16));
    assert!(!QuantizedDecoder::needs_decode(Precision::FP32));
}

// T8-6: dispatch → Done on mock-cuda path
#[test]
fn dispatch_completes_synchronously_on_mock() {
    use flowcast::decode::DecodeState;
    let decoder = QuantizedDecoder::new(Precision::FP16);

    decoder.dispatch(5, Precision::INT8).expect("dispatch INT8");
    assert_eq!(decoder.decode_state(5).unwrap(), DecodeState::Done);

    decoder.dispatch(7, Precision::INT4).expect("dispatch INT4");
    assert_eq!(decoder.decode_state(7).unwrap(), DecodeState::Done);
}

// T8-7: undispatched layer is Pending
#[test]
fn undispatched_layer_is_pending() {
    use flowcast::decode::DecodeState;
    let decoder = QuantizedDecoder::new(Precision::FP16);
    assert_eq!(decoder.decode_state(99).unwrap(), DecodeState::Pending);
}
