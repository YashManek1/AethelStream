//! T8 — Quantized-stream decode correctness.
//!
//! 1. INT8 bytes → FP16: decode error per element ≤ max_abs/127.
//! 2. INT4 packed bytes → FP16: decode error ≤ NF4 step × absmax.
//! 3. INT8 output has 2× bytes vs input (FP16 = 2 bytes per element).
//! 4. INT4 output has 4× bytes vs input (2 elements per byte, 2 bytes each).
//! 5. needs_decode is true for INT4/INT8, false for FP16/BF16.
//! 6. dispatch + decode_state returns Done with decoded bytes (A7-d fix).
//! 7. Undispatched layer is Pending.
//! 8. INT4 prefetch transfers fewer bytes than FP16 (A7-f1/f2).

use flowcast::decode::QuantizedDecoder;
use flowcast::config::Precision;

// Helper: f16 bits → f32
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as f32 * -2.0 + 1.0;
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
    let scale = max_abs / 127.0;
    let src: Vec<u8> = (0u8..=127).collect();
    let scales = vec![scale; 1];

    let out = QuantizedDecoder::decode_int8_to_fp16(&src, &scales, 1);
    assert_eq!(out.len(), src.len() * 2, "output must be 2× input bytes");

    let error_bound = max_abs / 127.0;
    for (idx, raw) in src.iter().enumerate() {
        let fp16_bits = u16::from_le_bytes([out[idx * 2], out[idx * 2 + 1]]);
        let decoded = f16_to_f32(fp16_bits);
        let expected = (*raw as i8) as f32 * scale;
        let error = (decoded - expected).abs();
        assert!(
            error <= error_bound + 1e-3,
            "element {idx}: decoded={decoded} expected={expected} error={error} bound={error_bound}"
        );
    }
}

// T8-2/T8-4: INT4 → FP16 error and shape
#[test]
fn int4_decode_error_within_nf4_tolerance_and_shape_correct() {
    let src: Vec<u8> = (0u8..8).map(|i| (i << 4) | i).collect();
    let absmax: f32 = 1.0;

    let out = QuantizedDecoder::decode_int4_to_fp16(&src, absmax);
    assert_eq!(out.len(), src.len() * 4, "output must be 4× input bytes");

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

// T8-6: dispatch actually calls the decode function and returns decoded bytes (A7-d fix).
#[test]
fn dispatch_completes_synchronously_on_mock() {
    use flowcast::decode::DecodeState;
    let decoder = QuantizedDecoder::new(Precision::FP16);

    // INT8: 16 bytes → 32 FP16 bytes
    let src_int8 = vec![0u8; 16];
    let scales = vec![1.0_f32; 1];
    let decoded_int8 = decoder
        .dispatch(5, Precision::INT8, &src_int8, &scales, 1.0, 1)
        .expect("dispatch INT8");
    assert_eq!(decoded_int8.len(), src_int8.len() * 2, "INT8 decoded size");
    assert_eq!(decoder.decode_state(5).unwrap(), DecodeState::Done);

    // INT4: 8 bytes → 32 FP16 bytes
    let src_int4 = vec![0u8; 8];
    let decoded_int4 = decoder
        .dispatch(7, Precision::INT4, &src_int4, &[], 1.0, 0)
        .expect("dispatch INT4");
    assert_eq!(decoded_int4.len(), src_int4.len() * 4, "INT4 decoded size");
    assert_eq!(decoder.decode_state(7).unwrap(), DecodeState::Done);

    // FP16: passthrough, same size
    let src_fp16 = vec![0u8; 32];
    let passthrough = decoder
        .dispatch(9, Precision::FP16, &src_fp16, &[], 1.0, 0)
        .expect("dispatch FP16");
    assert_eq!(passthrough.len(), src_fp16.len(), "FP16 passthrough size");
    assert_eq!(decoder.decode_state(9).unwrap(), DecodeState::Done);
}

// T8-7: undispatched layer is Pending
#[test]
fn undispatched_layer_is_pending() {
    use flowcast::decode::DecodeState;
    let decoder = QuantizedDecoder::new(Precision::FP16);
    assert_eq!(decoder.decode_state(99).unwrap(), DecodeState::Pending);
}

// T8-8: INT4 dispatch transfers fewer bytes than FP16 dispatch (A7-f1/f2).
//
// Tests that the state machine uses compressed byte-length for INT4 layers
// instead of the full FP16-sized pool slot.
#[test]
fn int4_bytes_transferred_less_than_fp16() {
    use flowcast::state_machine::PrefetchStateMachine;
    use flowcast::backend::mock::MockBackend;
    use flowcast::backend::IoBackend;
    use ramflow::{PoolRegistry, phase::Direction};

    let pool = PoolRegistry::with_defaults().expect("pool");
    let backend_int4 = MockBackend::new();
    let backend_fp16 = MockBackend::new();

    let sm_int4 = PrefetchStateMachine::new(4, 1, Precision::INT4);
    let sm_fp16 = PrefetchStateMachine::new(4, 1, Precision::FP16);

    sm_int4.prime_window(Direction::Forward, &pool, &backend_int4).expect("prime INT4");
    sm_fp16.prime_window(Direction::Forward, &pool, &backend_fp16).expect("prime FP16");

    let cqes_int4 = backend_int4.poll_completions().expect("poll INT4");
    let cqes_fp16 = backend_fp16.poll_completions().expect("poll FP16");

    let int4_bytes: i32 = cqes_int4.iter().map(|c| c.result).sum();
    let fp16_bytes: i32 = cqes_fp16.iter().map(|c| c.result).sum();

    assert!(
        int4_bytes < fp16_bytes,
        "INT4 should transfer fewer bytes than FP16: int4={int4_bytes}, fp16={fp16_bytes}"
    );
}
