// tests/test_overflow_check.rs — Sprint 2 overflow integration checks
//
// Run with:
//   cargo test --no-default-features --features mock-cuda test_overflow_check -- --nocapture

use ramflow::cuda_bridge::stream::check_overflow_fp16;
use ramflow::cuda_bridge::CudaStream;

fn fp16_one() -> u16 {
    0x3C00
}

fn fp16_nan() -> u16 {
    0x7C01
}

fn fp16_inf() -> u16 {
    0x7C00
}

fn fp16_neg_inf() -> u16 {
    0xFC00
}

#[test]
fn test_overflow_check_detects_expected_patterns() {
    let stream = CudaStream::new().expect("stream creation failed");

    let mut with_overflow = vec![fp16_one(); 10_000];
    with_overflow[111] = fp16_nan();
    with_overflow[2222] = fp16_nan();
    with_overflow[3333] = fp16_nan();
    with_overflow[4444] = fp16_nan();
    with_overflow[9999] = fp16_nan();
    with_overflow[10] = fp16_inf();
    with_overflow[20] = fp16_inf();
    with_overflow[30] = fp16_neg_inf();

    let has_overflow = check_overflow_fp16(with_overflow.as_ptr(), with_overflow.len(), &stream);
    assert!(has_overflow, "expected NaN/Inf array to report overflow");

    let clean = vec![fp16_one(); 10_000];
    let clean_overflow = check_overflow_fp16(clean.as_ptr(), clean.len(), &stream);
    assert!(!clean_overflow, "expected clean array to report no overflow");
}

#[test]
fn test_overflow_check_no_false_signals_1000_iterations() {
    let stream = CudaStream::new().expect("stream creation failed");

    for i in 0..1000usize {
        let mut with_nan = vec![fp16_one(); 10_000];
        with_nan[i % 10_000] = fp16_nan();
        let has_overflow = check_overflow_fp16(with_nan.as_ptr(), with_nan.len(), &stream);
        assert!(has_overflow, "false negative at iteration {i}");

        let clean = vec![fp16_one(); 10_000];
        let clean_overflow = check_overflow_fp16(clean.as_ptr(), clean.len(), &stream);
        assert!(!clean_overflow, "false positive at iteration {i}");
    }
}
