//! INT8 checkpoint compression round-trip test.
//!
//! Compresses an activation tensor via `store_checkpoint(..., true)` and
//! decompresses via `read_checkpoint`, then asserts the round-trip error is
//! within INT8 quantization tolerance.
//!
//! TIMINGS / THROUGHPUT numbers in this test are **mock** (not measured on GPU).
//!
//! Run: cargo test --features mock-cuda -p doublepass test_checkpoint_compress -- --nocapture
#![allow(clippy::unwrap_used, clippy::expect_used)]

use doublepass::checkpoint::{read_checkpoint, store_checkpoint};

/// INT8 round-trip via store_checkpoint + read_checkpoint (mock path).
///
/// INT8 quantization tolerance: max|Δ| ≤ max_abs / 127 (one quantization step).
#[test]
#[cfg(feature = "mock-cuda")]
fn test_int8_roundtrip_within_tolerance() {
    // Activation tensor: 32 elements, values in [-1, 1].
    let data: Vec<f32> = (0..32)
        .map(|i| ((i as f64 * 0.2).sin() * 0.8) as f32)
        .collect();

    let max_abs = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    // INT8 tolerance = one quantization step for this range.
    let tolerance = max_abs / 127.0 + 1e-6;

    let buf = store_checkpoint(&data, true).expect("store compressed");
    assert!(buf.is_compressed(), "compressed flag must be set");
    assert!(buf.len() > 0, "buffer must be non-empty");
    // Layout: 4 bytes scale + n bytes INT8.
    assert_eq!(
        buf.len(),
        core::mem::size_of::<f32>() + data.len(),
        "buffer size must be 4 + n_elements"
    );

    let recovered = read_checkpoint(&buf).expect("read compressed");
    assert_eq!(recovered.len(), data.len(), "recovered length must match");

    let max_diff: f32 = data
        .iter()
        .zip(&recovered)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!(
        "\nINT8 round-trip (mock): max|orig - recovered| = {:.3e}  tol = {:.3e}",
        max_diff, tolerance
    );
    assert!(
        max_diff <= tolerance,
        "INT8 round-trip exceeded tolerance: {:.3e} > {:.3e}",
        max_diff,
        tolerance
    );
    println!("INT8 round-trip PASSED (mock path)");
}

/// Uncompressed round-trip must be bit-identical.
#[test]
#[cfg(feature = "mock-cuda")]
fn test_uncompressed_roundtrip_bit_identical() {
    let data: Vec<f32> = (0..64)
        .map(|i| (i as f64 * 0.1 - 3.2).tanh() as f32)
        .collect();

    let buf = store_checkpoint(&data, false).expect("store uncompressed");
    assert!(!buf.is_compressed(), "uncompressed flag must not be set");
    let recovered = read_checkpoint(&buf).expect("read uncompressed");
    assert_eq!(
        data, recovered,
        "uncompressed round-trip must be bit-identical"
    );
    println!("Uncompressed round-trip PASSED (bit-identical)");
}
