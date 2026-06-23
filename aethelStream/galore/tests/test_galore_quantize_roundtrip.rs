//! Test 2 — 8-bit quantization round-trip fidelity.
//!
//! Simulate 100 AdamW steps to populate momentum/variance, then quantize/dequantize
//! and assert max relative error < 1% at scales 1e-5, 1e-3, 1e-1.
//!
//! Run: cargo test --no-default-features --features mock-cuda -p galore test_galore_quantize -- --nocapture

#![allow(clippy::unwrap_used)]

use galore::adamw::{adamw_lowrank_step, AdamWConfig, LowRankAdamState};
use galore::quantize::quantize_relative_error;

const STEPS: usize = 100;
const R: usize = 16;

fn random_matrix(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.max(1);
    (0..rows * cols)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u = (state >> 40) as f32 / (1u32 << 24) as f32;
            ((i as f32 * 0.017 + u).sin() * 0.1)
        })
        .collect()
}

fn simulate_adamw_state(scale: f32, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let m = 64usize;
    let n = 64usize;
    let p = random_matrix(m, R, seed);
    let q = random_matrix(n, R, seed.wrapping_add(1));
    let mut state = LowRankAdamState::new(R);
    state.dequantize_from_ram();

    let cfg = AdamWConfig {
        lr: 1e-3 * scale,
        ..AdamWConfig::default()
    };

    for step in 0..STEPS {
        let grad = random_matrix(m, n, seed.wrapping_add(step as u64 + 100));
        let scaled: Vec<f32> = grad.iter().map(|v| v * scale).collect();
        adamw_lowrank_step(&scaled, &p, &q, &mut state, m, n, R, &cfg, 1.0);
    }

    (state.momentum.clone(), state.variance.clone())
}

#[test]
fn test_galore_quantize_roundtrip_at_multiple_scales() {
    for &scale in &[1e-5f32, 1e-3, 1e-1] {
        let (momentum, variance) = simulate_adamw_state(scale, 0xDEAD_BEEF ^ (scale.to_bits() as u64));

        let rel_m = quantize_relative_error(&momentum);
        let rel_v = quantize_relative_error(&variance);

        println!(
            "scale={scale:.0e}: momentum rel_err={rel_m:.6}, variance rel_err={rel_v:.6}"
        );

        assert!(
            rel_m < 0.01,
            "momentum relative quant error {rel_m} exceeds 1% at scale {scale}"
        );
        assert!(
            rel_v < 0.01,
            "variance relative quant error {rel_v} exceeds 1% at scale {scale}"
        );
    }
}

#[test]
fn test_galore_quantize_roundtrip_via_lowrank_state() {
    let scale = 1e-3f32;
    let (momentum, variance) = simulate_adamw_state(scale, 42);

    let mut state = LowRankAdamState::new(R);
    state.momentum = momentum;
    state.variance = variance;
    state.quantize_to_ram();
    state.dequantize_from_ram();

    let rel_m = quantize_relative_error(&state.momentum);
    let rel_v = quantize_relative_error(&state.variance);

    assert!(rel_m < 0.01, "LowRankAdamState momentum round-trip: {rel_m}");
    assert!(rel_v < 0.01, "LowRankAdamState variance round-trip: {rel_v}");
}
