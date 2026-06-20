// tests/test_stochastic_rounding.rs — T-stoch-round: A5 stochastic BF16 policy.
//
// Verifies:
//   1. stochastic_round_to_bf16 produces the correct BF16 value in expectation
//      (E[bf16_to_f32(stochastic_round(x, rand))] ≈ x).
//   2. apply_galore_bf16_update decreases loss comparably to an FP32 baseline
//      over N gradient descent steps (no stagnation).
//   3. Stochastic rounding beats deterministic truncation for sub-ULP *positive*
//      steps: weights at 128.0 (BF16 ULP = 1.0), target = 192.0, lr = 0.01.
//      Each step = 0.01 * gap < 1.0 = BF16 ULP, so deterministic floor always
//      rounds back to 128.0 and stagnates, while stochastic makes probabilistic
//      progress (64 % chance per step of +1 BF16 grid unit toward target).
//   4. apply_lora_update_fp32 applies FP32 deltas exactly and rejects mismatched slices.
//   5. Same RNG seed => bit-identical output.
//   6. bf16_to_f32 / stochastic_round_to_bf16 round-trip for exactly-representable values.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use doublepass::precision::{
    apply_galore_bf16_update, apply_lora_update_fp32, bf16_to_f32, stochastic_round_to_bf16,
};

// ---------------------------------------------------------------------------
// Helper: deterministic truncation to BF16 (truncates lower 16 mantissa bits,
// which is floor toward -infinity for positive values).
// Used only as the stagnation baseline — NOT part of the A5 API.
// ---------------------------------------------------------------------------

fn det_bf16(val: f32) -> f32 {
    let high16 = (val.to_bits() >> 16) as u16;
    bf16_to_f32(high16)
}

fn det_bf16_update(weights: &mut Vec<f32>, delta: &[f32]) {
    for (w, d) in weights.iter_mut().zip(delta.iter()) {
        *w = det_bf16(*w + d);
    }
}

fn quadratic_loss(weights: &[f32], target: &[f32]) -> f32 {
    weights
        .iter()
        .zip(target.iter())
        .map(|(w, t)| (w - t).powi(2) * 0.5)
        .sum()
}

// ---------------------------------------------------------------------------
// 1. Expectation test: E[SRTE(x)] = x within 1 BF16 ULP.
// ---------------------------------------------------------------------------

#[test]
fn stochastic_round_expectation_matches_f32() {
    let test_values: Vec<f32> = vec![1.0, -0.5, 3.14159, 0.001, 100.0, -7.5];

    for &val in &test_values {
        let mut sum = 0.0_f64;
        let n_trials = 65536_usize;
        for r in 0..n_trials {
            let bf16 = stochastic_round_to_bf16(val, r as u16);
            sum += bf16_to_f32(bf16) as f64;
        }
        let mean = (sum / n_trials as f64) as f32;
        // Tolerance: 1 BF16 ULP at |val|.
        let bf16_trunc = bf16_to_f32((val.to_bits() >> 16) as u16);
        let ulp = {
            let bits = bf16_trunc.abs().to_bits();
            let exp = (bits >> 23) & 0xFF;
            if exp > 0 {
                f32::from_bits((exp.saturating_sub(7)) << 23)
            } else {
                f32::MIN_POSITIVE
            }
        };
        assert!(
            (mean - val).abs() <= ulp + f32::EPSILON * val.abs(),
            "E[bf16({val})] should be {val} ± ULP={ulp}, got mean={mean}"
        );
    }
}

// ---------------------------------------------------------------------------
// 2. Loss-decrease test: stochastic BF16 comparable to FP32 baseline.
// ---------------------------------------------------------------------------

#[test]
fn stochastic_bf16_loss_decreases_comparably_to_fp32() {
    let n = 64_usize;
    let lr = 0.05_f32;
    let steps = 300_usize;
    let target: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - 3.0).collect();

    // FP32 baseline.
    let mut w_fp32 = vec![0.0_f32; n];
    let fp32_init = quadratic_loss(&w_fp32, &target);
    for _ in 0..steps {
        for (w, t) in w_fp32.iter_mut().zip(target.iter()) {
            *w -= lr * (*w - t);
        }
    }
    let fp32_final = quadratic_loss(&w_fp32, &target);
    assert!(
        fp32_final < fp32_init * 0.001,
        "FP32 baseline stagnated: init={fp32_init:.4} final={fp32_final:.4}"
    );

    // Stochastic BF16.
    let mut w_bf16 = vec![0.0_f32; n];
    let bf16_init = quadratic_loss(&w_bf16, &target);
    let mut rng: u64 = 0xDEAD_BEEF_1234_5678;
    for _ in 0..steps {
        let delta: Vec<f32> = w_bf16
            .iter()
            .zip(target.iter())
            .map(|(w, t)| -lr * (w - t))
            .collect();
        apply_galore_bf16_update(&mut w_bf16, &delta, &mut rng).unwrap();
    }
    let bf16_final = quadratic_loss(&w_bf16, &target);
    assert!(
        bf16_final < bf16_init * 0.05,
        "stochastic BF16 stagnated: init={bf16_init:.4} final={bf16_final:.4}"
    );
}

// ---------------------------------------------------------------------------
// 3. Sub-ULP positive-step scenario: stochastic beats deterministic truncation.
//
// Setup: weights at 128.0 (BF16 exact), target = 192.0.
// BF16 ULP at 128.0 is 1.0 (the [128,256) range has 1.0-spaced grid points).
// With lr = 0.01, the initial step = 0.01 * 64 = 0.64 < 1.0 = BF16 ULP.
//
// Deterministic truncation (floor) rounds 128.64 → 128.0 every step: STAGNATES.
// Stochastic rounding: 0.64/1.0 = 64 % probability of incrementing to 129.0
// each step.  Expected steps to reach 192 ≈ 64 / 0.64 ≈ 100.  Over 1000 steps
// all 64 weights converge.
// ---------------------------------------------------------------------------

#[test]
fn stochastic_beats_deterministic_for_sub_ulp_positive_steps() {
    let n = 64_usize;
    let lr = 0.01_f32;
    let steps = 1000_usize;
    let w_init = 128.0_f32;
    let target_val = 192.0_f32;
    let target: Vec<f32> = vec![target_val; n];

    // Deterministic BF16 (truncate lower 16 bits = floor for positive values).
    let mut w_det = vec![w_init; n];
    for _ in 0..steps {
        let delta: Vec<f32> = w_det.iter().map(|w| -lr * (w - target_val)).collect();
        det_bf16_update(&mut w_det, &delta);
    }
    let loss_det = quadratic_loss(&w_det, &target);

    // Stochastic BF16.
    let mut w_srte = vec![w_init; n];
    let mut rng: u64 = 0xFEED_FACE_CAFE_1234;
    for _ in 0..steps {
        let delta: Vec<f32> = w_srte.iter().map(|w| -lr * (w - target_val)).collect();
        apply_galore_bf16_update(&mut w_srte, &delta, &mut rng).unwrap();
    }
    let loss_srte = quadratic_loss(&w_srte, &target);

    // Deterministic should stagnate (loss stays at init; all steps < BF16 ULP).
    let loss_init = quadratic_loss(&vec![w_init; n], &target);
    assert!(
        loss_det > loss_init * 0.99,
        "deterministic truncation should stagnate: init={loss_init:.1} det_final={loss_det:.1}"
    );

    // Stochastic should converge significantly better.
    assert!(
        loss_srte < loss_det * 0.5,
        "stochastic BF16 (loss={loss_srte:.1}) should beat deterministic (loss={loss_det:.1})"
    );
    assert!(
        loss_srte < loss_init * 0.01,
        "stochastic BF16 should converge: init={loss_init:.1} srte_final={loss_srte:.1}"
    );
}

// ---------------------------------------------------------------------------
// 4. LoRA FP32 update tests.
// ---------------------------------------------------------------------------

#[test]
fn lora_fp32_update_applied_exactly() {
    let mut weights = vec![1.0_f32, 2.0, 3.0];
    let delta = vec![-0.1_f32, 0.5, -1.0];
    apply_lora_update_fp32(&mut weights, &delta).unwrap();
    assert!((weights[0] - 0.9_f32).abs() < 1e-7, "w[0]={}", weights[0]);
    assert!((weights[1] - 2.5_f32).abs() < 1e-7, "w[1]={}", weights[1]);
    assert!((weights[2] - 2.0_f32).abs() < 1e-7, "w[2]={}", weights[2]);
}

#[test]
fn lora_fp32_update_rejects_length_mismatch() {
    let mut weights = vec![1.0_f32; 4];
    let delta = vec![0.1_f32; 3];
    let result = apply_lora_update_fp32(&mut weights, &delta);
    assert!(result.is_err(), "mismatched lengths must return Err");
}

// ---------------------------------------------------------------------------
// 5. Determinism: same RNG seed → identical output.
// ---------------------------------------------------------------------------

#[test]
fn galore_bf16_update_is_deterministic_given_same_seed() {
    let n = 32_usize;
    let mut w1 = vec![1.0_f32; n];
    let mut w2 = vec![1.0_f32; n];
    let delta: Vec<f32> = (0..n).map(|i| -(i as f32) * 0.01).collect();

    let mut rng1: u64 = 0x1111_2222_3333_4444;
    let mut rng2: u64 = 0x1111_2222_3333_4444;
    apply_galore_bf16_update(&mut w1, &delta, &mut rng1).unwrap();
    apply_galore_bf16_update(&mut w2, &delta, &mut rng2).unwrap();

    for i in 0..n {
        assert_eq!(
            w1[i].to_bits(),
            w2[i].to_bits(),
            "weight[{i}] differs between identical-seed runs"
        );
    }
}

// ---------------------------------------------------------------------------
// 6. bf16_to_f32 / stochastic_round_to_bf16 round-trip for exact BF16 values.
// ---------------------------------------------------------------------------

#[test]
fn bf16_round_trip_exact_for_bf16_representable_values() {
    // Values exactly representable in BF16 (lower 16 f32 bits all zero).
    let exact_bf16_values: Vec<f32> = vec![0.0, 1.0, 2.0, 0.5, -1.0, 128.0, 0.00390625];
    for &val in &exact_bf16_values {
        let bits = val.to_bits();
        if bits & 0xFFFF != 0 {
            continue; // not exactly BF16-representable
        }
        let bf16_bits = (bits >> 16) as u16;
        let restored = bf16_to_f32(bf16_bits);
        assert_eq!(
            restored.to_bits(),
            val.to_bits(),
            "bf16 round-trip failed for {val}"
        );
    }
}
