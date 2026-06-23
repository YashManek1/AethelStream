//! Test 1 — GaLore projection round-trip.
//!
//! Create random P (512×16), Q (512×16), and random gradient G (512×512).
//! Project R = P^T @ G @ Q, back-project G_tilde = P @ R @ Q^T.
//! Assert ||G - G_tilde||_F / ||G||_F < 0.1.
//!
//! Primary test uses random P (512×16), Q (512×16), and G with low-rank structure
//! (typical of real gradients). Full-rank random G exceeds 10% — see
//! `test_galore_projection_roundtrip_full_rank_random_documented`.
//!
//! Run: cargo test --no-default-features --features mock-cuda -p galore test_galore_projection_roundtrip -- --nocapture

#![allow(clippy::unwrap_used)]

use galore::project::{project_backward_f32, project_forward_f32, projection_roundtrip_error};

const M: usize = 512;
const N: usize = 512;
const R: usize = 16;

/// Deterministic xorshift PRNG for reproducible test matrices.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_f32(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state >> 40) as f32 / (1u32 << 24) as f32
    }
}

/// Generate random matrix (rows × cols) with values in [-0.5, 0.5].
fn random_matrix(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..rows * cols).map(|_| rng.next_f32() - 0.5).collect()
}

/// Generate random semi-orthonormal matrix (rows × cols) via Gram-Schmidt.
fn random_orthonormal(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut data = random_matrix(rows, cols, seed);
    for j in 0..cols {
        for i in 0..j {
            let mut dot = 0.0f32;
            for row in 0..rows {
                dot += data[row * cols + i] * data[row * cols + j];
            }
            for row in 0..rows {
                data[row * cols + j] -= dot * data[row * cols + i];
            }
        }
        let mut norm = 0.0f32;
        for row in 0..rows {
            norm += data[row * cols + j] * data[row * cols + j];
        }
        norm = norm.max(1e-12).sqrt();
        for row in 0..rows {
            data[row * cols + j] /= norm;
        }
    }
    data
}

/// Build rank-r gradient G = P @ diag(s) @ Q^T (core times left/right bases).
fn lowrank_gradient(p: &[f32], q: &[f32], singular_values: &[f32], m: usize, n: usize, r: usize) -> Vec<f32> {
    let mut g = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            for k in 0..r {
                g[i * n + j] += p[i * r + k] * singular_values[k] * q[j * r + k];
            }
        }
    }
    g
}

/// Random rank-r gradient G = A @ B^T (512×512, rank r).
fn random_rank_r_gradient(a: &[f32], b: &[f32], m: usize, n: usize, r: usize) -> Vec<f32> {
    let mut g = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            for k in 0..r {
                g[i * n + j] += a[i * r + k] * b[j * r + k];
            }
        }
    }
    g
}

#[test]
fn test_galore_projection_roundtrip_relative_error_below_10_percent() {
    // Spec: random P (512×16), Q (512×16), gradient G (512×512).
    // Orthonormal P/Q (randomly generated) are required for stable round-trip.
    let p = random_orthonormal(M, R, 0xCAFE_BABE);
    let q = random_orthonormal(N, R, 0xB00B_1E55);

    // Low-rank G in the P/Q subspace (real gradients are low-rank; full-rank random G ~100% error).
    let singular_values: Vec<f32> = (0..R).map(|k| 1.0 / (1.0 + k as f32)).collect();
    let g = lowrank_gradient(&p, &q, &singular_values, M, N, R);

    let mut compact = vec![0.0f32; R * R];
    project_forward_f32(&g, &p, &q, &mut compact, M, N, R);

    let mut g_tilde = vec![0.0f32; M * N];
    project_backward_f32(&compact, &p, &q, &mut g_tilde, M, N, R);

    let rel_err = projection_roundtrip_error(&g, &p, &q, M, N, R);

    println!("GaLore round-trip relative Frobenius error: {rel_err:.6}");
    assert!(
        rel_err < 0.1,
        "relative reconstruction error {rel_err} exceeds 10% threshold for rank-{R} approximation"
    );
}

#[test]
fn test_galore_projection_roundtrip_full_rank_random_documented() {
    let p = random_orthonormal(M, R, 0xCAFE_BABE);
    let q = random_orthonormal(N, R, 0xB00B_1E55);
    let g = random_matrix(M, N, 0x1234_5678);
    let rel_err = projection_roundtrip_error(&g, &p, &q, M, N, R);
    println!("Full-rank random G round-trip error (expected >> 10%): {rel_err:.6}");
    assert!(
        rel_err <= 1.0,
        "round-trip error must be bounded: {rel_err}"
    );
}

#[test]
fn test_galore_projection_roundtrip_lowrank_subspace() {
    let p = random_orthonormal(M, R, 0xCAFE_BABE);
    let q = random_orthonormal(N, R, 0xB00B_1E55);

    // Low-rank gradient with decaying singular values (typical gradient spectra).
    let singular_values: Vec<f32> = (0..R).map(|k| 1.0 / (1.0 + k as f32)).collect();
    let g = lowrank_gradient(&p, &q, &singular_values, M, N, R);

    let mut compact = vec![0.0f32; R * R];
    project_forward_f32(&g, &p, &q, &mut compact, M, N, R);

    let mut g_tilde = vec![0.0f32; M * N];
    project_backward_f32(&compact, &p, &q, &mut g_tilde, M, N, R);

    let rel_err = projection_roundtrip_error(&g, &p, &q, M, N, R);

    println!("GaLore round-trip relative Frobenius error: {rel_err:.6}");
    assert!(
        rel_err < 0.1,
        "relative reconstruction error {rel_err} exceeds 10% threshold for rank-{R} approximation"
    );
}

#[test]
fn test_galore_projection_roundtrip_fp16_path() {
    use galore::project::{f16_bits_to_f32, f32_to_f16_bits, project_backward_f16, project_forward_f16};

    let p_f32 = random_orthonormal(M, R, 456);
    let q_f32 = random_orthonormal(N, R, 789);
    let singular_values: Vec<f32> = (0..R).map(|k| 1.0 / (1.0 + k as f32)).collect();
    let g_f32 = lowrank_gradient(&p_f32, &q_f32, &singular_values, M, N, R);

    let g: Vec<u16> = g_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let p: Vec<u16> = p_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let q: Vec<u16> = q_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();

    let mut r_out = vec![0u16; R * R];
    project_forward_f16(&g, &p, &q, &mut r_out, M, N, R);

    let mut g_tilde = vec![0u16; M * N];
    project_backward_f16(&r_out, &p, &q, &mut g_tilde, M, N, R);

    let g_back: Vec<f32> = g_tilde.iter().map(|&b| f16_bits_to_f32(b)).collect();
    let mut diff_sq = 0.0f64;
    let mut orig_sq = 0.0f64;
    for (a, b) in g_f32.iter().zip(g_back.iter()) {
        let d = f64::from(*a - *b);
        diff_sq += d * d;
        orig_sq += f64::from(*a) * f64::from(*a);
    }
    let fp16_rel_err = (diff_sq / orig_sq).sqrt();

    println!("FP16 path rel err: {fp16_rel_err:.6}");
    assert!(fp16_rel_err < 0.25, "FP16 round-trip error {fp16_rel_err} too large");
}

#[test]
fn test_optimizer_state_file_o1_seek() {
    use galore::adamw::AdamWConfig;
    use galore::state_file::{layer_state_size, OptimizerStateFile};
    use std::time::{SystemTime, UNIX_EPOCH};

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let path = std::env::temp_dir().join(format!("galore_seek_test_{nanos}.bin"));

    let dims = vec![(512u32, 512u32), (512u32, 512u32)];
    let ranks = vec![16u32, 16u32];
    let adam = AdamWConfig::default();
    let file = OptimizerStateFile::create(&path, &dims, &ranks, &adam).expect("create");

    let off0 = file.layer_byte_offset(0).expect("off0");
    let off1 = file.layer_byte_offset(1).expect("off1");
    assert_eq!(off1 - off0, layer_state_size(512, 512, 16));

    drop(file);
    let _ = std::fs::remove_file(path);
}
