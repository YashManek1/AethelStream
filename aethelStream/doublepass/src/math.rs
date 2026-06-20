//! Basic linear algebra and activation helpers for transformer operations.
//!
//! Provides row-major matrix operations, RMSNorm, softmax, and SiLU
//! in pure Rust (mock-cuda). CUDA kernels would replace these in production.

#[allow(dead_code)]
/// Matrix multiply: c = a @ b, row-major.
/// a: [m, k], b: [k, n] → c: [m, n]
pub(crate) fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for l in 0..k {
            let a_il = a[i * k + l];
            for j in 0..n {
                c[i * n + j] += a_il * b[l * n + j];
            }
        }
    }
    c
}

/// Matrix multiply with transposed B: c = a @ b.T, row-major.
/// a: [m, k], b: [n, k] (stored as such, so b.T is [k, n]) → c: [m, n]
/// Equivalent to: c[i*n+j] = sum_l a[i*k+l] * b[j*k+l]
pub(crate) fn matmul_tb(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for l in 0..k {
                s += a[i * k + l] * b[j * k + l];
            }
            c[i * n + j] = s;
        }
    }
    c
}

/// Element-wise addition: c = a + b.
pub(crate) fn add_vecs(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

/// RMSNorm forward: normalise then scale.
/// For each token b in [0, bs):
///   mean_sq = sum(x[b*d+i]^2) / d
///   rms[b] = sqrt(mean_sq + eps)
///   x_hat[b*d+i] = x[b*d+i] / rms[b]
///   y[b*d+i] = x_hat[b*d+i] * w[i]
///
/// Returns (y, x_hat, rms).
pub(crate) fn rms_norm_fwd(
    x: &[f32],
    w: &[f32],
    bs: usize,
    d: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut y = vec![0.0f32; bs * d];
    let mut x_hat = vec![0.0f32; bs * d];
    let mut rms = vec![0.0f32; bs];
    let eps = 1e-6f32;

    for b in 0..bs {
        let mean_sq: f32 = x[b * d..(b + 1) * d].iter().map(|v| v * v).sum::<f32>() / d as f32;
        rms[b] = (mean_sq + eps).sqrt();

        for i in 0..d {
            x_hat[b * d + i] = x[b * d + i] / rms[b];
            y[b * d + i] = x_hat[b * d + i] * w[i];
        }
    }

    (y, x_hat, rms)
}

/// Softmax applied row-wise: for each row of `cols` elements, normalize with softmax.
/// Input: s[rows*cols], Output: a[rows*cols] where a[r*cols..] = softmax(s[r*cols..])
pub(crate) fn softmax_rows(s: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut a = vec![0.0f32; rows * cols];

    for r in 0..rows {
        let row_start = r * cols;
        let row_end = row_start + cols;
        let row = &s[row_start..row_end];

        // Compute max for numerical stability
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Compute exp and sum
        let mut exps = vec![0.0f32; cols];
        let mut sum_exp = 0.0f32;
        for (c, &val) in row.iter().enumerate() {
            exps[c] = (val - max_val).exp();
            sum_exp += exps[c];
        }

        // Normalize
        for c in 0..cols {
            a[row_start + c] = exps[c] / sum_exp;
        }
    }

    a
}

/// SiLU (Swish) activation: x / (1 + exp(-x)) = x * sigmoid(x).
pub(crate) fn silu_f(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Gradient of SiLU: sigma(x) * (1 + x * (1 - sigma(x))) where sigma = sigmoid.
pub(crate) fn silu_grad_f(x: f32) -> f32 {
    let sigma = 1.0 / (1.0 + (-x).exp());
    sigma * (1.0 + x * (1.0 - sigma))
}
