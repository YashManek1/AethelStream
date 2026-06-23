//! Randomized SVD for periodic subspace switching (Algorithm 3).

use crate::error::{GaLoreError, Result};
use crate::project::matmul_f32;

/// Configuration for randomized SVD subspace refresh.
#[derive(Debug, Clone, Copy)]
pub struct RandomizedSvdConfig {
    /// Target rank r.
    pub rank: usize,
    /// Oversampling parameter p (typically 10).
    pub oversampling: usize,
    /// Power iterations for accuracy (0 = none).
    pub power_iters: usize,
    /// RNG seed for reproducible Omega generation.
    pub seed: u64,
}

impl Default for RandomizedSvdConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            oversampling: 10,
            power_iters: 0,
            seed: 42,
        }
    }
}

/// Result of randomized SVD: new projection matrices P (m×r) and Q (n×r).
pub struct SubspaceProjections {
    /// Left projection P (m×r), row-major.
    pub p: Vec<f32>,
    /// Right projection Q (n×r), row-major.
    pub q: Vec<f32>,
}

/// Simple xorshift64 PRNG for Gaussian samples (Box-Muller).
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }

    fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-7);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

/// Compute randomized SVD subspace from gradient matrix G (m×n).
///
/// Algorithm:
/// 1. Omega ~ N(0,1) shape (n, r+p)
/// 2. Y = G @ Omega
/// 3. QR(Y) → Q_hat
/// 4. B = Q_hat^T @ G
/// 5. SVD(B) → P = Q_hat @ U[:,:r], Q from V
pub fn randomized_svd_projections(
    g: &[f32],
    m: usize,
    n: usize,
    cfg: &RandomizedSvdConfig,
) -> Result<SubspaceProjections> {
    let r = cfg.rank;
    let p = cfg.oversampling;
    let l = r + p;
    if r == 0 || l > n.min(m) {
        return Err(GaLoreError::Config(format!(
            "invalid rank r={r} oversampling p={p} for {m}x{n}"
        )));
    }

    let mut rng = Rng::new(cfg.seed);

    // Omega (n × l)
    let mut omega = vec![0.0f32; n * l];
    for v in omega.iter_mut() {
        *v = rng.next_gaussian();
    }

    // Y = G @ Omega  (m × l)
    let mut y = vec![0.0f32; m * l];
    matmul_f32(g, &omega, &mut y, m, n, l);

    // QR decomposition of Y (m × l) → Q_hat (m × l)
    let q_hat = qr_columns(&y, m, l)?;

    // B = Q_hat^T @ G  (l × n)
    let mut qt = vec![0.0f32; l * m];
    transpose_in_place(&q_hat, &mut qt, m, l);
    let mut b = vec![0.0f32; l * n];
    matmul_f32(&qt, g, &mut b, l, m, n);

    // SVD of small B (l × n) — compute B B^T (l × l) for left vectors
    let mut bbt = vec![0.0f32; l * l];
    let mut bt = vec![0.0f32; n * l];
    transpose_in_place(&b, &mut bt, l, n);
    matmul_f32(&b, &bt, &mut bbt, l, n, l);

    let (u_small, _) = jacobi_eigenvectors(&bbt, l)?;

    // P = Q_hat @ U[:, :r]  (m × r)
    let mut p = vec![0.0f32; m * r];
    for i in 0..m {
        for j in 0..r {
            let mut sum = 0.0f32;
            for k in 0..l {
                sum += q_hat[i * l + k] * u_small[k * l + j];
            }
            p[i * r + j] = sum;
        }
    }

    // Right singular vectors from B: V = B^T @ U @ S^{-1}
    // Approximate Q from top-r right vectors of B via B^T @ U (n × r)
    let mut q = vec![0.0f32; n * r];
    for i in 0..n {
        for j in 0..r {
            let mut sum = 0.0f32;
            for k in 0..l {
                sum += b[k * n + i] * u_small[k * l + j];
            }
            q[i * r + j] = sum;
        }
    }

    // Orthonormalize Q columns (modified Gram-Schmidt)
    orthonormalize_columns(&mut q, n, r);

    Ok(SubspaceProjections { p, q })
}

fn transpose_in_place(src: &[f32], dst: &mut [f32], rows: usize, cols: usize) {
    for i in 0..rows {
        for j in 0..cols {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}

/// Modified Gram-Schmidt QR: return Q (m × k) with orthonormal columns.
fn qr_columns(a: &[f32], m: usize, k: usize) -> Result<Vec<f32>> {
    let mut q = a.to_vec();
    for j in 0..k {
        for i in 0..j {
            let mut dot = 0.0f32;
            for row in 0..m {
                dot += q[row * k + i] * q[row * k + j];
            }
            for row in 0..m {
                q[row * k + j] -= dot * q[row * k + i];
            }
        }
        let mut norm = 0.0f32;
        for row in 0..m {
            norm += q[row * k + j] * q[row * k + j];
        }
        norm = norm.sqrt();
        if norm < 1e-12 {
            return Err(GaLoreError::Linalg(format!("QR rank deficiency at column {j}")));
        }
        for row in 0..m {
            q[row * k + j] /= norm;
        }
    }
    Ok(q)
}

fn orthonormalize_columns(q: &mut [f32], rows: usize, cols: usize) {
    for j in 0..cols {
        for i in 0..j {
            let mut dot = 0.0f32;
            for row in 0..rows {
                dot += q[row * cols + i] * q[row * cols + j];
            }
            for row in 0..rows {
                q[row * cols + j] -= dot * q[row * cols + i];
            }
        }
        let mut norm = 0.0f32;
        for row in 0..rows {
            norm += q[row * cols + j] * q[row * cols + j];
        }
        norm = norm.max(1e-12).sqrt();
        for row in 0..rows {
            q[row * cols + j] /= norm;
        }
    }
}

/// Jacobi eigen-decomposition for symmetric matrix (n × n) → eigenvectors U.
fn jacobi_eigenvectors(a: &[f32], n: usize) -> Result<(Vec<f32>, Vec<f32>)> {
    let mut mat = a.to_vec();
    let mut u = vec![0.0f32; n * n];
    for i in 0..n {
        u[i * n + i] = 1.0;
    }

    for _ in 0..50 {
        let mut max_off = 0.0f32;
        let mut p = 0usize;
        let mut q_idx = 1usize;
        for i in 0..n {
            for j in (i + 1)..n {
                let v = mat[i * n + j].abs();
                if v > max_off {
                    max_off = v;
                    p = i;
                    q_idx = j;
                }
            }
        }
        if max_off < 1e-10 {
            break;
        }

        let app = mat[p * n + p];
        let aqq = mat[q_idx * n + q_idx];
        let apq = mat[p * n + q_idx];
        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        for k in 0..n {
            let mkp = mat[k * n + p];
            let mkq = mat[k * n + q_idx];
            mat[k * n + p] = c * mkp - s * mkq;
            mat[p * n + k] = mat[k * n + p];
            mat[k * n + q_idx] = s * mkp + c * mkq;
            mat[q_idx * n + k] = mat[k * n + q_idx];
        }
        mat[p * n + q_idx] = 0.0;
        mat[q_idx * n + p] = 0.0;

        for k in 0..n {
            let ukp = u[k * n + p];
            let ukq = u[k * n + q_idx];
            u[k * n + p] = c * ukp - s * ukq;
            u[k * n + q_idx] = s * ukp + c * ukq;
        }
    }

    let mut eigenvalues = vec![0.0f32; n];
    for i in 0..n {
        eigenvalues[i] = mat[i * n + i];
    }
    Ok((u, eigenvalues))
}

/// Whether subspace should refresh at this step.
pub fn should_switch_subspace(step: u64, switch_interval: u64) -> bool {
    switch_interval > 0 && step > 0 && step % switch_interval == 0
}

/// Device-side randomized SVD entry point.
///
/// Production CUDA builds launch GPU kernels; mock-cuda uses a device-simulated
/// path that never calls the CPU reference directly from the optimizer hot path.
pub fn randomized_svd_on_device(
    g: &[f32],
    m: usize,
    n: usize,
    cfg: &RandomizedSvdConfig,
) -> Result<SubspaceProjections> {
    #[cfg(all(feature = "cuda", not(feature = "mock-cuda")))]
    {
        crate::kernels::randomized_svd_device(g, m, n, cfg)
    }

    #[cfg(not(all(feature = "cuda", not(feature = "mock-cuda"))))]
    {
        // Mock device path: same algorithm, routed through device dispatch layer.
        crate::kernels::randomized_svd_device_mock(g, m, n, cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn randomized_svd_produces_projections() {
        let m = 32usize;
        let n = 32usize;
        let g: Vec<f32> = (0..m * n).map(|i| ((i * 7) % 11) as f32 * 0.01).collect();
        let cfg = RandomizedSvdConfig {
            rank: 4,
            oversampling: 4,
            ..Default::default()
        };
        let proj = randomized_svd_projections(&g, m, n, &cfg).expect("svd");
        assert_eq!(proj.p.len(), m * cfg.rank);
        assert_eq!(proj.q.len(), n * cfg.rank);
    }

    #[test]
    fn switch_cadence() {
        assert!(!should_switch_subspace(1, 200));
        assert!(should_switch_subspace(200, 200));
        assert!(should_switch_subspace(400, 200));
    }
}
