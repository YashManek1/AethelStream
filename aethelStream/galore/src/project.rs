//! CPU reference and host-side GaLore projection (Algorithm 1).

use crate::error::{GaLoreError, Result};

/// Row-major matrix multiply: C(m×n) = A(m×k) @ B(k×n).
pub fn matmul_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// Transpose row-major A(m×n) into B(n×m).
pub fn transpose_f32(src: &[f32], dst: &mut [f32], rows: usize, cols: usize) {
    for i in 0..rows {
        for j in 0..cols {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}

/// Forward projection: R = P^T @ G @ Q.
///
/// - `g`: m×n gradient
/// - `p`: m×r left projection
/// - `q`: n×r right projection
/// - `r_out`: r×r output (must be zero-initialized or overwritten)
pub fn project_forward_f32(g: &[f32], p: &[f32], q: &[f32], r_out: &mut [f32], m: usize, n: usize, r: usize) {
    let mut temp = vec![0.0f32; r * n];
    // Temp = P^T @ G
    for i in 0..r {
        for j in 0..n {
            let mut sum = 0.0f32;
            for k in 0..m {
                sum += p[k * r + i] * g[k * n + j];
            }
            temp[i * n + j] = sum;
        }
    }
    // R = Temp @ Q
    matmul_f32(&temp, q, r_out, r, n, r);
}

/// Backward projection: G_tilde = P @ N @ Q^T.
pub fn project_backward_f32(n: &[f32], p: &[f32], q: &[f32], g_out: &mut [f32], m: usize, n_dim: usize, r: usize) {
    let mut temp = vec![0.0f32; m * r];
    // Temp = P @ N
    matmul_f32(p, n, &mut temp, m, r, r);
    // G_tilde = Temp @ Q^T
    let mut qt = vec![0.0f32; r * n_dim];
    transpose_f32(q, &mut qt, n_dim, r);
    matmul_f32(&temp, &qt, g_out, m, r, n_dim);
}

/// Round-trip relative Frobenius error: ||G - G_tilde||_F / ||G||_F.
pub fn projection_roundtrip_error(g: &[f32], p: &[f32], q: &[f32], m: usize, n: usize, r: usize) -> f64 {
    let mut compact = vec![0.0f32; r * r];
    project_forward_f32(g, p, q, &mut compact, m, n, r);

    let mut recon = vec![0.0f32; m * n];
    project_backward_f32(&compact, p, q, &mut recon, m, n, r);

    let mut diff_sq = 0.0f64;
    let mut orig_sq = 0.0f64;
    for (a, b) in g.iter().zip(recon.iter()) {
        let d = f64::from(*a - *b);
        diff_sq += d * d;
        orig_sq += f64::from(*a) * f64::from(*a);
    }
    if orig_sq <= 1e-30 {
        return 0.0;
    }
    (diff_sq / orig_sq).sqrt()
}

/// Validate matrix dimensions for projection.
pub fn validate_projection_dims(m: usize, n: usize, r: usize, g_len: usize, p_len: usize, q_len: usize, out_len: usize) -> Result<()> {
    if m == 0 || n == 0 || r == 0 {
        return Err(GaLoreError::Shape("m, n, r must be positive".into()));
    }
    if g_len != m * n {
        return Err(GaLoreError::Shape(format!("G len {g_len} != m*n={}", m * n)));
    }
    if p_len != m * r {
        return Err(GaLoreError::Shape(format!("P len {p_len} != m*r={}", m * r)));
    }
    if q_len != n * r {
        return Err(GaLoreError::Shape(format!("Q len {q_len} != n*r={}", n * r)));
    }
    if out_len != r * r {
        return Err(GaLoreError::Shape(format!("out len {out_len} != r*r={}", r * r)));
    }
    Ok(())
}

/// Convert f32 slice to f16 bits (IEEE half).
pub fn f32_to_f16_bits(x: f32) -> u16 {
    half::f32_to_f16(x)
}

/// Convert f16 bits to f32.
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    half::f16_to_f32(bits)
}

/// FP16 forward projection using f32 accumulation internally.
pub fn project_forward_f16(g: &[u16], p: &[u16], q: &[u16], r_out: &mut [u16], m: usize, n: usize, r: usize) {
    let gf: Vec<f32> = g.iter().map(|&b| f16_bits_to_f32(b)).collect();
    let pf: Vec<f32> = p.iter().map(|&b| f16_bits_to_f32(b)).collect();
    let qf: Vec<f32> = q.iter().map(|&b| f16_bits_to_f32(b)).collect();
    let mut rf = vec![0.0f32; r * r];
    project_forward_f32(&gf, &pf, &qf, &mut rf, m, n, r);
    for (o, v) in r_out.iter_mut().zip(rf.iter()) {
        *o = f32_to_f16_bits(*v);
    }
}

/// FP16 backward projection.
pub fn project_backward_f16(n_mat: &[u16], p: &[u16], q: &[u16], g_out: &mut [u16], m: usize, n_dim: usize, r: usize) {
    let nf: Vec<f32> = n_mat.iter().map(|&b| f16_bits_to_f32(b)).collect();
    let pf: Vec<f32> = p.iter().map(|&b| f16_bits_to_f32(b)).collect();
    let qf: Vec<f32> = q.iter().map(|&b| f16_bits_to_f32(b)).collect();
    let mut gf = vec![0.0f32; m * n_dim];
    project_backward_f32(&nf, &pf, &qf, &mut gf, m, n_dim, r);
    for (o, v) in g_out.iter_mut().zip(gf.iter()) {
        *o = f32_to_f16_bits(*v);
    }
}

/// Minimal IEEE half conversion (no external crate).
mod half {
    pub fn f32_to_f16(x: f32) -> u16 {
        let bits = x.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let frac = bits & 0x7F_FFFF;

        if exp == 0xFF {
            return sign | 0x7C00 | ((frac >> 13) as u16 & 0x3FF);
        }
        let new_exp = exp - 127 + 15;
        if new_exp >= 0x1F {
            return sign | 0x7C00;
        }
        if new_exp <= 0 {
            if new_exp < -10 {
                return sign;
            }
            let mant = (frac | 0x80_0000) >> (1 - new_exp);
            return sign | ((mant + 0x1000) >> 13) as u16;
        }
        sign | ((new_exp as u16) << 10) | ((frac + 0x1000) >> 13) as u16
    }

    pub fn f16_to_f32(h: u16) -> f32 {
        let sign = (h & 0x8000) as u32;
        let exp = ((h >> 10) & 0x1F) as u32;
        let frac = (h & 0x3FF) as u32;

        let bits = if exp == 0 {
            if frac == 0 {
                sign << 16
            } else {
                let mut e = 0u32;
                let mut f = frac;
                while (f & 0x400) == 0 {
                    f <<= 1;
                    e += 1;
                }
                f &= 0x3FF;
                (sign << 16) | ((127 - 15 - e) << 23) | (f << 13)
            }
        } else if exp == 0x1F {
            (sign << 16) | 0x7F80_0000 | (frac << 13)
        } else {
            (sign << 16) | ((exp + 127 - 15) << 23) | (frac << 13)
        };
        f32::from_bits(bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_small_matrix() {
        let m = 4usize;
        let n = 4usize;
        let r = 2usize;
        let g: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.01).collect();
        let p: Vec<f32> = (0..m * r).map(|i| ((i % 3) as f32) * 0.1).collect();
        let q: Vec<f32> = (0..n * r).map(|i| ((i % 5) as f32) * 0.1).collect();
        let err = projection_roundtrip_error(&g, &p, &q, m, n, r);
        assert!(err < 1.0, "roundtrip error {err}");
    }

    #[test]
    fn roundtrip_identity_projectors() {
        let m = 2usize;
        let n = 2usize;
        let r = 2usize;
        let p = vec![1.0, 0.0, 0.0, 1.0];
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let g = vec![1.0, 2.0, 3.0, 4.0];
        let err = projection_roundtrip_error(&g, &p, &q, m, n, r);
        assert!(err < 1e-5, "identity projector error {err}");
    }
}
