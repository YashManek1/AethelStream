//! Absmax INT8 quantisation for AdamW momentum and variance (Algorithm 2).

/// Compute absmax scale: max(abs(tensor)) / 127.0
pub fn absmax_scale(tensor: &[f32]) -> f32 {
    let max_abs = tensor
        .iter()
        .fold(0.0f32, |acc, &v| acc.max(v.abs()));
    if max_abs <= 0.0 {
        1.0
    } else {
        max_abs / 127.0
    }
}

/// Quantise FP32 tensor to INT8 using absmax scaling.
/// `quantized[i] = clamp(round(tensor[i] / scale), -127, 127)`
pub fn quantize_absmax(tensor: &[f32], out: &mut [i8]) -> f32 {
    assert_eq!(tensor.len(), out.len());
    let scale = absmax_scale(tensor);
    let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
    for (src, dst) in tensor.iter().zip(out.iter_mut()) {
        let q = (*src * inv).round();
        *dst = q.clamp(-127.0, 127.0) as i8;
    }
    scale
}

/// Dequantise INT8 tensor: tensor = quantized * scale
pub fn dequantize_absmax(quantized: &[i8], scale: f32, out: &mut [f32]) {
    assert_eq!(quantized.len(), out.len());
    for (src, dst) in quantized.iter().zip(out.iter_mut()) {
        *dst = f32::from(*src) * scale;
    }
}

/// Relative round-trip error: max|x - x̂| / max|x|.
pub fn quantize_relative_error(tensor: &[f32]) -> f32 {
    let mut q = vec![0i8; tensor.len()];
    let scale = quantize_absmax(tensor, &mut q);
    let mut recon = vec![0.0f32; tensor.len()];
    dequantize_absmax(&q, scale, &mut recon);

    let max_abs = tensor.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    if max_abs <= 0.0 {
        return 0.0;
    }
    let max_err = tensor
        .iter()
        .zip(recon.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    max_err / max_abs
}

/// Round-trip quantisation error (max absolute).
pub fn quantize_max_error(tensor: &[f32]) -> f32 {
    let mut q = vec![0i8; tensor.len()];
    let scale = quantize_absmax(tensor, &mut q);
    let mut recon = vec![0.0f32; tensor.len()];
    dequantize_absmax(&q, scale, &mut recon);
    tensor
        .iter()
        .zip(recon.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_dequantize_bounded_error() {
        let data: Vec<f32> = (-128..128).map(|i| i as f32 * 0.01).collect();
        let err = quantize_max_error(&data);
        assert!(err <= absmax_scale(&data) + 1e-5);
    }
}
