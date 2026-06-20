//! T-CE — Streaming Cut Cross-Entropy (A8) tests.
//!
//! 1. Loss matches a reference materialising the full [bs,V] tensor < 1e-4.
//! 2. Grad wrt hidden matches reference < 1e-3 (FP32).
//! 3. Peak logit memory is O(chunk): measured with V=128k, chunk=256.
//! 4. Determinism: same inputs -> identical loss/grad.
//! 5. Remainder chunk: V not divisible by chunk_size.
//! 6. Single-token sanity: bs=1, V=8, chunk=4 manual reference.
//!
//! Run: cargo test --features mock-cuda -p doublepass test_cut_ce -- --nocapture

#![allow(clippy::unwrap_used, clippy::expect_used)]

use doublepass::loss::streaming_cut_ce;

fn gen(seed: usize) -> f32 {
    let x = (seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)) as f64;
    (x * 7.4505806e-18_f64).sin() as f32 * 0.15
}

fn make_hidden(batch_seq: usize, d_model: usize) -> Vec<f32> {
    (0..batch_seq * d_model).map(|i| gen(i * 3 + 1)).collect()
}
fn make_lm_head(vocab_size: usize, d_model: usize) -> Vec<f32> {
    (0..vocab_size * d_model).map(|i| gen(i * 7 + 5)).collect()
}
fn make_labels(batch_seq: usize, vocab_size: usize) -> Vec<u32> {
    (0..batch_seq).map(|i| ((i * 31 + 7) % vocab_size) as u32).collect()
}

/// Reference CE: materialises full [bs, V] logit tensor (what streaming_cut_ce avoids).
fn reference_ce(
    hidden: &[f32], lm_head: &[f32], labels: &[u32],
    vocab_size: usize, d_model: usize,
) -> (f32, Vec<f32>) {
    let bs = labels.len();
    let mut logits = vec![0.0_f32; bs * vocab_size];
    for i in 0..bs {
        let h_i = &hidden[i * d_model..(i + 1) * d_model];
        for v in 0..vocab_size {
            let w_v = &lm_head[v * d_model..(v + 1) * d_model];
            logits[i * vocab_size + v] = h_i.iter().zip(w_v).map(|(a, b)| a * b).sum();
        }
    }
    let mut loss = 0.0_f32;
    let mut grad_h = vec![0.0_f32; bs * d_model];
    for i in 0..bs {
        let row = &logits[i * vocab_size..(i + 1) * vocab_size];
        let label = labels[i] as usize;
        let max_v = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|z| (z - max_v).exp()).collect();
        let sum_e: f32 = exps.iter().sum();
        loss += -(row[label] - max_v) + sum_e.ln();
        let gh_i = &mut grad_h[i * d_model..(i + 1) * d_model];
        for v in 0..vocab_size {
            let factor = exps[v] / sum_e - if v == label { 1.0_f32 } else { 0.0_f32 };
            let w_v = &lm_head[v * d_model..(v + 1) * d_model];
            for j in 0..d_model { gh_i[j] += factor * w_v[j]; }
        }
    }
    loss /= bs as f32;
    let inv_bs = 1.0_f32 / bs as f32;
    for g in grad_h.iter_mut() { *g *= inv_bs; }
    (loss, grad_h)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max)
}

// T-CE-1: Loss vs reference < 1e-4
#[test]
fn test_loss_matches_reference() {
    let (bs, d, v, c) = (6, 16, 128, 32);
    let h = make_hidden(bs, d); let w = make_lm_head(v, d); let l = make_labels(bs, v);
    let (ref_loss, _) = reference_ce(&h, &w, &l, v, d);
    let r = streaming_cut_ce(&h, &w, d, v, c, &l, None).expect("ok");
    let diff = (r.loss - ref_loss).abs();
    println!("\nT-CE-1: ref={:.8}  cut={:.8}  |diff|={:.2e}", ref_loss, r.loss, diff);
    assert!(diff < 1e-4_f32, "loss diff {diff:.2e} >= 1e-4");
    println!("T-CE-1 PASSED");
}

// T-CE-2: Grad wrt hidden < 1e-3
#[test]
fn test_grad_hidden_matches_reference() {
    let (bs, d, v, c) = (6, 16, 128, 32);
    let h = make_hidden(bs, d); let w = make_lm_head(v, d); let l = make_labels(bs, v);
    let (_, ref_grad) = reference_ce(&h, &w, &l, v, d);
    let r = streaming_cut_ce(&h, &w, d, v, c, &l, None).expect("ok");
    let diff = max_abs_diff(&r.grad_hidden, &ref_grad);
    println!("\nT-CE-2: max|delta_grad_h|={:.2e}  (tol=1e-3)", diff);
    assert!(diff < 1e-3_f32, "grad max_abs_diff {diff:.2e} >= 1e-3");
    println!("T-CE-2 PASSED");
}

// T-CE-3: O(chunk) peak logit bytes  [V=128k, chunk=256]
#[test]
fn test_peak_logit_bytes_o_chunk() {
    let (bs, d, v, c) = (2_usize, 16_usize, 131_072_usize, 256_usize);
    let h = make_hidden(bs, d); let w = make_lm_head(v, d); let l = make_labels(bs, v);
    let r = streaming_cut_ce(&h, &w, d, v, c, &l, None).expect("ok");
    let expected  = bs * c * std::mem::size_of::<f32>();
    let full_ov   = bs * v * std::mem::size_of::<f32>();
    println!("\nT-CE-3 (V=128k):  MEASURED peak_logit_bytes = {} B", r.peak_logit_bytes);
    println!("                  O(chunk) expected          = {} B  ({}x{} x4)", expected, bs, c);
    println!("                  O(V) would be              = {} B  ({:.0}x larger)", full_ov,
        full_ov as f64 / r.peak_logit_bytes as f64);
    assert_eq!(r.peak_logit_bytes, expected,
        "peak {} != O(chunk) {}", r.peak_logit_bytes, expected);
    assert!(r.peak_logit_bytes < full_ov / 100,
        "peak ({}) must be <1% of O(V) ({})", r.peak_logit_bytes, full_ov);
    println!("T-CE-3 PASSED — {}x reduction vs O(V)", full_ov / r.peak_logit_bytes);
}

// T-CE-4: Determinism
#[test]
fn test_determinism() {
    let (bs, d, v, c) = (4, 16, 64, 16);
    let h = make_hidden(bs, d); let w = make_lm_head(v, d); let l = make_labels(bs, v);
    let r1 = streaming_cut_ce(&h, &w, d, v, c, &l, None).expect("c1");
    let r2 = streaming_cut_ce(&h, &w, d, v, c, &l, None).expect("c2");
    assert_eq!(r1.loss.to_bits(), r2.loss.to_bits(),
        "loss not bit-identical: {:.10} vs {:.10}", r1.loss, r2.loss);
    assert_eq!(max_abs_diff(&r1.grad_hidden, &r2.grad_hidden), 0.0_f32,
        "grad_hidden not bit-identical");
    println!("\nT-CE-4 PASSED — bit-identical on two consecutive calls");
}

// T-CE-5: Remainder chunk (V % chunk != 0)
#[test]
fn test_remainder_chunk() {
    let (bs, d, v, c) = (4, 12, 70, 32);
    let h = make_hidden(bs, d); let w = make_lm_head(v, d); let l = make_labels(bs, v);
    let (ref_loss, ref_grad) = reference_ce(&h, &w, &l, v, d);
    let r = streaming_cut_ce(&h, &w, d, v, c, &l, None).expect("ok");
    let ld = (r.loss - ref_loss).abs();
    let gd = max_abs_diff(&r.grad_hidden, &ref_grad);
    println!("\nT-CE-5 (V=70 chunk=32): loss_diff={:.2e}  grad_diff={:.2e}", ld, gd);
    assert!(ld < 1e-4_f32, "loss diff {ld:.2e} >= 1e-4");
    assert!(gd < 1e-3_f32, "grad diff {gd:.2e} >= 1e-3");
    println!("T-CE-5 PASSED");
}

// T-CE-6: Single-token sanity (bs=1, V=8, chunk=4, d=2)
#[test]
fn test_single_token_sanity() {
    let h: Vec<f32> = vec![1.0, 0.0];
    let w: Vec<f32> = (0..8_usize).flat_map(|v| [v as f32 * 0.5, 0.0_f32]).collect();
    let l = vec![3_u32];
    let (d, v_size, chunk) = (2_usize, 8_usize, 4_usize);

    let max_v = 3.5_f32;
    let exps: Vec<f32> = (0..8_usize).map(|v| ((v as f32 * 0.5) - max_v).exp()).collect();
    let sum_e: f32 = exps.iter().sum();
    let ref_loss = -(1.5_f32 - max_v) + sum_e.ln();
    let mut ref_g0 = 0.0_f32;
    for v in 0..8_usize {
        let factor = exps[v] / sum_e - if v == 3 { 1.0 } else { 0.0 };
        ref_g0 += factor * (v as f32 * 0.5);
    }

    let r = streaming_cut_ce(&h, &w, d, v_size, chunk, &l, None).expect("ok");
    let ld  = (r.loss - ref_loss).abs();
    let g0d = (r.grad_hidden[0] - ref_g0).abs();
    let g1d = r.grad_hidden[1].abs();
    println!("\nT-CE-6: ref_loss={:.8}  cut={:.8}  |dl|={:.2e}", ref_loss, r.loss, ld);
    println!("        grad[0]: ref={:.8}  cut={:.8}  |dg|={:.2e}", ref_g0, r.grad_hidden[0], g0d);
    println!("        grad[1]: ref=0  cut={:.2e}", r.grad_hidden[1]);

    assert!(ld  < 1e-5_f32, "loss diff {ld:.2e} >= 1e-5");
    assert!(g0d < 1e-5_f32, "grad[0] diff {g0d:.2e} >= 1e-5");
    assert!(g1d < 1e-5_f32, "grad[1] should be 0, got {:.2e}", r.grad_hidden[1]);

    let expected_peak = 1 * chunk * std::mem::size_of::<f32>();
    assert_eq!(r.peak_logit_bytes, expected_peak,
        "peak {} != {} (1xcx4)", r.peak_logit_bytes, expected_peak);
    println!("T-CE-6 PASSED — peak_logit_bytes={} B", r.peak_logit_bytes);
}
