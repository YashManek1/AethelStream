//! A1 forward pass
use crate::math::{add_vecs, matmul_tb, rms_norm_fwd, softmax_rows, silu_f};
use crate::{Batch, Result};
use crate::plan::TrainingPlan;
use crate::state::RngState;
use ramflow::PinnedBuffer;

#[derive(Debug, Clone, Copy)]
/// Block config.
pub struct BlockConfig {
    /// d_model.
pub d_model: usize,
    /// n_heads.
pub n_heads: usize,
    /// d_ff.
pub d_ff: usize,
    /// seq_len.
pub seq_len: usize,
    /// batch.
pub batch: usize,
}

impl BlockConfig {
    /// d_head.
pub fn d_head(&self) -> usize { self.d_model / self.n_heads }
    /// bs.
pub fn bs(&self) -> usize { self.batch * self.seq_len }
}

#[derive(Debug, Clone)]
/// Weights.
pub struct BlockWeights {
    /// w.
pub rms1_w: Vec<f32>,
    /// w.
pub wq: Vec<f32>,
    /// w.
pub wk: Vec<f32>,
    /// w.
pub wv: Vec<f32>,
    /// w.
pub wo: Vec<f32>,
    /// w.
pub rms2_w: Vec<f32>,
    /// w.
pub wg: Vec<f32>,
    /// w.
pub wu: Vec<f32>,
    /// w.
pub wd: Vec<f32>,
}

impl BlockWeights {
    /// Init.
pub fn from_formula(cfg: &BlockConfig) -> Self {
        let init = |n: usize, offset: f64| -> Vec<f32> {
            (0..n).map(|i| ((i as f64 * 0.137 + offset).sin() * 0.1) as f32).collect()
        };
        let d = cfg.d_model;
        let ff = cfg.d_ff;
        BlockWeights {
            rms1_w: init(d, 0.0),
            wq: init(d * d, 1.0),
            wk: init(d * d, 2.0),
            wv: init(d * d, 3.0),
            wo: init(d * d, 4.0),
            rms2_w: init(d, 5.0),
            wg: init(ff * d, 6.0),
            wu: init(ff * d, 7.0),
            wd: init(d * ff, 8.0),
        }
    }
}

#[derive(Debug, Clone)]
/// Fwd out.
pub struct SingleLayerFwdOut {
    /// x.
pub x_in: Vec<f32>,
    /// rms.
pub rms1: Vec<f32>,
    /// x.
pub x_norm1: Vec<f32>,
    /// h.
pub h1: Vec<f32>,
    /// q.
pub q_heads: Vec<f32>,
    /// k.
pub k_heads: Vec<f32>,
    /// v.
pub v_heads: Vec<f32>,
    /// s.
pub attn_scores: Vec<f32>,
    /// w.
pub attn_weights: Vec<f32>,
    /// o.
pub attn_out: Vec<f32>,
    /// p.
pub out_proj: Vec<f32>,
    /// x.
pub x2: Vec<f32>,
    /// rms.
pub rms2: Vec<f32>,
    /// x.
pub x_norm2: Vec<f32>,
    /// h.
pub h2: Vec<f32>,
    /// g.
pub gate: Vec<f32>,
    /// u.
pub up: Vec<f32>,
    /// s.
pub silu_gate: Vec<f32>,
    /// h.
pub hidden: Vec<f32>,
    /// m.
pub mlp_out: Vec<f32>,
    /// o.
pub output: Vec<f32>,
}

/// Fwd.
pub fn single_layer_forward(cfg: &BlockConfig, w: &BlockWeights, input: &[f32]) -> SingleLayerFwdOut {
    let bs = cfg.bs();
    let d = cfg.d_model;
    let h = cfg.n_heads;
    let dh = cfg.d_head();
    let bh = cfg.batch * cfg.n_heads;
    let s = cfg.seq_len;
    let ff = cfg.d_ff;
    
    let (h1, x_norm1, rms1) = rms_norm_fwd(input, &w.rms1_w, bs, d);
    let q_flat = matmul_tb(&h1, &w.wq, bs, d, d);
    let k_flat = matmul_tb(&h1, &w.wk, bs, d, d);
    let v_flat = matmul_tb(&h1, &w.wv, bs, d, d);

    let reshape_to_heads = |flat: &[f32]| -> Vec<f32> {
        let mut heads = vec![0.0f32; bh * s * dh];
        for b in 0..cfg.batch {
            for hh in 0..h {
                for ss in 0..s {
                    for t in 0..dh {
                        heads[((b * h + hh) * s + ss) * dh + t] = flat[(b * s + ss) * d + hh * dh + t];
                    }
                }
            }
        }
        heads
    };

    let q_heads = reshape_to_heads(&q_flat);
    let k_heads = reshape_to_heads(&k_flat);
    let v_heads = reshape_to_heads(&v_flat);

    let mut attn_scores = vec![0.0f32; bh * s * s];
    let scale = (dh as f32).sqrt();
    for bh_i in 0..bh {
        for s1 in 0..s {
            for s2 in 0..s {
                let mut dot = 0.0f32;
                for t in 0..dh {
                    dot += q_heads[bh_i * s * dh + s1 * dh + t] * k_heads[bh_i * s * dh + s2 * dh + t];
                }
                attn_scores[bh_i * s * s + s1 * s + s2] = dot / scale;
            }
        }
    }

    let attn_weights = softmax_rows(&attn_scores, bh * s, s);

    let mut attn_out_heads = vec![0.0f32; bh * s * dh];
    for bh_i in 0..bh {
        for s1 in 0..s {
            for t in 0..dh {
                let mut v = 0.0f32;
                for s2 in 0..s {
                    v += attn_weights[bh_i * s * s + s1 * s + s2] * v_heads[bh_i * s * dh + s2 * dh + t];
                }
                attn_out_heads[bh_i * s * dh + s1 * dh + t] = v;
            }
        }
    }

    let reshape_from_heads = |heads: &[f32]| -> Vec<f32> {
        let mut flat = vec![0.0f32; bs * d];
        for b in 0..cfg.batch {
            for hh in 0..h {
                for ss in 0..s {
                    for t in 0..dh {
                        flat[(b * s + ss) * d + hh * dh + t] = heads[((b * h + hh) * s + ss) * dh + t];
                    }
                }
            }
        }
        flat
    };

    let attn_out = reshape_from_heads(&attn_out_heads);
    let out_proj = matmul_tb(&attn_out, &w.wo, bs, d, d);
    let x2 = add_vecs(input, &out_proj);

    let (h2, x_norm2, rms2) = rms_norm_fwd(&x2, &w.rms2_w, bs, d);

    let gate = matmul_tb(&h2, &w.wg, bs, d, ff);
    let up = matmul_tb(&h2, &w.wu, bs, d, ff);
    let silu_gate: Vec<f32> = gate.iter().map(|&g| silu_f(g)).collect();
    let hidden: Vec<f32> = silu_gate.iter().zip(&up).map(|(sg, u)| sg * u).collect();
    let mlp_out = matmul_tb(&hidden, &w.wd, bs, ff, d);
    let output = add_vecs(&x2, &mlp_out);

    SingleLayerFwdOut {
        x_in: input.to_vec(),
        rms1,
        x_norm1,
        h1,
        q_heads,
        k_heads,
        v_heads,
        attn_scores,
        attn_weights,
        attn_out,
        out_proj,
        x2,
        rms2,
        x_norm2,
        h2,
        gate,
        up,
        silu_gate,
        hidden,
        mlp_out,
        output,
    }
}

/// Out.
pub struct ForwardOutput {
    /// c.
pub checkpoints: Vec<(u32, u32, PinnedBuffer)>,
    /// r.
pub rng_states: Vec<RngState>,
    /// b.
pub weight_bytes: u64,
}

/// Run.
pub fn run_forward(
    _flowcast: &crate::FlowCast,
    _plan: &TrainingPlan,
    _batch: &Batch,
) -> Result<ForwardOutput> {
    unimplemented!("forward stub")
}
