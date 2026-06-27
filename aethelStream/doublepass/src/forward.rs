//! A1 forward pass
use crate::math::{add_vecs, matmul_tb, rms_norm_fwd, silu_f, softmax_rows};
use crate::plan::TrainingPlan;
use crate::state::RngState;
use crate::{Batch, Result};
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
    /// Dropout probability applied after attention out-projection (0.0 = disabled).
    pub dropout_p: f32,
}

impl BlockConfig {
    /// d_head.
    pub fn d_head(&self) -> usize {
        self.d_model / self.n_heads
    }
    /// bs.
    pub fn bs(&self) -> usize {
        self.batch * self.seq_len
    }
    /// Weight byte count for one transformer block (f32, 4 bytes per element).
    ///
    /// Equals (2*d + 4*d*d + 3*d*ff) * 4.
    pub fn bytes_per_layer(&self) -> usize {
        let d = self.d_model;
        let ff = self.d_ff;
        (2 * d + 4 * d * d + 3 * ff * d) * 4
    }
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
            (0..n)
                .map(|i| ((i as f64 * 0.137 + offset).sin() * 0.1) as f32)
                .collect()
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

    /// Init weights for layer `layer` (distinct from other layers via offset).
    pub fn from_formula_layered(cfg: &BlockConfig, layer: usize) -> Self {
        let off = layer as f64 * 100.0;
        let init = |n: usize, base: f64| -> Vec<f32> {
            (0..n)
                .map(|i| ((i as f64 * 0.137 + base + off).sin() * 0.1) as f32)
                .collect()
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
pub fn single_layer_forward(
    cfg: &BlockConfig,
    w: &BlockWeights,
    input: &[f32],
) -> SingleLayerFwdOut {
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
                        heads[((b * h + hh) * s + ss) * dh + t] =
                            flat[(b * s + ss) * d + hh * dh + t];
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
                    dot +=
                        q_heads[bh_i * s * dh + s1 * dh + t] * k_heads[bh_i * s * dh + s2 * dh + t];
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
                    v += attn_weights[bh_i * s * s + s1 * s + s2]
                        * v_heads[bh_i * s * dh + s2 * dh + t];
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
                        flat[(b * s + ss) * d + hh * dh + t] =
                            heads[((b * h + hh) * s + ss) * dh + t];
                    }
                }
            }
        }
        flat
    };

    let attn_out = reshape_from_heads(&attn_out_heads);
    let out_proj = matmul_tb(&attn_out, &w.wo, bs, d, d);
    let mut out_proj_dp = out_proj;
    crate::rng::apply_dropout(&mut out_proj_dp, cfg.dropout_p);
    let x2 = add_vecs(input, &out_proj_dp);

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
        out_proj: out_proj_dp,
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

/// An in-memory multi-layer transformer model for the forward pass.
///
/// In production the weights are streamed from NVMe via FlowCast; under mock
/// this `Vec<BlockWeights>` is the source of truth used by [`full_forward`].
pub struct Model {
    /// Per-layer weight tensors, indexed by `layer_idx`.
    pub layers: Vec<BlockWeights>,
    /// Shared block configuration for all layers.
    pub cfg: BlockConfig,
}

impl Model {
    /// Create a model with `n_layers` transformer blocks.
    ///
    /// Each layer is initialised with [`BlockWeights::from_formula_layered`]
    /// so adjacent layers have distinct weights.
    pub fn new(n_layers: usize, cfg: BlockConfig) -> Self {
        let layers = (0..n_layers)
            .map(|i| BlockWeights::from_formula_layered(&cfg, i))
            .collect();
        Self { layers, cfg }
    }

    /// Number of transformer layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}


impl Model {
    /// Load all transformer layers from a [`crate::WeightProvider`], building an in-memory model.
    ///
    /// Used to bridge M1 (shard_engine) weights into the M5 forward pass.
    /// `SyntheticProvider` works with no file I/O; `ShardEngineProvider` streams from NVMe.
    ///
    /// # Errors
    /// Propagates [`crate::weight_provider::ProviderError`] from each `load_layer_weights` call.
    pub fn from_provider(
        provider: &mut dyn crate::weight_provider::WeightProvider,
        cfg: BlockConfig,
    ) -> crate::weight_provider::ProviderResult<Self> {
        let n = provider.num_layers() as usize;
        let mut layers = Vec::with_capacity(n);
        for i in 0..n {
            layers.push(provider.load_layer_weights(i as u32, &cfg)?);
        }
        Ok(Self { layers, cfg })
    }
}/// Return value of [`full_forward`].
pub struct FullForwardResult {
    /// Sparse activation checkpoints: `(layer_idx, micro_batch_idx, PinnedBuffer)`.
    ///
    /// Stored immediately *before* running layer `layer_idx` for micro-batch
    /// `micro_batch_idx`, at each `layer_idx` where `layer_idx % checkpoint_freq == 0`.
    /// Holds the input activation to layer `layer_idx`.
    pub checkpoints: Vec<(u32, u32, ramflow::PinnedBuffer)>,
    /// Per-`(layer, micro_batch)` RNG states captured just before each layer's forward.
    pub rng_states: Vec<crate::state::RngState>,
    /// Final residual-stream outputs, one `Vec<f32>` per micro-batch.
    pub outputs: Vec<Vec<f32>>,
    /// Weight bytes that would be streamed in production for this step.
    ///
    /// Equals num_layers x cfg.bytes_per_layer(). Constant in G (grad-accum depth)
    /// because the outer loop is layer-major: each layer's weights are fetched once
    /// regardless of how many micro-batches pass through.
    pub weight_bytes_streamed: u64,
    /// Full per-layer-per-microbatch activations, stored by [`full_forward_with_retention`].
    ///
    /// Empty when constructed by the plain [`full_forward`]; populated for every layer
    /// when called via [`full_forward_with_retention`]. Used by the SARP executor
    /// (`sarp.rs`) for `RetainVram`, `PageCompressedRam`, `PageNvme`, and selective
    /// `Recompute` actions that need non-recomputed op outputs.
    ///
    /// Indexed as: `(layer_idx, micro_batch_idx, SingleLayerFwdOut)`.
    pub retained_activations: Vec<(u32, u32, SingleLayerFwdOut)>,
}

/// Layer-major multi-layer forward pass (pure math - no FlowCast I/O).
///
/// Implements A1: for each layer `i`, all `G = inputs.len()` micro-batches are
/// forwarded before moving to layer `i + 1`. An activation checkpoint is stored
/// in a [`ramflow::PinnedBuffer`] after each layer where
/// `layer_idx % plan.checkpoint_freq == 0`.
///
/// `compress` controls whether checkpoints are INT8-compressed
/// (via `CoScheduler::should_compress_checkpoints()` in the full engine;
/// passed directly here for testing).
///
/// # Errors
/// Propagates checkpoint allocation or kernel errors as [`crate::error::DoublePassError::Checkpoint`].
pub fn full_forward(
    model: &Model,
    inputs: &[Vec<f32>],
    plan: &crate::plan::TrainingPlan,
    compress: bool,
) -> crate::Result<FullForwardResult> {
    let l = model.layers.len();
    let g = inputs.len();
    let k = plan.checkpoint_freq as usize;

    let mut activations: Vec<Vec<f32>> = inputs.to_vec();
    let mut checkpoints: Vec<(u32, u32, ramflow::PinnedBuffer)> = Vec::new();
    let mut rng_states: Vec<crate::state::RngState> = Vec::new();

    // Weight bytes = L x bytes_per_layer; constant regardless of G.
    let weight_bytes_streamed = model
        .cfg
        .bytes_per_layer()
        .saturating_mul(l)
        .try_into()
        .unwrap_or(u64::MAX);

    for i in 0..l {
        for (m, act) in activations.iter_mut().enumerate().take(g) {
            // Capture RNG state before dropout draws in this layer (A10).
            let rng = crate::rng::capture(i as u32, m as u32)?;
            rng_states.push(rng);
            // Store checkpoint of the INPUT to layer i for micro-batch m.
            // Stored before the forward so backward recompute starts from this activation.
            if i % k == 0 {
                let buf = crate::checkpoint::store_checkpoint(act, compress)?;
                checkpoints.push((i as u32, m as u32, buf));
            }
            // Run the layer forward; dropout is applied inside when dropout_p > 0.
            let fwd = single_layer_forward(&model.cfg, &model.layers[i], act);
            *act = fwd.output.clone();
        }
    }

    Ok(FullForwardResult {
        checkpoints,
        rng_states,
        outputs: activations,
        weight_bytes_streamed,
        retained_activations: Vec::new(),
    })
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
/// Layer-major forward pass that also stores **all** per-layer activations.
///
/// Identical math to `full_forward`, but additionally populates
/// `FullForwardResult::retained_activations` with the complete
/// `SingleLayerFwdOut` for every `(layer_idx, micro_batch_idx)`. The SARP
/// executor uses these for `RetainVram`, `PageCompressedRam`, `PageNvme`, and
/// selective-region `Recompute` actions.
pub fn full_forward_with_retention(
    model: &Model,
    inputs: &[Vec<f32>],
    plan: &crate::plan::TrainingPlan,
    compress: bool,
) -> crate::Result<FullForwardResult> {
    let l = model.layers.len();
    let g = inputs.len();
    let k = plan.checkpoint_freq as usize;

    let mut activations: Vec<Vec<f32>> = inputs.to_vec();
    let mut checkpoints: Vec<(u32, u32, ramflow::PinnedBuffer)> = Vec::new();
    let mut rng_states: Vec<crate::state::RngState> = Vec::new();
    let mut retained_activations: Vec<(u32, u32, SingleLayerFwdOut)> = Vec::with_capacity(l * g);

    let weight_bytes_streamed = model
        .cfg
        .bytes_per_layer()
        .saturating_mul(l)
        .try_into()
        .unwrap_or(u64::MAX);

    for i in 0..l {
        for (m, act) in activations.iter_mut().enumerate().take(g) {
            let rng = crate::rng::capture(i as u32, m as u32)?;
            rng_states.push(rng);
            if i % k == 0 {
                let buf = crate::checkpoint::store_checkpoint(act, compress)?;
                checkpoints.push((i as u32, m as u32, buf));
            }
            let fwd = single_layer_forward(&model.cfg, &model.layers[i], act);
            retained_activations.push((i as u32, m as u32, fwd.clone()));
            *act = fwd.output.clone();
        }
    }

    Ok(FullForwardResult {
        checkpoints,
        rng_states,
        outputs: activations,
        weight_bytes_streamed,
        retained_activations,
    })
}

/// Recompute only the ops flagged in mask; reuse retained values for the rest.
///
/// Takes a retained `SingleLayerFwdOut` and re-runs only the stages where
/// `mask.should_recompute(op) == true`, using the retained values as inputs
/// to each stage. Stages not in the mask are kept as-is from retained.
pub fn selective_layer_forward(
    cfg: &BlockConfig,
    w: &BlockWeights,
    retained: &SingleLayerFwdOut,
    mask: &crate::plan::SelectiveRecomputeMask,
) -> SingleLayerFwdOut {
    use crate::plan::OpKind;

    let mut out = retained.clone();

    if mask.should_recompute(OpKind::Rms1) {
        let (h1, x_norm1, rms1) = rms_norm_fwd(&out.x_in, &w.rms1_w, cfg.bs(), cfg.d_model);
        out.rms1 = rms1;
        out.x_norm1 = x_norm1;
        out.h1 = h1;
    }

    if mask.should_recompute(OpKind::QkvProj) {
        let bs = cfg.bs();
        let d = cfg.d_model;
        let h = cfg.n_heads;
        let dh = cfg.d_head();
        let bh = cfg.batch * h;
        let s = cfg.seq_len;

        let q_flat = matmul_tb(&out.h1, &w.wq, bs, d, d);
        let k_flat = matmul_tb(&out.h1, &w.wk, bs, d, d);
        let v_flat = matmul_tb(&out.h1, &w.wv, bs, d, d);

        let reshape_to_heads = |flat: &[f32]| -> Vec<f32> {
            let mut heads = vec![0.0f32; bh * s * dh];
            for b in 0..cfg.batch {
                for hh in 0..h {
                    for ss in 0..s {
                        for t in 0..dh {
                            heads[((b * h + hh) * s + ss) * dh + t] =
                                flat[(b * s + ss) * d + hh * dh + t];
                        }
                    }
                }
            }
            heads
        };
        out.q_heads = reshape_to_heads(&q_flat);
        out.k_heads = reshape_to_heads(&k_flat);
        out.v_heads = reshape_to_heads(&v_flat);
    }

    if mask.should_recompute(OpKind::AttnSoftmax) {
        let bh = cfg.batch * cfg.n_heads;
        let s = cfg.seq_len;
        let dh = cfg.d_head();
        let d = cfg.d_model;
        let bs = cfg.bs();

        let scale = (dh as f32).sqrt();
        let mut attn_scores = vec![0.0f32; bh * s * s];
        for bh_i in 0..bh {
            for s1 in 0..s {
                for s2 in 0..s {
                    let mut dot = 0.0f32;
                    for t in 0..dh {
                        dot += out.q_heads[bh_i * s * dh + s1 * dh + t]
                            * out.k_heads[bh_i * s * dh + s2 * dh + t];
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
                        v += attn_weights[bh_i * s * s + s1 * s + s2]
                            * out.v_heads[bh_i * s * dh + s2 * dh + t];
                    }
                    attn_out_heads[bh_i * s * dh + s1 * dh + t] = v;
                }
            }
        }

        let mut attn_out = vec![0.0f32; bs * d];
        for b in 0..cfg.batch {
            for hh in 0..cfg.n_heads {
                for ss in 0..s {
                    for t in 0..dh {
                        attn_out[(b * s + ss) * d + hh * dh + t] =
                            attn_out_heads[((b * cfg.n_heads + hh) * s + ss) * dh + t];
                    }
                }
            }
        }

        out.attn_scores = attn_scores;
        out.attn_weights = attn_weights;
        out.attn_out = attn_out;
    }

    if mask.should_recompute(OpKind::OutProj) {
        let bs = cfg.bs();
        let d = cfg.d_model;
        let mut out_proj = matmul_tb(&out.attn_out, &w.wo, bs, d, d);
        crate::rng::apply_dropout(&mut out_proj, cfg.dropout_p);
        out.x2 = add_vecs(&out.x_in, &out_proj);
        out.out_proj = out_proj;
    }

    if mask.should_recompute(OpKind::Rms2) {
        let (h2, x_norm2, rms2) = rms_norm_fwd(&out.x2, &w.rms2_w, cfg.bs(), cfg.d_model);
        out.rms2 = rms2;
        out.x_norm2 = x_norm2;
        out.h2 = h2;
    }

    if mask.should_recompute(OpKind::MlpGateUp) {
        let bs = cfg.bs();
        let d = cfg.d_model;
        let ff = cfg.d_ff;
        let gate = matmul_tb(&out.h2, &w.wg, bs, d, ff);
        let up = matmul_tb(&out.h2, &w.wu, bs, d, ff);
        let silu_gate: Vec<f32> = gate.iter().map(|&g| silu_f(g)).collect();
        let hidden: Vec<f32> = silu_gate.iter().zip(&up).map(|(sg, u)| sg * u).collect();
        out.gate = gate;
        out.up = up;
        out.silu_gate = silu_gate;
        out.hidden = hidden;
    }

    if mask.should_recompute(OpKind::MlpDown) {
        let bs = cfg.bs();
        let d = cfg.d_model;
        let ff = cfg.d_ff;
        let mlp_out = matmul_tb(&out.hidden, &w.wd, bs, ff, d);
        out.output = add_vecs(&out.x2, &mlp_out);
        out.mlp_out = mlp_out;
    }

    out
}

/// Run forward (stub).
pub fn run_forward(
    _flowcast: &crate::FlowCast,
    _plan: &TrainingPlan,
    _batch: &Batch,
) -> Result<ForwardOutput> {
    unimplemented!("forward stub")
}
