//! A8 — Streaming Cut Cross-Entropy (Cut-CE).
//!
//! Computes the language-model cross-entropy loss and the gradient wrt the final
//! hidden state `h` without ever materialising the full `[batch_seq, vocab_size]`
//! logit tensor.  Peak logit allocation is `O(batch_seq × chunk_size)`.
//!
//! # Algorithm (two-pass online softmax)
//!
//! For each tile of `chunk_size` vocabulary positions the LM-head weight matrix
//! `W ∈ ℝ^{V×d}` is streamed once per pass (tile-wise, via FlowCast in production).
//!
//! **Pass 1 — max + sumexp accumulation**
//! ```text
//! for each chunk c in 0..ceil(V/chunk):
//!     logit_chunk[i,k] = dot(h_i, W[v_start+k])
//!     m_c[i] = max_k logit_chunk[i,k]
//!     new_max[i] = max(run_max[i], m_c[i])
//!     run_sumexp[i] = run_sumexp[i] * exp(run_max[i] - new_max[i])
//!                   + sum_k exp(logit_chunk[i,k] - new_max[i])
//!     run_max[i] = new_max[i]
//!     if label[i] in [v_start, v_start+c): capture label_logit[i]
//! ```
//!
//! **Loss (mean over batch_seq)**
//! ```text
//! L = mean_i( -label_logit[i] + log(run_sumexp[i]) + run_max[i] )
//! ```
//!
//! **Pass 2 — gradient wrt h**
//! ```text
//! for each chunk c:
//!     logit_chunk[i,k] = dot(h_i, W[v_start+k])     ← recomputed; no storage needed
//!     factor[i,k] = exp(logit_chunk[i,k] - run_max[i]) / run_sumexp[i]
//!     if v_start+k == label[i]: factor[i,k] -= 1
//!     grad_h[i] += factor[i,:] @ W_chunk             ← [1,c] × [c,d] = [1,d]
//! grad_h /= batch_seq
//! ```
//!
//! # Memory guarantee
//! The only logit buffer — `logit_chunk` of size `batch_seq × min(chunk_size, V)` — is
//! allocated **once** and reused across every tile in both passes.
//! `LossOutput::peak_logit_bytes` equals this allocation's byte size exactly.
//!
//! # Optional LM-head gradient projection (M4 hook)
//! If `lm_head_grad_hook` is `Some`, the function calls
//! `project_and_accumulate(&grad_w_chunk_flat, u32::MAX, "lm_head")` for each chunk,
//! letting M4 absorb the LM-head gradient tile without materialising `[V, d]`.

use crate::{DoublePassError, OptimizerBackend, Result};

/// Output of [`streaming_cut_ce`].
#[derive(Debug, Clone)]
pub struct LossOutput {
    /// Scalar mean cross-entropy loss (natural log, mean over `batch_seq`).
    pub loss: f32,
    /// Gradient wrt the final hidden state, shape `[batch_seq, d_model]` (f32).
    pub grad_hidden: Vec<f32>,
    /// **Peak** logit bytes allocated during this call.
    ///
    /// Always equals `batch_seq × min(chunk_size, vocab_size) × 4`.
    /// Assert this is `O(chunk_size)`, not `O(vocab_size)`, in tests.
    pub peak_logit_bytes: usize,
}

/// Compute streaming cut-cross-entropy loss and `∂L/∂h` without materialising
/// the full `[batch_seq, vocab_size]` logit tensor.
///
/// # Parameters
/// - `hidden`     — `[batch_seq × d_model]` f32, row-major.
/// - `lm_head`    — `[vocab_size × d_model]` f32, row-major.
///   In production this is streamed tile-wise via FlowCast; here the
///   caller passes the full matrix and this function slices it chunk-by-chunk.
/// - `d_model`    — hidden dimension (must match `hidden` and `lm_head` layouts).
/// - `vocab_size` — number of output classes.
/// - `chunk_size` — vocabulary tile width; only `batch_seq × chunk_size × 4` bytes
///   of logit memory are ever live simultaneously.
/// - `labels`     — `[batch_seq]` token ids in `[0, vocab_size)`.
/// - `lm_head_grad_hook` — optional M4 optimizer backend to receive per-chunk LM-head
///   gradient tiles via `project_and_accumulate`.  Pass `None` to skip.
///
/// # Errors
/// Returns [`DoublePassError::Config`] for invalid shapes or out-of-range labels.
pub fn streaming_cut_ce(
    hidden: &[f32],
    lm_head: &[f32],
    d_model: usize,
    vocab_size: usize,
    chunk_size: usize,
    labels: &[u32],
    lm_head_grad_hook: Option<&dyn OptimizerBackend>,
) -> Result<LossOutput> {
    // --- Validation ---------------------------------------------------------
    let batch_seq = labels.len();

    if batch_seq == 0 {
        return Err(DoublePassError::Config(
            "streaming_cut_ce: labels must not be empty".into(),
        ));
    }
    if d_model == 0 {
        return Err(DoublePassError::Config(
            "streaming_cut_ce: d_model must be > 0".into(),
        ));
    }
    if vocab_size == 0 {
        return Err(DoublePassError::Config(
            "streaming_cut_ce: vocab_size must be > 0".into(),
        ));
    }
    if chunk_size == 0 {
        return Err(DoublePassError::Config(
            "streaming_cut_ce: chunk_size must be > 0".into(),
        ));
    }
    if hidden.len() != batch_seq * d_model {
        return Err(DoublePassError::Config(format!(
            "streaming_cut_ce: hidden.len()={} != batch_seq*d_model={}*{}={}",
            hidden.len(),
            batch_seq,
            d_model,
            batch_seq * d_model,
        )));
    }
    if lm_head.len() != vocab_size * d_model {
        return Err(DoublePassError::Config(format!(
            "streaming_cut_ce: lm_head.len()={} != vocab_size*d_model={}*{}={}",
            lm_head.len(),
            vocab_size,
            d_model,
            vocab_size * d_model,
        )));
    }
    for (idx, &label) in labels.iter().enumerate() {
        if label as usize >= vocab_size {
            return Err(DoublePassError::Config(format!(
                "streaming_cut_ce: labels[{idx}]={label} >= vocab_size={vocab_size}"
            )));
        }
    }

    // --- Single logit buffer ------------------------------------------------
    //
    // This is the ONLY allocation proportional to vocabulary size.
    // It is overwritten in-place for every tile in both passes.
    // Peak logit bytes = batch_seq × actual_chunk × sizeof(f32).
    let actual_chunk = chunk_size.min(vocab_size);
    let mut logit_chunk = vec![0.0_f32; batch_seq * actual_chunk];

    // --- Pass 1: online-softmax + label-logit capture -----------------------
    let mut run_max = vec![f32::NEG_INFINITY; batch_seq];
    let mut run_sumexp = vec![0.0_f32; batch_seq];
    let mut label_logit = vec![0.0_f32; batch_seq];

    iter_chunks(vocab_size, chunk_size, |v_start, c| {
        let w_chunk = &lm_head[v_start * d_model..(v_start + c) * d_model];
        compute_chunk_logits(
            hidden,
            w_chunk,
            &mut logit_chunk[..batch_seq * c],
            batch_seq,
            d_model,
            c,
        );
        online_softmax_update(
            &mut run_max,
            &mut run_sumexp,
            &logit_chunk[..batch_seq * c],
            batch_seq,
            c,
        );
        capture_label_logits(
            &mut label_logit,
            &logit_chunk[..batch_seq * c],
            labels,
            v_start,
            batch_seq,
            c,
        );
    });

    // --- Scalar loss --------------------------------------------------------
    let mut loss = 0.0_f32;
    for i in 0..batch_seq {
        loss += -label_logit[i] + run_sumexp[i].ln() + run_max[i];
    }
    loss /= batch_seq as f32;

    // --- Pass 2: gradient wrt h (and optional LM-head grad hook) ------------
    let mut grad_hidden = vec![0.0_f32; batch_seq * d_model];

    iter_chunks(vocab_size, chunk_size, |v_start, c| {
        let w_chunk = &lm_head[v_start * d_model..(v_start + c) * d_model];
        // Recompute this tile's logits — avoids storing O(V) logits between passes.
        compute_chunk_logits(
            hidden,
            w_chunk,
            &mut logit_chunk[..batch_seq * c],
            batch_seq,
            d_model,
            c,
        );
        accumulate_grad_hidden(
            &mut grad_hidden,
            &logit_chunk[..batch_seq * c],
            w_chunk,
            &run_max,
            &run_sumexp,
            labels,
            v_start,
            batch_seq,
            d_model,
            c,
        );

        // Optional M4 hook: project per-chunk LM-head gradient tile through M4
        // without materialising the full [V, d] gradient matrix.
        if let Some(opt) = lm_head_grad_hook {
            let grad_w = compute_lm_head_grad_chunk(
                hidden,
                &logit_chunk[..batch_seq * c],
                &run_max,
                &run_sumexp,
                labels,
                v_start,
                batch_seq,
                d_model,
                c,
            );
            // u32::MAX is the sentinel layer index for the LM head (not a transformer
            // block); M4 routes it via param_name = "lm_head".
            opt.project_and_accumulate(&grad_w, u32::MAX, "lm_head");
        }
    });

    // Scale gradient by 1/batch_seq to match mean loss.
    let inv_bs = 1.0_f32 / batch_seq as f32;
    for g in grad_hidden.iter_mut() {
        *g *= inv_bs;
    }

    let peak_logit_bytes = batch_seq * actual_chunk * std::mem::size_of::<f32>();
    Ok(LossOutput {
        loss,
        grad_hidden,
        peak_logit_bytes,
    })
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Yield `(v_start, c)` for each vocab chunk where `c = min(chunk_size, V - v_start)`.
#[inline]
fn iter_chunks(vocab_size: usize, chunk_size: usize, mut f: impl FnMut(usize, usize)) {
    let mut v_start = 0;
    while v_start < vocab_size {
        let c = chunk_size.min(vocab_size - v_start);
        f(v_start, c);
        v_start += c;
    }
}

/// `logit_buf[i*c + k] = dot(hidden[i*d .. (i+1)*d], w_chunk[k*d .. (k+1)*d])`.
///
/// `w_chunk` is `[c, d]` row-major; `logit_buf` is `[bs, c]` row-major.
fn compute_chunk_logits(
    hidden: &[f32],
    w_chunk: &[f32],
    logit_buf: &mut [f32],
    batch_seq: usize,
    d_model: usize,
    chunk: usize,
) {
    for i in 0..batch_seq {
        let h_i = &hidden[i * d_model..(i + 1) * d_model];
        for k in 0..chunk {
            let w_k = &w_chunk[k * d_model..(k + 1) * d_model];
            let dot: f32 = h_i.iter().zip(w_k).map(|(a, b)| a * b).sum();
            logit_buf[i * chunk + k] = dot;
        }
    }
}

/// Numerically-stable online-softmax state update for one vocab chunk.
///
/// For token `i`, given chunk logits `row = logit_buf[i*c .. (i+1)*c]`:
/// ```text
/// new_max = max(run_max[i], max_k row[k])
/// run_sumexp[i] = run_sumexp[i] * exp(run_max[i] - new_max)
///              + sum_k exp(row[k] - new_max)
/// run_max[i] = new_max
/// ```
fn online_softmax_update(
    run_max: &mut [f32],
    run_sumexp: &mut [f32],
    logit_buf: &[f32],
    batch_seq: usize,
    chunk: usize,
) {
    for i in 0..batch_seq {
        let row = &logit_buf[i * chunk..(i + 1) * chunk];
        let chunk_max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let new_max = run_max[i].max(chunk_max);
        // Rescale existing sum to the new reference point.
        run_sumexp[i] *= (run_max[i] - new_max).exp();
        for &z in row {
            run_sumexp[i] += (z - new_max).exp();
        }
        run_max[i] = new_max;
    }
}

/// Set `label_logit[i]` when `label[i]` falls in `[v_start, v_start + chunk)`.
fn capture_label_logits(
    label_logit: &mut [f32],
    logit_buf: &[f32],
    labels: &[u32],
    v_start: usize,
    batch_seq: usize,
    chunk: usize,
) {
    for i in 0..batch_seq {
        let label = labels[i] as usize;
        if label >= v_start && label < v_start + chunk {
            label_logit[i] = logit_buf[i * chunk + (label - v_start)];
        }
    }
}

#[allow(clippy::too_many_arguments)]
/// Accumulate `grad_hidden` contribution from one vocab chunk.
///
/// ```text
/// factor[i,k] = exp(z[i,k] - run_max[i]) / run_sumexp[i]
/// factor[i, label[i]-v_start] -= 1.0          (subtract one-hot)
/// grad_hidden[i] += sum_k factor[i,k] * w_chunk[k]
/// ```
/// Caller must divide `grad_hidden` by `batch_seq` after all chunks.
fn accumulate_grad_hidden(
    grad_hidden: &mut [f32],
    logit_buf: &[f32],
    w_chunk: &[f32],
    run_max: &[f32],
    run_sumexp: &[f32],
    labels: &[u32],
    v_start: usize,
    batch_seq: usize,
    d_model: usize,
    chunk: usize,
) {
    for i in 0..batch_seq {
        let label = labels[i] as usize;
        let max_i = run_max[i];
        let se_i = run_sumexp[i];
        let gh_i = &mut grad_hidden[i * d_model..(i + 1) * d_model];

        for k in 0..chunk {
            let z = logit_buf[i * chunk + k];
            let mut factor = (z - max_i).exp() / se_i;
            if v_start + k == label {
                factor -= 1.0_f32;
            }
            let w_k = &w_chunk[k * d_model..(k + 1) * d_model];
            for j in 0..d_model {
                gh_i[j] += factor * w_k[j];
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
/// Compute per-chunk LM-head gradient tile `∂L/∂W_chunk` for the M4 hook.
///
/// `∂L/∂W[v_start+k, j] = (1/bs) × sum_i factor[i,k] × h_i[j]`.
/// Returns a flat `[chunk × d_model]` buffer.
fn compute_lm_head_grad_chunk(
    hidden: &[f32],
    logit_buf: &[f32],
    run_max: &[f32],
    run_sumexp: &[f32],
    labels: &[u32],
    v_start: usize,
    batch_seq: usize,
    d_model: usize,
    chunk: usize,
) -> Vec<f32> {
    let mut grad_w = vec![0.0_f32; chunk * d_model];
    let inv_bs = 1.0_f32 / batch_seq as f32;

    for i in 0..batch_seq {
        let label = labels[i] as usize;
        let max_i = run_max[i];
        let se_i = run_sumexp[i];
        let h_i = &hidden[i * d_model..(i + 1) * d_model];

        for k in 0..chunk {
            let z = logit_buf[i * chunk + k];
            let mut factor = (z - max_i).exp() / se_i;
            if v_start + k == label {
                factor -= 1.0_f32;
            }
            factor *= inv_bs;
            let gw_k = &mut grad_w[k * d_model..(k + 1) * d_model];
            for j in 0..d_model {
                gw_k[j] += factor * h_i[j];
            }
        }
    }
    grad_w
}
