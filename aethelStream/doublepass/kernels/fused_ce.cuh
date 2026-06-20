// fused_ce.cuh — Streaming Cut Cross-Entropy CUDA kernel declarations (A8).
//
// Two-pass online-softmax for O(chunk) peak logit memory:
//   Pass 1 (ce_p1): accumulate per-token running max + sumexp from one vocab tile.
//   Pass 2 (ce_p2): accumulate grad_hidden from one vocab tile given final softmax state.
//
// The Rust host code (loss.rs) calls these once per vocab chunk, maintaining the
// running state arrays (run_max, run_sumexp) on the host between calls.
//
// Compiled only under --features cuda.  The mock-cuda path uses the pure-Rust
// implementation in src/loss.rs exclusively.
//
// Logit recomputation: Pass 2 recomputes the tile's logits rather than storing them,
// achieving O(chunk) peak memory at the cost of 2× GEMM compute per chunk.
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Pass 1: update online-softmax running state from one vocab tile.
///
/// After all tiles complete, run_max[i] and run_sumexp[i] hold the global max and
/// sumexp (normalised to exp(z - run_max)) for token i.
///
/// @param hidden       [bs, d]  f32  final hidden states (device ptr)
/// @param w_chunk      [c, d]   f32  LM-head weight tile, row-major (device ptr)
/// @param run_max      [bs]     f32  running max per token  — in/out, init to -FLT_MAX
/// @param run_sumexp   [bs]     f32  running sumexp per token — in/out, init to 0
/// @param label_logit  [bs]     f32  logit at label position — set when label ∈ tile
/// @param labels       [bs]     i32  target token ids
/// @param bs           number of tokens (batch × seq)
/// @param d            d_model
/// @param v_start      first vocabulary index in this tile
/// @param c            tile width (vocab positions in this chunk; last tile may be < chunk_size)
void doublepass_ce_p1_chunk(
    const float*   hidden,
    const float*   w_chunk,
    float*         run_max,
    float*         run_sumexp,
    float*         label_logit,
    const int32_t* labels,
    int            bs,
    int            d,
    int            v_start,
    int            c
);

/// Pass 2: accumulate gradient wrt hidden from one vocab tile.
///
/// Must be called after ALL Pass 1 calls are complete so that run_max / run_sumexp
/// are final.  grad_h is accumulated (not zeroed) — caller must initialise to zero.
///
/// @param hidden       [bs, d]  f32  final hidden states
/// @param w_chunk      [c, d]   f32  LM-head weight tile (same tile as Pass 1 call)
/// @param run_max      [bs]     f32  final running max from Pass 1 (read-only)
/// @param run_sumexp   [bs]     f32  final running sumexp from Pass 1 (read-only)
/// @param labels       [bs]     i32  target token ids
/// @param grad_h       [bs, d]  f32  gradient wrt hidden, accumulated in-place
/// @param bs, d, v_start, c     same semantics as Pass 1
void doublepass_ce_p2_chunk(
    const float*   hidden,
    const float*   w_chunk,
    const float*   run_max,
    const float*   run_sumexp,
    const int32_t* labels,
    float*         grad_h,
    int            bs,
    int            d,
    int            v_start,
    int            c
);

#ifdef __cplusplus
}
#endif
