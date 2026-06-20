// fused_ce.cu — Streaming Cut Cross-Entropy CUDA kernels (A8).
//
// Two-pass online-softmax.  Each extern "C" function launches a grid where one
// thread block owns one token (batch_seq position); threads within the block
// cooperate across the d_model dimension using shared-memory reductions.
//
// Compilation: nvcc -arch=sm_75 --std=c++17 -O3 -Ikernels -c kernels/fused_ce.cu
// Only built under --features cuda; mock-cuda CI uses src/loss.rs exclusively.
//
// Thread-block layout:
//   grid  = (bs,)        — one block per token
//   block = (BLOCK_D,)   — threads collaborate across d_model for dot products
//
// BLOCK_D = 256 with stride loops for large d_model (e.g. 4096, 8192).

#include "fused_ce.cuh"
#include <float.h>
#include <math.h>

#define BLOCK_D 256

// ---------------------------------------------------------------------------
// Device helpers — block-wide reductions
// ---------------------------------------------------------------------------

__device__ float block_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));

    __shared__ float sdata[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    if (lane == 0) sdata[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x + 31) / 32) ? sdata[lane] : -FLT_MAX;
    if (wid == 0) {
        for (int offset = 16; offset > 0; offset >>= 1)
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ float block_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);

    __shared__ float sdata[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    if (lane == 0) sdata[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x + 31) / 32) ? sdata[lane] : 0.0f;
    if (wid == 0) {
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ---------------------------------------------------------------------------
// Pass-1 kernel: online-softmax update for one vocab tile
// ---------------------------------------------------------------------------
//
// Block: one per token (bs_idx = blockIdx.x).
// For each k in [0, c): compute dot(h_{bs_idx}, W_chunk[k]), update running
// max and sumexp.  The dot product is computed in two sub-passes: first to find
// chunk_max, then to accumulate chunk_sum relative to chunk_max.
// label_logit[bs_idx] is set when the label's vocab index falls in this tile.
__global__ void ce_p1_kernel(
    const float*   hidden,
    const float*   w_chunk,
    float*         run_max,
    float*         run_sumexp,
    float*         label_logit,
    const int32_t* labels,
    int bs, int d, int v_start, int c
) {
    int bs_idx = blockIdx.x;
    if (bs_idx >= bs) return;

    const float* h_i  = hidden + (long long)bs_idx * d;
    int label_off      = (int)labels[bs_idx] - v_start;

    // Sub-pass A: find chunk_max and capture label logit.
    float chunk_max = -FLT_MAX;
    __shared__ float dot_shared;

    for (int k = 0; k < c; k++) {
        const float* w_k = w_chunk + (long long)k * d;
        float dot = 0.0f;
        for (int j = threadIdx.x; j < d; j += blockDim.x)
            dot += h_i[j] * w_k[j];
        dot = block_reduce_sum(dot);
        if (threadIdx.x == 0) {
            dot_shared  = dot;
            chunk_max   = fmaxf(chunk_max, dot);
        }
        __syncthreads();
        if (threadIdx.x == 0 && k == label_off)
            label_logit[bs_idx] = dot_shared;
    }

    // Sub-pass B: accumulate sumexp relative to new_max.
    if (threadIdx.x == 0) {
        float old_max    = run_max[bs_idx];
        float old_sumexp = run_sumexp[bs_idx];
        float new_max    = fmaxf(old_max, chunk_max);
        float chunk_sum  = 0.0f;

        for (int k = 0; k < c; k++) {
            const float* w_k = w_chunk + (long long)k * d;
            float dot2 = 0.0f;
            for (int j = 0; j < d; j++)
                dot2 += h_i[j] * w_k[j];
            chunk_sum += expf(dot2 - new_max);
        }

        run_max[bs_idx]    = new_max;
        run_sumexp[bs_idx] = old_sumexp * expf(old_max - new_max) + chunk_sum;
    }
}

// ---------------------------------------------------------------------------
// Pass-2 kernel: grad_hidden accumulation for one vocab tile
// ---------------------------------------------------------------------------
//
// Recomputes logits for each k in [0, c) (same 2× compute as the Rust path),
// computes factor = softmax_prob - one_hot, then accumulates into grad_h.
__global__ void ce_p2_kernel(
    const float*   hidden,
    const float*   w_chunk,
    const float*   run_max,
    const float*   run_sumexp,
    const int32_t* labels,
    float*         grad_h,
    int bs, int d, int v_start, int c
) {
    int bs_idx = blockIdx.x;
    if (bs_idx >= bs) return;

    const float* h_i  = hidden + (long long)bs_idx * d;
    float*       gh_i = grad_h + (long long)bs_idx * d;
    float max_i       = run_max[bs_idx];
    float se_i        = run_sumexp[bs_idx];
    int label_off     = (int)labels[bs_idx] - v_start;

    __shared__ float factor_sh;

    for (int k = 0; k < c; k++) {
        const float* w_k = w_chunk + (long long)k * d;

        // Recompute dot product for this (token, vocab) pair.
        float dot = 0.0f;
        for (int j = threadIdx.x; j < d; j += blockDim.x)
            dot += h_i[j] * w_k[j];
        dot = block_reduce_sum(dot);

        if (threadIdx.x == 0) {
            float factor = expf(dot - max_i) / se_i;
            if (k == label_off) factor -= 1.0f;
            factor_sh = factor;
        }
        __syncthreads();

        float factor = factor_sh;
        for (int j = threadIdx.x; j < d; j += blockDim.x)
            atomicAdd(&gh_i[j], factor * w_k[j]);

        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// extern "C" entry points
// ---------------------------------------------------------------------------

extern "C" void doublepass_ce_p1_chunk(
    const float*   hidden,
    const float*   w_chunk,
    float*         run_max,
    float*         run_sumexp,
    float*         label_logit,
    const int32_t* labels,
    int bs, int d, int v_start, int c
) {
    if (bs <= 0 || d <= 0 || c <= 0) return;
    ce_p1_kernel<<<bs, BLOCK_D>>>(
        hidden, w_chunk, run_max, run_sumexp, label_logit, labels,
        bs, d, v_start, c);
}

extern "C" void doublepass_ce_p2_chunk(
    const float*   hidden,
    const float*   w_chunk,
    const float*   run_max,
    const float*   run_sumexp,
    const int32_t* labels,
    float*         grad_h,
    int bs, int d, int v_start, int c
) {
    if (bs <= 0 || d <= 0 || c <= 0) return;
    ce_p2_kernel<<<bs, BLOCK_D>>>(
        hidden, w_chunk, run_max, run_sumexp, labels, grad_h,
        bs, d, v_start, c);
}
