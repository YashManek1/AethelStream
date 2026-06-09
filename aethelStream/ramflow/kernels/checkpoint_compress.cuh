// kernels/checkpoint_compress.cuh — AethelStream Sprint 5
//
// Host-side declarations for the INT8 activation-checkpoint compression kernels.
//
// These kernels compress FP16 activation checkpoints to INT8 with per-channel
// scale factors, achieving ~2× RAM reduction for moderate (<0.1% gradient
// deviation) quality loss.
//
// ─── Algorithm ────────────────────────────────────────────────────────────────
//
//   Compress (FP16 → INT8):
//     Phase 1 — find_channel_scales: one block per channel, shared-memory
//       reduction over elems_per_channel elements → scale[c] = max_abs / 127.0
//     Phase 2 — compress_fp16_to_int8: quantize:
//       dst[c][i] = clamp(round(src[c][i] / scale[c]), -128, 127)
//
//   Decompress (INT8 → FP16):
//     decompress_int8_to_fp16: dst[c][i] = (float)src[c][i] * scale[c]
//
// ─── Buffer layout ────────────────────────────────────────────────────────────
//
//   src / dst: contiguous row-major, channels × elems_per_channel elements.
//   scales: n_channels float32 values (written by compress, read by decompress).
//
// ─── Grid dimensions ──────────────────────────────────────────────────────────
//
//   gridDim.x  = n_channels
//   blockDim.x = min(elems_per_channel, 256)
//   Each thread strides over elems_per_channel when blockDim.x < elems_per_channel.
//
// ─── Target ───────────────────────────────────────────────────────────────────
//
//   Minimum sm_75 (Turing).  Ampere/Ada inherit.
//   Requires: cuda_fp16.h, stdint.h (both from the CUDA Toolkit).

#pragma once

#include <cuda_fp16.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Compress FP16 activation checkpoints to INT8 with per-channel scales.
///
/// Launches two sequential kernels on `stream`:
///   1. find_channel_scales  — computes per-channel scale factors.
///   2. compress_fp16_to_int8 — quantises src to dst using those scales.
///
/// @param src_device       Device pointer to FP16 input (n_channels × elems_per_channel).
/// @param dst_device       Device pointer to INT8 output (same total element count).
/// @param scales_device    Device pointer to n_channels float32 scales (OUTPUT).
/// @param n_channels       Number of channels (gridDim.x for both kernels).
/// @param elems_per_channel Elements per channel.
/// @param stream           CUDA stream to submit kernels on.
/// @return 0 on success.  Inspect cudaGetLastError() after the call for asynchronous errors.
unsigned int ramflow_compress_checkpoint_fp16_to_int8(
    const __half* src_device,
    int8_t*       dst_device,
    float*        scales_device,
    int           n_channels,
    int           elems_per_channel,
    cudaStream_t  stream
);

/// Decompress INT8 activation checkpoints back to FP16 using stored scales.
///
/// Inverse of ramflow_compress_checkpoint_fp16_to_int8.
///
/// @param src_device       Device pointer to INT8 input.
/// @param dst_device       Device pointer to FP16 output.
/// @param scales_device    Device pointer to n_channels float32 scales (INPUT — from compress).
/// @param n_channels       Number of channels.
/// @param elems_per_channel Elements per channel.
/// @param stream           CUDA stream.
/// @return 0 on success.
unsigned int ramflow_decompress_checkpoint_int8_to_fp16(
    const int8_t* src_device,
    __half*       dst_device,
    const float*  scales_device,
    int           n_channels,
    int           elems_per_channel,
    cudaStream_t  stream
);

#ifdef __cplusplus
}
#endif
