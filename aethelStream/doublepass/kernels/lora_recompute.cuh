// lora_recompute.cuh — LoRA adapter recompute kernel interface
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Recompute hidden state with LoRA adapter applied.
///
/// @param x      [batch * seq, d_in] f32
/// @param a_mat  [d_in, rank]        f32 (LoRA A matrix)
/// @param h_out  [batch * seq, d_in] f32 (output: h + x @ A @ B)
void doublepass_lora_recompute_h(
    const float* x,
    const float* a_mat,
    float* h_out,
    int batch_seq,
    int d_in,
    int rank
);

#ifdef __cplusplus
}
#endif
