// galore_apply.cuh — GaLore optimizer apply kernel interface
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Apply one Adam step with GaLore low-rank gradient.
///
/// @param param       [n_elements] f32 (weights to update)
/// @param grad_lr     [n_elements] f32 (low-rank gradient)
/// @param m           [n_elements] f32 (first moment)
/// @param v           [n_elements] f32 (second moment)
/// @param clip_scale  global clipping scale factor
/// @param beta1       Adam beta1 (momentum decay)
/// @param beta2       Adam beta2 (second moment decay)
/// @param eps         Adam epsilon
/// @param lr          learning rate
/// @param n_elements  number of parameters
/// @param step        current optimizer step (for bias correction)
void doublepass_galore_apply(
    float* param,
    const float* grad_lr,
    float* m,
    float* v,
    float clip_scale,
    float beta1,
    float beta2,
    float eps,
    float lr,
    int n_elements,
    int step
);

#ifdef __cplusplus
}
#endif
