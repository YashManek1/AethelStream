// galore_project.cuh — GaLore gradient projection via cuBLAS HGEMM
#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Forward projection: R = P^T @ G @ Q  (all FP16, row-major storage).
///
/// @param G      [m×n] gradient matrix on device
/// @param P      [m×r] left projection on device
/// @param Q      [n×r] right projection on device
/// @param R      [r×r] output compact gradient on device
/// @param temp   [r×n] scratch buffer on device (caller-allocated)
/// @param m,n,r  matrix dimensions
/// @param stream CUDA stream for async execution
int galore_project_forward(
    const void* G,
    const void* P,
    const void* Q,
    void*       R,
    void*       temp,
    int         m,
    int         n,
    int         r,
    cudaStream_t stream
);

/// Backward projection: G_tilde = P @ N @ Q^T  (all FP16, row-major storage).
///
/// @param N       [r×r] normalized update on device
/// @param P       [m×r] left projection on device
/// @param Q       [n×r] right projection on device
/// @param G_tilde [m×n] full-size update on device
/// @param temp    [m×r] scratch buffer on device (caller-allocated)
/// @param m,n,r   matrix dimensions
/// @param stream  CUDA stream
int galore_project_backward(
    const void* N,
    const void* P,
    const void* Q,
    void*       G_tilde,
    void*       temp,
    int         m,
    int         n,
    int         r,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif
