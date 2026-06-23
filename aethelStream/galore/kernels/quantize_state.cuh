// quantize_state.cuh — absmax INT8 quantisation for optimizer states
#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Compute absmax scale and quantise FP32 tensor to INT8.
/// scale = max(abs(tensor)) / 127.0;  q = clamp(round(v/scale), -127, 127)
int galore_quantize_absmax_f32_to_i8(
    const float* src,
    int8_t*      dst,
    float*       scale_out,
    int          n_elements,
    cudaStream_t stream
);

/// Dequantise INT8 tensor to FP32: dst = q * scale
int galore_dequantize_i8_to_f32(
    const int8_t* src,
    float*        dst,
    float         scale,
    int           n_elements,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif
