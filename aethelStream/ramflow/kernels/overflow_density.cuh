#pragma once
// kernels/overflow_density.cuh — FP16 overflow element count kernel declarations
//
// Sprint 4: two-stage shared-memory reduction to count NaN/Inf elements.
// See overflow_density.cu for the full design commentary.
//
// Contrast with overflow_check.cuh (boolean flag only):
//   overflow_check.cu   -> bool: did ANY element overflow?
//   overflow_density.cu -> unsigned int: HOW MANY elements overflowed?
//
// The count feeds PerLayerScaleTable::update(layer_idx, n_total, n_overflow).

#include <cuda_fp16.h>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// ramflow_count_overflow_fp16
// ---------------------------------------------------------------------------
// Host-callable C wrapper used by Rust FFI.
// Allocates a device unsigned int, zeros it, launches count_overflow_fp16,
// synchronizes, copies count back to host, frees device memory.
//
// Parameters:
//   grad_device : device pointer to FP16 gradient tensor
//   n           : number of FP16 elements
//   stream      : CUDA stream (may be 0 for default stream)
//
// Returns:
//   number of NaN or Inf elements in [grad_device, grad_device + n)
unsigned int ramflow_count_overflow_fp16(
    const __half* grad_device,
    int           n,
    cudaStream_t  stream
);

#ifdef __cplusplus
} // extern "C"
#endif
