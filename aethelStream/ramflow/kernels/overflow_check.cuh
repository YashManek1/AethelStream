#pragma once
// kernels/overflow_check.cuh — FP16 overflow detection kernel declarations
//
// Sprint 2 Day 1.
//
// IEEE 754 FP16 layout (16 bits total):
//   Bit  15   : sign
//   Bits 14-10: exponent (5 bits, biased by 15)
//   Bits  9-0 : mantissa (10 bits)
//
// Overflow condition (NaN or Inf):
//   All 5 exponent bits are 1 (0x7C00 mask).
//   NaN  = exponent all-1 AND mantissa != 0
//   Inf  = exponent all-1 AND mantissa == 0
//   Both are detected by: (bits & 0x7C00u) == 0x7C00u
//
// The single-pass boolean kernel (overflow_check) is used by the Rust wrapper
// in src/cuda_bridge/stream.rs to decide whether to call PerLayerScaleTable::update.
// The density-count kernel (overflow_density.cu) is a separate Sprint 4 concern.

#include <cuda_fp16.h>
#include <cstdint>
#include <cstdbool>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// fused_overflow_check
// ---------------------------------------------------------------------------
// Scans every FP16 element in [grad, grad+n) for NaN or Inf.
// Sets *overflow_flag = true if ANY element is NaN or Inf.
// Multiple threads writing true is safe — idempotent write, never writes false.
//
// Launch config:
//   blockDim.x = 256
//   gridDim.x  = ceil(n / 256)
//
// Caller is responsible for:
//   1. Zeroing *overflow_flag (device memory) before launch.
//   2. Calling cudaStreamSynchronize before reading *overflow_flag back.
__global__ void fused_overflow_check(
    const __half* __restrict__ grad,
    int                        n,
    bool*                      overflow_flag
);

// ---------------------------------------------------------------------------
// ramflow_check_overflow_fp16
// ---------------------------------------------------------------------------
// Host-callable C wrapper used by Rust FFI (src/cuda_bridge/stream.rs).
// Allocates a device bool, launches the kernel, synchronizes, copies result back.
//
// Parameters:
//   grad_device : device pointer to FP16 gradient tensor
//   n           : number of elements
//   stream      : CUDA stream to launch on (may be 0 for default stream)
//
// Returns:
//   true  if any element is NaN or Inf
//   false if all elements are finite
//
// This is NOT exposed as __global__ — it is a plain C function that manages
// the device allocation, launch, and synchronization so the Rust caller
// doesn't need to manage device memory directly.
bool ramflow_check_overflow_fp16(
    const __half* grad_device,
    int           n,
    cudaStream_t  stream
);

#ifdef __cplusplus
} // extern "C"
#endif
