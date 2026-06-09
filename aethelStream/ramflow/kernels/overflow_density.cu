// kernels/overflow_density.cu — FP16 overflow element count kernel
//
// Sprint 4: replaces the Sprint 0 stub with the full two-stage reduction.
//
// Build: nvcc -O3 -arch=sm_75 --std=c++17
//              --relocatable-device-code=true
//              -Ikernels -c overflow_density.cu -o overflow_density.o
//        (build.rs handles this automatically)
//
// ─── WHY TWO-STAGE INSTEAD OF PER-THREAD ATOMICADD ───────────────────────────
//
// A naive kernel would call atomicAdd(overflow_count, 1) for each overflowed
// thread.  On a 4096×4096 gradient (16M elements) with, say, 3% overflow, that
// is ~480 000 global atomic operations.  The L2 atomic unit on sm_75 sustains
// ~200M atomicAdds/s — about 2.4 ms just in serialised atomic traffic.
//
// Two-stage shared-memory reduction:
//   Stage 1: each block accumulates its overflow count into sdata[blockDim.x]
//            via a binary tree reduction (__syncthreads() between each step).
//            Result: sdata[0] = block overflow count.
//   Stage 2: thread 0 calls atomicAdd(overflow_count, sdata[0]).
//            Only one global atomic per block (256 threads) → 65 536 atomics
//            instead of 480 000.  ~7× fewer global atomic operations.
//
// ─── __HALF_AS_USHORT VS UNION/POINTER CAST ──────────────────────────────────
//
// We use __half_as_ushort(grad[idx]) to reinterpret FP16 bits as uint16_t.
// The union/pointer cast (*(uint16_t*)&grad[idx]) is technically UB in C++
// under the strict-aliasing rule even though every compiler emits correct code.
// __half_as_ushort is the CUDA-provided intrinsic that avoids the UB.
//
// ─── OVERFLOW CONDITION ──────────────────────────────────────────────────────
//
// FP16 layout (IEEE 754):
//   Bit  15   : sign
//   Bits 14-10: exponent (5 bits, biased by 15)
//   Bits  9-0 : mantissa
//
// NaN  = exponent all-1 AND mantissa != 0  → (bits & 0x7C00u) == 0x7C00u
// Inf  = exponent all-1 AND mantissa == 0  → (bits & 0x7C00u) == 0x7C00u
// Both detected by: (bits & 0x7C00u) == 0x7C00u

#include "overflow_density.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// ===========================================================================
// Kernel — count_overflow_fp16
// ===========================================================================
//
// Launch configuration (set by the host wrapper):
//   blockDim.x = 256   (power of 2, required by the binary tree reduction)
//   gridDim.x  = ceil(n / 256)
//   shared mem = 256 * sizeof(unsigned int) = 1 024 bytes per block
//
// Threads with idx >= n write 0 to sdata[threadIdx.x] — they do not
// contribute to the count.  The guard ensures no out-of-bounds load.

__global__ void count_overflow_fp16(
    const __half* __restrict__ grad,
    int                        n,
    unsigned int*              overflow_count
) {
    // Dynamic shared memory: one unsigned int per thread in this block.
    extern __shared__ unsigned int sdata[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Stage 1a: each thread loads its overflow flag (0 or 1) into sdata.
    unsigned int local_flag = 0;
    if (idx < n) {
        // Reinterpret FP16 bits as uint16_t without UB — intrinsic, not cast.
        uint16_t bits = __half_as_ushort(grad[idx]);
        // All five exponent bits set → NaN or Inf → count it.
        if ((bits & 0x7C00u) == 0x7C00u) {
            local_flag = 1;
        }
    }
    sdata[threadIdx.x] = local_flag;
    __syncthreads();

    // Stage 1b: binary tree reduction within the block.
    // Each iteration halves the number of active threads.
    // blockDim.x must be a power of 2 (guaranteed: host always passes 256).
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Stage 2: one atomicAdd per block — ~256x fewer global atomic ops.
    if (threadIdx.x == 0) {
        atomicAdd(overflow_count, sdata[0]);
    }
}

// ===========================================================================
// Host-callable C wrapper
// ===========================================================================
//
// Full lifecycle:
//   1. Allocate one unsigned int on the device.
//   2. Zero it via cudaMemsetAsync (same stream → zeroed before kernel runs).
//   3. Launch count_overflow_fp16 with blockDim=256, sharedMem=1024 B.
//   4. Synchronise the stream.
//   5. Copy count to host.
//   6. Free device memory.
//
// The caller gets a clean unsigned int count without managing device memory.

unsigned int ramflow_count_overflow_fp16(
    const __half* grad_device,
    int           n,
    cudaStream_t  stream
) {
    unsigned int* d_count = nullptr;
    cudaMalloc(&d_count, sizeof(unsigned int));

    // Zero before launch so the kernel accumulates from a clean baseline.
    // cudaMemsetAsync is in-order on `stream`, so it completes before the
    // kernel launch on the same stream.
    cudaMemsetAsync(d_count, 0, sizeof(unsigned int), stream);

    const int block_dim   = 256;
    const int grid_dim    = (n + block_dim - 1) / block_dim;
    // 256 unsigned ints = 1 024 bytes of shared memory per block.
    const size_t shared_bytes = static_cast<size_t>(block_dim) * sizeof(unsigned int);

    count_overflow_fp16<<<grid_dim, block_dim, shared_bytes, stream>>>(
        grad_device, n, d_count
    );

    // Synchronise before reading back — cudaStreamSynchronize blocks the
    // calling CPU thread until all ops on `stream` have completed.
    cudaStreamSynchronize(stream);

    unsigned int h_result = 0;
    cudaMemcpy(&h_result, d_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_count);

    return h_result;
}
