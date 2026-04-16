// kernels/overflow_check.cu — FP16 overflow detection kernel (Sprint 2 Day 1)
//
// See overflow_check.cuh for the full design commentary.
//
// Build: nvcc -O3 -arch=sm_75 --std=c++17 -c overflow_check.cu -o overflow_check.o
//        (build.rs handles this; sm_75 = Turing minimum, Ampere/Ada are compatible)

#include "overflow_check.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// ===========================================================================
// Kernel
// ===========================================================================

// Each thread checks one FP16 element.
// Mask 0x7C00 selects exponent bits 14-10 (5 bits).
// If all five exponent bits are 1, the value is NaN or Inf — both are overflow.
//
// WHY __half_as_ushort and not *(uint16_t*)&grad[idx]:
//   The union/pointer-cast approach is type punning and is technically UB in
//   C++ (strict aliasing rules), even though every compiler emits correct code
//   for it in practice. __half_as_ushort is the CUDA-provided intrinsic that
//   reinterprets the bit pattern safely and without UB.
//
// WHY no atomics on overflow_flag:
//   The flag starts false (zeroed by host before launch).
//   No thread ever writes false during the kernel — only the host does that
//   before launch. All threads that find overflow write the same value (true).
//   Concurrent idempotent writes to the same address are well-defined in
//   CUDA's memory model — the value converges to true, which is correct.
//   Using an atomic would be correct too, but wastes L2 atomic bus bandwidth
//   when many threads find overflow simultaneously.

__global__ void fused_overflow_check(
    const __half* __restrict__ grad,
    int                        n,
    bool*                      overflow_flag
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Reinterpret FP16 bits as unsigned 16-bit integer (no UB).
    uint16_t bits = __half_as_ushort(grad[idx]);

    // If all 5 exponent bits are set, this is NaN or Inf.
    if ((bits & 0x7C00u) == 0x7C00u) {
        *overflow_flag = true;
    }
}

// ===========================================================================
// Host-callable C wrapper (used by Rust FFI)
// ===========================================================================

// Manages the full lifecycle:
//   1. Allocate a device bool.
//   2. Zero it (so the kernel starts from a clean false state).
//   3. Launch fused_overflow_check.
//   4. Synchronize the stream.
//   5. Copy result back to host.
//   6. Free the device bool.
//
// This keeps the Rust caller simple — it just passes a device pointer, count,
// and stream, and gets a bool back. No device memory management in Rust.
//
// Error handling: CUDA errors are ignored in Sprint 2 (the Rust layer catches
// stream errors via cudaGetLastError after sync). Sprint 4 adds proper
// propagation via cudaError_t return codes.

bool ramflow_check_overflow_fp16(
    const __half* grad_device,
    int           n,
    cudaStream_t  stream
) {
    // --- Step 1: Allocate one bool on the device ---
    // cudaMalloc on a single bool is fast — the CUDA allocator has a fast
    // path for small allocations that avoids the full heap search.
    bool* d_flag = nullptr;
    cudaMalloc(&d_flag, sizeof(bool));

    // --- Step 2: Zero the flag before the kernel reads/writes it ---
    // cudaMemsetAsync uses the same stream so the zero happens before the
    // kernel launch (streams are in-order for operations on the same stream).
    bool h_false = false;
    cudaMemcpyAsync(d_flag, &h_false, sizeof(bool), cudaMemcpyHostToDevice, stream);

    // --- Step 3: Launch the kernel ---
    // blockDim=256 is the standard choice: fills one SM warp pipeline,
    // and 256 threads per block × 65536 blocks = 16.7M elements/launch,
    // which covers a full 4096×4096 gradient tensor in one call.
    const int block_dim = 256;
    const int grid_dim  = (n + block_dim - 1) / block_dim;

    fused_overflow_check<<<grid_dim, block_dim, 0, stream>>>(grad_device, n, d_flag);

    // --- Step 4: Synchronize — wait for kernel + memcpy to complete ---
    // We must synchronize before reading h_result on the host.
    // cudaStreamSynchronize blocks the calling CPU thread until all
    // operations submitted to `stream` before this call have completed.
    cudaStreamSynchronize(stream);

    // --- Step 5: Copy result to host ---
    bool h_result = false;
    cudaMemcpy(&h_result, d_flag, sizeof(bool), cudaMemcpyDeviceToHost);

    // --- Step 6: Free device memory ---
    cudaFree(d_flag);

    return h_result;
}

// ---------------------------------------------------------------------------
// Sprint 0 stub symbol — keeps the archive non-empty if this TU is linked
// standalone during early build phases. Remove in Sprint 4 cleanup.
// ---------------------------------------------------------------------------
extern "C" void ramflow_overflow_check_stub(void) {}
