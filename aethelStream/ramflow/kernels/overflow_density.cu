// kernels/overflow_density.cu — per-layer overflow element count kernel
//
// Sprint 0 stub: declares the kernel signature only.  Real implementation
// goes in Sprint 4 Day 4.
//
// Algorithm: two-stage reduction (shared memory local → one global atomicAdd)
// to count FP16 overflow elements without saturating the L2 atomic bus.
//
// The kernel is launched alongside (or after) the boolean overflow_check
// kernel — they serve different consumers:
//   overflow_check.cu   → boolean flag: did ANY element overflow?
//   overflow_density.cu → count: HOW MANY elements overflowed?
//
// The fraction (count / n_total) feeds PerLayerScaleTable::update().

// ---------------------------------------------------------------------------
// Forward declaration (implementation in Sprint 4)
// ---------------------------------------------------------------------------

// __global__ void count_overflow_fp16(
//     const __half* grad,       // FP16 gradient tensor (device pointer)
//     int n,                    // total element count
//     unsigned int* overflow_count  // output: number of NaN/Inf elements
// );
//
// Launch configuration (Sprint 4):
//   blockDim = 256
//   gridDim  = ceil(n / 256)
//   sharedMem = blockDim * sizeof(int)   // for local block reduction
//
// Two-stage reduction:
//   Stage 1: each thread checks one element, writes 0 or 1 to sdata[tid].
//            __syncthreads(); parallel reduce sdata to sdata[0].
//   Stage 2: one thread per block: atomicAdd(overflow_count, sdata[0]).
//            This cuts global atomic traffic by blockDim (256x fewer atomics).
//
// Note: Use __half_as_ushort(grad[idx]) (not a union cast) to read bits.
//       The union cast is UB in C++ even though it works in practice.

extern "C" {
    // Placeholder symbol so build.rs can compile this file into a linkable
    // object without producing an empty translation unit warning.
    void ramflow_overflow_density_stub(void);
}

void ramflow_overflow_density_stub(void) {}
