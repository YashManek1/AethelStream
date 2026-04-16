// src/cuda_bridge/stream.rs — CUDA stream RAII wrapper + overflow check (Sprint 2)
//
// Sprint 2 adds:
//   - check_overflow_fp16(): Rust wrapper around the ramflow_check_overflow_fp16
//     C function compiled from kernels/overflow_check.cu.
//   - CudaStream now has a real (opaque) handle that gets passed to the kernel.
//
// In mock-cuda mode, check_overflow_fp16 is a pure-Rust stub that scans the
// host-side slice directly. This lets all Sprint 2 tests run on CI without a GPU.

use crate::{RamFlowError, Result};

// ===========================================================================
// CudaStream
// ===========================================================================

/// Owned handle to a `cudaStream_t`.
///
/// Automatically calls `cudaStreamDestroy` on drop (real CUDA path) or is a
/// no-op (mock-cuda path).
///
/// # Sprint 2 note
/// The inner `ptr` is a raw *mut c_void that aliases a `cudaStream_t`.
/// We store it as `*mut c_void` rather than importing the CUDA type so that
/// the mock-cuda path compiles without any CUDA headers.
#[derive(Debug)]
pub struct CudaStream {
    /// Opaque pointer to the underlying cudaStream_t.
    /// null in mock-cuda mode (the null stream / stream 0).
    ptr: *mut std::os::raw::c_void,
}

// Safety: CudaStream owns a stream handle that is bound to a device.
// Sending it across threads is safe because the CUDA runtime is thread-safe
// with respect to stream handles — the handle is just an integer ID.
unsafe impl Send for CudaStream {}

impl CudaStream {
    /// Create a new non-blocking CUDA stream.
    pub fn new() -> Result<Self> {
        #[cfg(not(feature = "mock-cuda"))]
        {
            let mut stream: *mut std::os::raw::c_void = std::ptr::null_mut();
            // cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)
            // cudaStreamNonBlocking = 1: the stream does not implicitly
            // synchronize with stream 0, avoiding hidden sync points.
            let rc = unsafe { cuda_stream_create_non_blocking(&mut stream) };
            if rc != 0 {
                return Err(RamFlowError::CudaError(rc));
            }
            Ok(CudaStream { ptr: stream })
        }
        #[cfg(feature = "mock-cuda")]
        {
            // null pointer represents the default stream in mock mode.
            Ok(CudaStream {
                ptr: std::ptr::null_mut(),
            })
        }
    }

    /// Synchronise: block the calling CPU thread until all GPU work on this
    /// stream completes.
    pub fn synchronize(&self) -> Result<()> {
        #[cfg(not(feature = "mock-cuda"))]
        {
            let rc = unsafe { cuda_stream_synchronize(self.ptr) };
            if rc != 0 {
                return Err(RamFlowError::CudaError(rc));
            }
        }
        // mock-cuda: stream operations are synchronous by definition (no GPU),
        // so synchronize is always a no-op.
        Ok(())
    }

    /// Raw stream handle for passing to CUDA API calls.
    ///
    /// # Safety
    /// The returned pointer is only valid while `self` is alive.
    pub(crate) fn as_raw(&self) -> *mut std::os::raw::c_void {
        self.ptr
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        #[cfg(not(feature = "mock-cuda"))]
        if !self.ptr.is_null() {
            unsafe {
                // Ignore the return code in Drop — we cannot propagate errors.
                // cudaStreamDestroy returns cudaErrorInvalidResourceHandle if
                // the stream was already destroyed, which should never happen
                // here because CudaStream enforces single ownership.
                let _ = cuda_stream_destroy(self.ptr);
            }
        }
    }
}

// ===========================================================================
// Overflow check — the Sprint 2 core deliverable
// ===========================================================================

/// Scan a device-side FP16 tensor for NaN or Inf elements.
///
/// # Real CUDA path
/// Calls `ramflow_check_overflow_fp16` from `kernels/overflow_check.cu`:
///   1. Allocates a device bool initialized to false.
///   2. Launches `fused_overflow_check` (blockDim=256, gridDim=ceil(n/256)).
///   3. Synchronizes the stream.
///   4. Returns the device bool value.
///
/// # mock-cuda path
/// Iterates the elements as a host slice using the same bitmask logic
/// (`(bits & 0x7C00) == 0x7C00`) so the detection logic is tested on CI
/// without a physical GPU.
///
/// # Parameters
/// - `grad_ptr`: **device** pointer to the FP16 gradient tensor.
///   In mock-cuda mode this is interpreted as a **host** pointer.
/// - `n`: number of `f16` elements.
/// - `stream`: the CUDA stream to launch on.
///
/// # Returns
/// `true` if any element is NaN or Inf, `false` otherwise.
pub fn check_overflow_fp16(
    grad_ptr: *const u16, // FP16 stored as u16 bits; avoids half-float dep in Rust
    n: usize,
    stream: &CudaStream,
) -> bool {
    if n == 0 {
        return false;
    }

    #[cfg(not(feature = "mock-cuda"))]
    {
        // Delegate to the C function compiled from overflow_check.cu.
        // The function manages its own device allocation and sync internally.
        unsafe {
            ramflow_check_overflow_fp16(
                grad_ptr as *const std::os::raw::c_void,
                n as i32,
                stream.as_raw(),
            )
        }
    }

    #[cfg(feature = "mock-cuda")]
    {
        // Pure-host fallback: same bitmask logic, operates on host memory.
        // grad_ptr is a host pointer in mock mode.
        let slice = unsafe { std::slice::from_raw_parts(grad_ptr, n) };
        for &bits in slice {
            if (bits & 0x7C00u16) == 0x7C00u16 {
                return true;
            }
        }
        false
    }
}

// ===========================================================================
// FFI declarations — real CUDA path only
// ===========================================================================

#[cfg(not(feature = "mock-cuda"))]
extern "C" {
    // From kernels/overflow_check.cu
    fn ramflow_check_overflow_fp16(
        grad_device: *const std::os::raw::c_void,
        n: i32,
        stream: *mut std::os::raw::c_void,
    ) -> bool;

    // From libcudart — stream lifecycle
    fn cudaStreamCreateWithFlags(stream: *mut *mut std::os::raw::c_void, flags: u32) -> i32;

    fn cudaStreamSynchronize(stream: *mut std::os::raw::c_void) -> i32;

    fn cudaStreamDestroy(stream: *mut std::os::raw::c_void) -> i32;
}

// Thin wrappers so the Rust code above doesn't need raw unsafe in-line.
// These just exist to give the extern functions nicer names at the call site.
#[cfg(not(feature = "mock-cuda"))]
unsafe fn cuda_stream_create_non_blocking(out: *mut *mut std::os::raw::c_void) -> i32 {
    const CUDA_STREAM_NON_BLOCKING: u32 = 1;
    cudaStreamCreateWithFlags(out, CUDA_STREAM_NON_BLOCKING)
}

#[cfg(not(feature = "mock-cuda"))]
unsafe fn cuda_stream_synchronize(stream: *mut std::os::raw::c_void) -> i32 {
    cudaStreamSynchronize(stream)
}

#[cfg(not(feature = "mock-cuda"))]
unsafe fn cuda_stream_destroy(stream: *mut std::os::raw::c_void) -> i32 {
    cudaStreamDestroy(stream)
}

// ===========================================================================
// Tests (mock-cuda only, run on CI without a GPU)
// ===========================================================================
//
// These are the tests from the Sprint 2 spec:
//   Test 2a: array with known NaN/Inf positions → must return true
//   Test 2b: clean array → must return false
//   1000 iterations with zero false negatives / false positives.
//
// Run with:
//   cargo test --no-default-features --features mock-cuda

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::ManuallyDrop;

    // Helper: build a u16 that represents a valid FP16 value.
    // 1.0 in FP16: sign=0, exp=01111 (15, biased), mantissa=0 → 0x3C00
    fn fp16_one() -> u16 {
        0x3C00
    }

    // NaN in FP16: exponent=11111, mantissa=0x001 (non-zero) → 0x7C01
    fn fp16_nan() -> u16 {
        0x7C01
    }

    // +Inf in FP16: exponent=11111, mantissa=0 → 0x7C00
    fn fp16_inf() -> u16 {
        0x7C00
    }

    // -Inf in FP16: sign=1, exponent=11111, mantissa=0 → 0xFC00
    fn fp16_neg_inf() -> u16 {
        0xFC00
    }

    // Largest finite FP16 value: sign=0, exp=11110, mantissa=all-1 → 0x7BFF
    fn fp16_max_finite() -> u16 {
        0x7BFF
    }

    fn make_stream() -> CudaStream {
        CudaStream::new().expect("stream creation failed")
    }

    #[test]
    fn overflow_check_detects_nan() {
        let mut data = vec![fp16_one(); 10_000];
        // Embed 5 NaN values at known positions
        data[100] = fp16_nan();
        data[1000] = fp16_nan();
        data[5000] = fp16_nan();
        data[7777] = fp16_nan();
        data[9999] = fp16_nan();

        let stream = make_stream();
        assert!(
            check_overflow_fp16(data.as_ptr(), data.len(), &stream),
            "NaN values not detected"
        );
    }

    #[test]
    fn overflow_check_detects_inf() {
        let mut data = vec![fp16_one(); 10_000];
        // Embed 3 Inf values (mix of +Inf and -Inf)
        data[50] = fp16_inf();
        data[3000] = fp16_neg_inf();
        data[8888] = fp16_inf();

        let stream = make_stream();
        assert!(
            check_overflow_fp16(data.as_ptr(), data.len(), &stream),
            "Inf values not detected"
        );
    }

    #[test]
    fn overflow_check_clean_array_returns_false() {
        // All values are either 1.0 or the max finite FP16 value.
        // Neither has all-1 exponent bits — must return false.
        let data: Vec<u16> = (0..10_000)
            .map(|i| {
                if i % 2 == 0 {
                    fp16_one()
                } else {
                    fp16_max_finite()
                }
            })
            .collect();

        let stream = make_stream();
        assert!(
            !check_overflow_fp16(data.as_ptr(), data.len(), &stream),
            "False positive: clean array triggered overflow"
        );
    }

    #[test]
    fn overflow_check_empty_is_false() {
        let stream = make_stream();
        // n=0 must return false without dereferencing the pointer.
        assert!(!check_overflow_fp16(std::ptr::null(), 0, &stream));
    }

    #[test]
    fn overflow_check_1000_iterations_no_false_negative() {
        let stream = make_stream();
        for iteration in 0..1000usize {
            let mut data = vec![fp16_one(); 10_000];
            // Single NaN at a position derived from the iteration count.
            data[iteration % 10_000] = fp16_nan();
            let result = check_overflow_fp16(data.as_ptr(), data.len(), &stream);
            assert!(
                result,
                "False negative at iteration {iteration}: NaN at position {} not detected",
                iteration % 10_000
            );
        }
    }

    #[test]
    fn overflow_check_1000_iterations_no_false_positive() {
        let stream = make_stream();
        for iteration in 0..1000usize {
            let data: Vec<u16> = (0..10_000)
                .map(|i| {
                    // Mix valid FP16 values, none with all-1 exponent.
                    // 0x3C00 = 1.0, 0x4000 = 2.0, 0x7BFF = max finite
                    match (i + iteration) % 3 {
                        0 => 0x3C00u16,
                        1 => 0x4000u16,
                        _ => 0x7BFFu16,
                    }
                })
                .collect();
            let result = check_overflow_fp16(data.as_ptr(), data.len(), &stream);
            assert!(!result, "False positive at iteration {iteration}");
        }
    }
}
