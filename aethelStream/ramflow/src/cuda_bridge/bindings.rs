// src/cuda_bridge/bindings.rs — raw FFI declarations to the CUDA runtime
//
// Sprint 1: real extern "C" blocks replace the Sprint 0 placeholder.
//
// ─── FLAG REFERENCE ────────────────────────────────────────────────────────
//
//   cudaHostRegisterDefault = 0
//     Pages the memory into the GPU driver's DMA aperture.
//     The CPU still holds the only virtual address for this memory.
//     Use this flag everywhere in Sprints 1-3.
//
//   cudaHostRegisterMapped  = 2
//     Additionally creates a device-side virtual address via CUDA UVA.
//     Required for zero-copy (GPU reads CPU RAM directly over NVLink/
//     integrated memory).  Sprint 4's zero_copy.rs uses this flag.
//     ⚠️  DO NOT use Mapped here — using it accidentally on DMA-only buffers
//     wastes the UVA mapping table and may confuse ZeroCopyRouter's
//     is_mapped check, causing it to return ZeroCopy for buffers that were
//     never intended for that path.
//
// ─── MOCK-CUDA vs REAL-CUDA ────────────────────────────────────────────────
//
//   When `--features mock-cuda` is passed:
//     • The `extern "C"` block is EXCLUDED.
//     • Two pure-Rust no-op functions take its place (always return 0).
//     • posix_memalign still runs — allocation size behavior is fully tested.
//     • This lets CI validate Sprint 1's precision claims without a GPU.
//
//   When `--features cuda` is passed (default):
//     • The real symbols are linked from libcudart.
//     • build.rs emits `cargo:rustc-link-lib=dylib=cudart`.
//
// ─── SAFETY CONTRACT ──────────────────────────────────────────────────────
//
//   These functions are `unsafe` because:
//     • `ptr` must point to memory that was allocated and is still live.
//     • `cuda_host_unregister` must be called before `libc::free`.
//       Calling free first leaves the CUDA driver holding a dangling
//       pointer; the subsequent unregister attempt produces silent
//       corruption or a deferred crash inside the driver.
//     • Callers are responsible for ensuring these invariants.
//       See PinnedBuffer::drop for the canonical call order.

use std::os::raw::c_void;

// ─── CUDA error code alias ─────────────────────────────────────────────────

/// Raw CUDA error code.  0 = cudaSuccess.  Non-zero = driver failure.
///
/// The specific non-zero values are the `cudaError_t` enum from cuda_runtime_api.h.
/// We surface the integer rather than an enum so we never have a version mismatch.
pub type CudaError = i32;

// ─── cudaHostRegisterDefault flag constant ─────────────────────────────────

/// Standard DMA access: map pages into GPU DMA aperture only.
///
/// Use this for all Sprint 1–3 allocations.  Sprint 4's `zero_copy.rs`
/// uses `CUDA_HOST_REGISTER_MAPPED` instead — the distinction is checked by
/// `PinnedBuffer::is_mapped()` before routing through `ZeroCopyRouter`.
pub const CUDA_HOST_REGISTER_DEFAULT: u32 = 0;

/// UVA-mapped access: DMA aperture + device virtual address creation.
///
/// Only `PinnedBuffer::alloc_mapped` (called from Sprint 4's slab packer)
/// should pass this flag.  Set here as a named constant so Sprint 4
/// developers have a single canonical reference.
pub const CUDA_HOST_REGISTER_MAPPED: u32 = 2;

// ===========================================================================
// REAL CUDA path
// ===========================================================================

/// Link the CUDA runtime functions when building with the real GPU.
///
/// `cuda_host_register` pins CPU memory so the GPU's DMA engine can reach it.
/// `cuda_host_unregister` releases that pin — must happen BEFORE `libc::free`.
#[cfg(not(feature = "mock-cuda"))]
extern "C" {
    /// Pin `size` bytes starting at `ptr` for GPU DMA access.
    ///
    /// # Arguments
    /// * `ptr`   — must be a valid, aligned, live CPU pointer.
    /// * `size`  — length in bytes (must match the allocation size).
    /// * `flags` — `CUDA_HOST_REGISTER_DEFAULT` or `CUDA_HOST_REGISTER_MAPPED`.
    ///
    /// # Returns
    /// `0` (cudaSuccess) on success; a non-zero `cudaError_t` on failure.
    pub fn cudaHostRegister(ptr: *mut c_void, size: usize, flags: u32) -> CudaError;

    /// Release the GPU's pin on `ptr`.
    ///
    /// **Call this BEFORE `libc::free`.**  If the pointer is freed first, the
    /// driver attempts to unregister memory it no longer owns — this is
    /// undefined behaviour in the CUDA driver and may cause silent heap
    /// corruption or a deferred crash.
    ///
    /// # Returns
    /// `0` (cudaSuccess) on success; a non-zero `cudaError_t` on failure.
    pub fn cudaHostUnregister(ptr: *mut c_void) -> CudaError;
}

// ===========================================================================
// MOCK-CUDA path — no GPU required
// ===========================================================================

/// No-op substitute for `cudaHostRegister`.
///
/// Memory is still allocated by the platform aligned allocator with full
/// alignment and exact size, so all allocation-precision tests remain
/// meaningful even on CI.
/// Named with camelCase to match the CUDA runtime symbol exactly.
#[cfg(feature = "mock-cuda")]
#[allow(non_snake_case)]
#[inline(always)]
pub unsafe fn cudaHostRegister(
    _ptr: *mut c_void,
    _size: usize,
    _flags: u32,
) -> CudaError {
    0 // cudaSuccess
}

/// No-op substitute for `cudaHostUnregister`.
///
/// Matches the real function's signature exactly.  The `Drop` impl in
/// `PinnedBuffer` always calls this before the platform aligned free;
/// in mock mode the call is free (inlined away) but the ordering
/// contract is still exercised.
/// Named with camelCase to match the CUDA runtime symbol exactly.
#[cfg(feature = "mock-cuda")]
#[allow(non_snake_case)]
#[inline(always)]
pub unsafe fn cudaHostUnregister(_ptr: *mut c_void) -> CudaError {
    0 // cudaSuccess
}

// ===========================================================================
// Convenience wrappers (used by PinnedBuffer only)
// ===========================================================================

/// Register `ptr` for DMA access (Sprint 1–3 path).
///
/// Thin wrapper that converts the raw CUDA error into `crate::RamFlowError::CudaError`.
///
/// # Safety
/// See module-level safety contract.
pub unsafe fn cuda_host_register(
    ptr: *mut c_void,
    size: usize,
    flags: u32,
) -> crate::Result<()> {
    let rc = cudaHostRegister(ptr, size, flags);
    if rc == 0 {
        Ok(())
    } else {
        Err(crate::RamFlowError::CudaError(rc))
    }
}

/// Unregister `ptr` (must be called before `libc::free`).
///
/// # Safety
/// See module-level safety contract.
pub unsafe fn cuda_host_unregister(ptr: *mut c_void) -> crate::Result<()> {
    let rc = cudaHostUnregister(ptr);
    if rc == 0 {
        Ok(())
    } else {
        Err(crate::RamFlowError::CudaError(rc))
    }
}
