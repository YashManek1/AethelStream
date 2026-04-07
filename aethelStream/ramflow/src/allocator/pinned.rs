// src/allocator/pinned.rs — PinnedBuffer: page-locked host memory
//
// Sprint 1: full implementation replacing Sprint 0 stubs.
//
// ─── WHAT THIS MODULE DOES ────────────────────────────────────────────────
//
//   PinnedBuffer wraps a single contiguous allocation of CPU RAM that:
//
//   1. Is aligned to 64 bytes (one CPU cache line).
//      The GPU's DMA engine fetches memory in 64-byte chunks.  Starting at a
//      64-byte boundary means the very first DMA read for this buffer costs
//      exactly one cache-line fetch — it never "straddles" two lines.
//      Without alignment, a 64-byte tensor that starts at offset +32 requires
//      two fetches (one for bytes 0-63, one for bytes 64-127), doubling DMA
//      latency for the first cache line of every tensor.
//
//   2. Is exactly the requested size.
//      PyTorch's CachingAllocator rounds up to the next power of two.
//      posix_memalign gives exactly the requested bytes (rounded up to the
//      OS page size by the kernel, but NOT to the next power of two).
//      For a 2.1 MB request: PyTorch → 4 MB; RamFlow → 2,097,152 + a few
//      hundred bytes of page overhead ≈ 2 MB.
//
//   3. Is pinned (page-locked) by the CUDA driver.
//      cudaHostRegister tells the kernel "never swap these pages to disk
//      while the GPU is alive."  That lets the DMA engine transfer data
//      asynchronously — the driver knows the physical address is stable.
//
// ─── REGISTRATION MODES ──────────────────────────────────────────────────
//
//   `is_mapped = false` (default, created by `PinnedBuffer::alloc`)
//     cudaHostRegisterDefault (flag = 0)
//     Pages are DMA-accessible.  The CPU holds the only virtual address.
//     Use for all standard offload/prefetch transfers in Sprints 1–3.
//
//   `is_mapped = true` (created by `PinnedBuffer::alloc_mapped`)
//     cudaHostRegisterMapped (flag = 2)
//     CUDA creates a *device* virtual address via UVA.  The GPU can read
//     this memory directly over NVLink without a cudaMemcpyAsync.
//     Use ONLY in Sprint 4's slab packer (ZeroCopyRouter path).
//     ⚠️  ZeroCopyRouter checks `is_mapped()` before calling
//     cudaHostGetDevicePointer.  A Default buffer routed to ZeroCopy
//     will return cudaErrorInvalidValue.
//
// ─── SEND / SYNC ──────────────────────────────────────────────────────────
//
//   Send:  The raw pointer can be sent to another thread (e.g. the CQE
//          poller thread or the CUDA callback thread).  The pool ring
//          enforces that only one thread holds a claim on the buffer at
//          a time.
//
//   !Sync: Two threads must NOT read/write the buffer concurrently without
//          external synchronisation.  The slab packer and pool ring provide
//          this synchronisation — PinnedBuffer itself does not.
//          Rust's default for *mut T is !Sync, so we do NOT add an impl.
//
// ─── DROP ORDER (CRITICAL) ────────────────────────────────────────────────
//
//   1. cudaHostUnregister(&ptr)  ← tell CUDA driver to release pin
//   2. libc::free(ptr)           ← release the memory to the OS
//
//   Reversing this order is undefined behaviour: `libc::free` returns the
//   pages to the OS, then `cudaHostUnregister` attempts to dereference the
//   now-invalid physical address.  The CUDA driver may not crash immediately
//   — it can silently corrupt the next allocation that reuses those pages.

use std::os::raw::c_void;

use crate::cuda_bridge::bindings::{
    cuda_host_register,
    cuda_host_unregister,
    CUDA_HOST_REGISTER_DEFAULT,
    CUDA_HOST_REGISTER_MAPPED,
};
use crate::{Result, RamFlowError};

// ===========================================================================
// Platform-specific aligned allocator
// ===========================================================================

/// Wrappers for aligned allocation/free that compile on both Linux and Windows.
///
/// Linux/macOS: `posix_memalign` + `libc::free`
///   • POSIX standard; alignment can be any power-of-two ≥ sizeof(void*).
///   • The pointer MUST be freed with `libc::free`.
///
/// Windows: `_aligned_malloc` + `_aligned_free`
///   • The CRT function that mirrors posix_memalign semantics.
///   • The pointer MUST be freed with `_aligned_free` (NOT `free`!).
///   • We use the libc crate's FFI binding to the Windows CRT.
mod platform {
    use crate::{Result, RamFlowError};

    /// Allocate `size` bytes aligned to `alignment`.
    ///
    /// Returns a non-null pointer on success.  The caller must free it with
    /// [`free_aligned`] — do NOT pass this pointer to any other free function.
    pub(super) fn allocate_aligned(size_bytes: usize, alignment: usize) -> Result<*mut u8> {
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        {
            // posix_memalign is the canonical POSIX way to get aligned memory.
            // Requirement: alignment is a power of two and a multiple of
            // sizeof(void*) (8 on 64-bit). 64 satisfies both.
            let mut ptr: *mut c_void = std::ptr::null_mut();
            // SAFETY: ptr is a valid out-pointer; alignment and size are valid.
            let rc = unsafe { libc::posix_memalign(&mut ptr, alignment, size_bytes) };
            if rc != 0 {
                return Err(RamFlowError::AllocationFailed(format!(
                    "posix_memalign({size_bytes} B, align {alignment}) failed, errno {rc}"
                )));
            }
            debug_assert!(!ptr.is_null());
            Ok(ptr as *mut u8)
        }

        #[cfg(target_os = "windows")]
        {
            // _aligned_malloc is the Windows CRT equivalent.
            // It is available via the `libc` crate as `aligned_malloc` on Windows.
            let ptr = unsafe {
                libc::aligned_malloc(size_bytes, alignment)
            };
            if ptr.is_null() {
                return Err(RamFlowError::AllocationFailed(format!(
                    "aligned_malloc({size_bytes} B, align {alignment}) returned null"
                )));
            }
            Ok(ptr as *mut u8)
        }

        #[cfg(not(any(unix, target_os = "windows")))]
        {
            // Fallback for exotic targets: use std::alloc with Layout.
            use std::alloc::{alloc, Layout};
            let layout = Layout::from_size_align(size_bytes, alignment)
                .map_err(|e| RamFlowError::AllocationFailed(format!("aligned layout error: {e}")))?;
            // SAFETY: layout.size() > 0 (checked by caller).
            let ptr = unsafe { alloc(layout) };
            if ptr.is_null() {
                return Err(RamFlowError::AllocationFailed(
                    "std::alloc::alloc returned null".into(),
                ));
            }
            Ok(ptr)
        }
    }

    /// Free a pointer returned by [`allocate_aligned`].
    ///
    /// # Safety
    /// `ptr` must have been returned by [`allocate_aligned`] and must not have
    /// been freed already.
    pub(super) unsafe fn free_aligned(ptr: *mut u8) {
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        {
            libc::free(ptr as *mut c_void);
        }

        #[cfg(target_os = "windows")]
        {
            libc::aligned_free(ptr as *mut _);
        }

        #[cfg(not(any(unix, target_os = "windows")))]
        {
            // Mirror the fallback alloc path.
            drop(Box::from_raw(ptr));
        }
    }
}

// ─── Alignment constant ────────────────────────────────────────────────────

/// 64 bytes = one CPU cache line.
///
/// The GPU DMA engine reads memory in 64-byte chunks aligned to 64-byte
/// boundaries.  Allocating at a 64-byte boundary ensures:
///   • The first DMA read for any buffer costs exactly one chunk.
///   • No buffer ever "straddles" a cache-line boundary on its first byte.
///
/// posix_memalign requires alignment to be a power of two AND a multiple of
/// `sizeof(void*)` (8 bytes on 64-bit).  64 satisfies both.
const PINNED_ALIGN: usize = 64;

// ===========================================================================
// PinnedBuffer
// ===========================================================================

/// Page-locked (pinned) host buffer visible to both CPU and GPU.
///
/// # Memory model
/// Allocated with `posix_memalign(64)` for exact sizing and DMA alignment.
/// Registered with the CUDA driver via `cudaHostRegister` to prevent kernel
/// page migration.
///
/// # Safety
/// This type holds a raw pointer.  All `unsafe` blocks within are justified
/// by the construction invariants: the pointer was returned by `posix_memalign`
/// and remains registered with `cudaHostRegister` until `drop` runs.
pub struct PinnedBuffer {
    /// Pointer to the start of the pinned allocation.
    ///
    /// Invariant: non-null, aligned to 64 bytes, allocated by posix_memalign,
    /// registered with cudaHostRegister (or mock no-op in mock-cuda mode).
    ptr: *mut u8,

    /// Allocation size in bytes — exactly what the caller requested.
    ///
    /// Invariant: matches the `size` argument passed to `posix_memalign` and
    /// `cudaHostRegister`.
    size_bytes: usize,

    /// Registration mode flag.
    ///
    /// `false` → registered with `cudaHostRegisterDefault` (DMA only).
    ///           Suitable for standard offload/prefetch transfers.
    ///
    /// `true`  → registered with `cudaHostRegisterMapped` (DMA + UVA).
    ///           Required for zero-copy (Sprint 4).  ZeroCopyRouter checks
    ///           this flag before calling `cudaHostGetDevicePointer`.
    is_mapped: bool,
}

// Safety: PinnedBuffer wraps a raw pointer that was allocated by posix_memalign
// and registered with cudaHostRegister.  The pool ring enforces single-owner
// semantics at runtime: only one thread holds a "claim" token for any buffer
// at a time.  Sending the buffer to another thread is therefore safe — it
// is equivalent to moving a Box<[u8]>.
unsafe impl Send for PinnedBuffer {}

// We do NOT implement Sync.  The default for *mut T is !Sync, which is
// correct: two threads must not concurrently read/write this buffer.
// The slab packer and pool ring are responsible for synchronisation.

// ===========================================================================
// impl PinnedBuffer
// ===========================================================================

impl PinnedBuffer {
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /// Allocate `bytes` of pinned memory for DMA access.
    ///
    /// Uses `posix_memalign(64)` for exact sizing, then registers the pages
    /// with `cudaHostRegisterDefault` (flag = 0) for DMA access.
    ///
    /// In `mock-cuda` mode, `cudaHostRegister` is a no-op — the allocation
    /// precision is still tested because `posix_memalign` runs normally.
    ///
    /// # Errors
    /// - `RamFlowError::AllocationFailed` if `posix_memalign` returns ENOMEM.
    /// - `RamFlowError::CudaError` if `cudaHostRegister` returns non-zero.
    pub fn alloc(bytes: usize) -> Result<Self> {
        // Zero-size allocations are not meaningful for DMA buffers.
        if bytes == 0 {
            return Err(RamFlowError::AllocationFailed(
                "zero-size PinnedBuffer requested".into(),
            ));
        }

        let ptr = Self::posix_memalign_alloc(bytes)?;

        // SAFETY: ptr is non-null, aligned to 64 bytes, allocated by
        // posix_memalign, and `bytes` matches the allocation size.
        unsafe {
            cuda_host_register(ptr as *mut c_void, bytes, CUDA_HOST_REGISTER_DEFAULT)
                .map_err(|e| {
                    // If registration fails, free the memory immediately to
                    // avoid a leak (the Drop impl would not call unregister
                    // since registration never succeeded).
                    // SAFETY: ptr was returned by posix_memalign_alloc.
                    platform::free_aligned(ptr);
                    e
                })?;
        }

        Ok(Self {
            ptr,
            size_bytes: bytes,
            is_mapped: false,
        })
    }

    /// Allocate `bytes` of pinned memory registered for UVA access.
    ///
    /// Uses `posix_memalign(64)` + `cudaHostRegisterMapped` (flag = 2).
    ///
    /// # When to use
    /// **Only the slab packer should call this** (Sprint 4).  All other
    /// allocations use [`PinnedBuffer::alloc`].  The `ZeroCopyRouter` checks
    /// `is_mapped()` before proceeding with UVA routing — calling `alloc_mapped`
    /// on a buffer that will be transferred via `cudaMemcpyAsync` wastes the
    /// UVA table entry and may cause `ZeroCopyRouter::route` to return the
    /// wrong strategy.
    ///
    /// # Errors
    /// Same as [`PinnedBuffer::alloc`].
    pub fn alloc_mapped(bytes: usize) -> Result<Self> {
        if bytes == 0 {
            return Err(RamFlowError::AllocationFailed(
                "zero-size PinnedBuffer (mapped) requested".into(),
            ));
        }

        let ptr = Self::posix_memalign_alloc(bytes)?;

        // SAFETY: Same as alloc. Flag is MAPPED for UVA.
        unsafe {
            cuda_host_register(ptr as *mut c_void, bytes, CUDA_HOST_REGISTER_MAPPED)
                .map_err(|e| {
                    // SAFETY: ptr was returned by posix_memalign_alloc.
                    platform::free_aligned(ptr);
                    e
                })?;
        }

        Ok(Self {
            ptr,
            size_bytes: bytes,
            is_mapped: true,
        })
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Call the platform-appropriate aligned allocator and return the pointer.
    ///
    /// | Platform | Function               | Notes                          |
    /// |----------|------------------------|--------------------------------|
    /// | Linux    | `posix_memalign(64)`   | POSIX standard, exact size     |
    /// | Windows  | `_aligned_malloc(64)`  | CRT aligned allocator          |
    ///
    /// Both return a pointer that must be freed with the corresponding free:
    /// - `libc::free` on Linux
    /// - `_aligned_free` on Windows (see `platform::aligned_free`)
    ///
    /// # Safety invariants on return
    /// The returned pointer is:
    ///   • Non-null (error was checked).
    ///   • Aligned to `PINNED_ALIGN` (64) bytes.
    ///   • Points to `size` bytes of writable, uninitialized memory.
    fn posix_memalign_alloc(size: usize) -> Result<*mut u8> {
        platform::allocate_aligned(size, PINNED_ALIGN)
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Size of this buffer in bytes — exactly what was requested at construction.
    #[inline]
    pub fn len(&self) -> usize {
        self.size_bytes
    }

    /// Returns `true` if the buffer is zero-sized.
    ///
    /// Note: `PinnedBuffer::alloc(0)` returns an error, so this is always
    /// `false` for successfully constructed buffers.  Provided for API
    /// completeness (to satisfy `clippy::len_without_is_empty`).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.size_bytes == 0
    }

    /// Whether this buffer was registered with `cudaHostRegisterMapped`.
    ///
    /// `ZeroCopyRouter` reads this flag before calling `cudaHostGetDevicePointer`.
    /// Only buffers created with [`PinnedBuffer::alloc_mapped`] return `true`.
    #[inline]
    pub fn is_mapped(&self) -> bool {
        self.is_mapped
    }

    /// Read-only byte slice over the entire buffer.
    ///
    /// # Safety
    /// SAFETY note for the implementation: `ptr` is valid, properly aligned,
    /// and initialized (posix_memalign allocates writable memory; it is the
    /// caller's responsibility to write before reading, as with any Vec<u8>).
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr is non-null, properly aligned, valid for `size_bytes`
        // reads.  The lifetime is tied to `&self`, which prevents the buffer
        // from being dropped while the slice exists.
        unsafe { std::slice::from_raw_parts(self.ptr, self.size_bytes) }
    }

    /// Mutable byte slice over the entire buffer.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: ptr is non-null, valid for `size_bytes` reads/writes.
        // `&mut self` guarantees exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size_bytes) }
    }

    /// Raw const pointer to the start of the buffer (for CUDA FFI).
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Raw mutable pointer to the start of the buffer (for CUDA FFI and io_uring).
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }
}

// ===========================================================================
// Drop implementation — ordering is critical
// ===========================================================================

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        // ⚠️  ORDERING: cudaHostUnregister MUST be called BEFORE libc::free.
        //
        //     Why: If we call libc::free first, the OS may immediately reuse
        //     those physical pages.  The CUDA driver still has an internal
        //     reference to the (now-invalid) physical address.  When we later
        //     call cudaHostUnregister, the driver touches that address to
        //     release the pin — but the page may now belong to a different
        //     allocation.  The resulting corruption is:
        //       • Silent (no immediate crash).
        //       • Non-deterministic (depends on OS page reuse timing).
        //       • Diagnosed only hours later when unrelated reads return
        //         wrong data or the driver triggers a deferred assertion.
        //
        // SAFETY:
        //   • `self.ptr` was allocated by `posix_memalign` — guaranteed by
        //     `PinnedBuffer::alloc` / `alloc_mapped` construction.
        //   • `cudaHostRegister` was called exactly once during construction.
        //     We call `cudaHostUnregister` exactly once here.
        //   • After `Drop` completes no code can observe `self.ptr` (the
        //     borrow checker prevents use-after-drop for the safe API surface).
        //   • We intentionally ignore `cuda_host_unregister`'s Result here:
        //     Drop cannot propagate errors, and the only failure case is a
        //     programming error that should have been caught in tests.
        unsafe {
            // Step 1: Release the CUDA driver's pin on these pages.
            let _ = cuda_host_unregister(self.ptr as *mut c_void);

            // Step 2: Return the memory to the OS.
            // Use platform::free_aligned — on Windows this calls _aligned_free
            // (NOT libc::free, which would be UB for _aligned_malloc pointers).
            platform::free_aligned(self.ptr);
        }
    }
}
