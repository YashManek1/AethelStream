// src/allocator/drop_guard.rs — RAII guard for raw CUDA + aligned-malloc pointers
//
// Sprint 1: full implementation replacing Sprint 0 opaque placeholder.
//
// PURPOSE:
//   PinnedDropGuard is a lightweight RAII wrapper for situations where you
//   have a raw pointer that needs the unregister-then-free teardown sequence
//   but you are NOT using the full PinnedBuffer API (e.g., during a failed
//   construction attempt or in test scaffolding).
//
//   For normal usage, prefer PinnedBuffer directly — it embeds the same Drop
//   logic.  PinnedDropGuard exists for cases where you want to track a raw
//   pointer separately before "promoting" it into a PinnedBuffer.
//
// DROP ORDER (mirrors PinnedBuffer::drop):
//   1. cuda_host_unregister(ptr)  — release CUDA driver pin
//   2. platform::aligned_free(ptr) — release memory to OS
//
//   See pinned.rs for the detailed explanation of why this order is critical.
//
// PLATFORM NOTE:
//   Memory is freed with the same call that was used to allocate it:
//     Linux:   libc::free (matches posix_memalign)
//     Windows: _aligned_free (matches _aligned_malloc)
//   Using libc::free on Windows for _aligned_malloc memory is undefined behaviour.

use std::os::raw::c_void;
use crate::cuda_bridge::bindings::cuda_host_unregister;

// ─── Platform-specific free ────────────────────────────────────────────────

/// Free a pointer that was allocated by the platform aligned allocator.
///
/// # Safety
/// `ptr` must have been allocated by `allocate_aligned` / `platform::allocate_aligned`
/// in `pinned.rs`.  Must not be called more than once for the same pointer.
unsafe fn free_aligned(ptr: *mut u8) {
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
        drop(Box::from_raw(ptr));
    }
}

// ===========================================================================
// PinnedDropGuard
// ===========================================================================

/// RAII wrapper that calls `cudaHostUnregister` + platform aligned free when dropped.
///
/// Useful for exception-safety during two-phase construction:
/// ```ignore
/// let ptr = platform::aligned_alloc(size, 64)?;
/// let mut guard = PinnedDropGuard::new(ptr, size); // freed if we return early
/// cuda_host_register(ptr, size, flags)?;
/// guard.mark_registered();
/// std::mem::forget(guard); // PinnedBuffer takes ownership
/// ```
///
/// In `mock-cuda` mode, `cudaHostUnregister` is a no-op, so `PinnedDropGuard`
/// still compiles and provides correct aligned-free semantics.
pub struct PinnedDropGuard {
    /// Pointer to the allocation.  `null_mut()` means "already freed" (used
    /// after `PinnedDropGuard::defuse` is called).
    ptr: *mut u8,

    /// Length in bytes (must match the cudaHostRegister call).
    size: usize,

    /// Whether cudaHostRegister was successfully called on this pointer.
    ///
    /// If `false` (registration failed or never happened), we skip
    /// `cudaHostUnregister` and only call platform aligned free.
    registered: bool,
}

// Safety: same reasoning as PinnedBuffer — the pool enforces single-owner
// semantics so sending PinnedDropGuard across threads is safe.
unsafe impl Send for PinnedDropGuard {}

impl PinnedDropGuard {
    /// Wrap `ptr` (allocated by the platform aligned allocator) before CUDA registration.
    ///
    /// After successful `cuda_host_register`, call [`PinnedDropGuard::mark_registered`]
    /// so that `drop` will issue the unregister call.
    pub fn new(ptr: *mut u8, size: usize) -> Self {
        Self {
            ptr,
            size,
            registered: false,
        }
    }

    /// Mark the pointer as successfully registered with `cudaHostRegister`.
    ///
    /// After this, `drop` will call `cudaHostUnregister` before aligned free.
    pub fn mark_registered(&mut self) {
        self.registered = true;
    }

    /// Consume the guard **without** freeing the memory.
    ///
    /// Call this after transferring ownership to a `PinnedBuffer` (which has
    /// its own `Drop`).  Leaving the guard alive after a move would cause
    /// a double-free.
    pub fn defuse(mut self) {
        self.ptr = std::ptr::null_mut();
    }

    /// Raw pointer (for passing to `PinnedBuffer` constructor internals).
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Allocation size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for PinnedDropGuard {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            // Already defused — nothing to do.
            return;
        }

        // SAFETY:
        //   • `ptr` was allocated by the platform aligned allocator.
        //   • `registered` is true only after a successful `cudaHostRegister`.
        //   • We call unregister before free for the same reason as PinnedBuffer.
        unsafe {
            if self.registered {
                // Step 1: release CUDA driver pin
                let _ = cuda_host_unregister(self.ptr as *mut c_void);
            }
            // Step 2: free memory using the platform-matched free function
            free_aligned(self.ptr);
        }

        self.ptr = std::ptr::null_mut();
    }
}
