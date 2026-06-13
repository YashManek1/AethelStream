// src/allocator/mmap_tier.rs — mmap-backed buffer for graceful-degradation fallback
//
// Feature gate: `mmap-fallback`
//
// MmapBuffer is allocated via mmap(MAP_PRIVATE|MAP_ANONYMOUS) + madvise(MADV_SEQUENTIAL).
// It is NOT registered with cudaHostRegister — DMA requires a pinned staging copy.
// This is 2-4× slower than pinned DMA but allows training on machines with <16 GB RAM.

use crate::{RamFlowError, Result};

/// mmap-backed host buffer for graceful-degradation mode.
///
/// Allocated with `mmap(MAP_PRIVATE|MAP_ANONYMOUS)` + `madvise(MADV_SEQUENTIAL|MADV_WILLNEED)`.
/// Never registered with `cudaHostRegister`; DMA must go through a pinned staging copy
/// (see `ZeroCopyRouter::route`). `is_pinned()` returns `false`.
///
/// On Windows, `alloc_mmap` always returns `RamFlowError::ConfigError`.
///
/// # Thread safety
/// `MmapBuffer` is `Send` but not `Sync`. Exclusive access must be enforced by the caller.
pub struct MmapBuffer {
    ptr: *mut u8,
    size_bytes: usize,
}

// SAFETY: MmapBuffer wraps a raw mmap pointer. Single-owner semantics are enforced
// externally (via PoolSlot or AnyBuffer); sending to another thread is safe.
unsafe impl Send for MmapBuffer {}

impl MmapBuffer {
    /// Allocate `size_bytes` of anonymous mmap memory with sequential access hints.
    ///
    /// On Linux/macOS: calls `mmap(MAP_PRIVATE|MAP_ANONYMOUS)` then
    /// `madvise(MADV_SEQUENTIAL)` and `madvise(MADV_WILLNEED)`. madvise failures
    /// are non-fatal (pages remain usable).
    ///
    /// On Windows: always returns `RamFlowError::ConfigError("not supported")`.
    ///
    /// # Errors
    /// - `RamFlowError::AllocationFailed` if `size_bytes == 0` or `mmap` fails.
    /// - `RamFlowError::ConfigError` on non-Unix platforms.
    pub fn alloc_mmap(size_bytes: usize) -> Result<Self> {
        if size_bytes == 0 {
            return Err(RamFlowError::AllocationFailed(
                "zero-size MmapBuffer requested".into(),
            ));
        }

        #[cfg(unix)]
        {
            let ptr = mmap_alloc(size_bytes)?;
            Ok(MmapBuffer { ptr, size_bytes })
        }

        #[cfg(not(unix))]
        {
            let _ = size_bytes;
            Err(RamFlowError::ConfigError(
                "mmap-fallback is not supported on Windows; \
                 use a Linux or macOS system for mmap-backed pool slots"
                    .into(),
            ))
        }
    }

    /// Returns `false`: mmap buffers are pageable. DMA requires a pinned staging copy.
    #[inline]
    pub fn is_pinned(&self) -> bool {
        false
    }

    /// Length of the buffer in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        self.size_bytes
    }

    /// Returns `true` if the buffer has zero length (always `false` for successfully
    /// constructed buffers — `alloc_mmap(0)` returns an error).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.size_bytes == 0
    }

    /// Raw const pointer to the start of the buffer.
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }

    /// Raw mutable pointer to the start of the buffer.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    /// Read-only byte slice over the entire buffer.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr is non-null, valid for size_bytes reads, and outlives &self.
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.size_bytes) }
    }

    /// Mutable byte slice over the entire buffer.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: ptr is non-null, valid for size_bytes reads/writes; &mut self
        // guarantees exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size_bytes) }
    }
}

impl Drop for MmapBuffer {
    fn drop(&mut self) {
        #[cfg(unix)]
        // SAFETY: ptr and size_bytes were returned by mmap(2) in alloc_mmap and
        // have not been munmap'd before. munmap must receive the exact values from mmap.
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.size_bytes);
        }
    }
}

// ---------------------------------------------------------------------------
// Platform helpers
// ---------------------------------------------------------------------------

#[cfg(unix)]
fn mmap_alloc(size_bytes: usize) -> Result<*mut u8> {
    // SAFETY: standard anonymous mmap. All arguments are valid:
    //   addr=NULL (OS chooses), size>0, prot=RW, flags=PRIVATE|ANON, fd=-1, offset=0.
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            size_bytes,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    if ptr == libc::MAP_FAILED {
        return Err(RamFlowError::AllocationFailed(format!(
            "mmap({size_bytes} B) failed for MmapBuffer: errno {}",
            std::io::Error::last_os_error()
        )));
    }
    let ptr = ptr as *mut u8;
    // MADV_SEQUENTIAL: kernel can prefetch pages in order; reduces page-fault latency
    // during sequential streaming reads. MADV_WILLNEED: warm up TLB preemptively.
    // Failures are non-fatal — pages are still accessible.
    // SAFETY: ptr and size_bytes come from the successful mmap above.
    unsafe {
        libc::madvise(
            ptr as *mut libc::c_void,
            size_bytes,
            libc::MADV_SEQUENTIAL,
        );
        libc::madvise(
            ptr as *mut libc::c_void,
            size_bytes,
            libc::MADV_WILLNEED,
        );
    }
    Ok(ptr)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    #[cfg(unix)]
    fn alloc_and_round_trip() {
        let mut buf = MmapBuffer::alloc_mmap(4096).expect("alloc_mmap");
        assert_eq!(buf.len(), 4096);
        assert!(!buf.is_pinned());
        for (index, byte) in buf.as_mut_slice().iter_mut().enumerate() {
            *byte = (index % 251) as u8;
        }
        for (index, byte) in buf.as_slice().iter().enumerate() {
            assert_eq!(*byte, (index % 251) as u8, "byte mismatch at {index}");
        }
    }

    #[test]
    fn zero_size_returns_error() {
        let result = MmapBuffer::alloc_mmap(0);
        assert!(
            matches!(result, Err(RamFlowError::AllocationFailed(_))),
            "zero-size must fail"
        );
    }

    #[test]
    #[cfg(unix)]
    fn is_pinned_is_always_false() {
        let buf = MmapBuffer::alloc_mmap(512).expect("alloc");
        assert!(!buf.is_pinned());
    }
}

