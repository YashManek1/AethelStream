// src/allocator/huge.rs — hugepage-backed allocation for large training buffers
//
// Feature gate: `hugepages`.  Platform gate: `target_os = "linux"`.
//
// Option A (transparent hugepages):
//   mmap(NULL, rounded_size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0)
//   madvise(ptr, rounded_size, MADV_HUGEPAGE)
//   cudaHostRegister(ptr, size, cudaHostRegisterDefault)
//
// Why this helps:
//   A 1 GB buffer backed by 4 KB pages requires 262,144 TLB entries.
//   With 2 MiB transparent hugepages the same buffer needs only 512 entries,
//   reducing TLB pressure on every DMA transfer.  cudaHostRegister cost also
//   falls because the driver page-table walk visits 512x fewer entries.
//
// Drop constraint:
//   mmap'd pages MUST be released with munmap(ptr, mmap_size), NOT free(ptr).
//   PinnedBuffer::Drop branches on AllocKind::Huge to call munmap_huge.

/// Pool routing threshold: slots >= this byte count use hugepage allocation
/// when the `hugepages` feature is active.
///
/// 2 MiB matches the Linux transparent hugepage size.  Smaller buffers see
/// no benefit (TLB coverage is the same whether backed by 4 KB or 2 MiB pages
/// when the total is < 2 MiB) but pay the page-alignment round-up overhead.
#[cfg(feature = "hugepages")]
pub const HUGEPAGE_THRESHOLD: usize = 2 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Linux implementation
// ---------------------------------------------------------------------------

#[cfg(all(feature = "hugepages", target_os = "linux"))]
pub(crate) mod linux {
    use crate::{RamFlowError, Result};

    /// 2 MiB: the Linux transparent hugepage granularity.
    pub const HUGE_PAGE_SIZE: usize = 2 * 1024 * 1024;

    /// Round `size` up to the next 2 MiB boundary.
    ///
    /// munmap requires the address + length to match the original mmap exactly.
    /// Rounding up here ensures the entire requested region fits in whole hugepages.
    #[inline]
    pub fn round_to_huge(size: usize) -> usize {
        (size + HUGE_PAGE_SIZE - 1) & !(HUGE_PAGE_SIZE - 1)
    }

    /// Allocate `size` bytes via `mmap` + `MADV_HUGEPAGE`.
    ///
    /// Returns `(ptr, mmap_size)` where `mmap_size >= size` (rounded to 2 MiB).
    /// The caller must pass **`mmap_size`** (not `size`) to [`munmap_huge`] on Drop.
    ///
    /// # Errors
    /// Returns [`RamFlowError::AllocationFailed`] if `mmap` returns `MAP_FAILED`
    /// (e.g. OOM or virtual-address exhaustion).  `madvise(MADV_HUGEPAGE)` is a
    /// best-effort hint; its return value is intentionally ignored — the memory is
    /// still valid even if the kernel cannot back it with hugepages right now.
    pub fn mmap_huge(size: usize) -> Result<(*mut u8, usize)> {
        let mmap_size = round_to_huge(size);
        // SAFETY: standard anonymous private mapping; -1 fd, 0 offset.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                mmap_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(RamFlowError::AllocationFailed(format!(
                "mmap({mmap_size} B) for hugepage allocation failed"
            )));
        }
        // SAFETY: ptr is a valid mmap result; mmap_size > 0.
        // MADV_HUGEPAGE is advisory — the kernel upgrades pages to 2 MiB entries
        // opportunistically; it never returns an error that invalidates the mapping.
        unsafe {
            libc::madvise(ptr, mmap_size, libc::MADV_HUGEPAGE);
        }
        Ok((ptr as *mut u8, mmap_size))
    }

    /// Release a hugepage allocation returned by [`mmap_huge`].
    ///
    /// # Safety
    /// * `ptr` must be the exact pointer returned by `mmap_huge`.
    /// * `mmap_size` must be the second element of that return value — the
    ///   rounded-up length, not the original `size_bytes`.
    /// * Must be called exactly once.  A second call is undefined behaviour
    ///   because the kernel may have reused those virtual addresses.
    pub unsafe fn munmap_huge(ptr: *mut u8, mmap_size: usize) {
        let rc = libc::munmap(ptr as *mut libc::c_void, mmap_size);
        // On debug builds, assert the munmap succeeded. In release builds, there
        // is nothing useful to do with the error (Drop cannot propagate it).
        debug_assert_eq!(rc, 0, "munmap({ptr:p}, {mmap_size}) failed — double-free or wrong mmap_size");
    }
}
