// src/allocator/mod.rs -- pinned-memory allocator module

/// RAII guard that frees a pinned allocation on drop even during panic.
pub mod drop_guard;
/// Hugepage-backed allocation via mmap + MADV_HUGEPAGE (Linux, feature = "hugepages").
pub mod huge;
/// mmap-backed buffer for graceful-degradation fallback (feature = "mmap-fallback").
pub mod mmap_tier;
/// NUMA topology detection and mbind page-binding (Linux, feature = "numa").
pub mod numa;
/// `PinnedBuffer`: page-locked host memory registered with the CUDA driver.
pub mod pinned;

#[cfg(feature = "hugepages")]
pub use huge::HUGEPAGE_THRESHOLD;
#[cfg(feature = "mmap-fallback")]
pub use mmap_tier::MmapBuffer;
pub use numa::NumaConfig;
pub use pinned::{AllocKind, PinnedBuffer};

// ---------------------------------------------------------------------------
// BufferAccess — common interface for pinned and mmap buffers
// ---------------------------------------------------------------------------

/// Common access interface for host buffers, regardless of allocation strategy.
///
/// Implemented by [`PinnedBuffer`] (always pinned) and [`MmapBuffer`] (pageable).
/// Callers that need to handle both use `&dyn BufferAccess` or [`AnyBuffer`].
pub trait BufferAccess {
    /// Raw const pointer to the first byte.
    fn as_ptr(&self) -> *const u8;
    /// Length of the buffer in bytes.
    fn len(&self) -> usize;
    /// Returns `true` if the memory is page-locked for DMA.
    fn is_pinned(&self) -> bool;
    /// Read-only byte slice.
    fn as_slice(&self) -> &[u8];
    /// Mutable byte slice.
    fn as_mut_slice(&mut self) -> &mut [u8];
    /// Returns `true` if the buffer is zero-sized.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl BufferAccess for PinnedBuffer {
    #[inline]
    fn as_ptr(&self) -> *const u8 {
        PinnedBuffer::as_ptr(self)
    }
    #[inline]
    fn len(&self) -> usize {
        PinnedBuffer::len(self)
    }
    #[inline]
    fn is_pinned(&self) -> bool {
        PinnedBuffer::is_pinned(self)
    }
    fn as_slice(&self) -> &[u8] {
        PinnedBuffer::as_slice(self)
    }
    fn as_mut_slice(&mut self) -> &mut [u8] {
        PinnedBuffer::as_mut_slice(self)
    }
}

#[cfg(feature = "mmap-fallback")]
impl BufferAccess for MmapBuffer {
    #[inline]
    fn as_ptr(&self) -> *const u8 {
        MmapBuffer::as_ptr(self)
    }
    #[inline]
    fn len(&self) -> usize {
        MmapBuffer::len(self)
    }
    #[inline]
    fn is_pinned(&self) -> bool {
        false
    }
    fn as_slice(&self) -> &[u8] {
        MmapBuffer::as_slice(self)
    }
    fn as_mut_slice(&mut self) -> &mut [u8] {
        MmapBuffer::as_mut_slice(self)
    }
}

// ---------------------------------------------------------------------------
// AnyBuffer — unified buffer enum for callers that handle both tiers
// ---------------------------------------------------------------------------

/// Either a pinned [`PinnedBuffer`] or an mmap-backed [`MmapBuffer`].
///
/// Internal type; not exported in the frozen public API of `lib.rs`.
/// Use `BufferAccess` trait methods to access contents without branching.
#[cfg(feature = "mmap-fallback")]
pub enum AnyBuffer {
    /// Page-locked buffer registered with `cudaHostRegister`.
    Pinned(PinnedBuffer),
    /// mmap-backed pageable buffer; DMA requires a pinned staging copy.
    Mmap(MmapBuffer),
}

// SAFETY: Both PinnedBuffer and MmapBuffer are Send; AnyBuffer inherits that.
#[cfg(feature = "mmap-fallback")]
unsafe impl Send for AnyBuffer {}

#[cfg(feature = "mmap-fallback")]
impl AnyBuffer {
    /// Returns `true` if the underlying buffer is page-locked.
    #[inline]
    pub fn is_pinned(&self) -> bool {
        match self {
            AnyBuffer::Pinned(buf) => buf.is_pinned(),
            AnyBuffer::Mmap(_) => false,
        }
    }

    /// Length in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            AnyBuffer::Pinned(buf) => buf.len(),
            AnyBuffer::Mmap(buf) => buf.len(),
        }
    }

    /// Returns `true` if the buffer has zero length.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Raw const pointer to the first byte.
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        match self {
            AnyBuffer::Pinned(buf) => buf.as_ptr(),
            AnyBuffer::Mmap(buf) => buf.as_ptr(),
        }
    }

    /// Read-only byte slice.
    pub fn as_slice(&self) -> &[u8] {
        match self {
            AnyBuffer::Pinned(buf) => buf.as_slice(),
            AnyBuffer::Mmap(buf) => buf.as_slice(),
        }
    }

    /// Mutable byte slice.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        match self {
            AnyBuffer::Pinned(buf) => buf.as_mut_slice(),
            AnyBuffer::Mmap(buf) => buf.as_mut_slice(),
        }
    }
}

#[cfg(feature = "mmap-fallback")]
impl BufferAccess for AnyBuffer {
    #[inline]
    fn as_ptr(&self) -> *const u8 { AnyBuffer::as_ptr(self) }
    #[inline]
    fn len(&self) -> usize { AnyBuffer::len(self) }
    #[inline]
    fn is_pinned(&self) -> bool { AnyBuffer::is_pinned(self) }
    fn as_slice(&self) -> &[u8] { AnyBuffer::as_slice(self) }
    fn as_mut_slice(&mut self) -> &mut [u8] { AnyBuffer::as_mut_slice(self) }
}


