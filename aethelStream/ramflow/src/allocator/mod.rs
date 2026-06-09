// src/allocator/mod.rs — pinned-memory allocator module

/// RAII guard that frees a pinned allocation on drop even during panic.
pub mod drop_guard;
/// `PinnedBuffer`: page-locked host memory registered with the CUDA driver.
pub mod pinned;

pub use pinned::PinnedBuffer;
