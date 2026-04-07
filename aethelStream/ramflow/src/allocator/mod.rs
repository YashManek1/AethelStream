// src/allocator/mod.rs — pinned-memory allocator module

pub mod drop_guard;
pub mod pinned;

pub use pinned::PinnedBuffer;
