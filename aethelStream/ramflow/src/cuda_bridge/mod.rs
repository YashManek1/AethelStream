// src/cuda_bridge/mod.rs — CUDA runtime bridge module

/// Raw FFI declarations and mock-cuda no-op substitutes for CUDA runtime functions.
pub mod bindings;
/// `CudaStream` RAII wrapper and FP16 overflow-check kernel interface.
pub mod stream;
/// `ZeroCopyRouter`: chooses DMA vs. UVA zero-copy based on tensor size.
pub mod zero_copy;

// Re-export the CUDA stream handle for ergonomic use
pub use stream::CudaStream;

// Re-export the CUDA binding helpers so crate internals can write:
//   use crate::cuda_bridge::{cuda_host_register, cuda_host_unregister};
//   use crate::cuda_bridge::bindings::{CUDA_HOST_REGISTER_DEFAULT, CUDA_HOST_REGISTER_MAPPED};
pub use bindings::{
    cuda_host_register, cuda_host_unregister, CUDA_HOST_REGISTER_DEFAULT, CUDA_HOST_REGISTER_MAPPED,
};
