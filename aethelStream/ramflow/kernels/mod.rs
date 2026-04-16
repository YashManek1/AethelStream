// src/kernels/mod.rs — thin Rust wrappers around compiled CUDA kernels (Sprint 2)
//
// Sprint 2 exposes: check_overflow_fp16
//
// Usage example:
//   use ramflow::kernels;
//   use ramflow::cuda_bridge::stream::CudaStream;
//
//   let stream = CudaStream::new()?;
//   // grad_device_ptr is a *const u16 pointing to device FP16 data (or host in mock mode)
//   let has_overflow = kernels::check_overflow_fp16(grad_device_ptr, n_elements, &stream);
//   if has_overflow {
//       scale_table.update(layer_idx, n_total, overflow_count);
//   }
//
// The function lives in cuda_bridge::stream but is re-exported here so callers
// have a single, stable import path for all kernel wrappers.

pub use crate::cuda_bridge::stream::check_overflow_fp16;
