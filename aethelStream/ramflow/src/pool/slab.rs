// src/pool/slab.rs — TensorSlab: per-layer slab packing for small tensors
//
// Algorithm 5 from the spec. All small tensors belonging to one layer
// (LayerNorm weights/biases, LoRA A/B, optimizer state fragments) are packed
// into a SINGLE pinned allocation with an offset table.
//
// This reduces pool ring buffer pressure from 4 claims per layer to 1 claim
// per layer for the entire small-tensor category, and allows all of them to
// share a single cudaHostGetDevicePointer call.
//
// Sprint 0: all types declared, all methods unimplemented!.

use std::collections::HashMap;
use crate::pool::tensor_location::TensorLocationDict;

/// A single pinned allocation holding all small tensors for one layer.
///
/// The backing buffer is registered with `cudaHostRegisterMapped` so that all
/// tensors within the slab can be accessed by the GPU via UVA without a copy.
///
/// # Sprint 0 contract
/// Compiles; all methods `unimplemented!`.
pub struct TensorSlab {
    /// The single contiguous pinned backing allocation.
    _backing: (),
    /// Maps tensor name → `(byte_offset_within_slab, byte_length)`.
    ///
    /// Pre-computed at startup; zero-cost lookup during training (one index into
    /// a `HashMap`, then a pointer add).
    _offsets: HashMap<String, (usize, usize)>,
}

impl TensorSlab {
    /// Build a slab for all small tensors in `layer_idx`.
    ///
    /// Tensors with `byte_length >= threshold` are excluded (they go through
    /// the DMA copy path instead).  Each tensor is padded to 64-byte alignment
    /// within the slab so every sub-buffer starts on a CPU cache-line boundary.
    ///
    /// `threshold` defaults to `ZERO_COPY_THRESHOLD_BYTES` (4 MiB on PCIe Gen4);
    /// the warm-up profiler writes the actual value for this machine to
    /// `hardware_profile.json`, which is read here at startup.
    #[allow(unused_variables)]
    pub fn build_for_layer(
        layer_idx: u32,
        dict: &TensorLocationDict,
        threshold: usize,
    ) -> crate::Result<Self> {
        unimplemented!("TensorSlab::build_for_layer — Sprint 0 stub")
    }

    /// Raw pointer (within the backing allocation) for the tensor named `name`.
    ///
    /// This is the pointer the io_uring engine writes into, and the pointer
    /// the GPU kernel reads from via UVA.
    #[allow(unused_variables)]
    pub fn ptr_for(&self, name: &str) -> *mut u8 {
        unimplemented!("TensorSlab::ptr_for — Sprint 0 stub")
    }

    /// GPU-side device pointer for `name`, obtained via
    /// `cudaHostGetDevicePointer`.
    ///
    /// Only valid if the backing buffer was registered with
    /// `cudaHostRegisterMapped`.  Panics in mock-cuda mode (Sprint 0 stub).
    #[allow(unused_variables)]
    pub fn device_ptr_for(&self, name: &str) -> *mut u8 {
        unimplemented!("TensorSlab::device_ptr_for — Sprint 0 stub")
    }

    /// Total size in bytes of the backing allocation.
    pub fn total_bytes(&self) -> usize {
        unimplemented!("TensorSlab::total_bytes — Sprint 0 stub")
    }

    /// Number of tensors packed into this slab.
    pub fn tensor_count(&self) -> usize {
        unimplemented!("TensorSlab::tensor_count — Sprint 0 stub")
    }
}
