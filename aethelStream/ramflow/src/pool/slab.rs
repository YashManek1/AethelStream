// src/pool/slab.rs — TensorSlab: per-layer slab packing for small tensors

use std::collections::HashMap;

use crate::allocator::PinnedBuffer;
use crate::cuda_bridge::zero_copy::device_pointer_for_mapped_buffer;
use crate::pool::tensor_location::TensorLocationDict;
use crate::{RamFlowError, Result};

const SLAB_ALIGNMENT_BYTES: usize = 64;

/// A single mapped pinned allocation holding small tensors for one layer.
///
/// Tensor offsets are deterministic: tensors are sorted by name and each start
/// offset is aligned to 64 bytes.
pub struct TensorSlab {
    layer_idx: u32,
    backing: Option<PinnedBuffer>,
    offsets: HashMap<String, (usize, usize)>,
    total_bytes: usize,
}

// Safety: TensorSlab exposes raw sub-pointers, but it does not provide shared
// mutable references to the backing buffer. Users must synchronize I/O/GPU
// access around those raw pointers.
unsafe impl Send for TensorSlab {}
unsafe impl Sync for TensorSlab {}

impl TensorSlab {
    /// Build a slab for all sub-threshold tensors in `layer_idx`.
    ///
    /// # Errors
    /// Returns [`RamFlowError::AllocationFailed`] if the mapped backing
    /// allocation fails.
    pub fn build_for_layer(
        layer_idx: u32,
        dict: &TensorLocationDict,
        threshold: usize,
    ) -> Result<Self> {
        let mut tensors = dict
            .tensors_for_layer(layer_idx)
            .filter(|tensor_info| tensor_info.byte_length < threshold)
            .collect::<Vec<_>>();
        tensors.sort_by(|left, right| left.name.cmp(&right.name));

        let mut offsets = HashMap::with_capacity(tensors.len());
        let mut cursor = 0usize;
        for tensor_info in tensors {
            cursor = align_up(cursor, SLAB_ALIGNMENT_BYTES);
            offsets.insert(tensor_info.name.clone(), (cursor, tensor_info.byte_length));
            cursor = cursor.saturating_add(tensor_info.byte_length);
        }

        let total_bytes = align_up(cursor, SLAB_ALIGNMENT_BYTES);
        let backing = if total_bytes == 0 {
            None
        } else {
            Some(PinnedBuffer::alloc_mapped(total_bytes)?)
        };

        Ok(TensorSlab {
            layer_idx,
            backing,
            offsets,
            total_bytes,
        })
    }

    /// Raw pointer within the slab backing for tensor `name`.
    ///
    /// # Errors
    /// Returns [`RamFlowError::TensorNotFound`] if `name` is not packed in this
    /// layer's slab.
    pub fn ptr_for(&self, name: &str) -> Result<*mut u8> {
        let (offset, _length) =
            self.offsets
                .get(name)
                .copied()
                .ok_or_else(|| RamFlowError::TensorNotFound {
                    layer_idx: self.layer_idx,
                    name: name.to_owned(),
                })?;
        let backing = self
            .backing
            .as_ref()
            .ok_or_else(|| RamFlowError::TensorNotFound {
                layer_idx: self.layer_idx,
                name: name.to_owned(),
            })?;
        Ok((backing.as_ptr() as *mut u8).wrapping_add(offset))
    }

    /// GPU-side device pointer for tensor `name`.
    ///
    /// # Errors
    /// Returns [`RamFlowError::TensorNotFound`] if `name` is absent, or
    /// [`RamFlowError::CudaError`] if CUDA rejects the mapped pointer lookup.
    pub fn device_ptr_for(&self, name: &str) -> Result<*mut u8> {
        let (offset, _length) =
            self.offsets
                .get(name)
                .copied()
                .ok_or_else(|| RamFlowError::TensorNotFound {
                    layer_idx: self.layer_idx,
                    name: name.to_owned(),
                })?;
        let backing = self
            .backing
            .as_ref()
            .ok_or_else(|| RamFlowError::TensorNotFound {
                layer_idx: self.layer_idx,
                name: name.to_owned(),
            })?;
        Ok(device_pointer_for_mapped_buffer(backing)?
            .as_ptr()
            .wrapping_add(offset))
    }

    /// Offset and byte length for `name`.
    pub fn offset_for(&self, name: &str) -> Result<(usize, usize)> {
        self.offsets
            .get(name)
            .copied()
            .ok_or_else(|| RamFlowError::TensorNotFound {
                layer_idx: self.layer_idx,
                name: name.to_owned(),
            })
    }

    /// Backing buffer used by io_uring reads.
    pub fn backing(&self) -> Option<&PinnedBuffer> {
        self.backing.as_ref()
    }

    /// Total size in bytes of the backing allocation.
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Number of tensors packed into this slab.
    pub fn tensor_count(&self) -> usize {
        self.offsets.len()
    }
}

fn align_up(value: usize, alignment: usize) -> usize {
    let remainder = value % alignment;
    if remainder == 0 {
        value
    } else {
        value + (alignment - remainder)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::pool::TensorLocationDict;

    #[test]
    fn slab_offsets_are_sorted_and_aligned() {
        let json = br#"{
            "layers": [
                {"index":0,"name":"z","path":"layer_0000.safetensors","byte_offset":0,"byte_length":31,"shape":[31],"dtype":"u8"},
                {"index":0,"name":"a","path":"layer_0000.safetensors","byte_offset":512,"byte_length":65,"shape":[65],"dtype":"u8"},
                {"index":0,"name":"large","path":"layer_0000.safetensors","byte_offset":1024,"byte_length":8192,"shape":[8192],"dtype":"u8"}
            ]
        }"#;
        let dict = TensorLocationDict::from_json_bytes(json, None).expect("dict");
        let slab = TensorSlab::build_for_layer(0, &dict, 4096).expect("slab");
        assert_eq!(slab.tensor_count(), 2);
        assert_eq!(slab.offset_for("a").expect("a").0, 0);
        assert_eq!(slab.offset_for("z").expect("z").0 % 64, 0);
        assert!(matches!(
            slab.ptr_for("missing"),
            Err(RamFlowError::TensorNotFound { .. })
        ));
    }
}
