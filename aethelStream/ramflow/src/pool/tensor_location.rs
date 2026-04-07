// src/pool/tensor_location.rs — TensorLocationDict and TensorInfo
//
// Populated by Module 1 from shard_index.json.  Every other module queries
// this to find out where a tensor lives on disk and what its shape is.

use std::path::PathBuf;
use std::collections::HashMap;

/// Shape, dtype, and on-disk location for a single tensor.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Name of the tensor (e.g. `"q_proj"`, `"layernorm_weight"`).
    pub name: String,
    /// Absolute path to the shard file containing this tensor.
    pub path: PathBuf,
    /// Byte offset within the shard file where the tensor starts.
    /// Must be a multiple of 512 (O_DIRECT alignment requirement).
    pub byte_offset: u64,
    /// Length of the tensor in bytes.
    pub byte_length: usize,
    /// Tensor shape as a list of dimensions.
    pub shape: Vec<usize>,
    /// Data type string (e.g. `"f16"`, `"bf16"`, `"i8"`).
    pub dtype: String,
}

/// Central registry of every tensor in every layer, loaded from
/// `shard_index.json` produced by Module 1.
///
/// # Sprint 0 contract
/// Struct compiles and can be constructed; iterator methods `unimplemented!`.
pub struct TensorLocationDict {
    /// (layer_index, tensor_name) → TensorInfo
    _inner: HashMap<(u32, String), TensorInfo>,
}

impl TensorLocationDict {
    /// Load from a `shard_index.json` at `path`.
    #[allow(unused_variables)]
    pub fn load(path: &std::path::Path) -> crate::Result<Self> {
        unimplemented!("TensorLocationDict::load — Sprint 0 stub")
    }

    /// Construct an empty dict (useful for testing without a real shard file).
    pub fn empty() -> Self {
        TensorLocationDict { _inner: HashMap::new() }
    }

    /// Iterate over all tensors belonging to `layer_idx`.
    #[allow(unused_variables)]
    pub fn tensors_for_layer(&self, layer_idx: u32) -> std::vec::IntoIter<&TensorInfo> {
        unimplemented!("TensorLocationDict::tensors_for_layer — Sprint 0 stub")
    }

    /// Look up a specific `(layer_idx, name)` pair.
    #[allow(unused_variables)]
    pub fn get(&self, layer_idx: u32, name: &str) -> Option<&TensorInfo> {
        unimplemented!("TensorLocationDict::get — Sprint 0 stub")
    }

    /// Total layer count in this model.
    pub fn num_layers(&self) -> usize {
        unimplemented!("TensorLocationDict::num_layers — Sprint 0 stub")
    }
}

/// Physical location of a tensor's data at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorLocation {
    /// Resident in pinned host (CPU-side) memory.
    PinnedHost,
    /// Mapped into GPU VRAM.
    GpuVram { device: u32 },
    /// Spilled to NVMe storage.
    NvmeSpill,
    /// Partially on host, partially on NVMe (transitional).
    Transitioning,
}
