// src/pool/tensor_location.rs — TensorLocationDict and TensorInfo
//
// Populated by Module 1 from shard_index.json.  Every other module queries
// this to find out where a tensor lives on disk and what its shape is.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::{RamFlowError, Result};

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
    /// xxHash3-64 digest of this tensor's raw bytes on disk.
    ///
    /// `None` when `shard_index.json` does not include a `"xxh3"` field for this
    /// tensor.  Populated by the `checksum_shard` binary or Module 1 at shard time.
    /// Only verified when the `checksums` feature is active.
    pub xxhash3: Option<u64>,
}

/// Central registry of every tensor in every layer, loaded from
/// `shard_index.json` produced by Module 1.
///
/// # Sprint 0 contract
/// Struct compiles and can be constructed; iterator methods `unimplemented!`.
pub struct TensorLocationDict {
    /// (layer_index, tensor_name) → TensorInfo
    inner: HashMap<(u32, String), TensorInfo>,
}

impl TensorLocationDict {
    /// Load from a `shard_index.json` at `path`.
    ///
    /// # Errors
    /// Returns [`RamFlowError::ConfigError`] when the file cannot be read,
    /// parsed, or contains duplicate `(layer_index, tensor_name)` entries.
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|read_error| {
            RamFlowError::ConfigError(format!(
                "failed to read shard index {}: {read_error}",
                path.display()
            ))
        })?;
        Self::from_json_bytes(&bytes, path.parent())
    }

    /// Construct an empty dict (useful for testing without a real shard file).
    pub fn empty() -> Self {
        TensorLocationDict {
            inner: HashMap::new(),
        }
    }

    /// Iterate over all tensors belonging to `layer_idx`.
    pub fn tensors_for_layer(&self, layer_idx: u32) -> std::vec::IntoIter<&TensorInfo> {
        let mut tensors = self
            .inner
            .iter()
            .filter_map(|((candidate_layer_index, _tensor_name), tensor_info)| {
                if *candidate_layer_index == layer_idx {
                    Some(tensor_info)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        tensors.sort_by(|left, right| left.name.cmp(&right.name));
        tensors.into_iter()
    }

    /// Look up a specific `(layer_idx, name)` pair.
    pub fn get(&self, layer_idx: u32, name: &str) -> Option<&TensorInfo> {
        self.inner.get(&(layer_idx, name.to_owned()))
    }

    /// Total layer count in this model.
    pub fn num_layers(&self) -> usize {
        self.inner
            .keys()
            .map(|(layer_index, _tensor_name)| *layer_index)
            .collect::<std::collections::HashSet<_>>()
            .len()
    }

    /// Parse a `shard_index.json` payload from memory.
    ///
    /// Relative tensor paths are resolved against `base_dir` when provided.
    ///
    /// # Errors
    /// Returns [`RamFlowError::ConfigError`] for malformed JSON, duplicate
    /// tensor entries, or entries with missing tensor names.
    pub fn from_json_bytes(bytes: &[u8], base_dir: Option<&Path>) -> Result<Self> {
        let shard_index: ShardIndex = serde_json::from_slice(bytes).map_err(|parse_error| {
            RamFlowError::ConfigError(format!("failed to parse shard_index.json: {parse_error}"))
        })?;

        let mut inner = HashMap::new();
        for layer_entry in shard_index.layers {
            match layer_entry {
                LayerEntry::Flat(flat_tensor) => {
                    insert_tensor(&mut inner, flat_tensor.index, flat_tensor, base_dir)?;
                }
                LayerEntry::Grouped(grouped_layer) => {
                    for tensor_entry in grouped_layer.tensors {
                        let flattened = FlatTensorEntry {
                            index: grouped_layer.index,
                            name: tensor_entry.name,
                            path: tensor_entry.path,
                            byte_offset: tensor_entry.byte_offset,
                            byte_length: tensor_entry.byte_length,
                            shape: tensor_entry.shape,
                            dtype: tensor_entry.dtype,
                            xxhash3: tensor_entry.xxhash3,
                        };
                        insert_tensor(&mut inner, grouped_layer.index, flattened, base_dir)?;
                    }
                }
                LayerEntry::Mapped(mapped_layer) => {
                    for (tensor_name, tensor_entry) in mapped_layer.tensors {
                        let flattened = FlatTensorEntry {
                            index: mapped_layer.index,
                            name: Some(tensor_entry.name.unwrap_or(tensor_name)),
                            path: tensor_entry.path,
                            byte_offset: tensor_entry.byte_offset,
                            byte_length: tensor_entry.byte_length,
                            shape: tensor_entry.shape,
                            dtype: tensor_entry.dtype,
                            xxhash3: tensor_entry.xxhash3,
                        };
                        insert_tensor(&mut inner, mapped_layer.index, flattened, base_dir)?;
                    }
                }
            }
        }

        Ok(TensorLocationDict { inner })
    }
}

#[derive(Debug, Deserialize)]
struct ShardIndex {
    layers: Vec<LayerEntry>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum LayerEntry {
    Mapped(MappedLayerEntry),
    Grouped(GroupedLayerEntry),
    Flat(FlatTensorEntry),
}

#[derive(Debug, Deserialize)]
struct GroupedLayerEntry {
    index: u32,
    tensors: Vec<TensorEntry>,
}

#[derive(Debug, Deserialize)]
struct MappedLayerEntry {
    index: u32,
    tensors: HashMap<String, TensorEntry>,
}

#[derive(Debug, Deserialize)]
struct TensorEntry {
    #[serde(alias = "tensor_name")]
    name: Option<String>,
    path: PathBuf,
    byte_offset: u64,
    byte_length: usize,
    shape: Vec<usize>,
    dtype: String,
    #[serde(rename = "xxh3", default)]
    xxhash3: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct FlatTensorEntry {
    index: u32,
    #[serde(alias = "tensor_name")]
    name: Option<String>,
    path: PathBuf,
    byte_offset: u64,
    byte_length: usize,
    shape: Vec<usize>,
    dtype: String,
    #[serde(rename = "xxh3", default)]
    xxhash3: Option<u64>,
}

fn insert_tensor(
    inner: &mut HashMap<(u32, String), TensorInfo>,
    layer_index: u32,
    entry: FlatTensorEntry,
    base_dir: Option<&Path>,
) -> Result<()> {
    let tensor_name = match entry.name {
        Some(name) if !name.is_empty() => name,
        _ => entry
            .path
            .file_stem()
            .and_then(|file_stem| file_stem.to_str())
            .filter(|file_stem| !file_stem.is_empty())
            .map(str::to_owned)
            .ok_or_else(|| {
                RamFlowError::ConfigError(format!(
                    "layer {layer_index} tensor entry is missing a non-empty name"
                ))
            })?,
    };

    let path = if entry.path.is_relative() {
        match base_dir {
            Some(base_dir) => base_dir.join(entry.path),
            None => entry.path,
        }
    } else {
        entry.path
    };

    let key = (layer_index, tensor_name.clone());
    let previous = inner.insert(
        key,
        TensorInfo {
            name: tensor_name.clone(),
            path,
            byte_offset: entry.byte_offset,
            byte_length: entry.byte_length,
            shape: entry.shape,
            dtype: entry.dtype,
            xxhash3: entry.xxhash3,
        },
    );
    if previous.is_some() {
        return Err(RamFlowError::ConfigError(format!(
            "duplicate tensor entry for layer {layer_index}, tensor {tensor_name}"
        )));
    }

    Ok(())
}

/// Physical location of a tensor's data at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorLocation {
    /// Resident in pinned host (CPU-side) memory.
    PinnedHost,
    /// Mapped into GPU VRAM.
    GpuVram {
        /// CUDA device index (0-indexed).
        device: u32,
    },
    /// Spilled to NVMe storage.
    NvmeSpill,
    /// Partially on host, partially on NVMe (transitional).
    Transitioning,
}
