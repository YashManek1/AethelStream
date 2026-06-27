use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Information about a single tensor in a shard file.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TensorInfo {
    /// Path to the safetensor file containing this tensor.
    pub file_path: String,

    /// Byte offset within the file where the tensor data starts.
    pub byte_offset: usize,

    /// Length in bytes of the tensor data.
    pub byte_length: usize,

    /// Shape of the tensor.
    pub shape: Vec<usize>,

    /// Data type (always "F16" after dequantization).
    pub dtype: String,

    /// Precision: "fp16", "bf16", or "nf4".
    pub precision: String,

    /// Byte offset of NF4 absmax data (for NF4 tensors).
    pub nf4_absmax_offset: Option<usize>,

    /// Length in bytes of NF4 absmax data.
    pub nf4_absmax_length: Option<usize>,

    /// Block size for NF4 quantization.
    pub nf4_block_size: Option<usize>,
}

/// Mapping from parameter names to their tensor info.
pub type ShardIndex = HashMap<String, TensorInfo>;

/// Mapping from layer index to shard file name.
pub type LayerRegistry = HashMap<String, String>;

/// Index store for loading and querying tensor metadata.
pub struct IndexStore {
    /// Tensor index: param name -> tensor info.
    pub shard_index: ShardIndex,

    /// Layer registry: layer index -> shard file.
    pub layer_registry: LayerRegistry,

    /// Base directory for model files.
    pub model_dir: PathBuf,
}

impl IndexStore {
    /// Load index from model directory.
    ///
    /// Expects `shard_index.json` and `layer_registry.json` in the model directory.
    pub fn load(model_dir: impl AsRef<Path>) -> crate::error::Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();

        let index_path = model_dir.join("shard_index.json");
        let index_data = std::fs::read_to_string(&index_path)?;
        let shard_index: ShardIndex = serde_json::from_str(&index_data)?;

        let registry_path = model_dir.join("layer_registry.json");
        let registry_data = std::fs::read_to_string(&registry_path)?;
        let layer_registry: LayerRegistry = serde_json::from_str(&registry_data)?;

        Ok(IndexStore {
            shard_index,
            layer_registry,
            model_dir,
        })
    }

    /// Get the shard file name for a given layer index.
    pub fn shard_file_for_layer(&self, layer_index: u32) -> crate::error::Result<&str> {
        let key = layer_index.to_string();
        self.layer_registry
            .get(&key)
            .map(|s| s.as_str())
            .ok_or(crate::error::ShardEngineError::LayerNotFound(layer_index))
    }

    /// Get tensor info for a given parameter name.
    pub fn tensor_info(&self, param_name: &str) -> crate::error::Result<&TensorInfo> {
        self.shard_index
            .get(param_name)
            .ok_or_else(|| crate::error::ShardEngineError::ParamNotFound(param_name.to_owned()))
    }
}
