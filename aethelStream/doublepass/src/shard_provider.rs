//! ShardEngineProvider — WeightProvider backed by M1 [`shard_engine::ShardLoader`].
//!
//! Maps M5 BlockWeights field names to LLaMA-style parameter key strings,
//! loads each tensor as FP16 bytes from the shard index, and converts to f32.

use std::collections::HashMap;
use std::path::Path;

use crate::forward::{BlockConfig, BlockWeights};
use crate::weight_provider::{ProviderError, ProviderResult, WeightProvider};

/// Maps M5 [`BlockWeights`] field names to M1 shard parameter key suffixes.
///
/// The full key is assembled as `{layer_prefix}.{layer_idx}.{suffix}`,
/// e.g. `model.layers.0.self_attn.q_proj.weight`.
pub struct ParamNameMap {
    field_to_suffix: HashMap<&'static str, &'static str>,
    layer_prefix: String,
}

impl ParamNameMap {
    /// Standard LLaMA-family mapping (LLaMA-2 / LLaMA-3 / Mistral layout).
    pub fn llama() -> Self {
        let mut m = HashMap::new();
        m.insert("rms1_w", "input_layernorm.weight");
        m.insert("wq", "self_attn.q_proj.weight");
        m.insert("wk", "self_attn.k_proj.weight");
        m.insert("wv", "self_attn.v_proj.weight");
        m.insert("wo", "self_attn.o_proj.weight");
        m.insert("rms2_w", "post_attention_layernorm.weight");
        m.insert("wg", "mlp.gate_proj.weight");
        m.insert("wu", "mlp.up_proj.weight");
        m.insert("wd", "mlp.down_proj.weight");
        Self {
            field_to_suffix: m,
            layer_prefix: "model.layers".to_string(),
        }
    }

    /// Build the full shard param key for `(layer_idx, field_name)`.
    ///
    /// Returns `None` if `field_name` is not in the map.
    pub fn param_key(&self, layer_idx: u32, field: &str) -> Option<String> {
        let suffix = self.field_to_suffix.get(field)?;
        Some(format!("{}.{}.{}", self.layer_prefix, layer_idx, suffix))
    }
}

/// Convert raw little-endian FP16 bytes to `Vec<f32>`.
fn fp16_bytes_to_f32(data: &[u8]) -> ProviderResult<Vec<f32>> {
    if !data.len().is_multiple_of(2) {
        return Err(ProviderError::ConversionError(format!(
            "FP16 data length {} is not even",
            data.len()
        )));
    }
    Ok(data
        .chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            half::f16::from_bits(bits).to_f32()
        })
        .collect())
}

/// [`WeightProvider`] backed by M1's [`shard_engine::ShardLoader`].
///
/// Loads each parameter tensor (FP16, post-dequantisation) from the shard
/// index and converts it to f32 for M5's math kernels.
pub struct ShardEngineProvider {
    loader: shard_engine::ShardLoader,
    param_map: ParamNameMap,
    num_layers: u32,
    d_model: usize,
    d_ff: usize,
    n_heads: usize,
}

impl ShardEngineProvider {
    /// Create a provider for the model at `model_dir` with the given architecture dims.
    pub fn new(
        model_dir: impl AsRef<Path>,
        param_map: ParamNameMap,
        num_layers: u32,
        d_model: usize,
        d_ff: usize,
        n_heads: usize,
    ) -> ProviderResult<Self> {
        let loader = shard_engine::ShardLoader::new(model_dir)
            .map_err(|e| ProviderError::IoError(e.to_string()))?;
        Ok(Self {
            loader,
            param_map,
            num_layers,
            d_model,
            d_ff,
            n_heads,
        })
    }
}

impl WeightProvider for ShardEngineProvider {
    fn load_layer_weights(
        &mut self,
        layer_idx: u32,
        _cfg: &BlockConfig,
    ) -> ProviderResult<BlockWeights> {
        let mut load = |field: &str| -> ProviderResult<Vec<f32>> {
            let key = self
                .param_map
                .param_key(layer_idx, field)
                .ok_or_else(|| ProviderError::NotFound(field.to_string()))?;
            let buf = self
                .loader
                .load_param(&key)
                .map_err(|e| ProviderError::IoError(e.to_string()))?;
            fp16_bytes_to_f32(&buf.data)
        };
        Ok(BlockWeights {
            rms1_w: load("rms1_w")?,
            wq: load("wq")?,
            wk: load("wk")?,
            wv: load("wv")?,
            wo: load("wo")?,
            rms2_w: load("rms2_w")?,
            wg: load("wg")?,
            wu: load("wu")?,
            wd: load("wd")?,
        })
    }

    fn num_layers(&self) -> u32 {
        self.num_layers
    }

    fn d_model(&self) -> usize {
        self.d_model
    }

    fn d_ff(&self) -> usize {
        self.d_ff
    }

    fn n_heads(&self) -> usize {
        self.n_heads
    }
}

