# AethelStream Module 1 (shard_engine) — Reference

## 1. Module Purpose

**shard_engine** is the quantized model shard preparation and runtime loading layer for AethelStream. It performs the initial pipeline stage: load a 7B–70B HuggingFace model from CPU, partition parameters by layer and role, quantize middle layers to NF4 (4-bit), write safetensors files with indexed metadata, and provide a zero-copy Rust loader for downstream consumers (Module 2).

The module bridges HuggingFace models (floating-point) and AethelStream's layered architecture by:
- **Ghost-loading** models on meta-device to verify they fit memory
- **Partitioning** parameters by semantic role (embedding, layer, expert, head, norm)
- **Quantizing** middle layers to NF4 and keeping edge/special layers at FP16
- **Writing** safetensors shards with inline metadata
- **Indexing** tensor locations and shapes for lazy mmap-based loading
- **Providing** Rust FFI bindings for C++ and Python to load tensors on-demand

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│ HuggingFace Model (e.g. meta-llama/Llama-2-7b)             │
└──────────────────────┬──────────────────────────────────────┘
                       │
       ┌───────────────┴───────────────┐
       │                               │
   [Python API]                   [Rust Runtime]
       │                               │
    ┌──▼────────────────┐       ┌──────▼──────────────┐
    │ Ghost Loader      │       │ Index Store        │
    │ Partitioner       │       │ ShardLoader        │
    │ Quantizer         │       │ NF4 Dequant        │
    │ Writer            │       │ PyShardLoader (FFI)│
    │ Verifier          │       └──────┬──────────────┘
    └──────┬────────────┘              │
           │ pipeline                   │ lazy mmap
           ▼                            ▼
    ┌──────────────────────────────────┐
    │ Shard Files (safetensors)        │
    │ + shard_index.json               │
    │ + layer_registry.json            │
    └──────┬───────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────┐
    │ Module 2 (ramflow)              │
    │ Loads tensors via Rust FFI      │
    │ Manages SSD↔RAM↔VRAM streaming  │
    └─────────────────────────────────┘
```

**Components:**

- **Python pipeline (`aethelStream/shard_engine/python/`)** — full model-to-shards workflow
  - `ghost_loader.py` — meta-device loading + memory audit
  - `partitioner.py` — parameter role classification + layer extraction
  - `quantizer.py` — NF4 quantization (vectorized) and fidelity checking
  - `writer.py` — shard file serialization + index generation
  - `verifier.py` — post-sharding correctness tests
  - `arch_registry.py` — 10 model family patterns for parameter extraction
  - `shard_engine.py` — CLI entry point (`shard` and `verify` commands)

- **Rust runtime (`aethelStream/shard_engine/src/`)** — loader + index
  - `lib.rs` — public API
  - `error.rs` — error types
  - `index.rs` — index structure and parsing
  - `loader.rs` — mmap-based shard loading with NF4 dequantization
  - `nf4.rs` — bitwise NF4 dequantization
  - `ffi.rs` — PyO3 bindings for Python access

---

## 3. Public Rust API

### lib.rs

```rust
pub mod error;
pub mod index;
pub mod loader;
pub mod nf4;

pub use error::{Result, ShardEngineError};
pub use index::{IndexStore, LayerRegistry, ShardIndex, TensorInfo};
pub use loader::{LayerBuffer, ShardLoader, TensorBuffer};

#[cfg(feature = "python-ffi")]
pub mod ffi;
#[cfg(feature = "python-ffi")]
pub use ffi::PyShardLoader;
```

**Result Type:** `pub type Result<T> = std::result::Result<T, ShardEngineError>;`

### error.rs

```rust
#[derive(Error, Debug)]
pub enum ShardEngineError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("layer index {0} not found in layer registry")]
    LayerNotFound(u32),

    #[error("parameter '{0}' not found in shard index")]
    ParamNotFound(String),

    #[error("malformed index entry '{0}': {1}")]
    MalformedIndex(String, String),
}
```

**Usage:** All functions return `Result<T>` which is `Result<T, ShardEngineError>`.

### index.rs

**TensorInfo** — metadata for a single tensor in a shard file.

```rust
pub struct TensorInfo {
    pub file_path: String,              // safetensors filename
    pub byte_offset: usize,             // absolute offset in file
    pub byte_length: usize,             // bytes (NF4: packed; FP16: raw)
    pub shape: Vec<usize>,              // original tensor shape
    pub dtype: String,                  // always "F16" after dequant
    pub precision: String,              // "fp16", "bf16", or "nf4"
    pub nf4_absmax_offset: Option<usize>,     // absmax array start (NF4 only)
    pub nf4_absmax_length: Option<usize>,     // absmax array byte length
    pub nf4_block_size: Option<usize>,        // block size for NF4
}
```

**ShardIndex** — type alias for parameter name → TensorInfo mapping.

```rust
pub type ShardIndex = HashMap<String, TensorInfo>;
```

**LayerRegistry** — type alias for layer index → shard filename mapping.

```rust
pub type LayerRegistry = HashMap<String, String>;
```

**IndexStore** — loads and queries shard metadata.

```rust
pub struct IndexStore {
    pub shard_index: ShardIndex,
    pub layer_registry: LayerRegistry,
    pub model_dir: PathBuf,
}

impl IndexStore {
    /// Load from model directory; expects shard_index.json and layer_registry.json.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> { ... }

    /// Get shard filename for layer_index; returns LayerNotFound if missing.
    pub fn shard_file_for_layer(&self, layer_index: u32) -> Result<&str> { ... }

    /// Get TensorInfo for param_name; returns ParamNotFound if missing.
    pub fn tensor_info(&self, param_name: &str) -> Result<&TensorInfo> { ... }
}
```

**Errors:** `LayerNotFound(u32)`, `ParamNotFound(String)`, `Json`, `Io`.

### loader.rs

**TensorBuffer** — FP16 tensor data.

```rust
pub struct TensorBuffer {
    pub data: Vec<u8>,      // FP16 bytes (2 bytes per element)
    pub shape: Vec<usize>,  // original shape
    pub dtype: String,      // always "F16"
    pub param_name: String,
}
```

**LayerBuffer** — raw safetensors bytes for entire layer shard.

```rust
pub struct LayerBuffer {
    pub data: Vec<u8>,      // safetensors bytes
    pub layer_index: u32,
    pub file_path: String,
}
```

**ShardLoader** — mmap-based lazy loader with NF4 dequantization.

```rust
pub struct ShardLoader {
    pub store: IndexStore,
    // mmap_cache private
}

impl ShardLoader {
    /// Create loader for model directory.
    /// Errors: Io, Json (from IndexStore::load)
    pub fn new(model_dir: impl AsRef<Path>) -> Result<Self> { ... }

    /// Load entire layer shard as raw safetensors bytes.
    /// Errors: LayerNotFound, Io (from mmap)
    pub fn load_layer(&mut self, layer_index: u32) -> Result<LayerBuffer> { ... }

    /// Load single parameter tensor as FP16 bytes.
    /// If precision is "nf4", dequantizes on-the-fly using absmax + packed data.
    /// If "fp16", returns copy of raw bytes.
    /// Errors: ParamNotFound, MalformedIndex (bad offsets/alignment),
    ///         errors from dequant_nf4_alloc
    pub fn load_param(&mut self, param_name: &str) -> Result<TensorBuffer> { ... }

    /// Private: get or open mmap for filename (cached).
    fn get_or_open_mmap(&mut self, filename: &str) -> Result<()> { ... }
}
```

**Safety Notes:**
- `load_param` with NF4 verifies bounds before slicing.
- Checks 4-byte alignment for f32 absmax array before unsafe slice conversion.
- Converts `Vec<f16>` to `Vec<u8>` via raw pointer (sound: f16 has no drop, same allocation).

### nf4.rs

**NF4_CODES** — 16-value codebook spanning [-1.0, 1.0].

```rust
pub const NF4_CODES: [f32; 16] = [
    -1.0, -0.6961928, -0.52507305, -0.3949175, -0.28444138, -0.18477343,
    -0.09105004, 0.0, 0.0795803, 0.1609302, 0.2461123, 0.33791524,
    0.44070983, 0.562617, 0.72295684, 1.0,
];
```

**dequant_nf4_into** — dequantize packed NF4 into pre-allocated f16 buffer.

```rust
pub fn dequant_nf4_into(
    packed: &[u8],               // 2 elements per byte (high/low nibbles)
    absmax: &[f32],              // scale per block
    output: &mut [half::f16],    // output buffer
    block_size: usize,
) -> Result<()> { ... }
```

**Packing format:** byte N contains elements 2N (high nibble) and 2N+1 (low nibble).

**Error handling:**
- Returns `MalformedIndex` if absmax or packed slices are undersized.
- Bounds checks before slicing; does not panic.

**dequant_nf4_alloc** — dequantize and allocate output buffer.

```rust
pub fn dequant_nf4_alloc(
    packed: &[u8],
    absmax: &[f32],
    block_size: usize,
) -> Result<Vec<half::f16>> { ... }
```

Returns vector of size `packed.len() * 2`.

### ffi.rs (python-ffi feature only)

**PyShardLoader** — PyO3 wrapper around ShardLoader.

```rust
#[pyclass]
pub struct PyShardLoader { ... }

#[pymethods]
impl PyShardLoader {
    /// Create from model directory.
    /// Raises PyIOError if load fails.
    #[new]
    fn new(model_dir: String) -> PyResult<Self> { ... }

    /// Load raw safetensors bytes for layer.
    /// Returns Python bytes object.
    /// Raises PyIOError.
    fn load_layer<'py>(&mut self, py: Python<'py>, layer_index: u32) 
        -> PyResult<&'py PyBytes> { ... }

    /// Load FP16 bytes for parameter.
    /// Raises PyKeyError if not found; PyIOError otherwise.
    fn load_param<'py>(&mut self, py: Python<'py>, param_name: &str) 
        -> PyResult<&'py PyBytes> { ... }

    /// Return parameter shape as Python list.
    fn param_shape(&self, param_name: &str) -> PyResult<Vec<usize>> { ... }

    /// Return precision string ("fp16", "nf4", etc).
    fn param_precision(&self, param_name: &str) -> PyResult<String> { ... }
}
```

**Module:** `#[pymodule] pub fn shard_engine(...)`

**Build:** Compile with `--features python-ffi` to activate.

---

## 4. Public Python API

### shard_engine.py

**CLI entry point.** Run with `python -m aethelStream.shard_engine <command> [args]`.

#### `shard <model_id> <output_dir> [--workers=N]`

Shards a HuggingFace model end-to-end:
1. Ghost-loads model to verify memory budget
2. Partitions parameters by semantic role
3. Quantizes middle layers to NF4
4. Writes safetensors shards
5. Verifies index integrity

**Arguments:**
- `model_id` — HuggingFace model ID or local path (e.g., `meta-llama/Llama-2-7b`)
- `output_dir` — directory for shard files + index JSON
- `--workers` — parallel workers for quantization (default 4; not yet used)

**Output:**
- Shard files (`layer_XXXX.safetensor`, `embed.safetensor`, etc.)
- `shard_index.json`
- `layer_registry.json`

**Returns:** 0 on success; 1 on error (ghost load fail, write error, verification fail).

**Example:**
```bash
python -m aethelStream.shard_engine shard meta-llama/Llama-2-7b /tmp/llama7b-shards
```

#### `verify <shard_dir>`

Verify shard integrity post-sharding.

**Arguments:**
- `shard_dir` — directory containing shard_index.json and shard files

**Returns:** 0 on success; 1 if shard_index.json missing, malformed, or verification fails.

**Example:**
```bash
python -m aethelStream.shard_engine verify /tmp/llama7b-shards
```

### ghost_loader.py

**ghost_load(model_id: str) → (model, config, MemStats)**

Load model on meta-device (zero allocation) to verify it fits memory.

**Returns:**
- `model` — PreTrainedModel with no actual weights (meta tensors only)
- `config` — PretrainedConfig (used for architecture inference)
- `MemStats` — dataclass with:
  - `vram_delta_mb: float` — GPU memory change (MB)
  - `ram_delta_mb: float` — system RAM change (MB)
  - `passed: bool` — True if delta < limits (50 MB VRAM, 200 MB RAM)
  - `message: str` — human-readable status or error

**Errors:** Raises HuggingFace exceptions if model not found or config invalid.

**Usage:** Called by `cmd_shard` to gate sharding pipeline.

### partitioner.py

**ShardSpec** — maps shard filename to list of (param_name, tensor) pairs.

```python
@dataclass
class ShardSpec:
    filename: str
    params: list[tuple[str, torch.Tensor]]
    layer_index: int | None = None
    precision: str = "fp16"
```

**partition(model: PreTrainedModel, model_type: str) → (shard_specs, num_layers)**

Partition model parameters into shard specs. Works on meta-device model (no weights loaded).

**Args:**
- `model` — meta-device PreTrainedModel
- `model_type` — model family (lowercase, e.g., "llama", "mistral")

**Returns:**
- `shard_specs: list[ShardSpec]` — list of specs with layer assignments and filenames
- `num_layers: int` — number of layers in model

**Parameter allocation:**
- Layer parameters → `layer_XXXX.safetensor`
- Expert parameters (MoE) → `expert_XXXX_YYY.safetensor`
- Router parameters (MoE) → `router_XXXX.safetensor`
- Embed parameters → `embed.safetensor`
- Head parameters → `lm_head.safetensor`
- Norm (non-layer) → `norm.safetensor`
- Misc (unclassified) → `misc.safetensor`

**Precision assignment:**
- FP16: edge layers (0–3, last 4), embed, head, norm, misc
- NF4: middle layers (4 to num_layers-5)

### quantizer.py

**NF4_CODES** — Python torch tensor, same 16-value codebook.

**BLOCK_SIZE** — default block size (64).

**Nf4Result** — dataclass holding quantization output.

```python
@dataclass
class Nf4Result:
    packed: torch.Tensor           # uint8, shape (ceil(N/2),)
    absmax: torch.Tensor           # float32, shape (num_blocks,)
    original_shape: list[int]
    block_size: int = 64
```

**quantize_nf4(tensor: torch.Tensor, block_size: int = 64) → Nf4Result**

Vectorized NF4 quantization (no Python element loops).

**Args:**
- `tensor` — any shape float tensor
- `block_size` — elements per block (default 64)

**Returns:** `Nf4Result` with packed indices, per-block scales, and metadata.

**Algorithm:**
1. Flatten and pad tensor to multiple of block_size
2. Reshape into blocks
3. Compute per-block absmax
4. Normalize each block
5. Find nearest NF4 code index for each element (vectorized via torch broadcasting)
6. Pack indices into uint8 (2 per byte: high nibble = even idx, low = odd)

**dequantize_nf4(result: Nf4Result) → torch.Tensor**

Reference dequantization (for testing, not used in production loading).

**check_fidelity(original: torch.Tensor, result: Nf4Result, threshold: float = 1e-3) → (passed: bool, mse: float)**

Compute MSE between original (FP16) and dequantized NF4. Used to decide fallback to FP16.

### writer.py

**ShardIndexEntry** — TypedDict for shard_index.json entries.

```python
class ShardIndexEntry(TypedDict, total=False):
    file_path: str
    byte_offset: int
    byte_length: int
    shape: list[int]
    dtype: str
    precision: str
    nf4_absmax_offset: int
    nf4_absmax_length: int
    nf4_block_size: int
```

**write_shards(model_id: str, output_dir: str | Path, *, num_proc: int = 4) → (shard_index, layer_registry)**

Full sharding pipeline: load model from HF Hub → partition → quantize → write → index.

**Args:**
- `model_id` — HuggingFace model ID
- `output_dir` — where to write shards + JSON indices
- `num_proc` — parallel workers (currently unused placeholder)

**Returns:**
- `shard_index: ShardIndex` — dict[param_name, ShardIndexEntry]
- `layer_registry: LayerRegistry` — dict[layer_key, filename]

**Side effects:**
- Writes shard_index.json and layer_registry.json to output_dir
- Streams one shard at a time to keep peak RAM bounded
- Deletes tensors from state_dict after writing

**Error handling:**
- Raises ValueError if shard files malformed after writing
- Raises HuggingFace exceptions if model not found

### verifier.py

**verify_zero_memory(stats: MemStats) → None**

Test 1: Assert ghost load consumed negligible memory.

**verify_completeness(shard_index, model_dir, expected_param_count) → None**

Test 2: Total element count across all shards equals original model.

**verify_nf4_fidelity(shard_index, model_dir, sample_count: int = 10, threshold: float = 1e-3) → None**

Test 3: Sample NF4 tensors, dequantize, check finiteness and bounded magnitude.

**verify_index_integrity(shard_index, model_dir) → None**

Test 5: Every shard_index entry has valid file, offsets within file bounds, correct shape.

**run_all_verifications(shard_index, model_dir, expected_param_count, mem_stats) → None**

Run all four tests in sequence.

### arch_registry.py

**ParamRole** — enum classifying parameter semantic role.

```python
class ParamRole(Enum):
    EMBED = "embed"
    HEAD = "head"
    NORM = "norm"
    LAYER = "layer"
    MOE_EXPERT = "moe_expert"
    MOE_ROUTER = "moe_router"
    MISC = "misc"
```

**ParamClassification** — dataclass output from classify_param.

```python
@dataclass
class ParamClassification:
    role: ParamRole
    layer_index: Optional[int] = None
    expert_index: Optional[int] = None
```

**ArchConfig** — regex patterns for parameter extraction.

```python
@dataclass
class ArchConfig:
    layer_pattern: str            # regex with capturing group for layer index
    expert_pattern: str = ""      # regex for MoE experts
    router_pattern: str = ""      # regex for MoE routers
    embed_patterns: list[str] = ...
    head_patterns: list[str] = ...
    norm_patterns: list[str] = ...
```

**ARCH_REGISTRY** — dict[model_type, ArchConfig] for 10 families:

```
llama, mistral, mixtral, phi, qwen2, gpt2, gpt_neox, bloom, opt, falcon
```

**get_arch_config(model_type: str) → ArchConfig**

Returns config for model_type; falls back to llama-style if unknown.

**classify_param(name: str, config: ArchConfig) → ParamClassification**

Extract layer index, expert index, and role from parameter name.

**Priority:** layer → expert → router → embed → head → norm → MISC.

**get_num_layers(model_config: object) → int**

Infer layer count from config. Tries `num_hidden_layers`, `n_layer`, `num_layers`.

Raises `ValueError` if none found.

---

## 5. Data Flow: HuggingFace to Shards to Loader

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. HuggingFace Model                                                 │
│    meta-llama/Llama-2-7b, mistralai/Mistral-7B, etc.                │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
       ┌───────────────┴───────────────┐
       │ ghost_load() + verify memory  │
       └───────────────┬───────────────┘
                       │
       ┌───────────────▼────────────────┐
       │ partition() → ShardSpec list   │
       │ - classify each parameter      │
       │ - assign filename + precision  │
       └───────────────┬────────────────┘
                       │
       ┌───────────────▼────────────────────────┐
       │ write_shards() → safetensors shards    │
       │ 1. load real weights                   │
       │ 2. for each shard:                     │
       │    - if NF4: quantize_nf4()            │
       │    - check_fidelity() → fallback if bad│
       │    - save to safetensors               │
       │ 3. parse offsets from safetensors hdr  │
       │ 4. build shard_index.json              │
       │ 5. build layer_registry.json           │
       └───────────────┬────────────────────────┘
                       │
       ┌───────────────▼────────────────────────┐
       │ verify_index_integrity()               │
       │ - check file existence                 │
       │ - check offset bounds                  │
       │ - check tensor shape                   │
       └───────────────┬────────────────────────┘
                       │
                      ▼
       ┌─────────────────────────────────────────┐
       │ Output Directory                        │
       │ ├─ layer_0000.safetensor                │
       │ ├─ layer_0001.safetensor                │
       │ ├─ ...                                  │
       │ ├─ embed.safetensor                     │
       │ ├─ lm_head.safetensor                   │
       │ ├─ norm.safetensor                      │
       │ ├─ shard_index.json                     │
       │ └─ layer_registry.json                  │
       └────────────────┬────────────────────────┘
                        │
       ┌────────────────▼─────────────────────┐
       │ Runtime: Rust ShardLoader            │
       │ 1. IndexStore::load(model_dir)       │
       │ 2. load_layer(layer_idx)             │
       │    → mmap file, return raw bytes     │
       │ 3. load_param(param_name)            │
       │    → mmap file, slice via offset     │
       │    → if NF4: dequant_nf4_into()      │
       │    → return FP16 bytes               │
       └────────────────┬─────────────────────┘
                        │
                       ▼
       ┌─────────────────────────────────┐
       │ Module 2 (ramflow)              │
       │ Consumes via PyShardLoader FFI  │
       └─────────────────────────────────┘
```

---

## 6. shard_index.json Format

**Type:** `dict[param_name: str, entry: ShardIndexEntry]`

**Example entry for FP16 parameter:**

```json
{
  "model.embed_tokens.weight": {
    "file_path": "embed.safetensor",
    "byte_offset": 256,
    "byte_length": 786432,
    "shape": [32000, 4096],
    "dtype": "F16",
    "precision": "fp16"
  }
}
```

**Example entry for NF4 parameter:**

```json
{
  "model.layers.15.self_attn.q_proj.weight": {
    "file_path": "layer_0015.safetensor",
    "byte_offset": 2048,
    "byte_length": 131072,
    "shape": [4096, 4096],
    "dtype": "U8",
    "precision": "nf4",
    "nf4_absmax_offset": 133120,
    "nf4_absmax_length": 1024,
    "nf4_block_size": 64
  }
}
```

**Field semantics:**

| Field | Type | Meaning |
|-------|------|---------|
| `file_path` | str | Safetensors filename (e.g., "layer_0015.safetensor") |
| `byte_offset` | int | Absolute byte offset within file where data starts |
| `byte_length` | int | Bytes of tensor data (NF4: packed; FP16: raw) |
| `shape` | list[int] | Original tensor shape (e.g., [4096, 4096]) |
| `dtype` | str | "F16" for FP16, "U8" for NF4 packed data |
| `precision` | str | "fp16", "bf16" (reserved), or "nf4" |
| `nf4_absmax_offset` | int (optional) | Byte offset of NF4 absmax array (NF4 only) |
| `nf4_absmax_length` | int (optional) | Byte length of absmax array (must be divisible by 4 for f32) |
| `nf4_block_size` | int (optional) | Number of elements per block (typically 64) |

**Notes:**
- For NF4 tensors, offsets point to the **packed** tensor stored in safetensors, and **nf4_absmax_offset** points to absmax stored as separate tensor.
- Byte offsets are absolute within the file (include safetensors header size).
- **dtype** indicates the **raw** data type in the file; after Rust loader dequantization, all tensors are FP16.

---

## 7. layer_registry.json Format

**Type:** `dict[layer_key: str, shard_filename: str]`

**Example:**

```json
{
  "0": "layer_0000.safetensor",
  "1": "layer_0001.safetensor",
  "2": "layer_0002.safetensor",
  "3": "layer_0003.safetensor",
  ...
  "31": "layer_0031.safetensor",
  "embed": "embed.safetensor",
  "head": "lm_head.safetensor",
  "norm": "norm.safetensor"
}
```

**Mapping:**

| Key | Value | Used for |
|-----|-------|----------|
| "0", "1", ... | layer_XXXX.safetensor | Layer blocks (e.g., self_attn, mlp) |
| "embed" | embed.safetensor | Token/position embeddings |
| "head" | lm_head.safetensor | Output projection (lm_head) |
| "norm" | norm.safetensor | Final layer norm |

**Purpose:** Maps layer semantic identity to physical shard file for quick lookup during inference.

---

## 8. NF4 Quantization

### Algorithm

**Codebook:** 16 normalized float values spanning [-1.0, 1.0]:

```
Index  Value
0      -1.0
1      -0.6961928
2      -0.52507305
3      -0.3949175
4      -0.28444138
5      -0.18477343
6      -0.09105004
7      0.0         (exact zero)
8      0.0795803
9      0.1609302
10     0.2461123
11     0.33791524
12     0.44070983
13     0.562617
14     0.72295684
15     1.0
```

**Quantization steps:**

1. **Block division:** Flatten tensor to 1D, pad to multiple of block_size (64), reshape to [num_blocks, 64].
2. **Per-block scaling:** Compute absmax = max(|block|) for each block (safe against zero by clamping to 1e-8).
3. **Normalization:** norm_block = block / absmax.
4. **Indexing:** For each element, find nearest NF4 code index (0–15) via argmin of distance.
5. **Packing:** Combine two 4-bit indices into one uint8 byte (high nibble = element 0, low = element 1).

**Inverse (reference):**

1. **Unpacking:** Extract high/low nibbles from packed bytes → indices.
2. **Lookup:** codes[index] → normalized value.
3. **Scaling:** result = value × absmax.

### Packing Format

One byte encodes two elements:

```
Byte = [high_nibble (4 bits) | low_nibble (4 bits)]
         Element 2N (idx 0)    Element 2N+1 (idx 1)

e.g. Byte 0x7F (binary 0111 1111)
     high=7 → code[7] = 0.0
     low=15 → code[15] = 1.0
     Encoded: [0.0, 1.0]
```

### Block Size

Default: **64 elements per block**. This gives:
- Dense per-block statistics (16 bits of precision in absmax)
- Small absmax array overhead (e.g., 4096×4096 → 4 blocks × 4 bytes = 16 bytes)

### Fidelity Check

After quantization, compare dequantized tensor vs. original (FP16) via MSE.

**Threshold:** 1e-3 (configurable in writer.py).

**Fallback:** If MSE > threshold, store parameter as FP16 instead of NF4.

### Compatibility with bitsandbytes

M1's NF4 codebook and packing format match **bitsandbytes exactly**. Any parameter quantized by M1 can be loaded by bitsandbytes and vice versa.

---

## 9. Precision Policy

**Default configuration:**

| Layer Type | Precision |
|------------|-----------|
| Embed (token, position) | FP16 |
| Norm (non-layer) | FP16 |
| Head (lm_head) | FP16 |
| Layer 0–3 (first 4) | FP16 |
| Layer 4 to (num_layers - 5) | NF4 |
| Layer (num_layers - 4) to end | FP16 |
| Expert (MoE) | NF4 (if layer is middle) |
| Router (MoE) | FP16 |

**Rationale:** Edge layers (first and last) and non-layer parameters are sensitive to quantization. Middle layers have more redundancy.

**Implemented in:** `partitioner.py::compute_precision()` and `quantizer.py::check_fidelity()`.

---

## 10. Architecture Registry

**Supported model families: 10**

#### llama
```
Pattern: model.layers.{layer_idx}.
Embed: model.embed_tokens.weight
Head: lm_head.weight
Norm: model.norm.weight
```

#### mistral
```
Same as llama.
```

#### mixtral
```
Pattern: model.layers.{layer_idx}.
Expert: model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.
Router: model.layers.{layer_idx}.block_sparse_moe.gate.
Embed: model.embed_tokens.weight
Head: lm_head.weight
Norm: model.norm.weight
```

#### phi
```
Pattern: model.layers.{layer_idx}.
Embed: model.embed_tokens.weight
Head: lm_head.weight
Norm: model.final_layernorm.weight, .bias
```

#### qwen2
```
Pattern: model.layers.{layer_idx}.
Embed: model.embed_tokens.weight
Head: lm_head.weight
Norm: model.norm.weight
```

#### gpt2
```
Pattern: transformer.h.{layer_idx}.
Embed: transformer.wte.weight, transformer.wpe.weight
Head: lm_head.weight
Norm: transformer.ln_f.weight, .bias
```

#### gpt_neox
```
Pattern: gpt_neox.layers.{layer_idx}.
Embed: gpt_neox.embed_in.weight
Head: embed_out.weight
Norm: gpt_neox.final_layer_norm.weight
```

#### bloom
```
Pattern: transformer.h.{layer_idx}.
Embed: word_embeddings.weight, word_embeddings_layernorm.weight
Head: lm_head.weight
Norm: ln_f.weight, .bias
```

#### opt
```
Pattern: model.decoder.layers.{layer_idx}.
Embed: model.decoder.embed_tokens.weight
Head: lm_head.weight
Norm: model.decoder.final_layer_norm.weight
```

#### falcon
```
Pattern: transformer.h.{layer_idx}.
Embed: transformer.word_embeddings.weight
Head: lm_head.weight
Norm: transformer.ln_f.weight, .bias
```

**Unknown models:** Fall back to llama-style patterns.

**Implementation:** `arch_registry.py` with regex-based extraction and priority matching (layer → expert → router → embed → head → norm → MISC).

---

## 11. Feature Flags

### python-ffi

**Enables:** PyO3 bindings for Python access to Rust loader.

**Build:**
```bash
cd aethelStream/shard_engine
maturin develop --features python-ffi
```

**Exports:** `PyShardLoader` class, importable as `import shard_engine; loader = shard_engine.PyShardLoader(model_dir)`.

**Without flag:** Rust crate can still be used directly in C++ or other Rust code; Python pipeline uses only the Python modules.

### mock-cuda

**Enables:** Mock CUDA runtime for testing without hardware.

**Current status:** Reserved for future use. Not implemented in v0.1.0.

---

## 12. Performance Characteristics

### Memory Usage

**Ghost load:** < 200 MB RAM, < 50 MB VRAM (meta-device tensors have no allocation).

**Sharding pipeline:**
- Loads one shard's parameters into RAM at a time.
- Each shard typically holds 1–10 parameters (e.g., 4096×4096 @ FP16 = 32 MB).
- Peak RAM: ~size of largest shard + safetensors write buffer (~100 MB per shard for 7B models).

**Rust loader:**
- mmap-based: zero-copy until dequantization.
- NF4 dequantization allocates temporary output buffer (2× packed size).
- Mmap cache grows with number of unique shard files (typically 30–40 for 7B–70B models).

### Speed

**Quantization:** Vectorized torch ops, ~100–200 MB/s on modern CPU.

**Loading (Rust):**
- FP16 mmap direct: nanosecond slice (zero copy).
- NF4 dequantization: ~1 GB/s (vectorized loop over blocks).
- First load touches disk; subsequent loads use OS page cache.

**File I/O:**
- Safetensors format: efficient (binary header, no per-tensor overhead).
- Shard file sizes: 50–500 MB (depends on layer size and quantization).

### Disk Space

**Compression ratio:**
- FP16: baseline (1.0×)
- NF4: ~0.25× (4 bits vs. 16 bits, plus ~0.5% absmax overhead)

**Example (Llama-2-7B):**
- Original (FP16): ~14 GB
- After quantization: ~4 GB (3× reduction; only middle layers quantized)

---

## 13. Integration with Module 2 (ramflow)

### Loading Interface

M2 consumes M1 shards via the Rust FFI:

```python
from shard_engine import PyShardLoader

loader = PyShardLoader(model_dir)
param_bytes = loader.load_param("model.layers.15.self_attn.q_proj.weight")  # FP16 bytes
shape = loader.param_shape("model.layers.15.self_attn.q_proj.weight")      # [4096, 4096]
```

M1 guarantees all returned bytes are **FP16** (NF4 is dequantized on-the-fly).

### Metadata Exchange

M2 reads JSON indices directly:

```python
import json
with open("shard_index.json") as f:
    shard_index = json.load(f)
with open("layer_registry.json") as f:
    layer_registry = json.load(f)
```

**TensorLocationDict** (M2 internal format) is populated from shard_index fields:
- `file_path` → physical shard file
- `byte_offset`, `byte_length` → data location
- `shape` → tensor shape
- `precision` → "fp16" or "nf4" (for M2 to decide dequant vs. direct load)

### Contract

- M1 produces safetensors shards + JSON indices
- M2 loads via PyShardLoader or direct JSON parsing
- All tensors arrive at M2 as FP16 (no dequant needed)
- M2 manages SSD ↔ RAM ↔ VRAM streaming from there

---

## 14. CLI Usage

### shard command

```bash
python -m aethelStream.shard_engine shard <model_id> <output_dir> [--workers N]
```

**Example:**

```bash
python -m aethelStream.shard_engine shard meta-llama/Llama-2-7b /data/llama7b
```

**Output:**

```
[M1] Ghost-loading meta-llama/Llama-2-7b...
[M1] VRAM Δ=12.34 MiB, RAM Δ=150.56 MiB
[M1] Sharding meta-llama/Llama-2-7b → /data/llama7b
[M1] Written 291 parameter entries
[M1] Test 1 PASS: VRAM Δ=12.34 MiB, RAM Δ=150.56 MiB
[M1] Test 2 PASS: 6738415616 elements verified across all shards
[M1] Test 3 PASS: 10 NF4 tensors verified: finite, bounded
[M1] Test 5 PASS: 291 index entries verified: files exist, offsets valid
```

**Return codes:**
- 0 = success
- 1 = ghost load failed, write error, or verification failed

### verify command

```bash
python -m aethelStream.shard_engine verify <shard_dir>
```

**Example:**

```bash
python -m aethelStream.shard_engine verify /data/llama7b
```

**Output:**

```
[M1] Test 1 PASS: VRAM Δ=0.00 MiB, RAM Δ=0.00 MiB
[M1] Test 2 PASS: 6738415616 elements verified across all shards
[M1] Test 3 PASS: 10 NF4 tensors verified: finite, bounded
[M1] Test 5 PASS: 291 index entries verified: files exist, offsets valid
```

**Return codes:**
- 0 = all verifications passed
- 1 = shard_index.json missing/malformed or verification failed

---

## 15. Test Coverage

### Python Tests

Located in `tests/` directory. Run with `pytest`.

#### test_ghost_load.py
**What:** Verifies ghost load memory audit.
**Coverage:** MemStats calculation, passed/failed thresholds.

#### test_completeness.py
**What:** Sums element counts across all shards; checks against original model.
**Coverage:** Partition correctness, tensor size tracking.

#### test_nf4_fidelity.py
**What:** Samples NF4 tensors, dequantizes, measures MSE vs. original FP16.
**Coverage:** Quantization quality, fallback-to-FP16 triggering.

#### test_index_integrity.py
**What:** Validates shard_index.json offsets against file sizes.
**Coverage:** JSON parsing, byte offset bounds checking.

#### test_rust_loader.py
**What:** Loads tensors via PyShardLoader FFI; compares vs. Python reference.
**Coverage:** Rust dequantization, FFI interface.

#### test_rust_roundtrip.py
**What:** Forward pass through model loaded via Rust loader; checks activation fidelity.
**Coverage:** End-to-end correctness (Test 4 in design doc).

### Rust Tests

Inline module tests in `src/nf4.rs`:

- `test_zero_at_index_7` — codebook correctness
- `test_roundtrip_accuracy` — quantize → dequantize round-trip
- `test_partial_block_handling` — edge case handling for incomplete blocks

### What's NOT Tested

- Actual hardware (CUDA) loading — assumes mock-cuda disabled
- Multi-layer MoE expert routing
- Very large models (>70B)
- Network errors during HF Hub download

---

## Appendix: Example Workflow

### Step 1: Shard a model

```bash
python -m aethelStream.shard_engine shard mistralai/Mistral-7B /tmp/mistral-shards
```

**Files created:**

```
/tmp/mistral-shards/
├── layer_0000.safetensor          (attention, MLP for layer 0)
├── layer_0001.safetensor
├── ...
├── layer_0031.safetensor
├── embed.safetensor               (token embeddings)
├── lm_head.safetensor             (output projection)
├── norm.safetensor                (final layer norm)
├── shard_index.json               (metadata for all tensors)
└── layer_registry.json            (layer → shard filename mapping)
```

### Step 2: Verify shards

```bash
python -m aethelStream.shard_engine verify /tmp/mistral-shards
```

All tests pass; shard files are ready for Module 2.

### Step 3: Load via Rust FFI in Python

```python
from shard_engine import PyShardLoader

loader = PyShardLoader("/tmp/mistral-shards")
q_proj_bytes = loader.load_param("model.layers.5.self_attn.q_proj.weight")
shape = loader.param_shape("model.layers.5.self_attn.q_proj.weight")
# q_proj_bytes is FP16 (if originally NF4, already dequantized)
# shape is [4096, 4096]
```

### Step 4: Module 2 takes over

M2 (ramflow) uses PyShardLoader to fetch tensors on-demand, managing the SSD → RAM → VRAM pipeline.

---

## Summary Table

| Aspect | Details |
|--------|---------|
| **Purpose** | Quantize & index HuggingFace models for AethelStream |
| **Input** | HF model ID (e.g., llama-2-7b) |
| **Output** | Safetensors shards + JSON indices |
| **Quantization** | NF4 middle layers; FP16 edges |
| **Codec** | bitsandbytes-compatible NF4 codebook |
| **Supported models** | 10 families (llama, mistral, mixtral, phi, qwen2, gpt2, gpt_neox, bloom, opt, falcon) |
| **Rust modules** | index, loader, nf4, error, ffi |
| **Python modules** | ghost_loader, partitioner, quantizer, writer, verifier, arch_registry |
| **Performance** | 100–200 MB/s quantization; ~1 GB/s dequant |
| **Space** | ~3–4× compression vs. FP16 |
| **Integration** | Via PyShardLoader FFI to Module 2 (ramflow) |

