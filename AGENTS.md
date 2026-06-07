# AethelStream Agent Guide

AethelStream is a model sharding and efficient inference system that splits large language models across memory and device boundaries, combining Python orchestration with a Rust runtime for fast tensor loading and dequantization.

## Architecture Overview

### Three Core Layers

1. **Python Orchestration** (`shard_model.py`): Loads transformers via HuggingFace, discovers model layers by architecture type, applies selective NF4 quantization to middle layers, and outputs sharded safetensors files with a unified index.

2. **Sharding Metadata** (`model_shards/shard_index.json`): Maps parameter names → file location + byte offsets. This flat index enables random access to any parameter across shards without loading full files.

3. **Rust Runtime** (`rust_runtime/`): PyO3-based extension module. Reads shard_index.json, memory-maps safetensors files, and performs lazy dequantization of NF4-quantized tensors on-demand, returning FP16 bytes to Python.

### FastAPI Service (`app.py`)

- `POST /shard-model` accepts `model_name` (HuggingFace model ID), returns sharding result with index path
- `GET /` health check
- **Key dependency**: Calls `sharding.shard.shard_model()` not the root `shard_model.py`—note the separate, simpler implementation

## Critical Patterns & Conventions

### Model Layer Discovery (`discover_layers()`)

**Pattern**: Model type → layer prefix mapping. Different architectures name layers differently:

```python
LAYER_PREFIXES = {
    "llama": "model.layers",
    "gpt2": "transformer.h",
    "gpt_neox": "gpt_neox.layers",
}
```

**Why**: safetensors parameter names must be parsed to assign roles (embed, head, layer_N, norm). Add new architectures to this dict first, or dynamic layer detection fails.

**Key insight**: Layer indices extracted from parameter names determine quantization strategy—first 4 and last 4 stay FP16, middle layers get NF4.

### Precision Schedule

**Critical Contract**: Layers 0-3 and final 4 layers always remain **FP16** (full precision). Middle layers quantized to **NF4** (4-bit). This two-tier approach preserves gradient flow for early input processing and final token prediction.

**Test coverage**: See `rust_runtime/src/lib.rs` `test_precision_schedule()` for layer allocation validation.

### Quantization Flow

1. **Quantize** (`quantize_nf4()`): Pads tensor to block_size=64, finds per-block absmax, encodes as 4-bit NF4 indices, packs 2 indices per byte. Returns: (packed_bytes, absmax_scale, original_shape).

2. **Store**: Three artifacts per quantized param in safetensors:
   - `{name}.packed`: 4-bit indices (half size)
   - `{name}.absmax`: per-block scales
   - `{name}.shape`: metadata to reconstruct original dimensions

3. **Dequantize** (Rust): Unpacks bytes → indices → NF4_CODES lookup with per-block absmax → FP16. Tight loop in `lib.rs` lines 120-149 critical for latency.

### Index Building (`verify_shards()`)

After saving safetensors, the index is rebuilt by:
1. Reading safetensors header to extract tensor byte offsets
2. Computing absolute file offsets (8-byte length prefix + header + data offset)
3. Storing (start, end, dtype) for each tensor and quantization component
4. Validating total parameter count matches original model

**Why rebuilding?** safetensors header contains relative offsets; we need absolute file positions for mmap-based random access.

## Rust Runtime Specifics

### Memory-Mapped Loading (`ShardLoader::load_layer()`)

- Opens file, mmaps entire shard, returns slice without copy
- Unix systems: Issues `MADV_WILLNEED` hint for prefetch (lines 74-83)
- Returns `LayerBuffer` (Vec<u8> wrapper) as raw bytes—Python handles casting

### NF4_CODES Alignment

Rust and Python **must** use identical NF4 code table:
```rust
const NF4_CODES: [f32; 16] = [-1.0000, -0.6961928, ..., 1.0];
```
**Mismatch breaks dequantization**. Keep in sync across `shard_model.py` line 23 and `lib.rs` line 10.

### Building/Testing

```powershell
# Inside rust_runtime/
cargo build --release
# Or for Python integration (replaces .so in site-packages):
maturin develop --release
```

Run `Test 4` (`test_rust_loader.py`) after maturin build to validate round-trip accuracy.

## Testing Philosophy

Five distinct tests embedded in workflow:

| Test | Location | Purpose |
|------|----------|---------|
| 1 | `ghost_load()` | Verify empty model init uses <50MB VRAM |
| 2 | `verify_shards()` | Total params match original (strict equality) |
| 3 | `quantize_nf4()` loop | MSE ≤ 1e-3 per NF4 tensor; fallback to FP16 if violated |
| 4 | `test_rust_loader.py` | Rust dequantization max diff <1e-3 vs PyTorch |
| 5 | `verify_shards()` | Byte offset bounds check in shard files |

**Convention**: Tests log `[Test N Pass/Fail]` format. Failures halt execution. **Do not suppress test output.**

## Development Workflow

1. **Add model type**: Update `LAYER_PREFIXES` in `shard_model.py`
2. **Modify quantization**: Change `block_size` (64), NF4_CODES table, or MSE threshold (1e-3)
3. **Tune precision schedule**: Edit layer ranges (lines 235, 219 in `shard_model.py`)
4. **Rust changes**: Rebuild with `maturin develop --release` after edits to `lib.rs`
5. **Run full pipeline**: `python shard_model.py gpt2` → `/shard-model` endpoint → `test_rust_loader.py`

## Key Files Reference

| File | Purpose |
|------|---------|
| `shard_model.py` | Core orchestration: load → discover → quantize → index → verify |
| `rust_runtime/src/lib.rs` | PyO3 loader; NF4 dequant kernel; tests |
| `app.py` | FastAPI wrapper (delegates to simpler `sharding.shard`) |
| `sharding/shard.py` | Fallback simple sharding (no quantization) |
| `model_shards/shard_index.json` | Master metadata: param → file + byte ranges |

## Common Pitfalls

- **Forgetting Test 5 checks**: Stale offsets corrupt Rust loading. Always rebuild index after format changes.
- **NF4_CODES drift**: Table values are sacred. Any rounding changes break dequantization fidelity.
- **Layer prefix wrong**: Models silently miscategorize parameters → wrong quantization applied.
- **MSE threshold violation**: High-variance layers fallback to FP16 transparently—watch logs for `[Test 3 Alert]`.
- **Rust not compiled**: Test 4 gracefully skips but production fails silently. Always run `maturin develop`.

## Integration Points

- **HuggingFace Transformers**: Loads model, provides `AutoConfig`, `AutoModelForCausalLM`
- **safetensors**: Non-PyTorch format, memory-safe, faster load. Headers store serialized JSON of tensor metadata.
- **PyO3**: Exposes Rust `ShardLoader` to Python as `rust_runtime.PyShardLoader`
- **memmap2**: Cross-platform memory-mapping library (no POSIX-only code in main path)
- **half crate**: f16 ↔ f32 conversion for quantization precision

---

*Last updated: April 2026. For questions on sharding architecture, see `shard_model.py` function docstrings and test comments.*

