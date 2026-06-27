"""
Shared test fixtures for shard_engine tests.
Uses a tiny synthetic model (2 layers, small dims) to avoid downloading real checkpoints.
GPT-2 tests are marked @pytest.mark.slow and skipped by default.
"""
from __future__ import annotations
import json
from pathlib import Path
import pytest
import torch
from safetensors.torch import save_file

# Tiny synthetic model params for fast unit tests
TINY_NUM_LAYERS = 6
TINY_DIM = 64
TINY_FFN_DIM = 128

def _make_tiny_fp16_shard(output_dir: Path, layer_index: int) -> dict:
    """Write a tiny FP16 shard and return shard_index entries for it."""
    filename = f"layer_{layer_index:04d}.safetensor"
    param_name = f"model.layers.{layer_index}.self_attn.q_proj.weight"
    tensor = torch.randn(TINY_DIM, TINY_DIM, dtype=torch.float16)
    save_file({param_name: tensor}, str(output_dir / filename))
    import struct
    with open(output_dir / filename, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header_bytes = f.read(header_len)
    import json as _json
    header = _json.loads(header_bytes.decode("utf-8"))
    data_start = 8 + header_len
    start, end = header[param_name]["data_offsets"]
    return {
        param_name: {
            "file_path": filename,
            "byte_offset": data_start + start,
            "byte_length": end - start,
            "shape": list(tensor.shape),
            "dtype": "F16",
            "precision": "fp16",
        }
    }

@pytest.fixture(scope="session")
def tiny_shard_dir(tmp_path_factory):
    """Session-scoped fixture: tiny synthetic shard directory with 6 layers."""
    shard_dir = tmp_path_factory.mktemp("shards")
    shard_index = {}
    layer_registry = {}
    for layer_idx in range(TINY_NUM_LAYERS):
        entries = _make_tiny_fp16_shard(shard_dir, layer_idx)
        shard_index.update(entries)
        layer_registry[str(layer_idx)] = f"layer_{layer_idx:04d}.safetensor"
    # Write embed shard
    embed_tensor = torch.randn(TINY_DIM, TINY_DIM, dtype=torch.float16)
    save_file({"embed.weight": embed_tensor}, str(shard_dir / "embed.safetensor"))
    with open(shard_dir / "shard_index.json", "w") as f:
        json.dump(shard_index, f)
    with open(shard_dir / "layer_registry.json", "w") as f:
        json.dump(layer_registry, f)
    return shard_dir

@pytest.fixture(scope="session")
def tiny_shard_index(tiny_shard_dir):
    with open(tiny_shard_dir / "shard_index.json") as f:
        return json.load(f)
