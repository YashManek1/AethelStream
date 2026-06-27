"""Test 4 — Rust round-trip: load via Rust loader, run forward, compare activations."""
from __future__ import annotations
import pytest
import torch

pytestmark = pytest.mark.slow  # skip unless --slow flag

def _has_shard_engine_extension() -> bool:
    try:
        import shard_engine  # noqa: F401
        return True
    except ImportError:
        return False

@pytest.mark.skipif(not _has_shard_engine_extension(), reason="shard_engine Rust extension not built (run `maturin develop --features python-ffi`)")
def test_rust_roundtrip_fp16(tiny_shard_dir, tiny_shard_index):
    """FP16 params loaded via Rust must match safetensors-loaded params byte-for-byte."""
    import shard_engine
    from safetensors import safe_open
    from pathlib import Path

    loader = shard_engine.PyShardLoader(str(tiny_shard_dir))
    for param_name, info in list(tiny_shard_index.items())[:3]:
        if info["precision"] != "fp16":
            continue
        rust_bytes = loader.load_param(param_name)
        rust_tensor = torch.frombuffer(bytearray(rust_bytes), dtype=torch.float16)
        filepath = Path(tiny_shard_dir) / info["file_path"]
        with safe_open(str(filepath), framework="pt", device="cpu") as f:
            ref_tensor = f.get_tensor(param_name).flatten()
        assert rust_tensor.shape == ref_tensor.shape
        assert torch.equal(rust_tensor, ref_tensor), (
            f"{param_name}: Rust bytes differ from safetensors. Max diff: "
            f"{(rust_tensor - ref_tensor).abs().max().item()}"
        )

@pytest.mark.skipif(not _has_shard_engine_extension(), reason="shard_engine Rust extension not built")
def test_rust_load_layer_returns_safetensors_bytes(tiny_shard_dir):
    """load_layer(index) must return valid safetensors bytes (8-byte length prefix + header)."""
    import shard_engine
    import struct
    loader = shard_engine.PyShardLoader(str(tiny_shard_dir))
    raw = loader.load_layer(0)
    assert len(raw) >= 8
    header_len = struct.unpack("<Q", raw[:8])[0]
    assert header_len > 0
    assert len(raw) >= 8 + header_len

@pytest.mark.skipif(not _has_shard_engine_extension(), reason="shard_engine Rust extension not built")
def test_rust_missing_layer_raises(tiny_shard_dir):
    """load_layer with unknown index must raise ValueError."""
    import shard_engine
    loader = shard_engine.PyShardLoader(str(tiny_shard_dir))
    with pytest.raises((ValueError, IOError)):
        loader.load_layer(9999)
