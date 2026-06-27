"""Test 2 — Shard completeness: total param count matches original."""
from __future__ import annotations
import pytest

def test_completeness_tiny_shards(tiny_shard_dir, tiny_shard_index):
    """Tiny shard set: verify total element count matches expected."""
    from python.pipeline.verifier import verify_completeness
    # Compute expected count from the tiny shard index itself (FP16 only in tiny fixture)
    from safetensors import safe_open
    from pathlib import Path
    total = 0
    for param_name, info in tiny_shard_index.items():
        filepath = Path(tiny_shard_dir) / info["file_path"]
        with safe_open(str(filepath), framework="pt", device="cpu") as f:
            tensor = f.get_tensor(param_name)
            total += tensor.numel()
    # verify_completeness should pass with the exact count
    verify_completeness(tiny_shard_index, tiny_shard_dir, total)

@pytest.mark.slow
def test_completeness_gpt2_real(tmp_path):
    """Integration: shard GPT-2 and verify param count == sum(p.numel() for p in model)."""
    import torch
    from transformers import AutoModelForCausalLM
    from python.pipeline.writer import write_shards
    from python.pipeline.verifier import verify_completeness
    model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float16, device_map="cpu")
    expected = sum(p.numel() for p in model.parameters())
    del model
    shard_index, _ = write_shards("gpt2", str(tmp_path))
    verify_completeness(shard_index, tmp_path, expected)
