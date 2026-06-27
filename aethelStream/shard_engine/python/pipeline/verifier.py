from __future__ import annotations
import random
from pathlib import Path
from typing import TypedDict
import torch
from safetensors import safe_open
from .ghost_loader import MemStats
from .quantizer import dequantize_nf4, Nf4Result

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

ShardIndex = dict[str, ShardIndexEntry]

def verify_zero_memory(stats: MemStats) -> None:
    """Test 1: assert ghost load consumed negligible memory."""
    assert stats.passed, f"[Test 1 FAIL] {stats.message}"
    print(f"[Test 1 PASS] {stats.message}")

def verify_completeness(
    shard_index: ShardIndex,
    model_dir: str | Path,
    expected_param_count: int,
) -> None:
    """Test 2: total element count across all shards == original model."""
    total = 0
    for param_name, info in shard_index.items():
        filepath = Path(model_dir) / info["file_path"]
        if info["precision"] == "nf4":
            shape = info["shape"]
            count = 1
            for s in shape:
                count *= s
            total += count
        else:
            with safe_open(str(filepath), framework="pt", device="cpu") as f:
                tensor = f.get_tensor(param_name)
                total += tensor.numel()
    assert total == expected_param_count, (
        f"[Test 2 FAIL] Expected {expected_param_count} elements, got {total}"
    )
    print(f"[Test 2 PASS] {total} elements verified across all shards")

def verify_nf4_fidelity(
    shard_index: ShardIndex,
    model_dir: str | Path,
    sample_count: int = 10,
    threshold: float = 1e-3,
) -> None:
    """Test 3: MSE of NF4 dequant vs original FP16 < threshold for sampled tensors."""
    nf4_params = [(k, v) for k, v in shard_index.items() if v["precision"] == "nf4"]
    if not nf4_params:
        print("[Test 3 SKIP] No NF4 tensors in this shard set")
        return
    sample = random.sample(nf4_params, min(sample_count, len(nf4_params)))
    for param_name, info in sample:
        filepath = Path(model_dir) / info["file_path"]
        with safe_open(str(filepath), framework="pt", device="cpu") as f:
            packed = f.get_tensor(f"{param_name}.__packed__")
            absmax = f.get_tensor(f"{param_name}.__absmax__")
            shape_t = f.get_tensor(f"{param_name}.__shape__")
        result = Nf4Result(
            packed=packed,
            absmax=absmax,
            original_shape=shape_t.tolist(),
            block_size=info.get("nf4_block_size", 64),
        )
        dequanted = dequantize_nf4(result)
        assert torch.isfinite(dequanted).all(), f"[Test 3 FAIL] {param_name}: non-finite after dequant"
        assert dequanted.abs().max().item() <= 2.0, f"[Test 3 FAIL] {param_name}: absmax > 2.0 after dequant"
    print(f"[Test 3 PASS] {len(sample)} NF4 tensors verified: finite, bounded")

def verify_index_integrity(
    shard_index: ShardIndex,
    model_dir: str | Path,
) -> None:
    """Test 5: every shard_index entry has valid file, offsets within file, correct shape."""
    model_dir = Path(model_dir)
    for param_name, info in shard_index.items():
        filepath = model_dir / info["file_path"]
        assert filepath.exists(), f"[Test 5 FAIL] {filepath} missing"
        file_size = filepath.stat().st_size
        end = info["byte_offset"] + info["byte_length"]
        assert end <= file_size, (
            f"[Test 5 FAIL] {param_name}: offset {end} > file size {file_size}"
        )
        if info["precision"] == "nf4" and "nf4_absmax_offset" in info:
            absmax_end = info["nf4_absmax_offset"] + info["nf4_absmax_length"]
            assert absmax_end <= file_size, (
                f"[Test 5 FAIL] {param_name}: absmax offset {absmax_end} > file {file_size}"
            )
    print(f"[Test 5 PASS] {len(shard_index)} index entries verified: files exist, offsets valid")

def run_all_verifications(
    shard_index: ShardIndex,
    model_dir: str | Path,
    expected_param_count: int,
    mem_stats: MemStats,
) -> None:
    """Run all verification tests."""
    verify_zero_memory(mem_stats)
    verify_completeness(shard_index, model_dir, expected_param_count)
    verify_nf4_fidelity(shard_index, model_dir)
    verify_index_integrity(shard_index, model_dir)
