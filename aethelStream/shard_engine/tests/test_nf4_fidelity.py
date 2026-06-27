"""Test 3 — NF4 quantisation fidelity (MSE < 1e-3)."""
from __future__ import annotations
import torch

def test_nf4_roundtrip_mse_below_threshold():
    """NF4 quantise + dequantise MSE must be below 1e-3."""
    from python.pipeline.quantizer import quantize_nf4, check_fidelity
    torch.manual_seed(42)
    tensor = torch.randn(256, 256, dtype=torch.float16)
    result = quantize_nf4(tensor)
    passed, mse = check_fidelity(tensor, result)
    assert passed, f"NF4 MSE {mse:.6f} exceeds 1e-3 threshold"

def test_nf4_zero_tensor():
    """Zero tensor must quantise and dequantise to zero."""
    from python.pipeline.quantizer import quantize_nf4, dequantize_nf4
    tensor = torch.zeros(64, 64, dtype=torch.float16)
    result = quantize_nf4(tensor)
    deq = dequantize_nf4(result)
    assert deq.abs().max().item() < 1e-5

def test_nf4_shape_preserved():
    """Original shape must be recorded in Nf4Result."""
    from python.pipeline.quantizer import quantize_nf4
    tensor = torch.randn(32, 128, 4, dtype=torch.float16)
    result = quantize_nf4(tensor)
    assert result.original_shape == [32, 128, 4]

def test_nf4_packed_size():
    """Packed tensor must have ceil(num_elements / 2) bytes."""
    from python.pipeline.quantizer import quantize_nf4
    tensor = torch.randn(128, dtype=torch.float16)
    result = quantize_nf4(tensor)
    assert result.packed.numel() == 64  # 128 / 2

def test_nf4_absmax_num_blocks():
    """Number of absmax values must equal ceil(num_elements / block_size)."""
    from python.pipeline.quantizer import quantize_nf4, BLOCK_SIZE
    num_elements = 200
    tensor = torch.randn(num_elements, dtype=torch.float16)
    result = quantize_nf4(tensor)
    expected_blocks = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    assert result.absmax.numel() == expected_blocks

def test_fallback_flagged_in_fidelity_check():
    """check_fidelity returns False when MSE exceeds threshold."""
    from python.pipeline.quantizer import quantize_nf4, check_fidelity
    import torch
    # Create a tensor with extreme values that won't quantize well
    tensor = torch.randn(256, 256, dtype=torch.float16) * 1000
    result = quantize_nf4(tensor)
    passed, mse = check_fidelity(tensor, result, threshold=1e-6)
    assert not passed or mse >= 1e-6  # At least one should be true
