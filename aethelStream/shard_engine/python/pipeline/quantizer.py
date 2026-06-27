from __future__ import annotations
from dataclasses import dataclass
import torch

NF4_CODES = torch.tensor([
    -1.0, -0.6961928, -0.52507305, -0.3949175,
    -0.28444138, -0.18477343, -0.09105004, 0.0,
    0.0795803, 0.1609302, 0.2461123, 0.33791524,
    0.44070983, 0.562617, 0.72295684, 1.0,
], dtype=torch.float32)

BLOCK_SIZE = 64

@dataclass
class Nf4Result:
    packed: torch.Tensor
    absmax: torch.Tensor
    original_shape: list[int]
    block_size: int = BLOCK_SIZE

def quantize_nf4(tensor: torch.Tensor, block_size: int = BLOCK_SIZE) -> Nf4Result:
    """
    Quantize tensor to NF4 format matching bitsandbytes.
    Uses fully vectorized torch ops — no Python loops on elements.
    """
    original_shape = list(tensor.shape)
    flat = tensor.detach().cpu().float().flatten()
    num_elements = flat.numel()

    pad = (block_size - num_elements % block_size) % block_size
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=torch.float32)])

    blocks = flat.view(-1, block_size)
    absmax = blocks.abs().max(dim=1).values
    safe_absmax = absmax.clamp(min=1e-8)

    norm_blocks = blocks / safe_absmax.unsqueeze(1)
    codes = NF4_CODES
    distances = (norm_blocks.unsqueeze(2) - codes.view(1, 1, 16)).abs()
    indices = distances.argmin(dim=2).to(torch.uint8)

    indices_flat = indices.flatten()[:num_elements]
    if indices_flat.numel() % 2 != 0:
        indices_flat = torch.cat([indices_flat, torch.zeros(1, dtype=torch.uint8)])
    packed = (indices_flat[0::2] << 4) | indices_flat[1::2]

    return Nf4Result(
        packed=packed.cpu(),
        absmax=absmax.cpu(),
        original_shape=original_shape,
        block_size=block_size,
    )

def dequantize_nf4(result: Nf4Result) -> torch.Tensor:
    """Reference dequantization for fidelity testing (Test 3)."""
    packed = result.packed
    absmax = result.absmax
    num_elements = 1
    for s in result.original_shape:
        num_elements *= s

    hi = (packed >> 4).to(torch.int64)
    lo = (packed & 0x0F).to(torch.int64)
    codes = NF4_CODES

    indices = torch.stack([hi, lo], dim=1).flatten()[:num_elements]

    num_blocks = absmax.numel()
    block_size = result.block_size
    unscaled = codes[indices].view(num_blocks, -1)
    unscaled_flat = unscaled.flatten()[:num_elements]
    absmax_expanded = absmax.repeat_interleave(block_size)[:num_elements]
    dequanted = (unscaled_flat * absmax_expanded).view(result.original_shape)
    return dequanted.to(torch.float16)

def check_fidelity(original: torch.Tensor, result: Nf4Result, threshold: float = 1e-3) -> tuple[bool, float]:
    """Returns (passed, mse). If mse > threshold, the caller should fall back to FP16."""
    dequanted = dequantize_nf4(result)
    mse = ((original.float() - dequanted.float()) ** 2).mean().item()
    return mse <= threshold, mse
