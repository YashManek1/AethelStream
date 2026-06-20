"""
reference_layer.py — Exact FP32/BF16 autograd oracle for one transformer block.

Used ONLY by the parity harness (T-PARITY-1). Never imported on the production path.
This is the ground-truth reference against which DoublePass gradients are validated.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional


class ReferenceTransformerBlock(nn.Module):
    """Single transformer block (attention + MLP + LayerNorm).

    Matches AethelStream's LayerShard layout:
      QKV projection, output projection, MLP up/gate/down, two LayerNorms.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp_up = nn.Linear(d_model, d_ff, bias=False)
        self.mlp_gate = nn.Linear(d_model, d_ff, bias=False)
        self.mlp_down = nn.Linear(d_ff, d_model, bias=False)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, S, D]
        # Attention
        h = self.ln1(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        # Simple scaled dot-product (no flash-attn; this is the reference oracle)
        B, S, D = q.shape
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        attn = torch.softmax(q @ k.transpose(-2, -1) / self.head_dim**0.5, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, S, D)
        x = x + self.o(out)
        # MLP (SwiGLU)
        h = self.ln2(x)
        x = x + self.mlp_down(torch.nn.functional.silu(self.mlp_gate(h)) * self.mlp_up(h))
        return x


def compute_reference_grad(
    block: ReferenceTransformerBlock,
    x: torch.Tensor,
    loss_grad: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Return per-parameter gradients from full autograd (the parity oracle).

    Args:
        block: The reference block with frozen weights.
        x: Input tensor [B, S, D] on CPU (or CUDA if available).
        loss_grad: Upstream gradient. If None, uses sum of output.
        dtype: Computation dtype (float32 or bfloat16).

    Returns:
        Dict mapping parameter name to gradient tensor (always float32).
    """
    block = block.to(dtype)
    x = x.to(dtype).requires_grad_(False)
    for p in block.parameters():
        p.requires_grad_(True)
        if p.grad is not None:
            p.grad.zero_()
    y = block(x)
    if loss_grad is None:
        loss = y.sum()
    else:
        loss = (y * loss_grad.to(dtype)).sum()
    loss.backward()
    return {
        name: p.grad.float().clone()
        for name, p in block.named_parameters()
        if p.grad is not None
    }
