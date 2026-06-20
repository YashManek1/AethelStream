"""
reference_train.py — Tiny full-training loop for T-CONV-5.

Trains a 2-layer reference model for N steps and records per-step loss.
Used ONLY by the convergence parity harness. Never on the production path.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from reference_layer import ReferenceTransformerBlock
from typing import Iterator


class TinyLM(nn.Module):
    """Two-layer causal LM for convergence testing."""

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            ReferenceTransformerBlock(d_model, n_heads, d_ff),
            ReferenceTransformerBlock(d_model, n_heads, d_ff),
        ])
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:  # -> [B, S, V]
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def reference_train(
    model: TinyLM,
    data_iter: Iterator[torch.Tensor],
    n_steps: int = 500,
    lr: float = 1e-3,
) -> list[float]:
    """Run `n_steps` of full-precision AdamW training.

    Returns a list of per-step cross-entropy loss values.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses: list[float] = []
    for step in range(n_steps):
        input_ids = next(data_iter)  # [B, S]
        labels = input_ids.roll(-1, dims=1)
        logits = model(input_ids)  # [B, S, V]
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses
