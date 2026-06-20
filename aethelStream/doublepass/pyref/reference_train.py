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


def config_125m() -> dict:
    """Return hyperparameters for the 125M model (scaled for speed with 2 layers)."""
    return {
        "vocab_size": 50257,
        "d_model": 768,
        "n_heads": 12,
        "d_ff": 3072,
        "n_layers": 2,  # Reduced from 12 for speed
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--output", type=str, default="pyref/reference_losses.json", help="Output JSON path"
    )
    args = parser.parse_args()

    cfg = config_125m()
    vocab_size = cfg["vocab_size"]
    d_model = cfg["d_model"]
    n_heads = cfg["n_heads"]
    d_ff = cfg["d_ff"]

    # Build model with 2 blocks
    model = TinyLM(vocab_size, d_model, n_heads, d_ff)
    model.to(torch.device("cpu"))

    # Random data generator
    def data_generator():
        while True:
            yield torch.randint(0, vocab_size, (1, 4))

    gen = data_generator()

    # Train for args.steps
    losses = reference_train(model, gen, n_steps=args.steps, lr=args.lr)

    # Write to JSON
    output_data = {
        "losses": losses,
        "step_count": args.steps,
        "config": "125M (2-layer reduction for speed)",
        "backend": "REFERENCE: PyTorch f32 AdamW, CPU-only",
    }
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(
        f"REFERENCE: PyTorch f32 AdamW, CPU-only — {args.steps} steps, "
        f"losses[0]={losses[0]:.6f}, losses[-1]={losses[-1]:.6f}"
    )
