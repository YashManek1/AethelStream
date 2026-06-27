"""
Test 4 — Rust round-trip activation fidelity.

Verifies that the shard_engine Rust loader produces FP16 weights whose
forward-pass activations match PyTorch within 1e-3 (the FP16 parity gate).

Build the extension first:
    maturin develop --features python-ffi
"""
import torch
from transformers import AutoModelForCausalLM


def run_test_4() -> None:
    print("Test 4: Rust round-trip fidelity")
    try:
        import shard_engine
    except ImportError:
        print(
            "[Test 4 Skip] 'shard_engine' module not found. "
            "Run `maturin develop --features python-ffi` inside aethelStream/shard_engine/."
        )
        return

    loader = shard_engine.PyShardLoader("tests/fixtures")
    target_param = "transformer.h.0.attn.c_attn.weight"
    rust_bytes = loader.load_layer(target_param)

    model = AutoModelForCausalLM.from_pretrained(
        "gpt2", torch_dtype=torch.float16, device_map="cpu"
    )
    reference_weight = model.state_dict()[target_param]

    rust_weight_1d = torch.frombuffer(rust_bytes, dtype=torch.float16)
    rust_weight = rust_weight_1d.view(reference_weight.shape)

    with torch.no_grad():
        dummy_input = torch.randn((1, 10, reference_weight.shape[0]), dtype=torch.float16)
        reference_output = dummy_input @ reference_weight
        rust_output = dummy_input @ rust_weight

    max_diff = (reference_output - rust_output).abs().max().item()
    assert max_diff < 1e-3, (
        f"[Test 4 FAIL] Activation difference {max_diff:.6f} exceeds FP16 parity gate 1e-3"
    )
    print(f"[Test 4 PASS] Rust loader round-trip OK — max activation diff: {max_diff:.6f}")


if __name__ == "__main__":
    run_test_4()
