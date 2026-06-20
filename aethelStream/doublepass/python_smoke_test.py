#!/usr/bin/env python3
"""
AethelStream M5 DoublePass — Python smoke test.

Requires:
    pip install maturin
    cd aethelStream/doublepass
    maturin develop --features "python-ffi mock-cuda"

Then run:
    python python_smoke_test.py
"""
import sys
import json

try:
    import doublepass_py
except ImportError:
    print("SKIP: doublepass_py not installed.")
    print("      Build with: maturin develop --features 'python-ffi mock-cuda'")
    sys.exit(0)

print("=== AethelStream M5 PyO3 Smoke Test ===")

# Construct engine: 2 layers, d_model=32, 2 heads, d_ff=64, seq_len=4, batch=1, vocab=256
dp = doublepass_py.PyDoublePass(
    n_layers=2,
    d_model=32,
    n_heads=2,
    d_ff=64,
    seq_len=4,
    batch=1,
    vocab_size=256,
    chunk_size=64,
)
print("PyDoublePass constructed OK")

# Run one step: inputs = 1 micro-batch of shape [batch*seq_len*d_model] = [128]
inputs = [[float(i % 10) * 0.01 for i in range(4 * 32)]]
labels = list(range(4))  # G * batch * seq_len = 4 labels

metrics_json = dp.step(inputs, labels)
metrics = json.loads(metrics_json)
print(f"step() OK — loss proxy via weight_bytes: {metrics['weight_bytes_streamed']}")
print(f"  step_index        : {metrics['step_index']}")
print(f"  prefetch_misses   : {metrics['prefetch_misses']}")
print(f"  grad_accum_steps  : {metrics['grad_accum_steps']}")
print(f"  tokens_processed  : {metrics['tokens_processed']}")

assert metrics["prefetch_misses"] == 0, "PrefetchMiss on mock path!"
assert metrics["step_index"] == 0, "step_index should be 0 for first step"

# Test snapshot
snap_json = dp.snapshot()
snap = json.loads(snap_json)
print(f"snapshot() OK — step={snap['step']}")
assert snap["step"] == 1, f"snapshot step should be 1 after one step, got {snap['step']}"

# Test parity_probe
stream_grad = [0.1, 0.2, -0.1]
ref_grad    = [0.1, 0.2, -0.1]
rel = dp.parity_probe(0, stream_grad, ref_grad)
print(f"parity_probe() OK — rel={rel:.2e}")
assert rel < 1e-6, f"identical grads should give ~0 rel error, got {rel}"

# Test apply_delta
import json as _json
delta = {"checkpoint_freq": 2, "w_max_hint": None, "precision_overrides": []}
dp.apply_delta(_json.dumps(delta))
print("apply_delta() OK")

# Second step
metrics2 = json.loads(dp.step(inputs, labels))
assert metrics2["step_index"] == 1, "second step should have step_index=1"
print(f"second step() OK — step_index={metrics2['step_index']}")

print()
print("ALL SMOKE TESTS PASSED [MOCK CPU f32 path]")
print("NOTE: All numbers are MOCK. Do NOT cite as measured GPU performance.")
