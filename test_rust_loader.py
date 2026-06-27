import torch
from transformers import AutoModelForCausalLM

def run_test_4():
    print("Test 4: Rust Round Trip")
    try:
        import rust_runtime
        loader = rust_runtime.PyShardLoader("model_shards")
        target_param = "transformer.h.0.attn.c_attn.weight"
        rust_bytes = loader.load_layer(target_param)
        
        # Original FP16 via basic pytorch comparison
        model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float16, device_map="cpu")
        standard_w = model.state_dict()[target_param]
        
        # Reform rust weights to Tensor
        # Rust returns raw fp16 bytes
        rust_w_1d = torch.frombuffer(rust_bytes, dtype=torch.float16)
        rust_w = rust_w_1d.view(standard_w.shape)
        
        # Forward pass on small dummy X
        # For simplicity, just matmul or MSE the weights depending on layer
        with torch.no_grad():
            x = torch.randn((1, 10, standard_w.shape[0]), dtype=torch.float16)
            
            # Using standard torch W
            out_standard = x @ standard_w
            
            # Using Rust W
            out_rust = x @ rust_w
            
        max_diff = (out_standard - out_rust).abs().max().item()
        
        assert max_diff < 1e-3, f"Test 4 fail: Activation difference {max_diff:.6f} > 1e-3"
        print(f"[Test 4 Pass] Rust loader round-trip successful! Max diff: {max_diff:.6f}")
        
    except ImportError:
        print("[Test 4 Skip] 'rust_runtime' module not configured via maturin in this environment.")
        print("To run, execute `maturin develop --release` inside the rust_runtime folder.")

if __name__ == "__main__":
    run_test_4()
