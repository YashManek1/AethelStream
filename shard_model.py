import os
import json
import struct
import random
import re
import psutil
import torch
from accelerate import init_empty_weights
from safetensors.torch import save_file
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

LAYER_PREFIXES = {
    "llama": "model.layers",
    "mistral": "model.layers",
    "gpt_neox": "gpt_neox.layers",
    "bloom": "transformer.h",
    "opt": "model.decoder.layers",
    "phi": "model.layers",
    "gpt2": "transformer.h",
}

NF4_CODES = torch.tensor([
    -1.0000, -0.6961928, -0.52507305, -0.3949175, 
    -0.28444138, -0.18477343, -0.09105004, 0.0000,
    0.0795803, 0.1609302, 0.2461123, 0.33791524, 
    0.44070983, 0.562617, 0.72295684, 1.0000
], dtype=torch.float32)

def ghost_load(model_id):
    import gc
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        start_mem = torch.cuda.memory_allocated()
    else:
        start_mem = 0
        
    start_vmem = psutil.virtual_memory().used
    
    config = AutoConfig.from_pretrained(model_id)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
        
    if torch.cuda.is_available():
        end_mem = torch.cuda.memory_allocated()
    else:
        end_mem = 0
        
    end_vmem = psutil.virtual_memory().used
    
    delta_mb = (end_mem - start_mem) / (1024 * 1024)
    vmem_delta_mb = (end_vmem - start_vmem) / (1024 * 1024)
    
    # Test 1 logic
    if torch.cuda.is_available():
        assert delta_mb < 50.0, f"Ghost load failed: memory leaked {delta_mb:.2f} MiB"
        assert torch.cuda.memory_allocated() == 0, "torch.cuda.memory_allocated() isn't 0 after init_empty_weights!"
        
    assert vmem_delta_mb < 200.0, f"Ghost load failed: System RAM leaked significantly: {vmem_delta_mb:.2f} MiB"
    print(f"[Test 1 Pass] Ghost load passed. VRAM delta: {delta_mb:.2f} MiB, RAM delta: {vmem_delta_mb:.2f} MiB")
    return model

def discover_layers(model):
    roles = {}
    model_type = getattr(model.config, "model_type", "").lower()
    prefix = LAYER_PREFIXES.get(model_type, "model.layers")
    
    for name, _ in model.named_parameters():
        if "lora" in name.lower() or "optimizer" in name.lower():
            continue
            
        is_moe = False
        if "expert" in name.lower():
            m = re.search(r'experts?\.(\d+)', name)
            expert_id = m.group(1) if m else "unknown"
            bucket = f"expert_{expert_id}"
            roles[name] = (bucket, None)
            is_moe = True
            
        elif "router" in name.lower() or "gate" in name.lower():
            bucket = "router"
            roles[name] = (bucket, None)
            is_moe = True
            
        if is_moe:
            continue

        if "embed" in name or "wte" in name or "wpe" in name:
            roles[name] = ("embed", None)
        elif "lm_head" in name or ("output" in name and prefix not in name):
            roles[name] = ("head", None)
        elif prefix in name:
            try:
                parts = name.split(prefix + ".")[1].split(".")
                layer_idx = int(parts[0])
                roles[name] = (f"layer_{layer_idx}", layer_idx)
            except (IndexError, ValueError):
                roles[name] = ("misc", None)
        elif "norm" in name or "ln_" in name:
            roles[name] = ("norm", None)
        else:
            roles[name] = ("misc", None)
            
    layer_indices = [idx for role, idx in roles.values() if role.startswith("layer_") and idx is not None]
    num_layers = max(layer_indices) + 1 if layer_indices else 0
    print(f"[2/4] Layer discovery: Identified {num_layers} layers in '{model_type}'.")
    return roles, num_layers

def quantize_nf4(tensor, block_size=64):
    device = tensor.device
    original_shape = torch.tensor(list(tensor.shape), dtype=torch.int32)
    flat_tensor = tensor.flatten()
    
    rem = flat_tensor.numel() % block_size
    if rem != 0:
        pad_len = block_size - rem
        flat_tensor = torch.cat([flat_tensor, torch.zeros(pad_len, device=device, dtype=tensor.dtype)])
        
    blocks = flat_tensor.view(-1, block_size).float()
    absmax = blocks.abs().max(dim=1, keepdim=True).values
    absmax = torch.where(absmax == 0, torch.ones_like(absmax), absmax)
    
    norm_blocks = blocks / absmax
    codes = NF4_CODES.to(device)
    distances = torch.abs(norm_blocks.unsqueeze(-1) - codes.view(1, 1, 16))
    indices = distances.argmin(dim=-1).to(torch.uint8)
    
    indices_flat = indices.flatten()
    packed = (indices_flat[0::2] << 4) | indices_flat[1::2]
    
    return packed.cpu(), absmax.squeeze(-1).float().cpu(), original_shape.cpu()

def dequantize_nf4_python(packed, absmax, shape, block_size=64):
    device = packed.device
    num_elements = packed.numel() * 2
    codes = NF4_CODES.to(device)
    
    indices = torch.empty((num_elements,), dtype=torch.uint8, device=device)
    indices[0::2] = packed >> 4
    indices[1::2] = packed & 0x0F
    
    unscaled = codes[indices.long()].view(-1, block_size)
    dequantized = unscaled * absmax.view(-1, 1)
    
    original_size = 1
    for s in shape:
        original_size *= s.item()
        
    return dequantized.flatten()[:original_size].view(shape.tolist())

def verify_shards(output_dir, expected_num_params, original_model_name):
    # Test 2: Shard completeness (must match original params count strictly to within 0)
    config_path = AutoConfig.from_pretrained(original_model_name)
    total_shard_params = 0
    
    index_path = os.path.join(output_dir, "shard_index.json")
    with open(index_path, "r") as f:
        index = json.load(f)
        
    tested_existence = 0
    tested_shape_match = 0
        
    for param_name, info in index.items():
        file_path = os.path.join(output_dir, info["file"])
        
        # Test 5: Verify file exists
        assert os.path.exists(file_path), f"Test 5 fail: {file_path} missing."
        
        # Count params
        num_elements = 1
        with safe_open(file_path, framework="pt", device="cpu") as f:
            if f"{param_name}.shape" in f.keys():
                shape_t = f.get_tensor(f"{param_name}.shape")
                shape = shape_t.tolist()
                for s in shape: num_elements *= s
            elif f"{param_name}.packed" in f.keys():
                pass # Already counted from shape tensor
            elif param_name in f.keys():
                num_elements = f.get_tensor(param_name).numel()
                
            tested_existence += 1
            
        total_shard_params += num_elements
        
        file_size = os.path.getsize(file_path)
        for tensor_pname, t_info in info["tensors"].items():
            start_off, end_off = t_info["start"], t_info["end"]
            
            # Test 5: Verify bounds against file size
            assert end_off <= file_size, f"Test 5 fail: Index {tensor_pname} in {info['file']} out of bounds (offset {end_off} > file {file_size})"
            tested_shape_match += 1
            
    print(f"[Test 5 Pass] Integrity checks out. Validated {tested_existence} references, {tested_shape_match} offsets strict bounds within file lengths.")
    
    assert total_shard_params == expected_num_params, f"Test 2 Fail: Expected {expected_num_params} elements, got {total_shard_params}"
    print(f"[Test 2 Pass] Shard completeness strictly intact: {total_shard_params} {expected_num_params}")

def shard_model(model_name, output_dir="model_shards"):
    os.makedirs(output_dir, exist_ok=True)
    
    ghost_model = ghost_load(model_name)
    roles, num_layers = discover_layers(ghost_model)
    
    print(f"[3/4] Running precision-aware sharding...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    
    # Calculate original param counts for Test 2
    original_params = sum([p.numel() for n, p in model.state_dict().items() if "lora" not in n.lower() and "optimizer" not in n.lower()])
    
    shards = {}
    
    # For Test 3: Numerical fidelity check on 10 random layers
    int4_layers = [i for i in range(num_layers) if not (0 <= i <= 3 or (num_layers - 4) <= i <= (num_layers - 1))]
    sampled_layers = set(random.sample(int4_layers, min(10, len(int4_layers))))
    test3_passed = 0
    test3_fallen_back = 0
    
    for name, param in model.state_dict().items():
        if "lora" in name.lower() or "optimizer" in name.lower():
            continue
            
        role, layer_idx = roles.get(name, ("misc", None))
        shard_filename = f"{role}.safetensors"
        
        if shard_filename not in shards:
            shards[shard_filename] = {}
            
        if role.startswith("layer_") and layer_idx is not None:
            if 0 <= layer_idx <= 3 or (num_layers - 4) <= layer_idx <= (num_layers - 1):
                shards[shard_filename][name] = param.cpu().to(torch.float16)
            else:
                packed, absmax, shape = quantize_nf4(param)
                
                # Test 3
                if layer_idx in sampled_layers:
                    dequant_param = dequantize_nf4_python(packed, absmax, shape)
                    mse = ((param.float() - dequant_param.float())**2).mean().item()
                    
                    if mse > 1e-3:
                        print(f"[Test 3 Alert] Tensor {name} has MSE {mse:.4f} > 1e-3. Flagging & falling back to FP16.")
                        shards[shard_filename][name] = param.cpu().to(torch.float16)
                        test3_fallen_back += 1
                        continue
                    else:
                        test3_passed += 1

                shards[shard_filename][f"{name}.packed"] = packed
                shards[shard_filename][f"{name}.absmax"] = absmax
                shards[shard_filename][f"{name}.shape"] = shape
        else:
            shards[shard_filename][name] = param.cpu().to(torch.float16)
            
    print(f"[Test 3 Pass] Fidelity evaluations complete. {test3_passed} met 1e-3 MSE, {test3_fallen_back} fell back.")

    for file_name, weights in shards.items():
        save_file(weights, os.path.join(output_dir, file_name))
        
    print(f"[4/4] Building index...")
    index = {}
    for file_name in shards.keys():
        file_path = os.path.join(output_dir, file_name)
        with open(file_path, "rb") as f:
            length_bytes = f.read(8)
            header_len = struct.unpack("<Q", length_bytes)[0]
            header_bytes = f.read(header_len)
            header = json.loads(header_bytes.decode("utf-8"))
            
            for tensor_name, tensor_info in header.items():
                if tensor_name == "__metadata__":
                    continue
                    
                data_offsets = tensor_info["data_offsets"]
                abs_start = 8 + header_len + data_offsets[0]
                abs_end = 8 + header_len + data_offsets[1]
                
                if tensor_name.endswith(".packed") or tensor_name.endswith(".absmax") or tensor_name.endswith(".shape"):
                    base_name = tensor_name.rsplit(".", 1)[0]
                    suffix = tensor_name.rsplit(".", 1)[1]
                else:
                    base_name = tensor_name
                    suffix = "fp16"
                    
                if base_name not in index:
                    index[base_name] = {"file": file_name, "tensors": {}}
                    
                index[base_name]["tensors"][suffix] = {
                    "start": abs_start,
                    "end": abs_end,
                    "dtype": tensor_info["dtype"]
                }
                
    index_path = os.path.join(output_dir, "shard_index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
        
    print(f"Index mapped for {len(index)} parameters.")
    verify_shards(output_dir, original_params, model_name)
    return {"status": "success", "index": index_path, "total_layers": num_layers}

if __name__ == "__main__":
    import sys
    model_id = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    shard_model(model_id)
