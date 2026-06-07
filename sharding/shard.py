import os
import json
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

DTYPE = torch.float16


def shard_model(model_name, output_dir="model_shards"):
    os.makedirs(output_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=DTYPE,
        device_map="cpu"
    )

    state_dict = model.state_dict()

    shards = {}
    index = {}

    for name, param in state_dict.items():
        param = param.to(DTYPE)

        if "h." in name:
            layer_id = name.split("h.")[1].split(".")[0]
            file_name = f"layer_{layer_id}.safetensors"

        elif "wte" in name:
            file_name = "embeddings.safetensors"

        elif "lm_head" in name:
            file_name = "lm_head.safetensors"

        else:
            file_name = "misc.safetensors"

        if file_name not in shards:
            shards[file_name] = {}

        shards[file_name][name] = param
        index[name] = file_name

    # save shards
    for file_name, weights in shards.items():
        save_file(weights, os.path.join(output_dir, file_name))

    # save index
    with open(os.path.join(output_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=2)

    return {
        "status": "success",
        "shards": list(shards.keys()),
        "total_layers": len(shards)
    }