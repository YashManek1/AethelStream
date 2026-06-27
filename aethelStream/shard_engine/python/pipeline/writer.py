from __future__ import annotations
import json
import struct
from pathlib import Path
from typing import TypedDict

import torch
from safetensors.torch import save_file
from transformers.modeling_utils import PreTrainedModel

from .partitioner import ShardSpec, partition
from .quantizer import quantize_nf4, check_fidelity, Nf4Result

# Type-safe aliases instead of using Any
class ShardIndexEntry(TypedDict, total=False):
    file_path: str
    byte_offset: int
    byte_length: int
    shape: list[int]
    dtype: str
    precision: str
    nf4_absmax_offset: int
    nf4_absmax_length: int
    nf4_block_size: int

ShardIndex = dict[str, ShardIndexEntry]
LayerRegistry = dict[str, str]

def _read_safetensors_offsets(filepath: str | Path) -> dict[str, tuple[int, int]]:
    """
    Parse safetensors header to get absolute byte offsets for each tensor.
    Returns {tensor_name: (abs_start, abs_end)}.
    Raises ValueError if header is malformed.
    """
    with open(filepath, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header_bytes = f.read(header_len)
    header = json.loads(header_bytes.decode("utf-8"))
    data_start = 8 + header_len
    offsets: dict[str, tuple[int, int]] = {}
    for tensor_name, info in header.items():
        if tensor_name == "__metadata__":
            continue
        if "data_offsets" not in info:
            raise ValueError(f"Tensor {tensor_name} missing data_offsets field")
        start, end = info["data_offsets"]
        offsets[tensor_name] = (data_start + start, data_start + end)
    return offsets

def _write_shard(
    spec: ShardSpec,
    output_dir: Path,
    model: PreTrainedModel,
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, ShardIndexEntry], str]:
    """
    Write one shard file. Returns (shard_index_entries, filename).
    Frees tensors from state_dict after writing to keep RAM low.
    """
    filepath = output_dir / spec.filename
    tensors_to_save: dict[str, torch.Tensor] = {}
    nf4_results: dict[str, Nf4Result] = {}
    fallback_to_fp16: set[str] = set()
    index_entries: dict[str, ShardIndexEntry] = {}
    param_shapes: dict[str, list[int]] = {}

    for param_name, _ in spec.params:
        tensor = state_dict[param_name].detach().cpu()
        shape = list(tensor.shape)
        param_shapes[param_name] = shape

        if spec.precision == "nf4":
            result = quantize_nf4(tensor)
            passed, mse = check_fidelity(tensor, result)
            if not passed:
                fallback_to_fp16.add(param_name)
                tensors_to_save[param_name] = tensor.to(torch.float16)
            else:
                nf4_results[param_name] = result
                tensors_to_save[f"{param_name}.__packed__"] = result.packed
                tensors_to_save[f"{param_name}.__absmax__"] = result.absmax
                tensors_to_save[f"{param_name}.__shape__"] = torch.tensor(
                    result.original_shape, dtype=torch.int32
                )
        else:
            tensors_to_save[param_name] = tensor.to(torch.float16)

        del state_dict[param_name]
        del tensor

    save_file(tensors_to_save, str(filepath))
    file_offsets = _read_safetensors_offsets(filepath)

    for param_name, _ in spec.params:
        shape = param_shapes[param_name]
        
        if param_name in nf4_results:
            packed_key = f"{param_name}.__packed__"
            absmax_key = f"{param_name}.__absmax__"
            if packed_key not in file_offsets:
                raise ValueError(f"Parameter {packed_key} not found in safetensors file after writing")
            if absmax_key not in file_offsets:
                raise ValueError(f"Parameter {absmax_key} not found in safetensors file after writing")
            packed_start, packed_end = file_offsets[packed_key]
            absmax_start, absmax_end = file_offsets[absmax_key]
            precision = "fp16" if param_name in fallback_to_fp16 else "nf4"
            index_entries[param_name] = ShardIndexEntry(
                file_path=spec.filename,
                byte_offset=packed_start,
                byte_length=packed_end - packed_start,
                shape=shape,
                dtype="U8" if precision == "nf4" else "F16",
                precision=precision,
                nf4_absmax_offset=absmax_start,
                nf4_absmax_length=absmax_end - absmax_start,
                nf4_block_size=nf4_results[param_name].block_size,
            )
        else:
            if param_name not in file_offsets:
                raise ValueError(f"Parameter {param_name} not found in safetensors file after writing")
            abs_start, abs_end = file_offsets[param_name]
            index_entries[param_name] = ShardIndexEntry(
                file_path=spec.filename,
                byte_offset=abs_start,
                byte_length=abs_end - abs_start,
                shape=shape,
                dtype="F16",
                precision="fp16",
            )

    del tensors_to_save
    return index_entries, spec.filename

def write_shards(
    model_id: str,
    output_dir: str | Path,
    *,
    num_proc: int = 4,
) -> tuple[ShardIndex, LayerRegistry]:
    """
    Full sharding pipeline: load model → partition → quantize → write → index.
    
    Streams one shard at a time to keep peak RAM bounded.
    Returns (shard_index, layer_registry) — also writes shard_index.json and layer_registry.json.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    state_dict = model.state_dict()
    model_type = getattr(model.config, "model_type", "llama").lower()

    from accelerate import init_empty_weights
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_id)
    with init_empty_weights():
        ghost_model = AutoModelForCausalLM.from_config(config)

    specs, num_layers = partition(ghost_model, model_type)
    del ghost_model

    shard_index: ShardIndex = {}
    layer_registry: LayerRegistry = {}

    for spec in specs:
        real_params: list[tuple[str, torch.Tensor]] = []
        for param_name, _ in spec.params:
            if param_name in state_dict:
                real_params.append((param_name, state_dict[param_name]))
        spec.params = real_params

        entries, filename = _write_shard(spec, output_dir, model, state_dict)
        shard_index.update(entries)

        if spec.layer_index is not None:
            layer_registry[str(spec.layer_index)] = filename
        elif "embed" in filename:
            layer_registry["embed"] = filename
        elif "lm_head" in filename or "head" in filename:
            layer_registry["head"] = filename
        elif "norm" in filename and "layer" not in filename:
            layer_registry["norm"] = filename

    with open(output_dir / "shard_index.json", "w") as f:
        json.dump(shard_index, f, indent=2)
    with open(output_dir / "layer_registry.json", "w") as f:
        json.dump(layer_registry, f, indent=2)

    return shard_index, layer_registry
