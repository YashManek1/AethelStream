from __future__ import annotations
from dataclasses import dataclass, field
import torch
from transformers.modeling_utils import PreTrainedModel
from ..models.arch_registry import (
    ParamRole, classify_param, get_arch_config, get_num_layers,
)

@dataclass
class ShardSpec:
    """Maps a shard filename to the list of (param_name, tensor) it should contain."""
    filename: str
    params: list[tuple[str, torch.Tensor]] = field(default_factory=list)
    layer_index: int | None = None
    precision: str = "fp16"

def compute_precision(layer_index: int | None, num_layers: int, role: ParamRole) -> str:
    """FP16 for edge layers and non-layer params; NF4 for middle layers."""
    if role != ParamRole.LAYER and role != ParamRole.MOE_EXPERT:
        return "fp16"
    if layer_index is None:
        return "fp16"
    if layer_index <= 3 or layer_index >= num_layers - 4:
        return "fp16"
    return "nf4"

def partition(model: PreTrainedModel, model_type: str) -> tuple[list[ShardSpec], int]:
    """
    Partition model parameters into ShardSpecs.
    Returns (shard_specs, num_layers).
    Does NOT load actual weights — works on meta-device model for planning.
    """
    config = get_arch_config(model_type)
    num_layers = get_num_layers(model.config)
    
    shard_map: dict[str, ShardSpec] = {}

    for name, param in model.named_parameters():
        classification = classify_param(name, config)
        role = classification.role
        layer_idx = classification.layer_index
        expert_idx = classification.expert_index
        precision = compute_precision(layer_idx, num_layers, role)

        if role == ParamRole.LAYER:
            filename = f"layer_{layer_idx:04d}.safetensor"
        elif role == ParamRole.MOE_EXPERT:
            filename = f"expert_{layer_idx:04d}_{expert_idx:03d}.safetensor"
        elif role == ParamRole.MOE_ROUTER:
            filename = f"router_{layer_idx:04d}.safetensor"
        elif role == ParamRole.EMBED:
            filename = "embed.safetensor"
        elif role == ParamRole.HEAD:
            filename = "lm_head.safetensor"
        elif role == ParamRole.NORM:
            filename = "norm.safetensor"
        else:
            filename = "misc.safetensor"

        if filename not in shard_map:
            shard_map[filename] = ShardSpec(filename=filename, layer_index=layer_idx, precision=precision)
        shard_map[filename].params.append((name, param))

    return list(shard_map.values()), num_layers
