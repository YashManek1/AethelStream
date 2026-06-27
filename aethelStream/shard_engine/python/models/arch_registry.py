from __future__ import annotations
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class ParamRole(Enum):
    EMBED = "embed"
    HEAD = "head"
    NORM = "norm"
    LAYER = "layer"
    MOE_EXPERT = "moe_expert"
    MOE_ROUTER = "moe_router"
    MISC = "misc"

@dataclass
class ParamClassification:
    role: ParamRole
    layer_index: Optional[int] = None
    expert_index: Optional[int] = None

@dataclass
class ArchConfig:
    layer_pattern: str
    expert_pattern: str = ""
    router_pattern: str = ""
    embed_patterns: list[str] = field(default_factory=list)
    head_patterns: list[str] = field(default_factory=list)
    norm_patterns: list[str] = field(default_factory=list)

ARCH_REGISTRY: dict[str, ArchConfig] = {
    "llama": ArchConfig(
        layer_pattern=r"model\.layers\.(\d+)\.",
        embed_patterns=["model.embed_tokens.weight"],
        head_patterns=["lm_head.weight"],
        norm_patterns=["model.norm.weight"],
    ),
    "mistral": ArchConfig(
        layer_pattern=r"model\.layers\.(\d+)\.",
        embed_patterns=["model.embed_tokens.weight"],
        head_patterns=["lm_head.weight"],
        norm_patterns=["model.norm.weight"],
    ),
    "mixtral": ArchConfig(
        layer_pattern=r"model\.layers\.(\d+)\.",
        expert_pattern=r"model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.",
        router_pattern=r"model\.layers\.(\d+)\.block_sparse_moe\.gate\.",
        embed_patterns=["model.embed_tokens.weight"],
        head_patterns=["lm_head.weight"],
        norm_patterns=["model.norm.weight"],
    ),
    "phi": ArchConfig(
        layer_pattern=r"model\.layers\.(\d+)\.",
        embed_patterns=["model.embed_tokens.weight"],
        head_patterns=["lm_head.weight"],
        norm_patterns=["model.final_layernorm.weight", "model.final_layernorm.bias"],
    ),
    "qwen2": ArchConfig(
        layer_pattern=r"model\.layers\.(\d+)\.",
        embed_patterns=["model.embed_tokens.weight"],
        head_patterns=["lm_head.weight"],
        norm_patterns=["model.norm.weight"],
    ),
    "gpt2": ArchConfig(
        layer_pattern=r"transformer\.h\.(\d+)\.",
        embed_patterns=["transformer.wte.weight", "transformer.wpe.weight"],
        head_patterns=["lm_head.weight"],
        norm_patterns=["transformer.ln_f.weight", "transformer.ln_f.bias"],
    ),
    "gpt_neox": ArchConfig(
        layer_pattern=r"gpt_neox\.layers\.(\d+)\.",
        embed_patterns=["gpt_neox.embed_in.weight"],
        head_patterns=["embed_out.weight"],
        norm_patterns=["gpt_neox.final_layer_norm.weight"],
    ),
    "bloom": ArchConfig(
        layer_pattern=r"transformer\.h\.(\d+)\.",
        embed_patterns=["word_embeddings.weight", "word_embeddings_layernorm.weight"],
        head_patterns=["lm_head.weight"],
        norm_patterns=["ln_f.weight", "ln_f.bias"],
    ),
    "opt": ArchConfig(
        layer_pattern=r"model\.decoder\.layers\.(\d+)\.",
        embed_patterns=["model.decoder.embed_tokens.weight"],
        head_patterns=["lm_head.weight"],
        norm_patterns=["model.decoder.final_layer_norm.weight"],
    ),
    "falcon": ArchConfig(
        layer_pattern=r"transformer\.h\.(\d+)\.",
        embed_patterns=["transformer.word_embeddings.weight"],
        head_patterns=["lm_head.weight"],
        norm_patterns=["transformer.ln_f.weight", "transformer.ln_f.bias"],
    ),
}

def get_arch_config(model_type: str) -> ArchConfig:
    """Returns config or falls back to llama-style if unknown."""
    model_type_lower = model_type.lower()
    if model_type_lower in ARCH_REGISTRY:
        return ARCH_REGISTRY[model_type_lower]
    return ARCH_REGISTRY["llama"]

def classify_param(name: str, config: ArchConfig) -> ParamClassification:
    """Returns ParamClassification for a param name. Try layer match first, then expert, then router, then embed/head/norm, then MISC."""
    layer_match = re.search(config.layer_pattern, name)
    if layer_match and config.layer_pattern:
        layer_idx = int(layer_match.group(1))
        return ParamClassification(role=ParamRole.LAYER, layer_index=layer_idx)

    if config.expert_pattern:
        expert_match = re.search(config.expert_pattern, name)
        if expert_match:
            layer_idx = int(expert_match.group(1))
            expert_idx = int(expert_match.group(2))
            return ParamClassification(role=ParamRole.MOE_EXPERT, layer_index=layer_idx, expert_index=expert_idx)

    if config.router_pattern:
        router_match = re.search(config.router_pattern, name)
        if router_match:
            layer_idx = int(router_match.group(1))
            return ParamClassification(role=ParamRole.MOE_ROUTER, layer_index=layer_idx)

    for pattern in config.embed_patterns:
        if pattern in name:
            return ParamClassification(role=ParamRole.EMBED)

    for pattern in config.head_patterns:
        if pattern in name:
            return ParamClassification(role=ParamRole.HEAD)

    for pattern in config.norm_patterns:
        if pattern in name:
            return ParamClassification(role=ParamRole.NORM)

    return ParamClassification(role=ParamRole.MISC)

def get_num_layers(model_config: object) -> int:
    """Tries config.num_hidden_layers, config.n_layer, config.num_layers, else raises ValueError."""
    for attr in ["num_hidden_layers", "n_layer", "num_layers"]:
        if hasattr(model_config, attr):
            return getattr(model_config, attr)
    raise ValueError(f"Cannot determine num_layers from config {model_config}")
