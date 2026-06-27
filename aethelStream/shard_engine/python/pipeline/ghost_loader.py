from __future__ import annotations
from dataclasses import dataclass
import gc
from typing import TYPE_CHECKING
import psutil
import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel

@dataclass
class MemStats:
    vram_delta_mb: float
    ram_delta_mb: float
    passed: bool
    message: str

def ghost_load(model_id: str) -> tuple[PreTrainedModel, object, MemStats]:
    """
    Load model on meta device (zero allocation).
    Returns (ghost_model, config, mem_stats).
    Asserts VRAM delta < 50 MB and RAM delta < 200 MB.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        vram_before = torch.cuda.memory_allocated()
    else:
        vram_before = 0
    ram_before = psutil.virtual_memory().used

    config = AutoConfig.from_pretrained(model_id)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    if torch.cuda.is_available():
        vram_after = torch.cuda.memory_allocated()
    else:
        vram_after = 0
    ram_after = psutil.virtual_memory().used

    vram_delta = (vram_after - vram_before) / (1024 * 1024)
    ram_delta = (ram_after - ram_before) / (1024 * 1024)

    passed = vram_delta < 50.0 and ram_delta < 200.0
    message = (
        f"VRAM Δ={vram_delta:.2f} MiB, RAM Δ={ram_delta:.2f} MiB"
        if passed
        else f"FAIL: VRAM Δ={vram_delta:.2f} MiB (limit 50), RAM Δ={ram_delta:.2f} MiB (limit 200)"
    )
    return model, config, MemStats(vram_delta, ram_delta, passed, message)
