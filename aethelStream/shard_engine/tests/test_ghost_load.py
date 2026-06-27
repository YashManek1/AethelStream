"""Test 1 — Zero-memory ghost initialization."""
from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest

def test_ghost_load_zero_vram(monkeypatch):
    """Ghost load must not allocate VRAM."""
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    
    mock_vmem = MagicMock()
    mock_vmem.used = 1000 * 1024 * 1024  # 1 GB before
    with patch("psutil.virtual_memory", return_value=mock_vmem):
        from python.pipeline.ghost_loader import ghost_load
        # Use a mock so we don't download anything
        with patch("python.pipeline.ghost_loader.AutoConfig") as mock_config, \
             patch("python.pipeline.ghost_loader.AutoModelForCausalLM") as mock_model, \
             patch("python.pipeline.ghost_loader.init_empty_weights") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda s: None
            mock_ctx.return_value.__exit__ = lambda s, *a: False
            mock_config.from_pretrained.return_value = MagicMock()
            mock_model.from_config.return_value = MagicMock()
            _, _, stats = ghost_load("gpt2")
    assert stats.vram_delta_mb < 50.0
    assert stats.ram_delta_mb < 200.0
    assert stats.passed

def test_ghost_load_fails_on_large_vram_delta(monkeypatch):
    """MemStats.passed must be False when VRAM delta > 50 MB."""
    import torch
    call_count = 0
    def mock_memory_allocated():
        nonlocal call_count
        call_count += 1
        return 0 if call_count == 1 else 200 * 1024 * 1024  # 200 MB after
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_allocated", mock_memory_allocated)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    mock_vmem = MagicMock()
    mock_vmem.used = 0
    with patch("psutil.virtual_memory", return_value=mock_vmem):
        from python.pipeline.ghost_loader import ghost_load
        with patch("python.pipeline.ghost_loader.AutoConfig") as mock_config, \
             patch("python.pipeline.ghost_loader.AutoModelForCausalLM") as mock_model, \
             patch("python.pipeline.ghost_loader.init_empty_weights") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda s: None
            mock_ctx.return_value.__exit__ = lambda s, *a: False
            mock_config.from_pretrained.return_value = MagicMock()
            mock_model.from_config.return_value = MagicMock()
            _, _, stats = ghost_load("gpt2")
    assert not stats.passed

@pytest.mark.slow
def test_ghost_load_gpt2_real():
    """Integration: real GPT-2 ghost load on actual HF model."""
    from python.pipeline.ghost_loader import ghost_load
    _, config, stats = ghost_load("gpt2")
    assert stats.passed, stats.message
    assert config.model_type == "gpt2"
