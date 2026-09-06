"""
tests/test_architecture_lock.py – Architecture Lock Test

Verifies that the new Turn-Level Semantic Memory Bank architecture
contains all locked components, parameters, and methods.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn as nn

from models.gpt2_memory_model import GPT2MemoryModel
from models.tiny_memory_bank import MemoryState, TinyMemoryBank, TinyMemoryConfig

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gpt2-indo-instruct-tuned")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "izzulgod/gpt2-indo-instruct-tuned"


@pytest.fixture
def bank_fixture():
    config = TinyMemoryConfig(memory_capacity=16, memory_dim=8, hidden_size=8)
    bank = TinyMemoryBank(config=config)
    return bank, config


class TestLockedComponents:
    """Verify all architecture-locked components exist in PyTorch."""

    def test_locked_methods(self, bank_fixture):
        bank, _ = bank_fixture
        for name in ['read', 'add_memory', 'step_turn', 'evict_lifecycle', 'reset_memory']:
            assert callable(getattr(bank, name, None)), f"{name}() missing from TinyMemoryBank"

    def test_locked_model_projections(self):
        config = TinyMemoryConfig(memory_dim=768, hidden_size=768)
        model = GPT2MemoryModel(model_name_or_path=MODEL_PATH, memory_config=config, freeze_backbone=True)
        assert hasattr(model, 'c_proj'), "LOCKED PROJECTION MISSING: c_proj"
        assert hasattr(model, 'fusion_proj'), "LOCKED PROJECTION MISSING: fusion_proj"
        assert isinstance(model.c_proj, nn.Linear)
        assert isinstance(model.fusion_proj, nn.Linear)
        assert model.c_proj.weight.shape == (768, 768)
        assert model.fusion_proj.weight.shape == (768, 1536)

    def test_architecture_lock_json(self):
        lock_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'architecture_lock.json'
        )
        assert os.path.exists(lock_path), "architecture_lock.json missing"
        with open(lock_path) as f:
            lock = json.load(f)
        for comp in ['TinyMemoryBank', 'GPT2MemoryModel', 'c_proj', 'fusion_proj', 'add_memory', 'read', 'step_turn', 'evict_lifecycle']:
            assert comp in lock['locked_components'], f"{comp} not in architecture_lock.json"


class TestLockedPipeline:
    """Verify locked differentiable pipeline executes correctly."""

    def test_read_pipeline(self, bank_fixture):
        bank, config = bank_fixture
        # Add sample memory
        v = torch.randn(config.memory_dim)
        bank.add_memory(v)

        q = torch.randn(1, config.memory_dim)
        m_bar, top_idx = bank.read(q, top_k=1)
        assert m_bar.shape == (1, config.memory_dim)
        assert len(top_idx) == 1

    def test_evict_pipeline(self, bank_fixture):
        bank, config = bank_fixture
        for _ in range(5):
            bank.add_memory(torch.randn(config.memory_dim))
        for _ in range(3):
            bank.step_turn()
        bank.read_counts = [50, 50, 50, 50, 0]
        evicted = bank.evict_lifecycle(threshold_ratio=0.05, min_age=3)
        assert evicted == 1
        assert bank.num_memories == 4
