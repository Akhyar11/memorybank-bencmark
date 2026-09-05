"""
tests/test_architecture_lock.py – PyTorch Architecture Lock Test

Verifies that TinyMemoryBank contains all architecture-locked components in PyTorch.
This test MUST pass. If any locked component is removed or renamed, FAIL.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn as nn

from models.tiny_memory_bank import TinyMemoryBank, TinyMemoryConfig, sparsemax
from tests.conftest import init_bank, apply_write, apply_read, apply_decay, apply_fuse


@pytest.fixture
def bank_fixture():
    config = TinyMemoryConfig(
        memory_capacity=16, memory_dim=8, hidden_size=8,
    )
    bank = init_bank(config, seed=0)
    return bank, config


class TestLockedComponents:
    """Verify all architecture-locked components exist in PyTorch."""

    def test_locked_projections(self, bank_fixture):
        """q_proj, k_proj, v_proj, write_gate_proj, novelty_proj, fusion_proj, fusion_gate_proj must exist."""
        bank, _ = bank_fixture
        for name in ['q_proj', 'k_proj', 'v_proj', 'write_gate_proj', 'novelty_proj', 'fusion_proj', 'fusion_gate_proj']:
            assert hasattr(bank, name), f"LOCKED COMPONENT MISSING: {name}"
            assert isinstance(getattr(bank, name), nn.Linear), f"{name} must be nn.Linear"

    def test_locked_memory_tensors(self, bank_fixture):
        """All memory state tensors must exist as buffers."""
        bank, _ = bank_fixture
        buffers = dict(bank.named_buffers())
        for name in ['mem_keys', 'mem_vals', 'mem_importance', 'mem_confidence', 'global_step']:
            assert name in buffers, f"LOCKED MEMORY BUFFER MISSING: {name}"

    def test_global_step_exists(self, bank_fixture):
        bank, _ = bank_fixture
        buffers = dict(bank.named_buffers())
        assert 'global_step' in buffers, "global_step must exist"

    def test_locked_methods(self, bank_fixture):
        bank, _ = bank_fixture
        for name in ['decay_memory', 'read', 'write', 'fuse']:
            assert callable(getattr(bank, name, None)), f"{name}() missing"

    def test_projections_have_weights(self, bank_fixture):
        bank, _ = bank_fixture
        for name in ['q_proj', 'k_proj', 'v_proj', 'write_gate_proj', 'novelty_proj', 'fusion_proj', 'fusion_gate_proj']:
            layer = getattr(bank, name)
            assert hasattr(layer, 'weight'), f"{name} must have weight attribute"
            assert layer.weight is not None

    def test_memory_shapes(self, bank_fixture):
        bank, config = bank_fixture
        cap = config.memory_capacity
        dim = config.memory_dim
        assert bank.mem_keys.shape       == (cap, dim)
        assert bank.mem_vals.shape       == (cap, dim)
        assert bank.mem_importance.shape == (cap,)
        assert bank.mem_confidence.shape == (cap,)
        assert bank.global_step.numel()  == 1

    def test_architecture_lock_json(self):
        lock_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'architecture_lock.json'
        )
        assert os.path.exists(lock_path), "architecture_lock.json missing"
        with open(lock_path) as f:
            lock = json.load(f)
        for comp in ['q_proj', 'k_proj', 'v_proj', 'write_gate_proj', 'novelty_proj', 'fusion_proj', 'fusion_gate_proj',
                     'mem_keys', 'mem_vals', 'mem_importance', 'mem_confidence', 'global_step']:
            assert comp in lock['locked_components'], f"{comp} not in architecture_lock.json"


class TestLockedPipeline:
    """Verify locked differentiable pipeline executes correctly."""

    def test_read_uses_q_proj(self, bank_fixture):
        bank, config = bank_fixture
        h = torch.ones((1, config.hidden_size))
        state = bank.empty_memory_state(1, h.device, h.dtype)
        out, attn, scores = bank.read(h, state)
        assert out.shape == (1, config.memory_dim)
        assert attn.shape == (1, config.memory_capacity)

    def test_fuse_uses_fusion_proj(self, bank_fixture):
        bank, config = bank_fixture
        h = torch.ones((2, config.hidden_size))
        m = torch.ones((2, config.memory_dim))
        out, gate = bank.fuse(h, m)
        assert out.shape == (2, config.hidden_size)
        assert gate.shape == (2, config.hidden_size)

    def test_write_uses_kv_proj(self, bank_fixture):
        bank, config = bank_fixture
        h = torch.ones((1, config.hidden_size))
        state = bank.empty_memory_state(1, h.device, h.dtype)
        keys_before = state["keys"].clone()
        next_state, diag = bank.write(h, state)
        assert not torch.allclose(keys_before, next_state["keys"])

    def test_causal_projections(self, bank_fixture):
        """Verify that negating projection weights causally changes outputs."""
        bank, config = bank_fixture
        h = torch.ones((1, config.hidden_size))
        state = bank.empty_memory_state(1, h.device, h.dtype)
        m = torch.ones((1, config.memory_dim))

        # Test fusion_proj causal effect
        out1, _ = bank.fuse(h, m)
        bank.fusion_proj.weight.data.neg_()
        out2, _ = bank.fuse(h, m)
        assert not torch.allclose(out1, out2), "fusion_proj did not causally affect output"
        bank.fusion_proj.weight.data.neg_()  # restore

        # Test q_proj causal effect
        q1 = bank.q_proj(h).clone()
        bank.q_proj.weight.data.neg_()
        q2 = bank.q_proj(h).clone()
        assert not torch.allclose(q1, q2), "q_proj did not causally affect query"
        bank.q_proj.weight.data.neg_()

        # Test k_proj causal effect
        k1 = bank.k_proj(h).clone()
        bank.k_proj.weight.data.neg_()
        k2 = bank.k_proj(h).clone()
        assert not torch.allclose(k1, k2), "k_proj did not causally affect key"
        bank.k_proj.weight.data.neg_()

        # Test v_proj causal effect
        v1 = bank.v_proj(h).clone()
        bank.v_proj.weight.data.neg_()
        v2 = bank.v_proj(h).clone()
        assert not torch.allclose(v1, v2), "v_proj did not causally affect value"
        bank.v_proj.weight.data.neg_()
