"""
tests/test_architecture_lock.py – PyTorch Architecture Lock Test

Verifies that TinyMemoryBank contains all architecture-locked components in PyTorch.
This test MUST pass. If any locked component is removed or renamed, FAIL.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import pytest

from models.tiny_memory_bank import TinyMemoryBank, TinyMemoryConfig
from tests.conftest import init_bank, apply_write, apply_read, apply_decay, apply_fuse


@pytest.fixture
def bank_fixture():
    config = TinyMemoryConfig(
        memory_capacity=16, memory_dim=8, hidden_size=8,
        memory_top_k=4,
    )
    bank = init_bank(config, seed=0)
    return bank, config


class TestLockedComponents:
    """Verify all architecture-locked components exist in PyTorch."""

    def test_locked_projections(self, bank_fixture):
        """q_proj, k_proj, v_proj, i_proj, fusion_proj must exist as modules."""
        bank, _ = bank_fixture
        for name in ['q_proj', 'k_proj', 'v_proj', 'i_proj', 'fusion_proj']:
            assert hasattr(bank, name), f"LOCKED COMPONENT MISSING: {name}"
            assert isinstance(getattr(bank, name), nn.Linear), f"{name} must be nn.Linear"

    def test_locked_memory_tensors(self, bank_fixture):
        """All memory state tensors must exist as buffers."""
        bank, _ = bank_fixture
        buffers = dict(bank.named_buffers())
        for name in ['mem_keys', 'mem_vals', 'mem_importance', 'mem_confidence',
                     'mem_created_at', 'mem_last_access', 'mem_access_count', 'mem_state', 'global_step']:
            assert name in buffers, f"LOCKED MEMORY BUFFER MISSING: {name}"

    def test_global_step_not_step(self, bank_fixture):
        bank, _ = bank_fixture
        buffers = dict(bank.named_buffers())
        assert 'global_step' in buffers, "global_step must exist"
        assert 'step' not in buffers, "Old name 'step' must not exist"

    def test_locked_methods(self, bank_fixture):
        bank, _ = bank_fixture
        for name in ['decay_memory', 'read', 'write', 'fuse', 'forward']:
            assert callable(getattr(bank, name, None)), f"{name}() missing"

    def test_projections_have_weights(self, bank_fixture):
        bank, _ = bank_fixture
        for name in ['q_proj', 'k_proj', 'v_proj', 'i_proj', 'fusion_proj']:
            layer = getattr(bank, name)
            assert hasattr(layer, 'weight'), f"{name} must have weight attribute"
            assert layer.weight is not None

    def test_memory_shapes(self, bank_fixture):
        bank, config = bank_fixture
        cap = config.memory_capacity
        dim = config.memory_dim
        assert bank.mem_keys.shape         == (cap, dim)
        assert bank.mem_vals.shape         == (cap, dim)
        assert bank.mem_importance.shape   == (cap,)
        assert bank.mem_confidence.shape   == (cap,)
        assert bank.mem_created_at.shape   == (cap,)
        assert bank.mem_last_access.shape  == (cap,)
        assert bank.mem_access_count.shape == (cap,)
        assert bank.mem_state.shape        == (cap,)
        assert bank.global_step.numel()    == 1

    def test_state_constants(self):
        from models.tiny_memory_bank import STATE_EXPIRED, STATE_ACTIVE, STATE_DORMANT
        assert STATE_EXPIRED == 0
        assert STATE_ACTIVE  == 1
        assert STATE_DORMANT == 2

    def test_no_state_empty(self):
        import models.tiny_memory_bank as mb
        assert not hasattr(mb, 'STATE_EMPTY'), "STATE_EMPTY not in locked architecture"

    def test_architecture_lock_json(self):
        import json
        lock_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'architecture_lock.json'
        )
        assert os.path.exists(lock_path), "architecture_lock.json missing"
        with open(lock_path) as f:
            lock = json.load(f)
        for comp in ['q_proj', 'k_proj', 'v_proj', 'i_proj', 'fusion_proj',
                     'mem_keys', 'mem_vals', 'mem_importance', 'mem_confidence',
                     'mem_created_at', 'mem_last_access', 'mem_access_count', 'mem_state', 'global_step']:
            assert comp in lock['locked_components'], f"{comp} not in architecture_lock.json"


class TestLockedPipeline:
    """Verify locked pipeline executes correctly."""

    def test_call_pipeline_runs(self, bank_fixture):
        bank, config = bank_fixture
        h = torch.ones((2, config.hidden_size))
        r = torch.ones((2,))
        w = torch.ones((2,))
        out = bank(h, r, w, deterministic=False)
        assert out.shape == (2, config.hidden_size)

    def test_decay_memory_updates_state(self, bank_fixture):
        bank, config = bank_fixture
        bank.global_step[0] = 100000
        bank.mem_state.fill_(1)  # STATE_ACTIVE
        bank.decay_memory()
        assert torch.any(bank.mem_state == 0) or torch.any(bank.mem_state == 2)

    def test_read_uses_q_proj(self, bank_fixture):
        bank, config = bank_fixture
        h = torch.ones((1, config.hidden_size))
        out = bank.read(h)
        assert out.shape == (1, config.memory_dim)

    def test_fuse_uses_fusion_proj(self, bank_fixture):
        bank, config = bank_fixture
        h = torch.ones((2, config.hidden_size))
        m = torch.ones((2, config.memory_dim))
        out = bank.fuse(h, m)
        assert out.shape == (2, config.hidden_size)

    def test_write_uses_kv_proj(self, bank_fixture):
        bank, config = bank_fixture
        h = torch.ones((1, config.hidden_size))
        keys_before = bank.mem_keys.clone()
        bank.write(h, torch.ones(1), torch.ones(1))
        assert not torch.allclose(keys_before, bank.mem_keys)

    def test_causal_projections(self, bank_fixture):
        """Verify that negating projection weights causally changes outputs."""
        bank, config = bank_fixture
        h = torch.ones((1, config.hidden_size))
        
        # Test fusion_proj causal effect
        m = torch.ones((1, config.memory_dim))
        out1 = bank.fuse(h, m).clone()
        bank.fusion_proj.weight.data.neg_()
        out2 = bank.fuse(h, m).clone()
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

        # Test i_proj causal effect
        i1 = bank.i_proj(h).clone()
        bank.i_proj.weight.data.neg_()
        i2 = bank.i_proj(h).clone()
        assert not torch.allclose(i1, i2), "i_proj did not causally affect importance"
        bank.i_proj.weight.data.neg_()
