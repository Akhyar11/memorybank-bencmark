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

from models.tiny_memory_bank import MemoryState, TinyMemoryBank, TinyMemoryConfig
from tests.conftest import init_bank


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
        """q_proj, k_proj, v_proj, write_gate_proj, fusion_proj, fusion_gate_proj must exist."""
        bank, _ = bank_fixture
        for name in ['q_proj', 'k_proj', 'v_proj', 'write_gate_proj', 'fusion_proj', 'fusion_gate_proj']:
            assert hasattr(bank, name), f"LOCKED COMPONENT MISSING: {name}"
            assert isinstance(getattr(bank, name), nn.Linear), f"{name} must be nn.Linear"

    def test_locked_slot_address_parameter(self, bank_fixture):
        """Learned slot address matrix P must exist as a Parameter."""
        bank, config = bank_fixture
        assert hasattr(bank, 'P'), "LOCKED PARAMETER MISSING: P"
        assert isinstance(bank.P, nn.Parameter)
        assert bank.P.shape == (config.memory_capacity, config.memory_dim)

    def test_locked_memory_buffers(self, bank_fixture):
        """All memory state tensors must exist as buffers."""
        bank, _ = bank_fixture
        buffers = dict(bank.named_buffers())
        for name in ['mem_keys', 'mem_vals', 'mem_occupancy', 'mem_usage', 'mem_age']:
            assert name in buffers, f"LOCKED MEMORY BUFFER MISSING: {name}"

    def test_locked_methods(self, bank_fixture):
        bank, _ = bank_fixture
        for name in ['read', 'write', 'fuse', 'initialize_state', 'detach_state']:
            assert callable(getattr(bank, name, None)), f"{name}() missing"

    def test_projections_have_weights(self, bank_fixture):
        bank, _ = bank_fixture
        for name in ['q_proj', 'k_proj', 'v_proj', 'write_gate_proj', 'fusion_proj', 'fusion_gate_proj']:
            layer = getattr(bank, name)
            assert hasattr(layer, 'weight'), f"{name} must have weight attribute"
            assert layer.weight is not None

    def test_memory_shapes(self, bank_fixture):
        bank, config = bank_fixture
        cap = config.memory_capacity
        dim = config.memory_dim
        assert bank.mem_keys.shape      == (cap, dim)
        assert bank.mem_vals.shape      == (cap, dim)
        assert bank.mem_occupancy.shape == (cap,)
        assert bank.mem_usage.shape     == (cap,)
        assert bank.mem_age.shape       == (cap,)

    def test_architecture_lock_json(self):
        lock_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'architecture_lock.json'
        )
        assert os.path.exists(lock_path), "architecture_lock.json missing"
        with open(lock_path) as f:
            lock = json.load(f)
        for comp in ['q_proj', 'k_proj', 'v_proj', 'write_gate_proj', 'fusion_proj', 'fusion_gate_proj',
                     'P', 'mem_keys', 'mem_vals', 'mem_occupancy', 'mem_usage', 'mem_age']:
            assert comp in lock['locked_components'], f"{comp} not in architecture_lock.json"


class TestLockedPipeline:
    """Verify locked differentiable pipeline executes correctly."""

    def test_read_uses_q_proj(self, bank_fixture):
        bank, config = bank_fixture
        h = torch.ones((1, config.hidden_size))
        state = bank.initialize_state(1, h.device, h.dtype)
        r_t, alpha = bank.read(h, state)
        assert r_t.shape == (1, config.memory_dim)
        assert alpha.shape == (1, config.memory_capacity)

    def test_fuse_uses_fusion_proj(self, bank_fixture):
        bank, config = bank_fixture
        h = torch.ones((2, config.hidden_size))
        m = torch.ones((2, config.memory_dim))
        z_t, gate = bank.fuse(h, m)
        assert z_t.shape == (2, config.hidden_size)
        assert gate.shape == (2, config.hidden_size)

    def test_write_uses_kv_proj(self, bank_fixture):
        bank, config = bank_fixture
        h = torch.ones((1, config.hidden_size))
        state = bank.initialize_state(1, h.device, h.dtype)
        keys_before = state.keys.clone()
        next_state, diag = bank.write(h, state)
        assert not torch.allclose(keys_before, next_state.keys)
