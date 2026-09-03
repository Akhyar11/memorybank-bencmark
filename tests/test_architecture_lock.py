"""
tests/test_architecture_lock.py – STEP 10

Verifies that TinyMemoryBank contains all architecture-locked components.
This test MUST pass. If any locked component is removed or renamed, FAIL.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import flax.core
import pytest

from models.tiny_memory_bank import TinyMemoryBank, TinyMemoryConfig
from tests.conftest import init_bank, apply_write, apply_read, apply_decay, apply_fuse


@pytest.fixture
def bank_and_vars():
    config = TinyMemoryConfig(
        memory_capacity=16, memory_dim=8, hidden_size=8,
        memory_top_k=4,
    )
    bank, vars_ = init_bank(config, seed=0)
    return bank, vars_, config


class TestLockedComponents:
    """Verify all architecture-locked components exist."""

    def test_locked_projections(self, bank_and_vars):
        """q_proj, k_proj, v_proj, i_proj, fusion_proj must exist in params."""
        _, vars_, _ = bank_and_vars
        params = vars_['params']
        for name in ['q_proj', 'k_proj', 'v_proj', 'i_proj', 'fusion_proj']:
            assert name in params, f"LOCKED COMPONENT MISSING: {name}"

    def test_locked_memory_tensors(self, bank_and_vars):
        """All memory state tensors must exist."""
        _, vars_, _ = bank_and_vars
        memory = vars_['memory']
        for name in ['keys', 'vals', 'importance', 'confidence',
                     'created_at', 'last_access', 'access_count', 'state', 'global_step']:
            assert name in memory, f"LOCKED MEMORY TENSOR MISSING: {name}"

    def test_global_step_not_step(self, bank_and_vars):
        _, vars_, _ = bank_and_vars
        assert 'global_step' in vars_['memory'], "global_step must exist"
        assert 'step' not in vars_['memory'], "Old name 'step' must not exist"

    def test_locked_methods(self, bank_and_vars):
        bank, _, _ = bank_and_vars
        for name in ['decay_memory', 'read', 'write', 'fuse', '__call__']:
            assert callable(getattr(bank, name, None)), f"{name}() missing"

    def test_q_proj_is_dense(self, bank_and_vars):
        _, vars_, _ = bank_and_vars
        assert 'kernel' in vars_['params']['q_proj'], "q_proj must be Dense"

    def test_k_proj_is_dense(self, bank_and_vars):
        _, vars_, _ = bank_and_vars
        assert 'kernel' in vars_['params']['k_proj'], "k_proj must be Dense"

    def test_v_proj_is_dense(self, bank_and_vars):
        _, vars_, _ = bank_and_vars
        assert 'kernel' in vars_['params']['v_proj'], "v_proj must be Dense"

    def test_i_proj_is_dense(self, bank_and_vars):
        _, vars_, _ = bank_and_vars
        assert 'kernel' in vars_['params']['i_proj'], "i_proj must be Dense"

    def test_fusion_proj_is_dense(self, bank_and_vars):
        _, vars_, _ = bank_and_vars
        assert 'kernel' in vars_['params']['fusion_proj'], "fusion_proj must be Dense"

    def test_memory_shapes(self, bank_and_vars):
        _, vars_, config = bank_and_vars
        cap = config.memory_capacity
        dim = config.memory_dim
        mem = vars_['memory']
        assert mem['keys'].shape         == (cap, dim)
        assert mem['vals'].shape         == (cap, dim)
        assert mem['importance'].shape   == (cap,)
        assert mem['confidence'].shape   == (cap,)
        assert mem['created_at'].shape   == (cap,)
        assert mem['last_access'].shape  == (cap,)
        assert mem['access_count'].shape == (cap,)
        assert mem['state'].shape        == (cap,)
        assert mem['global_step'].shape  == ()

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

    def test_call_pipeline_runs(self, bank_and_vars):
        bank, vars_, config = bank_and_vars
        h   = jnp.ones((2, config.hidden_size))
        r   = jnp.ones((2,))
        w   = jnp.ones((2,))
        out, new_mem = bank.apply(vars_, h, r, w, False, mutable=['memory'])
        assert out.shape == (2, config.hidden_size)

    def test_decay_memory_updates_state(self, bank_and_vars):
        bank, vars_, config = bank_and_vars
        unfrozen = flax.core.unfreeze(vars_)
        unfrozen['memory']['global_step'] = jnp.array(100000, dtype=jnp.int32)
        unfrozen['memory']['state'] = jnp.ones((config.memory_capacity,), dtype=jnp.int32)
        vars_mod = flax.core.freeze(unfrozen)
        vars_mod = apply_decay(bank, vars_mod)
        assert jnp.any(vars_mod['memory']['state'] == 0) or jnp.any(vars_mod['memory']['state'] == 2)

    def test_read_uses_q_proj(self, bank_and_vars):
        bank, vars_, config = bank_and_vars
        h = jnp.ones((1, config.hidden_size))
        out, _ = apply_read(bank, vars_, h)
        assert out.shape == (1, config.memory_dim)

    def test_fuse_uses_fusion_proj(self, bank_and_vars):
        bank, vars_, config = bank_and_vars
        h = jnp.ones((2, config.hidden_size))
        m = jnp.ones((2, config.memory_dim))
        out = apply_fuse(bank, vars_, h, m)
        assert out.shape == (2, config.hidden_size)

    def test_write_uses_kv_proj(self, bank_and_vars):
        bank, vars_, config = bank_and_vars
        h = jnp.ones((1, config.hidden_size))
        keys_before = vars_['memory']['keys'].copy()
        vars_after  = apply_write(bank, vars_, h)
        assert not jnp.allclose(keys_before, vars_after['memory']['keys'])

    def test_causal_projections(self, bank_and_vars):
        """Verify that negating projection weights causally changes outputs."""
        bank, vars_, config = bank_and_vars
        h = jnp.ones((1, config.hidden_size))
        
        # Test fusion_proj causal effect
        out1 = apply_fuse(bank, vars_, h, h)
        unfrozen = flax.core.unfreeze(vars_)
        unfrozen['params']['fusion_proj']['kernel'] = -unfrozen['params']['fusion_proj']['kernel']
        vars_neg = flax.core.freeze(unfrozen)
        out2 = apply_fuse(bank, vars_neg, h, h)
        assert not jnp.allclose(out1, out2), "fusion_proj did not causally affect output"

        # Test q_proj causal effect
        q1 = bank.apply(vars_, h, method=lambda mdl, x: mdl.q_proj(x))
        unfrozen = flax.core.unfreeze(vars_)
        unfrozen['params']['q_proj']['kernel'] = -unfrozen['params']['q_proj']['kernel']
        vars_neg = flax.core.freeze(unfrozen)
        q2 = bank.apply(vars_neg, h, method=lambda mdl, x: mdl.q_proj(x))
        assert not jnp.allclose(q1, q2), "q_proj did not causally affect query"

        # Test k_proj causal effect
        k1 = bank.apply(vars_, h, method=lambda mdl, x: mdl.k_proj(x))
        unfrozen = flax.core.unfreeze(vars_)
        unfrozen['params']['k_proj']['kernel'] = -unfrozen['params']['k_proj']['kernel']
        vars_neg = flax.core.freeze(unfrozen)
        k2 = bank.apply(vars_neg, h, method=lambda mdl, x: mdl.k_proj(x))
        assert not jnp.allclose(k1, k2), "k_proj did not causally affect key"

        # Test v_proj causal effect
        v1 = bank.apply(vars_, h, method=lambda mdl, x: mdl.v_proj(x))
        unfrozen = flax.core.unfreeze(vars_)
        unfrozen['params']['v_proj']['kernel'] = -unfrozen['params']['v_proj']['kernel']
        vars_neg = flax.core.freeze(unfrozen)
        v2 = bank.apply(vars_neg, h, method=lambda mdl, x: mdl.v_proj(x))
        assert not jnp.allclose(v1, v2), "v_proj did not causally affect value"

        # Test i_proj causal effect
        i1 = bank.apply(vars_, h, method=lambda mdl, x: mdl.i_proj(x))
        unfrozen = flax.core.unfreeze(vars_)
        unfrozen['params']['i_proj']['kernel'] = -unfrozen['params']['i_proj']['kernel']
        vars_neg = flax.core.freeze(unfrozen)
        i2 = bank.apply(vars_neg, h, method=lambda mdl, x: mdl.i_proj(x))
        assert not jnp.allclose(i1, i2), "i_proj did not causally affect importance"
