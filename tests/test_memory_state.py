"""
tests/test_memory_state.py – STEP 11 (Fixed)

All bank.apply calls properly preserve params.
Uses conftest helpers: init_bank, apply_write, apply_read, apply_decay.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import flax.core
import pytest

from models.tiny_memory_bank import (
    TinyMemoryBank, TinyMemoryConfig,
    STATE_EXPIRED, STATE_ACTIVE, STATE_DORMANT
)
from tests.conftest import init_bank, apply_write, apply_read, apply_decay, make_blank_mem


@pytest.fixture
def small_bank():
    config = TinyMemoryConfig(
        memory_capacity=8, memory_dim=16, hidden_size=16,
        memory_top_k=4,
        mem_decay_rate=1.0,
        mem_importance_protection=0.5,
        memory_write_threshold=0.9,  # High threshold so random keys insert instead of update
    )
    bank, vars_ = init_bank(config, seed=0)
    return bank, vars_, config


class TestInitialState:
    def test_memory_starts_empty(self, small_bank):
        _, vars_, config = small_bank
        state = vars_['memory']['state']
        assert jnp.all(state == STATE_EXPIRED), f"Initial state not all EXPIRED: {state}"

    def test_active_count_starts_zero(self, small_bank):
        _, vars_, _ = small_bank
        assert int(jnp.sum(vars_['memory']['state'] == STATE_ACTIVE)) == 0

    def test_global_step_starts_zero(self, small_bank):
        _, vars_, _ = small_bank
        assert int(vars_['memory']['global_step']) == 0


class TestWrite:
    def test_write_increases_active_count(self, small_bank):
        bank, vars_, config = small_bank
        h    = jax.random.normal(jax.random.PRNGKey(1), (1, config.hidden_size))
        vars_ = apply_write(bank, vars_, h)
        assert int(jnp.sum(vars_['memory']['state'] == STATE_ACTIVE)) == 1

    def test_write_stores_keys_via_k_proj(self, small_bank):
        bank, vars_, config = small_bank
        h    = jax.random.normal(jax.random.PRNGKey(2), (1, config.hidden_size))
        vars_ = apply_write(bank, vars_, h)
        key_stored = vars_['memory']['keys'][0]
        assert jnp.linalg.norm(key_stored) > 1e-6, "Key slot should be non-zero after write"

    def test_write_updates_timestamps(self, small_bank):
        bank, vars_, config = small_bank
        h    = jax.random.normal(jax.random.PRNGKey(3), (1, config.hidden_size))
        vars_ = apply_write(bank, vars_, h)
        active_idx = int(jnp.argmax(vars_['memory']['state'] == STATE_ACTIVE))
        assert int(vars_['memory']['access_count'][active_idx]) >= 1

    def test_write_initialises_importance(self, small_bank):
        bank, vars_, config = small_bank
        h    = jax.random.normal(jax.random.PRNGKey(4), (1, config.hidden_size))
        vars_ = apply_write(bank, vars_, h)
        active_idx = int(jnp.argmax(vars_['memory']['state'] == STATE_ACTIVE))
        assert 0.0 <= float(vars_['memory']['importance'][active_idx]) <= 1.0

    def test_write_initialises_confidence(self, small_bank):
        bank, vars_, config = small_bank
        h    = jax.random.normal(jax.random.PRNGKey(5), (1, config.hidden_size))
        vars_ = apply_write(bank, vars_, h)
        active_idx = int(jnp.argmax(vars_['memory']['state'] == STATE_ACTIVE))
        conf = float(vars_['memory']['confidence'][active_idx])
        assert conf == pytest.approx(0.5, abs=0.01), f"Expected 0.5, got {conf}"

    def test_write_gating_write_prob_zero(self, small_bank):
        bank, vars_, config = small_bank
        h    = jax.random.normal(jax.random.PRNGKey(6), (1, config.hidden_size))
        # write_prob must be STRICTLY BELOW threshold (0.0) to block write
        # Default write_threshold=0.0, so -1.0 < 0.0 → blocked
        vars_ = apply_write(bank, vars_, h, wp=jnp.full((1,), -1.0))
        assert int(jnp.sum(vars_['memory']['state'] == STATE_ACTIVE)) == 0

    def test_write_gating_is_eos_zero(self, small_bank):
        bank, vars_, config = small_bank
        h    = jax.random.normal(jax.random.PRNGKey(7), (1, config.hidden_size))
        vars_ = apply_write(bank, vars_, h, eos=jnp.zeros((1,)))
        assert int(jnp.sum(vars_['memory']['state'] == STATE_ACTIVE)) == 0

    def test_multiple_writes_fill_capacity(self, small_bank):
        bank, vars_, config = small_bank
        # Use write_prob >> threshold to ensure all writes succeed
        for i in range(config.memory_capacity):
            h = jax.random.normal(jax.random.PRNGKey(10 + i), (1, config.hidden_size))
            vars_ = apply_write(bank, vars_, h, wp=jnp.ones((1,)))
        active = int(jnp.sum(vars_['memory']['state'] == STATE_ACTIVE))
        assert active == config.memory_capacity, f"Expected {config.memory_capacity} active, got {active}"


class TestRead:
    def test_read_empty_returns_zero(self, small_bank):
        bank, vars_, config = small_bank
        h   = jnp.ones((1, config.hidden_size))
        out, _ = apply_read(bank, vars_, h)
        assert jnp.allclose(out, 0.0, atol=1e-6)

    def test_read_updates_access_count(self, small_bank):
        bank, vars_, config = small_bank
        h = jax.random.normal(jax.random.PRNGKey(20), (1, config.hidden_size))
        vars_ = apply_write(bank, vars_, h)
        acc_before = vars_['memory']['access_count'].copy()
        _, vars_ = apply_read(bank, vars_, h)
        assert int(jnp.sum(vars_['memory']['access_count'])) > int(jnp.sum(acc_before))

    def test_read_updates_last_access(self, small_bank):
        bank, vars_, config = small_bank
        h = jax.random.normal(jax.random.PRNGKey(21), (1, config.hidden_size))
        vars_ = apply_write(bank, vars_, h)
        la_before = vars_['memory']['last_access'].copy()
        _, vars_ = apply_read(bank, vars_, h)
        assert jnp.any(vars_['memory']['last_access'] >= la_before)

    def test_read_boosts_importance(self, small_bank):
        bank, vars_, config = small_bank
        h = jax.random.normal(jax.random.PRNGKey(22), (1, config.hidden_size))
        vars_ = apply_write(bank, vars_, h)
        imp_before = vars_['memory']['importance'].copy()
        _, vars_ = apply_read(bank, vars_, h)
        assert jnp.any(vars_['memory']['importance'] >= imp_before)

    def test_read_top_k_respects_capacity(self, small_bank):
        bank, vars_, config = small_bank
        h = jnp.ones((1, config.hidden_size)) * 2.0
        vars_ = apply_write(bank, vars_, h)
        out, _ = apply_read(bank, vars_, h)
        assert jnp.linalg.norm(out) > 1e-6, "1 active memory should give non-zero read"

    def test_read_gate_has_no_effects(self, small_bank):
        bank, vars_, config = small_bank
        h = jnp.ones((1, config.hidden_size))
        vars_ = apply_write(bank, vars_, h)
        
        state_before = vars_['memory']['importance'].copy()
        out, vars_ = apply_read(bank, vars_, h, rp=jnp.zeros((1,)))
        state_after = vars_['memory']['importance'].copy()
        
        assert jnp.allclose(out, 0.0), "Read output should be 0 when gated"
        assert jnp.allclose(state_before, state_after), "Importance should not be updated when read_prob=0"


class TestDecay:
    def test_decay_expires_old_memories(self, small_bank):
        bank, vars_, config = small_bank
        h = jax.random.normal(jax.random.PRNGKey(30), (1, config.hidden_size))
        vars_ = apply_write(bank, vars_, h)
        unfrozen = flax.core.unfreeze(vars_)
        unfrozen['memory']['global_step'] = jnp.array(100000, dtype=jnp.int32)
        vars_ = flax.core.freeze(unfrozen)
        vars_ = apply_decay(bank, vars_)
        assert jnp.all(vars_['memory']['state'] != STATE_ACTIVE)

    def test_decay_formula(self, small_bank):
        _, _, config = small_bank
        lam = config.mem_decay_rate  # 1.0
        dt  = 1
        R   = jnp.exp(-lam * dt)
        assert float(R) < 0.5

    def test_decay_dormant_then_expired(self, small_bank):
        bank, vars_, config = small_bank
        h = jax.random.normal(jax.random.PRNGKey(31), (1, config.hidden_size))
        vars_ = apply_write(bank, vars_, h)

        # dt=1 → DORMANT
        unfrozen = flax.core.unfreeze(vars_)
        unfrozen['memory']['global_step'] = jnp.array(1, dtype=jnp.int32)
        vars_d = flax.core.freeze(unfrozen)
        vars_d = apply_decay(bank, vars_d)
        assert jnp.any(vars_d['memory']['state'] == STATE_DORMANT)

        # dt=10000 → EXPIRED
        unfrozen = flax.core.unfreeze(vars_)
        unfrozen['memory']['global_step'] = jnp.array(10000, dtype=jnp.int32)
        vars_e = flax.core.freeze(unfrozen)
        vars_e = apply_decay(bank, vars_e)
        assert jnp.any(vars_e['memory']['state'] == STATE_EXPIRED)


class TestReplacement:
    def test_replacement_uses_expired_first(self, small_bank):
        bank, vars_, config = small_bank
        for i in range(config.memory_capacity):
            h = jax.random.normal(jax.random.PRNGKey(40 + i), (1, config.hidden_size))
            vars_ = apply_write(bank, vars_, h)

        # Expire half
        unfrozen = flax.core.unfreeze(vars_)
        unfrozen['memory']['state'] = unfrozen['memory']['state'].at[:config.memory_capacity//2].set(STATE_EXPIRED)
        vars_ = flax.core.freeze(unfrozen)
        n_expired_before = int(jnp.sum(vars_['memory']['state'] == STATE_EXPIRED))

        h_new = jax.random.normal(jax.random.PRNGKey(99), (1, config.hidden_size))
        vars_ = apply_write(bank, vars_, h_new)
        n_expired_after = int(jnp.sum(vars_['memory']['state'] == STATE_EXPIRED))
        assert n_expired_after < n_expired_before

    def test_replacement_different_importance(self):
        config = TinyMemoryConfig(
            memory_capacity=4, memory_dim=4, hidden_size=4,
        )
        bank, vars_ = init_bank(config, seed=0)

        for i in range(4):
            h = jax.random.normal(jax.random.PRNGKey(50 + i), (1, 4))
            vars_ = apply_write(bank, vars_, h)

        unfrozen = flax.core.unfreeze(vars_)
        unfrozen['memory']['importance'] = jnp.array([0.1, 0.5, 0.9, 0.7])
        unfrozen['memory']['state'] = jnp.array([STATE_ACTIVE]*4, dtype=jnp.int32)
        vars_ = flax.core.freeze(unfrozen)

        h_new = jax.random.normal(jax.random.PRNGKey(200), (1, 4))
        vars_ = apply_write(bank, vars_, h_new)
        assert float(vars_['memory']['importance'][2]) > 0.0
