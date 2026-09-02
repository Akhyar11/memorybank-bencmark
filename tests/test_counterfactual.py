"""
tests/test_counterfactual.py – STEP 12 (Fixed)

All apply calls properly preserve params using conftest helpers.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import flax.core
import pytest

from models.tiny_memory_bank import TinyMemoryBank, TinyMemoryConfig, STATE_EXPIRED
from tests.conftest import init_bank, apply_write, apply_read, make_blank_mem


def cosine_sim(a, b):
    import numpy as np
    a = np.array(a).flatten(); b = np.array(b).flatten()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8: return 0.0
    return float(np.dot(a/na, b/nb))


class TestCounterfactualCausal:
    @pytest.fixture
    def setup(self):
        config = TinyMemoryConfig(
            memory_capacity=16, memory_dim=8, hidden_size=8,
            memory_top_k=4,
            mem_decay_rate=0.0001,
        )
        bank, vars_ = init_bank(config, seed=0)
        return bank, vars_, config



    def test_same_key_same_value_high_similarity(self, setup):
        """Same key + same params → identical retrieval."""
        bank, vars_, config = setup
        K = jax.random.normal(jax.random.PRNGKey(43), (1, config.hidden_size))

        vars_A = {'params': vars_['params'], 'memory': make_blank_mem(config)}
        vars_A = apply_write(bank, vars_A, K)
        RA1, _ = apply_read(bank, vars_A, K)

        vars_B = {'params': vars_['params'], 'memory': make_blank_mem(config)}
        vars_B = apply_write(bank, vars_B, K)
        RA2, _ = apply_read(bank, vars_B, K)

        assert jnp.allclose(RA1, RA2, atol=1e-5), "Same key + same value → identical retrievals"

    def test_true_causality_of_stored_value(self, setup):
        """Manually inject different value into slot → different read output. Proves true causality."""
        bank, vars_, config = setup
        K = jax.random.normal(jax.random.PRNGKey(44), (1, config.hidden_size))

        vars_w = {'params': vars_['params'], 'memory': make_blank_mem(config)}
        vars_w = apply_write(bank, vars_w, K)
        R_original, _ = apply_read(bank, vars_w, K)

        # Inject negated value in active slot
        active_slots = jnp.where(vars_w['memory']['state'] == 1)[0]
        assert len(active_slots) > 0
        idx = int(active_slots[0])
        unfrozen = flax.core.unfreeze(vars_w)
        unfrozen['memory']['vals'] = unfrozen['memory']['vals'].at[idx].set(
            -unfrozen['memory']['vals'][idx]
        )
        vars_mod = flax.core.freeze(unfrozen)

        R_modified, _ = apply_read(bank, vars_mod, K)
        assert not jnp.allclose(R_original, R_modified, atol=1e-4), \
            "Changing stored value must change read output"

        sim = cosine_sim(R_original[0], R_modified[0])
        assert sim < 0.5, f"Negated value should produce different retrieval. sim={sim:.4f}"
