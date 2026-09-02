"""
tests/conftest.py – Shared pytest configuration and fixtures.
"""
import os
import sys
# Make project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp

from models.tiny_memory_bank import TinyMemoryBank, TinyMemoryConfig, STATE_EXPIRED


def make_blank_mem(config):
    """Create a fresh, completely empty memory dict (all slots EXPIRED)."""
    cap = config.memory_capacity
    dim = config.memory_dim
    return {
        'keys':         jnp.zeros((cap, dim),  dtype=jnp.float32),
        'vals':         jnp.zeros((cap, dim),  dtype=jnp.float32),
        'importance':   jnp.zeros((cap,),      dtype=jnp.float32),
        'confidence':   jnp.zeros((cap,),      dtype=jnp.float32),
        'created_at':   jnp.zeros((cap,),      dtype=jnp.int32),
        'last_access':  jnp.zeros((cap,),      dtype=jnp.int32),
        'access_count': jnp.zeros((cap,),      dtype=jnp.int32),
        'state':        jnp.full((cap,), STATE_EXPIRED, dtype=jnp.int32),
        'global_step':  jnp.zeros((),          dtype=jnp.int32),
    }


def init_bank(config, seed=0):
    """
    Create TinyMemoryBank + properly initialised variables.
    Returns (bank, vars) where vars = {'params': ..., 'memory': empty_state}.
    
    Key: we call __call__ which touches all projections (including k/v/i via *0),
    ensuring ALL parameters are registered, then replace memory with blank state.
    """
    bank = TinyMemoryBank(config=config)
    rng  = jax.random.PRNGKey(seed)
    h    = jnp.ones((1, config.hidden_size))
    # __call__ touches k_proj, v_proj, i_proj via the *0.0 path
    raw_vars = bank.init(rng, h, jnp.ones((1,)), jnp.ones((1,)), False)
    # Start with truly blank memory (not init'd dummy values)
    vars_ = {'params': raw_vars['params'], 'memory': make_blank_mem(config)}
    return bank, vars_


def apply_write(bank, vars_, h, eos=None, wp=None):
    """
    Apply bank.write and return updated vars_ dict (params preserved).
    """
    if eos is None: eos = jnp.ones((h.shape[0],))
    if wp  is None: wp  = jnp.ones((h.shape[0],))
    _, new_mem = bank.apply(vars_, h, eos, wp, method=bank.write, mutable=['memory'])
    return {'params': vars_['params'], 'memory': new_mem['memory']}


def apply_read(bank, vars_, h):
    """
    Apply bank.read and return (output, updated vars_).
    """
    out, new_mem = bank.apply(vars_, h, method=bank.read, mutable=['memory'])
    return out, {'params': vars_['params'], 'memory': new_mem['memory']}


def apply_decay(bank, vars_):
    """
    Apply bank.decay_memory and return updated vars_.
    """
    _, new_mem = bank.apply(vars_, method=bank.decay_memory, mutable=['memory'])
    return {'params': vars_['params'], 'memory': new_mem['memory']}


def apply_fuse(bank, vars_, h, m):
    """
    Apply bank.fuse and return output.
    """
    out, _ = bank.apply(vars_, h, m, method=bank.fuse, mutable=['memory'])
    return out


def apply_v_proj(bank, vars_, h):
    """Apply v_proj and return output."""
    out, _ = bank.apply(vars_, h, method=lambda module, x: module.v_proj(x), mutable=['memory'])
    return out


def apply_q_proj(bank, vars_, h):
    """Apply q_proj and return output."""
    out, _ = bank.apply(vars_, h, method=lambda module, x: module.q_proj(x), mutable=['memory'])
    return out
