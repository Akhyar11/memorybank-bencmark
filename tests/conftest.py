"""
tests/conftest.py – Shared pytest configuration and fixtures (PyTorch Version).
"""
import os
import sys
# Make project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.tiny_memory_bank import TinyMemoryBank, TinyMemoryConfig, STATE_EXPIRED, STATE_ACTIVE, STATE_DORMANT


def make_blank_mem(config):
    """Create a fresh, completely empty memory dict (all slots EXPIRED)."""
    bank = TinyMemoryBank(config=config)
    return bank.empty_memory_state()


def init_bank(config, seed=0):
    """
    Create TinyMemoryBank + initialized state.
    Returns bank with blank memory state.
    """
    torch.manual_seed(seed)
    bank = TinyMemoryBank(config=config)
    bank.load_memory_state(bank.empty_memory_state())
    return bank


def apply_write(bank, h, eos=None, wp=None):
    """Apply bank.write and return target_indices."""
    if eos is None:
        eos = torch.ones(h.shape[0], device=h.device)
    if wp is None:
        wp = torch.ones(h.shape[0], device=h.device)
    target_indices = bank.write(h, eos, wp)
    return target_indices


def apply_read(bank, h, rp=None):
    """Apply bank.read and return output."""
    return bank.read(h, read_prob=rp)


def apply_decay(bank):
    """Apply bank.decay_memory and return effective_R."""
    return bank.decay_memory()


def apply_fuse(bank, h, m):
    """Apply bank.fuse and return output."""
    return bank.fuse(h, m)


def apply_v_proj(bank, h):
    """Apply v_proj and return output."""
    return bank.v_proj(h)
