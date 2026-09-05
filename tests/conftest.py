"""
tests/conftest.py – Shared pytest configuration and fixtures.
"""
import os
import sys

# Make project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.tiny_memory_bank import TinyMemoryBank, TinyMemoryConfig


def make_blank_mem(config, device=torch.device("cpu"), dtype=torch.float32):
    """Create a fresh, empty continuous memory state dict."""
    bank = TinyMemoryBank(config=config)
    return bank.empty_memory_state(batch_size=1, device=device, dtype=dtype)


def init_bank(config, seed=0, device=torch.device("cpu"), dtype=torch.float32):
    """
    Create TinyMemoryBank + initialized state.
    Returns bank with blank memory state.
    """
    torch.manual_seed(seed)
    bank = TinyMemoryBank(config=config)
    bank.load_memory_state(bank.empty_memory_state(batch_size=1, device=device, dtype=dtype))
    return bank


def apply_write(bank, h, state=None):
    """Apply bank.write and return next_state, diag."""
    if state is None:
        state = bank.empty_memory_state(batch_size=h.size(0), device=h.device, dtype=h.dtype)
    return bank.write(h, state)


def apply_read(bank, h, state=None):
    """Apply bank.read and return retrieved, attn, scores."""
    if state is None:
        state = bank.empty_memory_state(batch_size=h.size(0), device=h.device, dtype=h.dtype)
    return bank.read(h, state)


def apply_decay(bank, state=None):
    """Apply bank.decay_memory."""
    return bank.decay_memory(state)


def apply_fuse(bank, h, m):
    """Apply bank.fuse and return fused output, gate."""
    return bank.fuse(h, m)


def apply_v_proj(bank, h):
    """Apply v_proj and return output."""
    return bank.v_proj(h)
