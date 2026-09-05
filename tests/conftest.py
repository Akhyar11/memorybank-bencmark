"""
tests/conftest.py – Shared pytest configuration and fixtures.
"""
import os
import sys

# Make project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.tiny_memory_bank import MemoryState, TinyMemoryBank, TinyMemoryConfig


def make_blank_mem(config, device=torch.device("cpu"), dtype=torch.float32):
    """Create a fresh, empty continuous memory state."""
    bank = TinyMemoryBank(config=config)
    return bank.initialize_state(batch_size=1, device=device, dtype=dtype)


def init_bank(config, seed=0, device=torch.device("cpu"), dtype=torch.float32):
    """
    Create TinyMemoryBank + initialized state.
    Returns bank.
    """
    torch.manual_seed(seed)
    bank = TinyMemoryBank(config=config)
    return bank
