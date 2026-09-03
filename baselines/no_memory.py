"""
baselines/no_memory.py (Pure PyTorch Version)

Baseline with absolutely no memory contribution.
Passes the encoder query representation directly to downstream decoder.
"""
import torch
import torch.nn as nn


class NoMemory(nn.Module):
    """
    No-Memory Baseline: passes query representation through with zero memory contribution.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, h_eos: torch.Tensor):
        # Query passes directly; memory contribution is strictly zero
        return h_eos
