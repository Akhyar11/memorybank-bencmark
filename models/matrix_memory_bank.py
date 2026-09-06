"""
models/matrix_memory_bank.py
============================
Differentiable Memory Matrix Engine based on the continuous linear associative design:
  - Memory Matrix: M in R^(128 x 768), non-trainable state (requires_grad = False).
  - Query dot product: s = M @ q  (128-dimensional continuous activations).
  - Memory reconstruction: m = M^T @ s = (M^T @ M) @ q  (768-dimensional reconstructed vector).
  - No cosine similarity, no softmax, no argmax/top-k truncation on memory read.
  - 100% differentiable linear gradient flow: d(m)/d(q) = (1/sqrt(d)) * M^T @ M.
"""

import math
from typing import Any, Optional, Tuple
import torch
import torch.nn as nn


class DifferentiableMemoryMatrix(nn.Module):
    """
    Continuous Differentiable Memory Matrix Bank.
    
    Attributes:
        capacity (int): Maximum number of memory slots (default: 128).
        memory_dim (int): Vector dimension of each memory slot (default: 768).
        scale_factor (float): 1 / sqrt(memory_dim) to prevent magnitude explosion.
        M (torch.Tensor): State buffer of shape (capacity, memory_dim) with requires_grad=False.
    """

    def __init__(
        self,
        capacity: int = 128,
        memory_dim: int = 768,
        scaling: Any = True,
    ):
        super().__init__()
        self.capacity = capacity
        self.memory_dim = memory_dim
        self.scaling = scaling

        if scaling is True or scaling == "sqrt":
            self.scale_factor = 1.0 / math.sqrt(memory_dim)
        elif scaling == "dim":
            self.scale_factor = 1.0 / float(memory_dim)
        elif scaling is False or scaling == "none":
            self.scale_factor = 1.0
        elif isinstance(scaling, (int, float)):
            self.scale_factor = float(scaling)
        else:
            self.scale_factor = 1.0 / math.sqrt(memory_dim)

        # Memory state buffer: Non-trainable (requires_grad = False)
        self.register_buffer(
            "M",
            torch.zeros(capacity, memory_dim, dtype=torch.float32),
            persistent=True,
        )
        self.M.requires_grad_(False)

        self._active_count: int = 0

    @property
    def num_memories(self) -> int:
        """Returns the number of actively populated memory slots."""
        return self._active_count

    @property
    def memory_matrix(self) -> torch.Tensor:
        """Returns the active memory matrix tensor."""
        return self.M

    def reset_memory(self):
        """Clears all stored memories to zero and resets slot pointer."""
        self.M.zero_()
        self._active_count = 0

    def write(self, vector: torch.Tensor):
        """
        Writes a new hidden state representation to the memory matrix.
        Operates without gradient tracking (pure state update).
        
        Args:
            vector: Tensor of shape (memory_dim,) or (1, memory_dim).
        """
        vec = vector.detach().view(self.memory_dim).to(device=self.M.device, dtype=self.M.dtype)

        if self._active_count < self.capacity:
            # Fill next empty slot
            self.M[self._active_count].copy_(vec)
            self._active_count += 1
        else:
            # FIFO rolling replacement: discard oldest (slot 0), append to slot 127
            self.M[:-1].copy_(self.M[1:].clone())
            self.M[-1].copy_(vec)

    def add_memory(self, vector: torch.Tensor):
        """Alias for write() for backward compatibility."""
        self.write(vector)

    def read(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiable matrix read via pure continuous matrix multiplication:
          1. s = query @ M^T          (activations across all 128 slots)
          2. m = (s @ M) * scale       (reconstructed memory vector in 768-D)
          
        Args:
            query: Query tensor q of shape (..., memory_dim).
            
        Returns:
            m: Reconstructed memory representation of shape (..., memory_dim).
            s: Activation scores across all 128 slots of shape (..., capacity).
        """
        # Ensure query is on the same device and dtype
        q = query.to(device=self.M.device, dtype=self.M.dtype)
        
        # Step 1: Compute continuous activation across all 128 slots simultaneously
        # s = q @ M^T -> shape (..., capacity)
        s = torch.matmul(q, self.M.t())

        # Step 2: Linear combination of all memory slots weighted by activations
        # m = s @ M -> shape (..., memory_dim)
        m = torch.matmul(s, self.M)

        if self.scaling:
            m = m * self.scale_factor

        return m, s

    def extra_repr(self) -> str:
        return (
            f"capacity={self.capacity}, memory_dim={self.memory_dim}, "
            f"active_count={self._active_count}, scaling={self.scaling} "
            f"(scale_factor={self.scale_factor:.6f})"
        )
