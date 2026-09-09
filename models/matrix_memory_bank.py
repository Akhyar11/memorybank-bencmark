"""
models/matrix_memory_bank.py
============================
Differentiable Memory Matrix Engine with Softmax Attention:
  - Memory Matrix: M in R^(128 x 768), non-trainable state (requires_grad = False).
  - Query dot product: s = (q @ M^T) / sqrt(d)  (activations across memory slots).
  - Softmax Attention: attn = Softmax(s)  (normalized probability distribution over active slots).
  - Memory retrieval: m = attn @ M  (768-dimensional retrieved vector, convex combination of memory slots).
  - In column-vector notation: m = M^T @ Softmax(M @ (W_q @ q)).
  - 100% differentiable attention gradient flow without vanishing gradients.
"""

import math
from typing import Any, Optional, Tuple
import torch
import torch.nn as nn


class DifferentiableMemoryMatrix(nn.Module):
    """
    Continuous Differentiable Memory Matrix Bank with Softmax Attention.
    
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

        # Memory state buffer: Non-trainable runtime state (persistent=False allows arbitrary slot expansion)
        self.register_buffer(
            "M",
            torch.zeros(capacity, memory_dim, dtype=torch.float32),
            persistent=False,
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
        Differentiable matrix read via Softmax Attention:
          1. s = (query @ M^T) * scale_factor     (scores across memory slots)
          2. Mask inactive slots (if active_count < capacity) with -1e9
          3. attn = Softmax(s, dim=-1)            (normalized distribution over active slots)
          4. m = attn @ M                         (convex combination of active slots)
          
        Args:
            query: Query tensor q of shape (..., memory_dim).
            
        Returns:
            m: Retrieved memory representation of shape (..., memory_dim).
            attn: Attention distribution across memory slots of shape (..., capacity).
        """
        # Ensure query is on the same device and dtype
        q = query.to(device=self.M.device, dtype=self.M.dtype)

        # When no memories are stored, return zeros
        if self._active_count == 0:
            m = q * 0.0
            attn = torch.zeros(*q.shape[:-1], self.capacity, device=self.M.device, dtype=self.M.dtype)
            return m, attn

        # Step 1: Compute scaled dot-product attention scores across slots
        s = torch.matmul(q, self.M.t())
        if self.scaling:
            s = s * self.scale_factor

        # Step 2: Mask inactive empty slots so they do not absorb softmax probability
        if self._active_count < self.capacity:
            mask = torch.zeros(self.capacity, device=self.M.device, dtype=torch.bool)
            mask[self._active_count:] = True
            s = s.masked_fill(mask, -1e9)

        # Step 3: Softmax over slots
        attn = torch.softmax(s, dim=-1)

        # Step 4: Weighted combination of memory slots
        # m = attn @ M -> shape (..., memory_dim)
        m = torch.matmul(attn, self.M)

        return m, attn

    def extra_repr(self) -> str:
        return (
            f"capacity={self.capacity}, memory_dim={self.memory_dim}, "
            f"active_count={self._active_count}, scaling={self.scaling} "
            f"(scale_factor={self.scale_factor:.6f})"
        )
