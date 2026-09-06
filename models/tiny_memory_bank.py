from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MemoryState:
    """
    Snapshot of the Turn-level Semantic Memory Bank state.
    """
    memories: List[torch.Tensor] = field(default_factory=list)
    read_counts: List[int] = field(default_factory=list)
    ages: List[int] = field(default_factory=list)

    def detach(self) -> "MemoryState":
        return MemoryState(
            memories=[m.detach() for m in self.memories],
            read_counts=list(self.read_counts),
            ages=list(self.ages),
        )


@dataclass
class TinyMemoryConfig:
    """
    Configuration for Turn-level Semantic Memory Bank.
    """
    memory_dim: int = 768            # Dimension of memory representations
    hidden_size: int = 768           # Backbone hidden size
    memory_capacity: int = 128       # Max slot capacity (ceiling)
    top_k: int = 1                   # Number of top memories to retrieve
    eviction_threshold_ratio: float = 0.05  # 5% of mean read count
    min_age_for_eviction: int = 3    # Grace period in turns before eligible for deletion
    temperature: float = 1.0         # Temperature for Top-K softmax weighting
    eps: float = 1e-8

    # Backwards-compatibility fields for legacy scripts
    tau_read: float = 0.05
    tau_write: float = 0.05
    lambda_replace: float = 1.0


class TinyMemoryBank(nn.Module):
    """
    Turn-level Semantic Memory Bank.
    
    Key Characteristics:
      1. No micro token-level write operations.
      2. Exactly 2 memories saved per turn:
         - Memory 1: User prompt last hidden state (768).
         - Memory 2: AI generated response final token hidden state (768).
      3. Retrieval: Cosine similarity Top-K between Query C and stored memories.
      4. Eviction: Drops memories with read_count < 5% of mean(read_count) for age >= min_age.
    """

    def __init__(self, config: Optional[TinyMemoryConfig] = None):
        super().__init__()
        if config is None:
            config = TinyMemoryConfig()
        self.config = config
        self.memories: List[torch.Tensor] = []
        self.read_counts: List[int] = []
        self.ages: List[int] = []

    @property
    def num_memories(self) -> int:
        return len(self.memories)

    @property
    def mem_occupancy(self) -> torch.Tensor:
        """Compatibility property for benchmark scripts measuring active memory slots."""
        return torch.tensor([float(len(self.memories))])

    @property
    def mem_usage(self) -> torch.Tensor:
        """Compatibility property returning read counts as tensor."""
        if len(self.read_counts) == 0:
            return torch.tensor([0.0])
        return torch.tensor([float(r) for r in self.read_counts])

    @property
    def mem_age(self) -> torch.Tensor:
        """Compatibility property returning memory ages as tensor."""
        if len(self.ages) == 0:
            return torch.tensor([0.0])
        return torch.tensor([float(a) for a in self.ages])

    def reset_memory(self):
        """Clears all stored memories, read counts, and ages."""
        self.memories.clear()
        self.read_counts.clear()
        self.ages.clear()

    def add_memory(self, vector: torch.Tensor):
        """
        Appends a 768-dim hidden state representation to the memory bank.
        Vector can be of shape (768,) or (1, 768).
        """
        if vector.dim() == 2:
            vector = vector.squeeze(0)
        
        vec = vector.detach().clone()

        if len(self.memories) >= self.config.memory_capacity:
            self.memories.pop(0)
            self.read_counts.pop(0)
            self.ages.pop(0)

        self.memories.append(vec)
        self.read_counts.append(0)
        self.ages.append(0)

    def read(
        self,
        query: torch.Tensor,
        top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Reads from memory using Cosine Similarity between Query C and stored memories.
        
        Args:
            query: Query tensor C of shape (..., D) where D = memory_dim (768).
            top_k: Number of memories to retrieve (defaults to config.top_k).
            
        Returns:
            M_bar: Fused memory representation of shape (..., D).
            top_indices: List of retrieved memory indices.
        """
        orig_shape = query.shape[:-1]
        d = self.config.memory_dim

        if len(self.memories) == 0:
            return torch.zeros(*orig_shape, d, device=query.device, dtype=query.dtype), []

        k = top_k if top_k is not None else self.config.top_k
        k = min(k, len(self.memories))

        query_flat = query.reshape(-1, d)

        # Stack memories to tensor on query device: shape (M, D)
        mem_tensor = torch.stack([m.to(query.device) for m in self.memories], dim=0)

        # Cosine similarity: (N, D) x (M, D) -> (N, M)
        q_norm = F.normalize(query_flat, p=2, dim=-1, eps=self.config.eps)
        m_norm = F.normalize(mem_tensor, p=2, dim=-1, eps=self.config.eps)
        sim = torch.matmul(q_norm, m_norm.t())  # (N, M)

        # Retrieve Top-K
        top_vals, top_indices_tensor = torch.topk(sim, k=k, dim=-1)  # (N, K)

        # Track read counts for the selected memories
        top_idx_flat = top_indices_tensor.flatten().tolist()
        for idx in top_idx_flat:
            if idx < len(self.read_counts):
                self.read_counts[idx] += 1

        # Softmax weights over Top-K
        tau = max(self.config.temperature, 1e-4)
        weights = F.softmax(top_vals / tau, dim=-1)  # (N, K)

        # Weighted combination: (N, K, 1) * (N, K, D) -> (N, D)
        selected_mems = mem_tensor[top_indices_tensor]  # (N, K, D)
        M_bar_flat = torch.sum(weights.unsqueeze(-1) * selected_mems, dim=1)  # (N, D)

        M_bar = M_bar_flat.reshape(*orig_shape, d)
        return M_bar, top_idx_flat

    def step_turn(self):
        """Increments turn age for all memories currently stored in the bank."""
        self.ages = [age + 1 for age in self.ages]

    def evict_lifecycle(
        self,
        threshold_ratio: Optional[float] = None,
        min_age: Optional[int] = None,
    ) -> int:
        """
        Lifecycle-based eviction:
        Deletes memories whose read count is < (threshold_ratio * mean_read_count),
        given that the memory is mature (age >= min_age).
        
        Returns:
            Number of evicted memory items.
        """
        if len(self.memories) == 0:
            return 0

        if threshold_ratio is None:
            threshold_ratio = self.config.eviction_threshold_ratio
        if min_age is None:
            min_age = self.config.min_age_for_eviction

        mean_reads = sum(self.read_counts) / len(self.read_counts)
        cutoff = threshold_ratio * mean_reads

        kept_mems = []
        kept_reads = []
        kept_ages = []
        evicted = 0

        for m, r, a in zip(self.memories, self.read_counts, self.ages):
            if a >= min_age and r < cutoff:
                evicted += 1
                continue
            kept_mems.append(m)
            kept_reads.append(r)
            kept_ages.append(a)

        self.memories = kept_mems
        self.read_counts = kept_reads
        self.ages = kept_ages
        return evicted

    def get_runtime_state(self) -> MemoryState:
        return MemoryState(
            memories=[m.clone() for m in self.memories],
            read_counts=list(self.read_counts),
            ages=list(self.ages),
        )

    def persist_runtime_state(self, state: MemoryState):
        self.memories = [m.clone() for m in state.memories]
        self.read_counts = list(state.read_counts)
        self.ages = list(state.ages)
