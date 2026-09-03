"""
baselines/simple_memory.py (Pure PyTorch Version)

A simple baseline memory that stores K-V pairs sequentially (FIFO circular buffer).
No decay, no importance, no confidence, no multi-factor scoring.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMemory(nn.Module):
    """
    Sequential FIFO Memory Baseline (PyTorch Version).
    """
    def __init__(self, capacity: int, dim: int, top_k: int = 4):
        super().__init__()
        self.capacity = capacity
        self.dim = dim
        self.top_k = top_k

        self.register_buffer('mem_keys', torch.zeros(capacity, dim))
        self.register_buffer('mem_vals', torch.zeros(capacity, dim))
        self.register_buffer('head', torch.zeros(1, dtype=torch.long))

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.fusion_proj = nn.Linear(dim * 2, dim, bias=False)

    def write(self, h_eos: torch.Tensor):
        with torch.no_grad():
            k_new = self.k_proj(h_eos).detach()
            v_new = self.v_proj(h_eos).detach()
            batch_size = h_eos.size(0)

            for b in range(batch_size):
                idx = int(self.head.item())
                self.mem_keys[idx] = k_new[b]
                self.mem_vals[idx] = v_new[b]
                self.head[0] = (idx + 1) % self.capacity

    def read(self, h_eos: torch.Tensor, return_scores: bool = False):
        batch_size = h_eos.size(0)
        q = self.q_proj(h_eos)

        q_norm = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)
        k_norm = self.mem_keys / (torch.norm(self.mem_keys, dim=-1, keepdim=True) + 1e-8)

        sim = torch.matmul(q_norm, k_norm.T)

        k = min(self.top_k, self.capacity)
        topk_sim, topk_indices = torch.topk(sim, k, dim=-1)
        attn_weights = F.softmax(topk_sim, dim=-1)

        expanded_indices = topk_indices.unsqueeze(-1).expand(-1, -1, self.dim)
        topk_vals = torch.gather(self.mem_vals.unsqueeze(0).expand(batch_size, -1, -1), 1, expanded_indices)

        read_val = torch.sum(attn_weights.unsqueeze(-1) * topk_vals, dim=1)
        if return_scores:
            return read_val, sim, topk_indices
        return read_val

    def forward(self, h_eos: torch.Tensor):
        read_val = self.read(h_eos)
        fused = self.fusion_proj(torch.cat([h_eos, read_val], dim=-1))
        return fused
