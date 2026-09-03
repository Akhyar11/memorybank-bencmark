"""
baselines/nearest_neighbor.py (Pure PyTorch Version)

Baseline: Standard Key-Value Nearest Neighbor Associative Memory.
Independent from Memory Bank specific mechanisms:
- NO importance scoring
- NO confidence scoring
- NO temporal decay or recency
- NO read reinforcement
- NO i_proj or composite formulas
Uses pure cosine similarity between query projection and stored keys to retrieve associated values.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NearestNeighborMemory(nn.Module):
    """
    Standard Key-Value Nearest Neighbor Memory Baseline (PyTorch Version).
    Retrieves from a stored key-value buffer using pure Cosine Similarity.
    """
    def __init__(self, capacity: int, dim: int, top_k: int = 4):
        super().__init__()
        self.capacity = capacity
        self.dim = dim
        self.top_k = top_k

        # Key and Value buffers (non-trainable persistent store)
        self.register_buffer('mem_keys', torch.zeros(capacity, dim))
        self.register_buffer('mem_vals', torch.zeros(capacity, dim))
        self.register_buffer('active_mask', torch.zeros(capacity, dtype=torch.bool))

        # Projections
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.fusion_proj = nn.Linear(dim * 2, dim, bias=False)

    def write(self, h_k: torch.Tensor, h_v: torch.Tensor = None, slot_idx: int = 0):
        """Write key-value representation into a specified slot."""
        with torch.no_grad():
            k = self.k_proj(h_k).detach()
            v = self.v_proj(h_v if h_v is not None else h_k).detach()
            self.mem_keys[slot_idx] = k[0] if k.ndim > 1 else k
            self.mem_vals[slot_idx] = v[0] if v.ndim > 1 else v
            self.active_mask[slot_idx] = True

    def read(self, h_eos: torch.Tensor, return_scores: bool = False):
        """
        Pure nearest neighbor retrieval via cosine similarity:
        Query -> dot product with stored keys -> top-k softmax -> weighted values.
        """
        batch_size = h_eos.size(0)
        q = self.q_proj(h_eos)

        q_norm = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)
        k_norm = self.mem_keys / (torch.norm(self.mem_keys, dim=-1, keepdim=True) + 1e-8)

        # Pure cosine similarity (batch, capacity)
        sim = torch.matmul(q_norm, k_norm.T)

        # Mask inactive slots
        active = self.active_mask.unsqueeze(0).expand(batch_size, -1)
        sim_masked = torch.where(active, sim, torch.tensor(-1e9, device=sim.device))

        k = min(self.top_k, self.capacity)
        topk_sim, topk_indices = torch.topk(sim_masked, k, dim=-1)

        # Softmax over top-k
        attn_weights = F.softmax(topk_sim, dim=-1)

        # Gather vals
        expanded_indices = topk_indices.unsqueeze(-1).expand(-1, -1, self.dim)
        topk_vals = torch.gather(self.mem_vals.unsqueeze(0).expand(batch_size, -1, -1), 1, expanded_indices)

        read_val = torch.sum(attn_weights.unsqueeze(-1) * topk_vals, dim=1)

        if return_scores:
            return read_val, sim_masked, topk_indices
        return read_val

    def forward(self, h_eos: torch.Tensor):
        read_val = self.read(h_eos)
        fused = self.fusion_proj(torch.cat([h_eos, read_val], dim=-1))
        return fused
