"""
TinyMemoryBank – Locked Architecture Implementation (PyTorch Version)
===================================================================
ARCHITECTURE LOCK: DO NOT MODIFY COMPONENT NAMES OR LOGIC.
Source reference: /home/akhyar/Dokumen/Code/python/MemoryBank/mamoe/memory/bank.py
Locked components: q_proj, k_proj, v_proj, i_proj, fusion_proj,
                   mem_keys, mem_vals, mem_importance, mem_confidence,
                   mem_created_at, mem_last_access, mem_access_count,
                   mem_state, global_step
Locked flow: decay_memory → read → write → fuse → forward
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# State Constants (LOCKED – do not add or rename)
# ---------------------------------------------------------------------------
STATE_EXPIRED = 0   # Memory slot yang sudah kadaluarsa
STATE_ACTIVE  = 1   # Memory slot aktif/fresh
STATE_DORMANT = 2   # Memory slot dormant (belum expired tapi tidak aktif)


@dataclass
class TinyMemoryConfig:
    """Configuration for TinyMemoryBank."""
    memory_capacity: int = 128
    memory_dim: int = 32
    hidden_size: int = 32

    # Decay parameters
    mem_decay_rate: float = 0.001           # lambda – decay rate
    mem_importance_protection: float = 0.5  # rho – importance protection factor

    # Score weights (alpha, beta, gamma, delta)
    mem_alpha: float = 1.0    # weight for cosine similarity
    mem_beta: float = 0.5     # weight for importance
    mem_gamma: float = 0.1    # weight for recency
    mem_delta: float = 0.2    # weight for confidence

    # Retrieval params
    memory_top_k: int = 4
    memory_threshold: float = -1e9      # tau – relevance threshold for read (allow all by default)
    memory_read_threshold: float = 0.0  # gate threshold for read_prob
    memory_write_threshold: float = 0.0 # tau – threshold for write gate

    # Reinforcement
    mem_reinforcement_rate: float = 0.05  # eta_a – importance boost on access


class TinyMemoryBank(nn.Module):
    """
    Locked Neural Memory Bank (PyTorch Version).
    """
    def __init__(self, config: TinyMemoryConfig):
        super().__init__()
        self.config = config
        capacity = config.memory_capacity
        dim      = config.memory_dim

        # --- Locked Memory State Tensors ---
        self.register_buffer('mem_keys',         torch.zeros(capacity, dim))
        self.register_buffer('mem_vals',         torch.zeros(capacity, dim))
        self.register_buffer('mem_importance',   torch.zeros(capacity))
        self.register_buffer('mem_confidence',   torch.zeros(capacity))
        self.register_buffer('mem_created_at',   torch.zeros(capacity, dtype=torch.int32))
        self.register_buffer('mem_last_access',  torch.zeros(capacity, dtype=torch.int32))
        self.register_buffer('mem_access_count', torch.zeros(capacity, dtype=torch.int32))
        self.register_buffer('mem_state',        torch.zeros(capacity, dtype=torch.int32))
        self.register_buffer('global_step',      torch.zeros(1, dtype=torch.int32))

        # --- Locked Learned Projections ---
        self.q_proj      = nn.Linear(dim, dim, bias=False)
        self.k_proj      = nn.Linear(dim, dim, bias=False)
        self.v_proj      = nn.Linear(dim, dim, bias=False)
        self.i_proj      = nn.Linear(dim, 1)          # importance logit
        self.fusion_proj = nn.Linear(config.hidden_size + dim, config.hidden_size, bias=False)

    # -----------------------------------------------------------------------
    # LOCKED OPERATION 1: decay_memory
    # -----------------------------------------------------------------------
    def decay_memory(self):
        step        = self.global_step.item()
        last_access = self.mem_last_access
        importance  = self.mem_importance
        state       = self.mem_state

        dt = torch.clamp(torch.tensor(step) - last_access, min=0).float()

        lam = self.config.mem_decay_rate
        R   = torch.exp(-lam * dt)

        rho         = self.config.mem_importance_protection
        effective_R = R * (1.0 + rho * importance)

        is_expired = effective_R < 0.1
        is_dormant = (effective_R < 0.5) & (~is_expired)

        new_state = torch.where(is_expired, torch.tensor(STATE_EXPIRED, device=state.device), state)
        new_state = torch.where(is_dormant, torch.tensor(STATE_DORMANT, device=state.device), new_state)

        self.mem_state.copy_(new_state)
        return effective_R

    def empty_memory_state(self):
        cap = self.config.memory_capacity
        dim = self.config.memory_dim
        device = self.mem_keys.device
        return {
            'keys':         torch.zeros((cap, dim),  dtype=torch.float32, device=device),
            'vals':         torch.zeros((cap, dim),  dtype=torch.float32, device=device),
            'importance':   torch.zeros((cap,),      dtype=torch.float32, device=device),
            'confidence':   torch.zeros((cap,),      dtype=torch.float32, device=device),
            'created_at':   torch.zeros((cap,),      dtype=torch.int32, device=device),
            'last_access':  torch.zeros((cap,),      dtype=torch.int32, device=device),
            'access_count': torch.zeros((cap,),      dtype=torch.int32, device=device),
            'state':        torch.full((cap,), STATE_EXPIRED, dtype=torch.int32, device=device),
            'global_step':  torch.zeros((1,),        dtype=torch.int32, device=device),
        }

    # -----------------------------------------------------------------------
    # LOCKED OPERATION 2: read
    # -----------------------------------------------------------------------
    def read(self, h_eos: torch.Tensor, read_prob: torch.Tensor = None) -> torch.Tensor:
        step = self.global_step.item()
        cfg  = self.config

        # 1. Query projection
        q    = self.q_proj(h_eos)       # (batch, dim)
        keys = self.mem_keys            # (capacity, dim)
        vals = self.mem_vals            # (capacity, dim)

        importance  = self.mem_importance
        confidence  = self.mem_confidence
        last_access = self.mem_last_access
        state       = self.mem_state

        # 2. Cosine similarity
        q_norm = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)
        k_norm = keys / (torch.norm(keys, dim=-1, keepdim=True) + 1e-8)
        sim = torch.matmul(q_norm, k_norm.T)  # (batch, capacity)

        # 3. Recency
        dt      = torch.clamp(torch.tensor(step) - last_access, min=0).float()
        recency = torch.exp(-cfg.mem_decay_rate * dt)  # (capacity,)

        # 4. Broadcast metadata to (batch, capacity)
        importance_bd = importance.unsqueeze(0).expand_as(sim)
        confidence_bd = confidence.unsqueeze(0).expand_as(sim)
        recency_bd    = recency.unsqueeze(0).expand_as(sim)

        # 5. Score formula
        score = (cfg.mem_alpha * sim
               + cfg.mem_beta  * importance_bd
               + cfg.mem_gamma * recency_bd
               + cfg.mem_delta * confidence_bd)

        # 6. Mask EXPIRED slots
        active_mask = (state != STATE_EXPIRED).unsqueeze(0)  # (1, capacity)
        score = torch.where(active_mask, score, torch.tensor(-1e9, device=score.device))

        # 7. Top-K selection
        k = cfg.memory_top_k
        topk_scores, topk_indices = torch.topk(score, k, dim=-1)  # (batch, k)

        # 8. Threshold filter
        tau        = cfg.memory_threshold
        valid_mask = topk_scores > tau  # (batch, k)
        
        valid_mask = valid_mask & (topk_scores > -1e8)

        if read_prob is not None:
            is_read = read_prob > cfg.memory_read_threshold
            valid_mask = valid_mask & is_read.unsqueeze(-1)

        # 9. Softmax aggregation
        filtered_scores = torch.where(valid_mask, topk_scores, torch.tensor(-1e9, device=score.device))
        attn_weights    = F.softmax(filtered_scores, dim=-1)

        attn_weights = attn_weights * valid_mask.to(attn_weights.dtype)

        # 10. Weighted aggregation
        batch_size = topk_indices.size(0)
        # Gather vals for each index: topk_indices is (batch, k)
        # We want topk_vals of shape (batch, k, dim)
        expanded_indices = topk_indices.unsqueeze(-1).expand(-1, -1, vals.size(-1))
        topk_vals = torch.gather(vals.unsqueeze(0).expand(batch_size, -1, -1), 1, expanded_indices)

        attn_sum    = torch.sum(attn_weights, dim=-1, keepdim=True)  # (batch, 1)
        read_result = torch.sum(attn_weights.unsqueeze(-1) * topk_vals, dim=1)  # (batch, dim)
        read_result = torch.where(attn_sum > 0, read_result, torch.zeros_like(read_result))

        # 11. Access Reinforcement
        eta_a = cfg.mem_reinforcement_rate
        
        # We can update the memory buffers in-place in PyTorch
        if self.training or not self.training: # Always reinforce on read
            for b in range(batch_size):
                for i in range(k):
                    if valid_mask[b, i]:
                        idx = topk_indices[b, i].item()
                        self.mem_last_access[idx] = step
                        self.mem_access_count[idx] += 1
                        self.mem_importance[idx] = torch.clamp(self.mem_importance[idx] + eta_a, min=0.0, max=1.0)

        return read_result

    # -----------------------------------------------------------------------
    # LOCKED OPERATION 3: write
    # -----------------------------------------------------------------------
    def write(self, h_eos: torch.Tensor, is_eos: torch.Tensor, write_prob: torch.Tensor):
        step = self.global_step.item()
        cfg  = self.config
        batch_size = h_eos.size(0)

        # Project inputs
        k_new = self.k_proj(h_eos)                                    # (batch, dim)
        v_new = self.v_proj(h_eos)                                    # (batch, dim)
        i_new = torch.sigmoid(self.i_proj(h_eos).squeeze(-1))         # (batch,)
        c_new = torch.ones_like(i_new) * 0.5                          # (batch,) default confidence

        target_indices = []

        # We must process batch sequentially so writes see preceding updates
        for b in range(batch_size):
            k_n = k_new[b]
            v_n = v_new[b]
            i_n = i_new[b]
            c_n = c_new[b]
            is_e = is_eos[b]
            w_p = write_prob[b]

            do_write = (is_e > 0.5) and (w_p >= cfg.memory_write_threshold)
            
            k_n_norm = k_n / (torch.norm(k_n) + 1e-8)
            k_norm = self.mem_keys / (torch.norm(self.mem_keys, dim=-1, keepdim=True) + 1e-8)

            sim_ = torch.matmul(k_norm, k_n_norm)
            
            valid_search = (self.mem_state != STATE_EXPIRED)
            sim_masked   = torch.where(valid_search, sim_, torch.tensor(-1.0, device=sim_.device))
            
            max_sim     = torch.max(sim_masked)
            nearest_idx = torch.argmax(sim_masked).item()
            
            tau = cfg.memory_write_threshold
            is_update = (max_sim >= tau) and valid_search[nearest_idx]
            
            # INSERT logic
            sort_keys = torch.where(
                self.mem_state == STATE_EXPIRED, torch.tensor(0, device=sim_.device),
                torch.where(self.mem_state == STATE_DORMANT, torch.tensor(1, device=sim_.device), torch.tensor(2, device=sim_.device))
            )
            tiebreak   = self.mem_importance / (torch.max(self.mem_importance) + 1e-8) * 0.5
            sort_score = sort_keys.float() + tiebreak
            insert_idx = torch.argmin(sort_score).item()
            
            target_idx = nearest_idx if is_update else insert_idx
            target_indices.append(target_idx)
            
            if do_write:
                self.mem_keys[target_idx] = k_n
                if is_update:
                    eta = self.mem_confidence[nearest_idx].item()
                    self.mem_vals[target_idx] = (1.0 - eta) * self.mem_vals[nearest_idx] + eta * v_n
                    self.mem_importance[target_idx] = torch.max(self.mem_importance[nearest_idx], i_n)
                    self.mem_confidence[target_idx] = torch.clamp(self.mem_confidence[nearest_idx] + 0.1, min=0.0, max=1.0)
                    self.mem_state[target_idx] = STATE_ACTIVE
                    self.mem_last_access[target_idx] = step
                    self.mem_access_count[target_idx] += 1
                else:
                    self.mem_vals[target_idx] = v_n
                    self.mem_importance[target_idx] = i_n
                    self.mem_confidence[target_idx] = c_n
                    self.mem_state[target_idx] = STATE_ACTIVE
                    self.mem_last_access[target_idx] = step
                    self.mem_created_at[target_idx] = step
                    self.mem_access_count[target_idx] = 1

        if target_indices:
            return torch.tensor(target_indices, dtype=torch.long, device=h_eos.device)
        return torch.tensor([], dtype=torch.long, device=h_eos.device)

    # -----------------------------------------------------------------------
    # LOCKED OPERATION 4: fuse
    # -----------------------------------------------------------------------
    def fuse(self, h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        concatenated = torch.cat([h, m], dim=-1)
        fused        = self.fusion_proj(concatenated)
        return fused

    # -----------------------------------------------------------------------
    # LOCKED OPERATION 5: forward (decay → read → fuse)
    # -----------------------------------------------------------------------
    def forward(self, h_eos: torch.Tensor, read_prob: torch.Tensor,
                 write_prob: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        self.global_step[0] += 1

        self.decay_memory()
        read_val = self.read(h_eos, read_prob=read_prob)

        if deterministic:
            is_read = read_prob > self.config.memory_read_threshold
            m_eff   = torch.where(is_read.unsqueeze(-1), read_val, torch.zeros_like(read_val))
        else:
            m_eff = read_val * read_prob.unsqueeze(-1)

        fused_h = self.fuse(h_eos, m_eff)
        return fused_h
