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
    memory_update_threshold: float = 0.95 # tau – similarity threshold for updating slot vs inserting new slot

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

    @property
    def active_count(self) -> int:
        """Number of currently non-expired (active or dormant) memory slots."""
        return int((self.mem_state != STATE_EXPIRED).sum().item())

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

    def load_memory_state(self, state_dict):
        """Load memory state from dictionary."""
        for key in ['keys', 'vals', 'importance', 'confidence', 'created_at', 'last_access', 'access_count', 'state']:
            buffer_name = f'mem_{key}'
            if key in state_dict and hasattr(self, buffer_name):
                getattr(self, buffer_name).copy_(state_dict[key])
        if 'global_step' in state_dict:
            self.global_step.copy_(state_dict['global_step'])

    def reset_memory(self):
        """Reset all memory slots and buffers to initial empty state."""
        self.load_memory_state(self.empty_memory_state())

    def get_memory_state(self):
        """Export current memory state as a dictionary."""
        return {
            'keys': self.mem_keys.clone(),
            'vals': self.mem_vals.clone(),
            'importance': self.mem_importance.clone(),
            'confidence': self.mem_confidence.clone(),
            'created_at': self.mem_created_at.clone(),
            'last_access': self.mem_last_access.clone(),
            'access_count': self.mem_access_count.clone(),
            'state': self.mem_state.clone(),
            'global_step': self.global_step.clone(),
        }

    # -----------------------------------------------------------------------
    # LOCKED OPERATION 2: read
    # -----------------------------------------------------------------------
    def read(self, h_eos: torch.Tensor, read_prob: torch.Tensor = None, return_scores: bool = False):
        step = self.global_step.item()
        cfg  = self.config

        # 1. Query projection
        q    = self.q_proj(h_eos)             # (batch, dim)
        keys = self.mem_keys.clone()          # (capacity, dim)
        vals = self.mem_vals.clone()          # (capacity, dim)

        importance  = self.mem_importance.clone()
        confidence  = self.mem_confidence.clone()
        last_access = self.mem_last_access.clone()
        state       = self.mem_state.clone()

        # 2. Cosine similarity
        q_norm = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)
        k_norm = keys / (torch.norm(keys, dim=-1, keepdim=True) + 1e-8)
        sim = torch.matmul(q_norm, k_norm.T)  # (batch, capacity)

        # 3. Recency
        dt      = torch.clamp(torch.tensor(step, device=last_access.device) - last_access, min=0).float()
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
        k = min(cfg.memory_top_k, score.size(-1))
        topk_scores, topk_indices = torch.topk(score, k, dim=-1)  # (batch, k)

        # 8. Differentiable Softmax Attention
        # Softmax dengan temperatur memberikan turunan analitik mulus di setiap slot
        attn_weights = F.softmax(topk_scores, dim=-1)

        # 9. Weighted Value Aggregation
        batch_size = topk_indices.size(0)
        expanded_indices = topk_indices.unsqueeze(-1).expand(-1, -1, vals.size(-1))
        topk_vals = torch.gather(vals.unsqueeze(0).expand(batch_size, -1, -1), 1, expanded_indices)

        has_active = (state != STATE_EXPIRED).any()
        if has_active:
            read_result = torch.sum(attn_weights.unsqueeze(-1) * topk_vals, dim=1)  # (batch, dim)
        else:
            read_result = torch.zeros((batch_size, vals.size(-1)), device=score.device)

        # 10. Continuous Read Gate Modulation (Differentiable)
        if read_prob is not None:
            read_result = read_result * read_prob.unsqueeze(-1)

        # 11. Access Reinforcement (Episodic buffer update under torch.no_grad, proportional to attention weight)
        eta_a = cfg.mem_reinforcement_rate
        with torch.no_grad():
            for b in range(batch_size):
                for i in range(k):
                    idx = topk_indices[b, i].item()
                    w_i = attn_weights[b, i].item()
                    self.mem_last_access[idx] = step
                    self.mem_access_count[idx] += 1
                    self.mem_importance[idx] = torch.clamp(self.mem_importance[idx] + eta_a * w_i, min=0.0, max=1.0)

        # Cache actual composite retrieval score for evaluation & scientific ranking
        self.last_scores = score.detach()
        self.last_topk_indices = topk_indices.detach()
        self.last_valid_mask = (topk_scores > -1e8).detach()

        if return_scores:
            return read_result, score
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

        # Differentiable memory state mutation under torch.no_grad
        with torch.no_grad():
            k_new_d = k_new.detach()
            v_new_d = v_new.detach()
            i_new_d = i_new.detach()
            c_new_d = c_new.detach()

            for b in range(batch_size):
                k_n = k_new_d[b]
                v_n = v_new_d[b]
                i_n = i_new_d[b]
                c_n = c_new_d[b]
                w_p = write_prob[b]

                # 1. Content-based similarity addressing
                k_n_norm = k_n / (torch.norm(k_n) + 1e-8)
                k_norm = self.mem_keys / (torch.norm(self.mem_keys, dim=-1, keepdim=True) + 1e-8)
                sim = torch.matmul(k_norm, k_n_norm)
                valid_search = (self.mem_state != STATE_EXPIRED)
                sim_content = torch.where(valid_search, sim, torch.tensor(-10.0, device=sim.device))
                content_w = F.softmax(sim_content / 0.1, dim=-1)
                max_sim = sim_content.max()

                # 2. Usage/Allocation addressing (Differentiable DNC formulation - Graves et al.)
                # Menjamin probabilitas alokasi terkonsentrasi pada slot kosong berikutnya tanpa terdilusi 1/N
                usage = torch.where(self.mem_state != STATE_EXPIRED, torch.tensor(1.0, device=sim.device), torch.tensor(0.0, device=sim.device))
                shifted_usage = torch.cat([torch.tensor([1.0], device=sim.device), usage[:-1]])
                cumprod_usage = torch.cumprod(shifted_usage, dim=0)
                alloc_w = (1.0 - usage) * cumprod_usage
                alloc_sum = alloc_w.sum()
                if alloc_sum > 0:
                    alloc_w = alloc_w / alloc_sum
                else:
                    # Jika seluruh slot terisi penuh, pilih slot dengan importance terendah
                    alloc_w = F.softmax(-self.mem_importance / 0.1, dim=-1)

                # 3. Continuous routing gate between allocation and content update
                # Baseline cosine similarity embedding GPT-2 adalah ~0.75-0.85 untuk kalimat berbeda,
                # dan >0.93 untuk fakta yang sama/identik.
                gate_alloc = torch.sigmoid((0.93 - max_sim) * 15.0)
                write_w = gate_alloc * alloc_w + (1.0 - gate_alloc) * content_w

                # 4. Continuous Erase & Add operation (DNC/NTM style)
                erase_factor = i_n
                alpha = w_p
                write_w_2d = write_w.unsqueeze(-1)

                self.mem_vals.copy_((1.0 - write_w_2d * (alpha * erase_factor)) * self.mem_vals + (write_w_2d * alpha) * v_n)
                self.mem_keys.copy_((1.0 - write_w_2d * (alpha * erase_factor)) * self.mem_keys + (write_w_2d * alpha) * k_n)
                self.mem_importance.copy_((1.0 - write_w * alpha) * self.mem_importance + (write_w * alpha) * i_n)
                self.mem_confidence.copy_((1.0 - write_w * alpha) * self.mem_confidence + (write_w * alpha) * c_n)

                target_idx = torch.argmax(write_w).item()
                target_indices.append(target_idx)

                self.mem_state[target_idx] = STATE_ACTIVE
                self.mem_last_access[target_idx] = step
                self.mem_created_at[target_idx] = step
                self.mem_access_count[target_idx] += 1

        if target_indices:
            return torch.tensor(target_indices, dtype=torch.long, device=h_eos.device)
        return torch.tensor([], dtype=torch.long, device=h_eos.device)


    # -----------------------------------------------------------------------
    # LOCKED OPERATION 4: fuse
    # -----------------------------------------------------------------------
    def fuse(self, h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        concatenated = torch.cat([h, m], dim=-1)
        fused        = self.fusion_proj(concatenated)
        return h + fused

    # -----------------------------------------------------------------------
    # LOCKED OPERATION 5: forward (decay → read → fuse)
    # -----------------------------------------------------------------------
    def forward(self, h_eos: torch.Tensor, read_prob: torch.Tensor = None,
                 write_prob: torch.Tensor = None, deterministic: bool = False, return_scores: bool = False):
        self.global_step[0] += 1

        self.decay_memory()
        if return_scores:
            read_val, score = self.read(h_eos, read_prob=None, return_scores=True)
        else:
            read_val = self.read(h_eos, read_prob=None, return_scores=False)

        fused_h = self.fuse(h_eos, read_val)
        if return_scores:
            return fused_h, score
        return fused_h
