import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

STATE_EXPIRED = 0
STATE_ACTIVE = 1
STATE_DORMANT = 2


@dataclass
class TinyMemoryConfig:
    memory_capacity: int = 128
    memory_dim: int = 768
    hidden_size: int = 768

    # Temperatures for differentiable read, write, and vacancy competition
    read_temperature: float = 1.0
    write_temperature: float = 1.0
    vacancy_temperature: float = 1.0

    # Learnable gate calibration defaults
    initial_novelty_bias: float = 0.0
    initial_write_bias: float = 0.0

    # Optional secondary decay (disabled by default so memory learns through NTP)
    enable_decay: bool = False
    mem_decay_rate: float = 0.001


class TinyMemoryBank(nn.Module):
    """
    Genuine differentiable causal Memory Bank with competitive slot allocation.

    Strict Causal Order:
      READ(M_t) -> FUSE -> predict x_{t+1} -> WRITE -> M_{t+1}

    Clean Key/Query/Value formulation:
      q_t = Q(h_t)
      k_t = K(h_t)
      v_t = V(h_t)

    Stored Memory State M_t:
      K_t in R^{B x C x D}
      V_t in R^{B x C x D}
      confidence_t in R^{B x C}
      importance_t in R^{B x C}
      step in R^{B x 1}
    """

    def __init__(self, config: TinyMemoryConfig):
        super().__init__()
        self.config = config
        d = config.memory_dim
        h = config.hidden_size
        c = config.memory_capacity

        # Problem 2: Clean Q, K, V projections from hidden state h_t
        self.q_proj = nn.Linear(h, d, bias=False)
        self.k_proj = nn.Linear(h, d, bias=False)
        self.v_proj = nn.Linear(h, d, bias=False)

        # Write policy: gate determining write strength in [0, 1]
        self.write_gate_proj = nn.Linear(h, 1)
        nn.init.constant_(self.write_gate_proj.bias, config.initial_write_bias)

        # Novelty policy: determines trade-off between updating existing slot vs allocating vacant slot
        self.novelty_proj = nn.Linear(h, 1)
        nn.init.constant_(self.novelty_proj.bias, config.initial_novelty_bias)
        self.novelty_sim_scale = nn.Parameter(torch.tensor(1.0))

        # Problem 1: Competitive slot allocation mechanism
        # Learned slot biases break symmetry among empty slots:
        # Initialized with a decaying rank so slot 0 is preferred over slot 1, etc., when empty
        slot_ranks = torch.linspace(2.0, -2.0, steps=c)
        self.slot_bias = nn.Parameter(slot_ranks)

        # Differentiable vacancy score weights
        self.confidence_weight = nn.Parameter(torch.tensor(3.0))
        self.importance_weight = nn.Parameter(torch.tensor(1.0))

        # Problem 4: Meaningful fusion projections
        # r_t = sum_i alpha_i V_i
        # g_t = sigmoid(W_g [h_t ; r_t])
        # fused_t = h_t + g_t * W_f(r_t)
        self.fusion_proj = nn.Linear(d, h, bias=False)
        self.fusion_gate_proj = nn.Linear(h + d, h)

        # Persistent runtime buffer for single-stream interactive inference
        self.register_buffer("mem_keys", torch.zeros(c, d))
        self.register_buffer("mem_vals", torch.zeros(c, d))
        self.register_buffer("mem_confidence", torch.zeros(c))
        self.register_buffer("mem_importance", torch.zeros(c))
        self.register_buffer("global_step", torch.zeros(1, dtype=torch.int32))

        self.last_diagnostics: Dict[str, Any] = {}

    @property
    def active_count(self) -> int:
        return int((self.mem_confidence > 0.5).sum().item())

    def empty_memory_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        c = self.config.memory_capacity
        d = self.config.memory_dim
        return {
            "keys": torch.zeros(batch_size, c, d, device=device, dtype=dtype),
            "vals": torch.zeros(batch_size, c, d, device=device, dtype=dtype),
            "confidence": torch.zeros(batch_size, c, device=device, dtype=dtype),
            "importance": torch.zeros(batch_size, c, device=device, dtype=dtype),
            "step": torch.zeros(batch_size, 1, device=device, dtype=torch.int32),
        }

    def detach_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Explicit truncated-BPTT boundary: detaches all state tensors."""
        return {k: v.detach() for k, v in state.items()}

    def get_memory_state(self) -> Dict[str, torch.Tensor]:
        device = self.mem_keys.device
        dtype = self.mem_keys.dtype
        state = self.empty_memory_state(batch_size=1, device=device, dtype=dtype)
        state["keys"][0] = self.mem_keys
        state["vals"][0] = self.mem_vals
        state["confidence"][0] = self.mem_confidence
        state["importance"][0] = self.mem_importance
        state["step"][0] = self.global_step
        return state

    def load_memory_state(self, state: Dict[str, torch.Tensor]) -> None:
        self.mem_keys.copy_(state["keys"][0].detach())
        self.mem_vals.copy_(state["vals"][0].detach())
        self.mem_confidence.copy_(state["confidence"][0].detach())
        self.mem_importance.copy_(state["importance"][0].detach())
        self.global_step.copy_(state["step"][0].detach().to(torch.int32))

    def persist_runtime_state(self, state: Dict[str, torch.Tensor]) -> None:
        if state["keys"].size(0) != 1:
            return
        with torch.no_grad():
            self.mem_keys.copy_(state["keys"][0].detach())
            self.mem_vals.copy_(state["vals"][0].detach())
            self.mem_confidence.copy_(state["confidence"][0].detach())
            self.mem_importance.copy_(state["importance"][0].detach())
            self.global_step.copy_(state["step"][0].detach().to(torch.int32))

    def reset_memory(self) -> None:
        with torch.no_grad():
            self.mem_keys.zero_()
            self.mem_vals.zero_()
            self.mem_confidence.zero_()
            self.mem_importance.zero_()
            self.global_step.zero_()

    def decay_memory(self, memory_state: Optional[Dict[str, torch.Tensor]] = None) -> Optional[Dict[str, torch.Tensor]]:
        """Optional secondary decay, clearly separated from differentiable learning."""
        rate = self.config.mem_decay_rate
        decay_factor = math.exp(-rate)
        if memory_state is not None:
            return {
                "keys": memory_state["keys"] * decay_factor,
                "vals": memory_state["vals"] * decay_factor,
                "confidence": memory_state["confidence"] * decay_factor,
                "importance": memory_state["importance"] * decay_factor,
                "step": memory_state["step"] + 1,
            }
        with torch.no_grad():
            self.mem_keys.mul_(decay_factor)
            self.mem_vals.mul_(decay_factor)
            self.mem_confidence.mul_(decay_factor)
            self.mem_importance.mul_(decay_factor)
            self.global_step.add_(1)
        return None

    def read(
        self,
        h_t: torch.Tensor,
        memory_state: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        READ:
          q_t = Q(h_t)
          score_i = sim(q_t, K_i) + lambda * importance_i
          alpha = softmax(score / tau_r)
          r_t = sum_i alpha_i V_i
        """
        keys = memory_state["keys"]                # (B, C, D)
        vals = memory_state["vals"]                # (B, C, D)
        importance = memory_state["importance"]    # (B, C)

        # Query projection from hidden state
        q_t = self.q_proj(h_t)                     # (B, D)

        # Direct similarity between query and stored memory keys without re-projection
        scores = torch.einsum("bd,bcd->bc", q_t, keys) / math.sqrt(self.config.memory_dim)

        # Importance modulates retention and retrieval relevance
        scores = scores + self.importance_weight * importance

        tau_r = max(self.config.read_temperature, 1e-4)
        attn = torch.softmax(scores / tau_r, dim=-1)  # (B, C)

        retrieved = torch.einsum("bc,bcd->bd", attn, vals)  # (B, D)
        return retrieved, attn, scores

    def fuse(self, h_t: torch.Tensor, r_t: torch.Tensor) -> torch.Tensor:
        """
        FUSION:
          g_t = sigmoid(W_g [h_t ; r_t])
          fused_t = h_t + g_t * W_f(r_t)
        """
        cat_feat = torch.cat([h_t, r_t], dim=-1)
        gate = torch.sigmoid(self.fusion_gate_proj(cat_feat))
        return h_t + gate * self.fusion_proj(r_t)

    def write(
        self,
        h_t: torch.Tensor,
        memory_state: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        WRITE:
          k_t = K(h_t)
          v_t = V(h_t)
          write_gate = sigmoid(W_w h_t)
          content_score_i = sim(k_t, K_i)
          vacancy_score_i = - w_conf * confidence_i + slot_bias_i
          allocation = (1 - novelty) * softmax(content_score / tau_w) + novelty * softmax(vacancy_score / tau_v)
          K_{t+1, i} = (1 - write_strength_i) K_i + write_strength_i k_t
          V_{t+1, i} = (1 - write_strength_i) V_i + write_strength_i v_t
        """
        keys = memory_state["keys"]                # (B, C, D)
        vals = memory_state["vals"]                # (B, C, D)
        confidence = memory_state["confidence"]    # (B, C)
        importance = memory_state["importance"]    # (B, C)

        # Clean Write Key and Write Value projections (Problem 2)
        k_t = self.k_proj(h_t)                     # (B, D)
        v_t = self.v_proj(h_t)                     # (B, D)

        # Write strength policy in [0, 1]
        write_gate = torch.sigmoid(self.write_gate_proj(h_t)).squeeze(-1)  # (B,)

        # 1. Content-based similarity: compare write key k_t directly with stored keys
        content_sim = torch.einsum("bd,bcd->bc", k_t, keys) / math.sqrt(self.config.memory_dim)
        tau_w = max(self.config.write_temperature, 1e-4)
        content_alloc = torch.softmax(content_sim / tau_w, dim=-1)  # (B, C)

        # 2. Competitive vacancy score with learned slot bias (Problem 1)
        # Empty slots have confidence ~ 0, so slot_bias breaks symmetry and concentrates allocation
        vacancy_score = -self.confidence_weight * confidence + self.slot_bias  # (B, C)
        tau_v = max(self.config.vacancy_temperature, 1e-4)
        vacancy_alloc = torch.softmax(vacancy_score / tau_v, dim=-1)  # (B, C)

        # 3. Novelty mechanism: determines if information is novel vs redundant
        max_sim = content_sim.max(dim=-1, keepdim=True).values  # (B, 1)
        novelty = torch.sigmoid(self.novelty_proj(h_t) - self.novelty_sim_scale * max_sim)  # (B, 1)

        # Combined differentiable competitive allocation
        allocation = (1.0 - novelty) * content_alloc + novelty * vacancy_alloc  # (B, C)

        # 4. Differentiable memory update (Problem 3) - NO detaching during forward!
        write_strength = write_gate.unsqueeze(-1) * allocation  # (B, C)
        ws_3d = write_strength.unsqueeze(-1)                   # (B, C, 1)

        new_keys = (1.0 - ws_3d) * keys + ws_3d * k_t.unsqueeze(1)
        new_vals = (1.0 - ws_3d) * vals + ws_3d * v_t.unsqueeze(1)

        # 5. Differentiable metadata update (Problem 5)
        new_confidence = (1.0 - write_strength) * confidence + write_strength
        new_importance = (1.0 - write_strength) * importance + write_strength * write_gate.unsqueeze(-1)
        step = memory_state["step"] + 1

        # Optional secondary decay (Problem 6)
        if self.config.enable_decay:
            decay_factor = math.exp(-self.config.mem_decay_rate)
            new_keys = new_keys * decay_factor
            new_vals = new_vals * decay_factor
            new_confidence = new_confidence * decay_factor
            new_importance = new_importance * decay_factor

        next_state = {
            "keys": new_keys,
            "vals": new_vals,
            "confidence": new_confidence,
            "importance": new_importance,
            "step": step,
        }

        diag = {
            "write_gate": write_gate,
            "novelty": novelty.squeeze(-1) if novelty.dim() > 1 else novelty,
            "allocation": allocation,
        }
        return next_state, diag

    def init_or_expand_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        memory_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        if memory_state is not None:
            return memory_state

        if batch_size == 1:
            return self.get_memory_state()

        return self.empty_memory_state(batch_size=batch_size, device=device, dtype=dtype)

