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

    # Differentiable read/write temperatures
    read_temperature: float = 1.0
    write_temperature: float = 1.0
    novelty_temperature: float = 1.0
    empty_temperature: float = 1.0

    # Learnable gate calibration defaults
    initial_novelty_bias: float = 0.0

    # Runtime bookkeeping only
    mem_decay_rate: float = 0.001


class TinyMemoryBank(nn.Module):
    """
    Differentiable causal memory bank.

    READ:
      q_t = W_q h_t
      alpha = softmax(sim(q_t, W_k K_t) / tau_r)
      r_t = sum_i alpha_i V_t,i

    WRITE:
      v_t = W_v h_t
      w_t = sigmoid(W_i h_t)
      a_content = softmax(sim(W_k h_t, W_k K_t) / tau_w)
      a_empty = softmax(-confidence / tau_e)
      novelty = sigmoid((b_novelty - max_sim) / tau_n)
      a = (1 - novelty) * a_content + novelty * a_empty
      K_{t+1,i} = (1 - a_i w_t) K_{t,i} + (a_i w_t) k_t
      V_{t+1,i} = (1 - a_i w_t) V_{t,i} + (a_i w_t) v_t

    Causal order (implemented in GPT2MemoryModel):
      READ(M_t) -> FUSE -> predict x_{t+1} -> WRITE -> M_{t+1}
    """

    def __init__(self, config: TinyMemoryConfig):
        super().__init__()
        self.config = config
        d = config.memory_dim
        h = config.hidden_size
        c = config.memory_capacity

        self.q_proj = nn.Linear(h, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(h, d, bias=False)
        self.i_proj = nn.Linear(h, 1)
        self.fusion_proj = nn.Linear(d, h, bias=False)
        self.fusion_gate_proj = nn.Linear(h + d, h)

        self.novelty_bias = nn.Parameter(torch.tensor(float(config.initial_novelty_bias)))

        # Persistent runtime state (single stream), not trainable parameters.
        self.register_buffer("mem_keys", torch.zeros(c, d))
        self.register_buffer("mem_vals", torch.zeros(c, d))
        self.register_buffer("mem_importance", torch.zeros(c))
        self.register_buffer("mem_confidence", torch.zeros(c))
        self.register_buffer("mem_created_at", torch.zeros(c, dtype=torch.int32))
        self.register_buffer("mem_last_access", torch.zeros(c, dtype=torch.int32))
        self.register_buffer("mem_access_count", torch.zeros(c, dtype=torch.int32))
        self.register_buffer("mem_state", torch.full((c,), STATE_EXPIRED, dtype=torch.int32))
        self.register_buffer("global_step", torch.zeros(1, dtype=torch.int32))

        self.last_diagnostics: Dict[str, Any] = {}

    @property
    def active_count(self) -> int:
        return int((self.mem_state == STATE_ACTIVE).sum().item())

    def empty_memory_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        c = self.config.memory_capacity
        d = self.config.memory_dim
        zeros_cd = torch.zeros(batch_size, c, d, device=device, dtype=dtype)
        zeros_c = torch.zeros(batch_size, c, device=device, dtype=dtype)
        zeros_ci = torch.zeros(batch_size, c, device=device, dtype=torch.int32)
        zeros_1i = torch.zeros(batch_size, 1, device=device, dtype=torch.int32)
        return {
            "keys": zeros_cd,
            "vals": zeros_cd.clone(),
            "importance": zeros_c,
            "confidence": zeros_c.clone(),
            "created_at": zeros_ci,
            "last_access": zeros_ci.clone(),
            "access_count": zeros_ci.clone(),
            "state": torch.full((batch_size, c), STATE_EXPIRED, device=device, dtype=torch.int32),
            "global_step": zeros_1i,
        }

    def get_memory_state(self) -> Dict[str, torch.Tensor]:
        device = self.mem_keys.device
        dtype = self.mem_keys.dtype
        state = self.empty_memory_state(batch_size=1, device=device, dtype=dtype)
        state["keys"][0] = self.mem_keys
        state["vals"][0] = self.mem_vals
        state["importance"][0] = self.mem_importance
        state["confidence"][0] = self.mem_confidence
        state["created_at"][0] = self.mem_created_at
        state["last_access"][0] = self.mem_last_access
        state["access_count"][0] = self.mem_access_count
        state["state"][0] = self.mem_state
        state["global_step"][0] = self.global_step
        return state

    def load_memory_state(self, state: Dict[str, torch.Tensor]) -> None:
        s = state
        self.mem_keys.copy_(s["keys"][0].detach())
        self.mem_vals.copy_(s["vals"][0].detach())
        self.mem_importance.copy_(s["importance"][0].detach())
        self.mem_confidence.copy_(s["confidence"][0].detach())
        self.mem_created_at.copy_(s["created_at"][0].detach().to(torch.int32))
        self.mem_last_access.copy_(s["last_access"][0].detach().to(torch.int32))
        self.mem_access_count.copy_(s["access_count"][0].detach().to(torch.int32))
        self.mem_state.copy_(s["state"][0].detach().to(torch.int32))
        self.global_step.copy_(s["global_step"][0].detach().to(torch.int32))

    def detach_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.detach() for k, v in state.items()}

    def reset_memory(self) -> None:
        with torch.no_grad():
            self.mem_keys.zero_()
            self.mem_vals.zero_()
            self.mem_importance.zero_()
            self.mem_confidence.zero_()
            self.mem_created_at.zero_()
            self.mem_last_access.zero_()
            self.mem_access_count.zero_()
            self.mem_state.fill_(STATE_EXPIRED)
            self.global_step.zero_()

    def decay_memory(self) -> None:
        with torch.no_grad():
            step = self.global_step.item()
            dt = torch.clamp(torch.tensor(step, device=self.mem_last_access.device) - self.mem_last_access, min=0).float()
            decay = torch.exp(-self.config.mem_decay_rate * dt)
            self.mem_importance.mul_(decay)
            self.mem_confidence.mul_(decay)
            self.mem_state.copy_(
                torch.where(
                    self.mem_confidence > 0.5,
                    torch.tensor(STATE_ACTIVE, device=self.mem_state.device, dtype=torch.int32),
                    torch.where(
                        self.mem_confidence > 0.05,
                        torch.tensor(STATE_DORMANT, device=self.mem_state.device, dtype=torch.int32),
                        torch.tensor(STATE_EXPIRED, device=self.mem_state.device, dtype=torch.int32),
                    ),
                )
            )

    def read(
        self,
        h_t: torch.Tensor,
        memory_state: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        keys = memory_state["keys"]
        vals = memory_state["vals"]

        q_t = self.q_proj(h_t)
        k_slots = self.k_proj(keys)

        scores = torch.einsum("bd,bcd->bc", q_t, k_slots) / math.sqrt(k_slots.size(-1))
        scores = scores + memory_state["importance"]
        attn = torch.softmax(scores / self.config.read_temperature, dim=-1)
        retrieved = torch.einsum("bc,bcd->bd", attn, vals)
        return retrieved, attn, scores

    def write(
        self,
        h_t: torch.Tensor,
        memory_state: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        keys = memory_state["keys"]
        vals = memory_state["vals"]
        confidence = memory_state["confidence"]
        importance = memory_state["importance"]

        write_key = self.q_proj(h_t)
        write_val = self.v_proj(h_t)
        write_gate = torch.sigmoid(self.i_proj(h_t)).squeeze(-1)

        key_scores = torch.einsum("bd,bcd->bc", write_key, self.k_proj(keys)) / math.sqrt(keys.size(-1))
        content_alloc = torch.softmax(key_scores / self.config.write_temperature, dim=-1)

        empty_scores = -confidence
        empty_alloc = torch.softmax(empty_scores / self.config.empty_temperature, dim=-1)

        max_sim = key_scores.max(dim=-1).values
        novelty = torch.sigmoid((self.novelty_bias - max_sim) / self.config.novelty_temperature)

        alloc = (1.0 - novelty.unsqueeze(-1)) * content_alloc + novelty.unsqueeze(-1) * empty_alloc
        slot_gate = alloc * write_gate.unsqueeze(-1)
        slot_gate_e = slot_gate.unsqueeze(-1)

        new_keys = (1.0 - slot_gate_e) * keys + slot_gate_e * write_key.unsqueeze(1)
        new_vals = (1.0 - slot_gate_e) * vals + slot_gate_e * write_val.unsqueeze(1)

        write_strength = slot_gate
        new_importance = (1.0 - write_strength) * importance + write_strength * write_gate.unsqueeze(-1)
        new_confidence = (1.0 - write_strength) * confidence + write_strength

        step = memory_state["global_step"] + 1
        access_boost = (slot_gate > (1.0 / self.config.memory_capacity)).to(torch.int32)
        new_access_count = memory_state["access_count"] + access_boost
        new_last_access = torch.where(access_boost > 0, step.expand_as(memory_state["last_access"]), memory_state["last_access"])
        new_created_at = torch.where(
            (memory_state["confidence"] <= 1e-6) & (slot_gate > 0),
            step.expand_as(memory_state["created_at"]),
            memory_state["created_at"],
        )
        new_state_codes = torch.where(
            new_confidence > 0.5,
            torch.tensor(STATE_ACTIVE, device=new_confidence.device, dtype=torch.int32),
            torch.where(
                new_confidence > 0.05,
                torch.tensor(STATE_DORMANT, device=new_confidence.device, dtype=torch.int32),
                torch.tensor(STATE_EXPIRED, device=new_confidence.device, dtype=torch.int32),
            ),
        )

        next_state = {
            "keys": new_keys,
            "vals": new_vals,
            "importance": new_importance,
            "confidence": new_confidence,
            "created_at": new_created_at,
            "last_access": new_last_access,
            "access_count": new_access_count,
            "state": new_state_codes,
            "global_step": step,
        }

        diag = {
            "write_gate": write_gate,
            "novelty": novelty,
            "allocation": alloc,
        }
        return next_state, diag

    def fuse(self, h_t: torch.Tensor, r_t: torch.Tensor) -> torch.Tensor:
        # fused_t = h_t + g_t * W_f(r_t), g_t = sigmoid(W_g([h_t; r_t]))
        gated = torch.sigmoid(self.fusion_gate_proj(torch.cat([h_t, r_t], dim=-1)))
        return h_t + gated * self.fusion_proj(r_t)

    def persist_runtime_state(self, state: Dict[str, torch.Tensor]) -> None:
        if state["keys"].size(0) != 1:
            return
        with torch.no_grad():
            self.mem_keys.copy_(state["keys"][0].detach())
            self.mem_vals.copy_(state["vals"][0].detach())
            self.mem_importance.copy_(state["importance"][0].detach())
            self.mem_confidence.copy_(state["confidence"][0].detach())
            self.mem_created_at.copy_(state["created_at"][0].detach().to(torch.int32))
            self.mem_last_access.copy_(state["last_access"][0].detach().to(torch.int32))
            self.mem_access_count.copy_(state["access_count"][0].detach().to(torch.int32))
            self.mem_state.copy_(state["state"][0].detach().to(torch.int32))
            self.global_step.copy_(state["global_step"][0].detach().to(torch.int32))

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
