from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MemoryState:
    """
    Canonical state representation for Differentiable Causal Memory Bank.
    Each batch element maintains independent memory state.
    """
    keys: torch.Tensor       # K_t in R^(B x C x D)
    values: torch.Tensor     # V_t in R^(B x C x D)
    occupancy: torch.Tensor  # O_t in R^(B x C)
    usage: torch.Tensor      # U_t in R^(B x C)
    age: torch.Tensor        # A_t in R^(B x C)

    def detach(self) -> "MemoryState":
        """Explicit TBPTT boundary: detaches all state tensors."""
        return MemoryState(
            keys=self.keys.detach(),
            values=self.values.detach(),
            occupancy=self.occupancy.detach(),
            usage=self.usage.detach(),
            age=self.age.detach(),
        )


@dataclass
class TinyMemoryConfig:
    memory_capacity: int = 128   # C: number of memory slots
    memory_dim: int = 768        # D: memory key/value dimension
    hidden_size: int = 768       # H: backbone hidden dimension

    # Explicit positive temperatures
    tau_read: float = 1.0        # tau_read > 0
    tau_write: float = 1.0       # tau_write > 0

    # Replacement weight
    lambda_replace: float = 1.0  # lambda_replace >= 0

    # Numerical constant
    eps: float = 1e-8


class TinyMemoryBank(nn.Module):
    """
    Fully Differentiable Causal Memory Bank.
    Strictly implements the explicit mathematical specification.

    Order of operations at timestep t:
      1. r_t = READ(h_t, M_t)
      2. z_t = FUSE(h_t, r_t)
      3. logits_t = LMHead(z_t)   # (executed in GPT2MemoryModel before WRITE)
      4. M_{t+1} = WRITE(h_t, M_t, alpha_t, g_t)
    """

    def __init__(self, config: TinyMemoryConfig):
        super().__init__()
        self.config = config
        h = config.hidden_size
        d = config.memory_dim
        c = config.memory_capacity
        self.eps = config.eps
        self.tau_read = config.tau_read
        self.tau_write = config.tau_write
        self.lambda_replace = config.lambda_replace

        # Section 5: Projections
        # q_t = W_q h_t, k_t = W_k h_t, v_t = W_v h_t
        self.q_proj = nn.Linear(h, d, bias=False)
        self.k_proj = nn.Linear(h, d, bias=False)
        self.v_proj = nn.Linear(h, d, bias=False)

        # Section 9: Fusion projections
        # m_t = W_m r_t, g_t = sigmoid(W_g [h_t ; r_t] + b_g)
        self.fusion_proj = nn.Linear(d, h, bias=False)
        self.fusion_gate_proj = nn.Linear(h + d, h, bias=True)

        # Section 12: Write gate projection
        # g_write = sigmoid(W_write h_t + b_write)
        self.write_gate_proj = nn.Linear(h, 1, bias=True)

        # Section 6: Learned Slot Addresses P in R^(C x D)
        # Symmetry-breaking initialization (orthogonal or random normal)
        self.P = nn.Parameter(torch.empty(c, d))
        if c <= d:
            nn.init.orthogonal_(self.P)
        else:
            nn.init.normal_(self.P, mean=0.0, std=1.0 / (d ** 0.5))

        # Runtime persistence buffers for single-stream interactive inference
        self.register_buffer("mem_keys", torch.zeros(c, d))
        self.register_buffer("mem_vals", torch.zeros(c, d))
        self.register_buffer("mem_occupancy", torch.zeros(c))
        self.register_buffer("mem_usage", torch.zeros(c))
        self.register_buffer("mem_age", torch.zeros(c))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Helper to normalize vectors: xbar = x / (||x||_2 + eps)."""
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        return x / (norm + self.eps)

    def initialize_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> MemoryState:
        """
        Section 7: Memory State Initialization.
        At the beginning of a new memory stream:
          K_0 = 0, V_0 = 0, O_0 = 0, U_0 = 0, A_0 = 0
        """
        c = self.config.memory_capacity
        d = self.config.memory_dim
        return MemoryState(
            keys=torch.zeros(batch_size, c, d, device=device, dtype=dtype),
            values=torch.zeros(batch_size, c, d, device=device, dtype=dtype),
            occupancy=torch.zeros(batch_size, c, device=device, dtype=dtype),
            usage=torch.zeros(batch_size, c, device=device, dtype=dtype),
            age=torch.zeros(batch_size, c, device=device, dtype=dtype),
        )

    def empty_memory_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> MemoryState:
        """Alias for initialize_state."""
        return self.initialize_state(batch_size=batch_size, device=device, dtype=dtype)

    def detach_state(self, state: MemoryState) -> MemoryState:
        """Section 23: Explicit TBPTT boundary."""
        return state.detach()

    def get_runtime_state(self) -> MemoryState:
        """Retrieve single-stream runtime memory state."""
        device = self.mem_keys.device
        dtype = self.mem_keys.dtype
        state = self.initialize_state(batch_size=1, device=device, dtype=dtype)
        state.keys[0] = self.mem_keys
        state.values[0] = self.mem_vals
        state.occupancy[0] = self.mem_occupancy
        state.usage[0] = self.mem_usage
        state.age[0] = self.mem_age
        return state

    def persist_runtime_state(self, state: MemoryState) -> None:
        """Persist single-stream runtime state for interactive inference (detached)."""
        if state.keys.size(0) != 1:
            return
        with torch.no_grad():
            self.mem_keys.copy_(state.keys[0].detach())
            self.mem_vals.copy_(state.values[0].detach())
            self.mem_occupancy.copy_(state.occupancy[0].detach())
            self.mem_usage.copy_(state.usage[0].detach())
            self.mem_age.copy_(state.age[0].detach())

    def reset_memory(self) -> None:
        """Reset runtime memory buffers to zeros."""
        with torch.no_grad():
            self.mem_keys.zero_()
            self.mem_vals.zero_()
            self.mem_occupancy.zero_()
            self.mem_usage.zero_()
            self.mem_age.zero_()

    def read(
        self,
        h_t: torch.Tensor,
        state: MemoryState,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Section 8: Differentiable READ Operation.
          q_t = W_q h_t
          qbar_t = q_t / (||q_t||_2 + eps)
          s_i^read = qbar_t . kbar_i
          content_weight_i = exp(s_i^read / tau_read) * O_i
          Z = sum_i content_weight_i
          alpha_i = content_weight_i / (Z + eps)
          r_t = sum_i alpha_i V_i
          read_presence = valid_mass / (valid_mass + eps)
          r_t = read_presence * r_t
        """
        # q_t = W_q h_t
        q_t = self.q_proj(h_t)
        # qbar_t = q_t / (||q_t||_2 + eps)
        qbar_t = self._normalize(q_t)  # (B, D)

        # kbar_i = K_i / (||K_i||_2 + eps)
        kbar_stored = self._normalize(state.keys)  # (B, C, D)

        # s_i^read = qbar_t . kbar_i
        s_read = torch.einsum("bd,bcd->bc", qbar_t, kbar_stored)  # (B, C)

        # Numerically stable content_weight_i: exp((s_read - max) / tau_read) * O_i
        s_read_shifted = (s_read - s_read.max(dim=-1, keepdim=True).values.detach()) / self.tau_read
        # content_weight_i = exp(s_i^read / tau_read) * O_i
        content_weight = torch.exp(s_read_shifted) * state.occupancy  # (B, C)

        # Z = sum_i content_weight_i
        Z = content_weight.sum(dim=-1, keepdim=True)  # (B, 1)

        # alpha_i = content_weight_i / (Z + eps)
        alpha = content_weight / (Z + self.eps)  # (B, C)

        # r_t = sum_i alpha_i V_i
        r_t_raw = torch.einsum("bc,bcd->bd", alpha, state.values)  # (B, D)

        # valid_mass = sum_i O_i
        valid_mass = state.occupancy.sum(dim=-1, keepdim=True)  # (B, 1)

        # read_presence = valid_mass / (valid_mass + eps)
        read_presence = valid_mass / (valid_mass + self.eps)  # (B, 1)

        # r_t = read_presence * r_t
        r_t = read_presence * r_t_raw  # (B, D)

        return r_t, alpha

    def fuse(
        self,
        h_t: torch.Tensor,
        r_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Section 9: Memory FUSION.
          fusion_input = [h_t ; r_t]
          g_t = sigmoid(W_g [h_t ; r_t] + b_g)
          m_t = W_m r_t
          z_t = h_t + g_t * m_t
        """
        # fusion_input = [h_t ; r_t]
        fusion_input = torch.cat([h_t, r_t], dim=-1)  # (B, H + D)

        # g_t = sigmoid(W_g [h_t ; r_t] + b_g)
        g_t = torch.sigmoid(self.fusion_gate_proj(fusion_input))  # (B, H)

        # m_t = W_m r_t
        m_t = self.fusion_proj(r_t)  # (B, H)

        # z_t = h_t + g_t * m_t
        z_t = h_t + g_t * m_t  # (B, H)

        return z_t, g_t

    def write(
        self,
        h_t: torch.Tensor,
        state: MemoryState,
        alpha_read: Optional[torch.Tensor] = None,
        fusion_gate: Optional[torch.Tensor] = None,
    ) -> Tuple[MemoryState, Dict[str, Any]]:
        """
        Sections 10-19: Differentiable WRITE Operation.
          1. k_t = W_k h_t, kbar_t = normalize(k_t), v_t = W_v h_t
          2. c_i = kbar_t . kbar_i
          3. d_i = kbar_t . pbar_i
          4. b_i = O_i * c_i + (1 - O_i) * d_i
          5. replace_i = O_i * (1 - U_i) + (1 - O_i)
          6. write_score_i = b_i + lambda_replace * replace_i
          7. allocation_i = softmax(write_score_i / tau_write)
          8. g_write = sigmoid(W_write h_t + b_write)
          9. w_i = g_write * allocation_i
         10. K'_{t+1, i} = (1 - w_i) K_{t, i} + w_i kbar_t; K_{t+1, i} = normalize(K')
         11. V_{t+1, i} = (1 - w_i) V_{t, i} + w_i v_t
         12. O_{t+1, i} = O_{t, i} + w_i (1 - O_{t, i})
         13. usage_i = alpha_i * mean_H(g_t); U_{t+1, i} = (1 - w_i) U_{t, i} + w_i usage_i
         14. A_{t+1, i} = (1 - w_i) (A_{t, i} + 1)
        """
        # 1. k_t = W_k h_t, v_t = W_v h_t
        k_t = self.k_proj(h_t)
        v_t = self.v_proj(h_t)
        # kbar_t = k_t / (||k_t||_2 + eps)
        kbar_t = self._normalize(k_t)  # (B, D)

        # kbar_i = K_i / (||K_i||_2 + eps)
        kbar_stored = self._normalize(state.keys)  # (B, C, D)

        # pbar_i = P_i / (||P_i||_2 + eps)
        pbar = self._normalize(self.P)  # (C, D)

        # 2. Section 10.1: c_i = kbar_t . kbar_i
        c = torch.einsum("bd,bcd->bc", kbar_t, kbar_stored)  # (B, C)

        # 3. Section 10.2: d_i = kbar_t . pbar_i
        d = torch.einsum("bd,cd->bc", kbar_t, pbar)  # (B, C)

        # 4. Section 10.3: b_i = O_i * c_i + (1 - O_i) * d_i
        b = state.occupancy * c + (1.0 - state.occupancy) * d  # (B, C)

        # 5. Section 11: replace_i = O_i * (1 - U_i) + (1 - O_i)
        replace = state.occupancy * (1.0 - state.usage) + (1.0 - state.occupancy)  # (B, C)

        # 6. Section 11.1: write_score_i = b_i + lambda_replace * replace_i
        write_score = b + self.lambda_replace * replace  # (B, C)

        # 7. Section 11.1: allocation_i = softmax(write_score_i / tau_write)
        allocation = torch.softmax(write_score / self.tau_write, dim=-1)  # (B, C)

        # 8. Section 12: g_write = sigmoid(W_write h_t + b_write)
        g_write = torch.sigmoid(self.write_gate_proj(h_t))  # (B, 1)

        # 9. Section 12: w_i = g_write * allocation_i
        w = g_write * allocation  # (B, C)
        w_3d = w.unsqueeze(-1)    # (B, C, 1)

        # 10. Section 13: K'_i = (1 - w_i) K_i + w_i kbar_t
        K_prime = (1.0 - w_3d) * state.keys + w_3d * kbar_t.unsqueeze(1)  # (B, C, D)
        # Normalize non-empty resulting keys: K_(t+1, i) = K'_i / (||K'_i||_2 + eps)
        norm_K = torch.norm(K_prime, p=2, dim=-1, keepdim=True) + self.eps
        K_next = K_prime / norm_K  # (B, C, D)

        # 11. Section 14: V_(t+1, i) = (1 - w_i) V_i + w_i v_t
        V_next = (1.0 - w_3d) * state.values + w_3d * v_t.unsqueeze(1)  # (B, C, D)

        # 12. Section 15: O_(t+1, i) = O_i + w_i (1 - O_i)
        O_next = state.occupancy + w * (1.0 - state.occupancy)  # (B, C)

        # 13. Section 16: usage_i = alpha_i * G_t, where G_t = mean_H(g_t)
        if alpha_read is not None and fusion_gate is not None:
            G_t = fusion_gate.mean(dim=-1, keepdim=True)  # (B, 1)
            usage_signal = alpha_read * G_t               # (B, C)
        elif alpha_read is not None:
            usage_signal = alpha_read
        else:
            usage_signal = torch.zeros_like(state.usage)

        # U_(t+1, i) = (1 - w_i) U_i + w_i * usage_i
        U_next = (1.0 - w) * state.usage + w * usage_signal  # (B, C)

        # 14. Section 17: A_(t+1, i) = (1 - w_i) (A_i + 1)
        A_next = (1.0 - w) * (state.age + 1.0)  # (B, C)

        next_state = MemoryState(
            keys=K_next,
            values=V_next,
            occupancy=O_next,
            usage=U_next,
            age=A_next,
        )

        with torch.no_grad():
            diag = {
                "write_gate": g_write.squeeze(-1),
                "allocation": allocation,
                "w": w,
                "occupancy_sum": O_next.sum(dim=-1),
                "occupancy_mean": O_next.mean(dim=-1),
                "usage_mean": U_next.mean(dim=-1),
                "age_mean": A_next.mean(dim=-1),
            }

        return next_state, diag
