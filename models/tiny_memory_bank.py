import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparsemaxFunction(torch.autograd.Function):
    """
    Exact Euclidean projection onto the probability simplex (Martins & Astudillo, ICML 2016).

    Given input z in R^C:
      sparsemax(z) = argmin_{p in Delta^{C-1}} ||p - z||^2 = max(z - tau(z), 0)

    Sparsity arises naturally from the geometry of the projection.
    Zero manual thresholds, zero epsilon filtering.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, dim: int = -1) -> torch.Tensor:
        ctx.dim = dim

        # Shift input by max along dim for numerical stability
        max_val = input.max(dim=dim, keepdim=True).values
        input_shifted = input - max_val

        # Sort descending along dimension
        sorted_input, _ = torch.sort(input_shifted, descending=True, dim=dim)
        input_cumsum = torch.cumsum(sorted_input, dim=dim)

        # 1-based index vector
        k_indices = torch.arange(1, input.size(dim) + 1, device=input.device, dtype=input.dtype)
        view_shape = [1] * input.dim()
        view_shape[dim] = -1
        k_indices = k_indices.view(view_shape)

        # Support condition: 1 + k * sorted_input > input_cumsum
        support = (1.0 + k_indices * sorted_input) > input_cumsum
        k_z = support.sum(dim=dim, keepdim=True).clamp(min=1)

        # Threshold tau(z) = (sum_{j in S} z_j - 1) / |S|
        tau = (input_cumsum.gather(dim, k_z.long() - 1) - 1.0) / k_z.to(input.dtype)
        output = torch.clamp(input_shifted - tau, min=0.0)

        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        (output,) = ctx.saved_tensors
        dim = ctx.dim

        # Active support is non-zero output
        non_zeros = (output > 0).to(grad_output.dtype)
        num_nonzeros = non_zeros.sum(dim=dim, keepdim=True).clamp(min=1.0)

        # Exact subgradient: dL/dz_i = non_zeros * (grad_i - sum_{j in S} grad_j / |S|)
        sum_grad = (grad_output * non_zeros).sum(dim=dim, keepdim=True)
        grad_input = non_zeros * (grad_output - sum_grad / num_nonzeros)
        return grad_input, None


def sparsemax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Applies the Sparsemax activation function along the specified dimension."""
    return SparsemaxFunction.apply(input, dim)


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

    # Continuous utility tracking EMA rate
    utility_decay_rate: float = 0.1

    # Optional secondary decay (disabled by default so memory learns through NTP)
    enable_decay: bool = False
    mem_decay_rate: float = 0.001


class TinyMemoryBank(nn.Module):
    """
    Differentiable Causal Memory Bank with Competitive Slot Allocation.

    Strict Causal Ordering:
      READ(M_t) -> FUSE(h_t, r_t) -> PREDICT x_{t+1} -> WRITE(h_t, M_t) -> M_{t+1}

    Stored Memory State M_t:
      keys in R^{B x C x D}
      vals in R^{B x C x D}
      confidence in R^{B x C}  (continuous occupancy in [0, 1])
      importance in R^{B x C}  (continuous utility accumulated from NTP prediction retrieval)
      step in R^{B x 1}
    """

    def __init__(self, config: TinyMemoryConfig):
        super().__init__()
        self.config = config
        d = config.memory_dim
        h = config.hidden_size
        c = config.memory_capacity

        # Learned Q, K, V projections from hidden state h_t
        self.q_proj = nn.Linear(h, d, bias=False)
        self.k_proj = nn.Linear(h, d, bias=False)
        self.v_proj = nn.Linear(h, d, bias=False)

        # Continuous write gate policy in [0, 1]
        self.write_gate_proj = nn.Linear(h, 1)
        nn.init.constant_(self.write_gate_proj.bias, config.initial_write_bias)

        # Continuous novelty policy in [0, 1]
        self.novelty_proj = nn.Linear(h, 1)
        nn.init.constant_(self.novelty_proj.bias, config.initial_novelty_bias)
        self.novelty_sim_scale = nn.Parameter(torch.tensor(1.0))

        # Learned slot biases break initial symmetry among empty slots
        slot_ranks = torch.linspace(2.0, -2.0, steps=c)
        self.slot_bias = nn.Parameter(slot_ranks)

        # Learned positive vacancy weights for confidence (occupancy) and utility (importance)
        self.conf_weight = nn.Parameter(torch.tensor(2.0))
        self.imp_weight = nn.Parameter(torch.tensor(1.0))

        # Learned modulation of retrieval score by memory utility
        self.read_utility_weight = nn.Parameter(torch.tensor(1.0))

        # Gated residual fusion projections
        self.fusion_proj = nn.Linear(d, h, bias=False)
        self.fusion_gate_proj = nn.Linear(h + d, h)

        # Persistent runtime buffers for single-stream interactive inference
        self.register_buffer("mem_keys", torch.zeros(c, d))
        self.register_buffer("mem_vals", torch.zeros(c, d))
        self.register_buffer("mem_confidence", torch.zeros(c))
        self.register_buffer("mem_importance", torch.zeros(c))
        self.register_buffer("global_step", torch.zeros(1, dtype=torch.int32))

        self.last_diagnostics: Dict[str, Any] = {}

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
        """Explicit Truncated-BPTT boundary: detaches all state tensors across chunks."""
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
        """Continuous exponential decay, cleanly separated from the differentiable path."""
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
        Differentiable READ:
          q_t = Q(h_t)
          sim_i = <q_t, K_i> / sqrt(D)
          score_i = sim_i + softplus(read_utility_weight) * importance_i
          alpha_read = sparsemax(score / tau_r)
          r_t = sum_i alpha_read_i * V_i
        """
        keys = memory_state["keys"]                # (B, C, D)
        vals = memory_state["vals"]                # (B, C, D)
        importance = memory_state["importance"]    # (B, C)

        q_t = self.q_proj(h_t)                     # (B, D)

        # Scaled dot-product similarity
        sim = torch.einsum("bd,bcd->bc", q_t, keys) / math.sqrt(self.config.memory_dim)

        # Continuous utility modulation of retrieval score
        scores = sim + F.softplus(self.read_utility_weight) * importance

        tau_r = max(self.config.read_temperature, 1e-4)
        attn = sparsemax(scores / tau_r, dim=-1)   # (B, C)

        retrieved = torch.einsum("bc,bcd->bd", attn, vals)  # (B, D)
        return retrieved, attn, scores

    def fuse(self, h_t: torch.Tensor, r_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiable Gated Residual FUSION:
          g_t = sigmoid(W_g [h_t ; r_t])
          fused_t = h_t + g_t * W_f(r_t)
        """
        cat_feat = torch.cat([h_t, r_t], dim=-1)
        gate = torch.sigmoid(self.fusion_gate_proj(cat_feat))
        fused_t = h_t + gate * self.fusion_proj(r_t)
        return fused_t, gate

    def write(
        self,
        h_t: torch.Tensor,
        memory_state: Dict[str, torch.Tensor],
        alpha_read: Optional[torch.Tensor] = None,
        fusion_gate: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Differentiable Competitive WRITE:
          1. k_t = K(h_t), v_t = V(h_t)
          2. write_gate = sigmoid(W_w h_t)
          3. Content Allocation:
               content_sim_i = <k_t, K_i> / sqrt(D)
               content_alloc = sparsemax(content_sim / tau_w)
          4. Vacancy / Replacement Allocation:
               vacancy_score_i = slot_bias_i - softplus(conf_w) * conf_i - softplus(imp_w) * imp_i
               vacancy_alloc = sparsemax(vacancy_score / tau_v)
          5. Continuous Novelty:
               match_sim = sum_i (content_alloc_i * content_sim_i)
               novelty = sigmoid(W_nov h_t - softplus(scale) * match_sim + b_nov)
          6. Blended Allocation:
               allocation = (1 - novelty) * content_alloc + novelty * vacancy_alloc
          7. Soft Memory Update:
               write_strength_i = write_gate * allocation_i
               K_{t+1, i} = (1 - write_strength_i) * K_{t, i} + write_strength_i * k_t
               V_{t+1, i} = (1 - write_strength_i) * V_{t, i} + write_strength_i * v_t
          8. Continuous State Updates:
               conf_{t+1, i} = conf_{t, i} + (1 - conf_{t, i}) * write_strength_i
               imp_{t+1, i}  = (1 - lambda_u) * imp_{t, i} + lambda_u * (gate_mean * alpha_read_i)
        """
        keys = memory_state["keys"]                # (B, C, D)
        vals = memory_state["vals"]                # (B, C, D)
        confidence = memory_state["confidence"]    # (B, C)
        importance = memory_state["importance"]    # (B, C)

        k_t = self.k_proj(h_t)                     # (B, D)
        v_t = self.v_proj(h_t)                     # (B, D)

        # 1. Continuous write gate in [0, 1]
        write_gate = torch.sigmoid(self.write_gate_proj(h_t)).squeeze(-1)  # (B,)

        # 2. Content-based allocation via Sparsemax
        content_sim = torch.einsum("bd,bcd->bc", k_t, keys) / math.sqrt(self.config.memory_dim)
        tau_w = max(self.config.write_temperature, 1e-4)
        content_alloc = sparsemax(content_sim / tau_w, dim=-1)  # (B, C)

        # 3. Vacancy suitability allocation via Sparsemax
        w_conf = F.softplus(self.conf_weight)
        w_imp = F.softplus(self.imp_weight)
        vacancy_score = self.slot_bias.unsqueeze(0) - w_conf * confidence - w_imp * importance  # (B, C)
        tau_v = max(self.config.vacancy_temperature, 1e-4)
        vacancy_alloc = sparsemax(vacancy_score / tau_v, dim=-1)  # (B, C)

        # 4. Continuous Differentiable Novelty
        match_sim = (content_alloc * content_sim).sum(dim=-1, keepdim=True)  # (B, 1)
        nov_scale = F.softplus(self.novelty_sim_scale)
        novelty = torch.sigmoid(self.novelty_proj(h_t) - nov_scale * match_sim)  # (B, 1)

        # 5. Continuous Blended Allocation
        allocation = (1.0 - novelty) * content_alloc + novelty * vacancy_alloc  # (B, C)

        # 6. Soft Memory Update (Zero detach, fully differentiable)
        write_strength = write_gate.unsqueeze(-1) * allocation  # (B, C)
        ws_3d = write_strength.unsqueeze(-1)                   # (B, C, 1)

        new_keys = (1.0 - ws_3d) * keys + ws_3d * k_t.unsqueeze(1)
        new_vals = (1.0 - ws_3d) * vals + ws_3d * v_t.unsqueeze(1)

        # 7. Continuous State Dynamics
        # Confidence accumulates occupancy continuously in [0, 1]
        new_confidence = confidence + (1.0 - confidence) * write_strength

        # Importance accumulates continuous utility from prediction retrieval
        if alpha_read is not None:
            if fusion_gate is not None:
                gate_factor = fusion_gate.mean(dim=-1, keepdim=True)  # (B, 1)
                utility_signal = gate_factor * alpha_read              # (B, C)
            else:
                utility_signal = alpha_read
        else:
            utility_signal = torch.zeros_like(importance)

        lam_u = self.config.utility_decay_rate
        new_importance = (1.0 - lam_u) * importance + lam_u * utility_signal

        step = memory_state["step"] + 1

        # Optional secondary continuous decay
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

        # Continuous Diagnostics (Non-invasive, zero thresholds)
        with torch.no_grad():
            eff_write_slots = 1.0 / (allocation.pow(2).sum(dim=-1).clamp(min=1e-8))  # (B,)
            write_sparsity = (allocation == 0.0).float().mean(dim=-1)
            alloc_entropy = -(allocation * torch.log(allocation.clamp(min=1e-12))).sum(dim=-1)

            diag = {
                "write_gate": write_gate,
                "novelty": novelty.squeeze(-1) if novelty.dim() > 1 else novelty,
                "allocation": allocation,
                "effective_write_slots": eff_write_slots,
                "write_sparsity": write_sparsity,
                "allocation_entropy": alloc_entropy,
                "confidence_sum": new_confidence.sum(dim=-1),
                "confidence_mean": new_confidence.mean(dim=-1),
                "importance_mean": new_importance.mean(dim=-1),
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
