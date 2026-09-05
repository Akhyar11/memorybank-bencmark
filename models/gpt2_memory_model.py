from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from models.tiny_memory_bank import MemoryState, TinyMemoryBank, TinyMemoryConfig


class GPT2MemoryModel(nn.Module):
    """
    GPT-2 + Differentiable Causal Memory Bank.

    Strict Causal Loop for each token position t:
      1) READ from M_t:       r_t = READ(h_t, M_t)
      2) FUSE with h_t:       z_t = FUSE(h_t, r_t)
      3) PREDICT x_{t+1}:     logits_t = lm_head(z_t)
      4) WRITE to get M_{t+1}: M_{t+1} = WRITE(h_t, M_t, alpha_t, g_t)

    Training objective: strictly standard causal Next Token Prediction (NTP) loss.
    No auxiliary labels, losses, or manual thresholds.
    """

    def __init__(
        self,
        model_name_or_path: str,
        memory_config: Optional[TinyMemoryConfig] = None,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.gpt2 = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.config = self.gpt2.config
        embed_dim = self.config.n_embd

        # Frozen backbone by default (Section 1)
        if freeze_backbone:
            for p in self.gpt2.parameters():
                p.requires_grad = False

        if memory_config is None:
            memory_config = TinyMemoryConfig(memory_capacity=128, memory_dim=embed_dim, hidden_size=embed_dim)
        else:
            memory_config.hidden_size = embed_dim
            memory_config.memory_dim = embed_dim

        self.memory_config = memory_config
        self.bank = TinyMemoryBank(memory_config)

        self.last_diagnostics: Dict[str, Any] = {}

    def _causal_memory_loop(
        self,
        hidden_states: torch.Tensor,
        state: MemoryState,
        use_memory: bool = True,
    ) -> Dict[str, Any]:
        """
        Non-negotiable causal sequence (Section 2 & 26):
          For t = 0 ... T-1:
            r_t = READ(h_t, M_t)
            z_t = FUSE(h_t, r_t)
            logits_t = LMHead(z_t)
            M_{t+1} = WRITE(h_t, M_t)
        """
        bsz, seqlen, _ = hidden_states.shape
        fused_steps = []
        logits_steps = []

        mean_write_gate = []
        mean_occupancy = []
        mean_usage = []

        for t in range(seqlen):
            h_t = hidden_states[:, t, :]

            if use_memory:
                # 1. READ from M_t
                r_t, alpha_t = self.bank.read(h_t, state)
                # 2. FUSE with h_t
                z_t, g_t = self.bank.fuse(h_t, r_t)
            else:
                r_t = torch.zeros(bsz, self.memory_config.memory_dim, device=h_t.device, dtype=h_t.dtype)
                alpha_t = torch.zeros(bsz, self.memory_config.memory_capacity, device=h_t.device, dtype=h_t.dtype)
                z_t = h_t
                g_t = torch.zeros_like(h_t)

            fused_steps.append(z_t)

            # 3. PREDICT x_{t+1}: strictly before write
            logits_t = self.gpt2.lm_head(z_t)
            logits_steps.append(logits_t)

            if use_memory:
                # 4. WRITE to obtain M_{t+1}
                state, write_diag = self.bank.write(
                    h_t,
                    state,
                    alpha_read=alpha_t,
                    fusion_gate=g_t,
                )
                mean_write_gate.append(write_diag["write_gate"])
                mean_occupancy.append(write_diag["occupancy_mean"])
                mean_usage.append(write_diag["usage_mean"])

        fused_hidden = torch.stack(fused_steps, dim=1)
        logits = torch.stack(logits_steps, dim=1)

        diagnostics = {
            "avg_write_gate": torch.stack(mean_write_gate, dim=1).mean() if mean_write_gate else torch.tensor(0.0, device=hidden_states.device),
            "occupancy_sum": state.occupancy.sum(dim=-1).mean(),
            "occupancy_mean": state.occupancy.mean(),
            "usage_mean": state.usage.mean(),
            "age_mean": state.age.mean(),
        }

        return {
            "fused_hidden": fused_hidden,
            "logits": logits,
            "memory_state": state,
            "diagnostics": diagnostics,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        memory_state: Optional[MemoryState] = None,
        use_memory: bool = True,
        persist_memory: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass with strictly standard Next Token Prediction (NTP) loss.
        """
        transformer_outputs = self.gpt2.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = transformer_outputs.last_hidden_state

        if memory_state is not None:
            state = memory_state
        elif persist_memory and hidden_states.size(0) == 1:
            state = self.bank.get_runtime_state()
        else:
            state = self.bank.initialize_state(
                batch_size=hidden_states.size(0),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

        mem_out = self._causal_memory_loop(hidden_states, state, use_memory=use_memory)
        fused_hidden = mem_out["fused_hidden"]
        logits = mem_out["logits"]
        next_state = mem_out["memory_state"]

        # Section 24: NTP Loss only
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        # Single-stream persistent buffer for interactive inference (detached)
        if persist_memory and memory_state is None and use_memory and hidden_states.size(0) == 1:
            self.bank.persist_runtime_state(next_state)

        self.last_diagnostics = {k: float(v.detach().cpu().item()) for k, v in mem_out["diagnostics"].items()}

        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": fused_hidden,
            "memory_state": next_state,
            "memory_diagnostics": mem_out["diagnostics"],
        }

    def print_trainable_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        ratio = (trainable_params / total_params) * 100

        print("=" * 55)
        print("   PARAMETER AUDIT: GPT-2 + DIFFERENTIABLE MEMORY")
        print("=" * 55)
        print(f"Total Parameters     : {total_params:,}")
        print(f"Frozen Parameters    : {frozen_params:,}")
        print(f"Trainable Parameters : {trainable_params:,}")
        print(f"Trainable Ratio      : {ratio:.2f}%")
        print("=" * 55)
        return trainable_params, total_params

    def reset_memory(self):
        self.bank.reset_memory()

    def detach_memory_state(self, memory_state: MemoryState) -> MemoryState:
        """Section 23: Explicit TBPTT boundary."""
        return memory_state.detach()

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        stop_token_ids: Optional[list] = None,
    ) -> torch.Tensor:
        """
        Section 29: Generation follows the exact same causal memory loop:
          READ(M_t) -> FUSE(h_t, r_t) -> PREDICT x_{t+1} -> WRITE(h_t, M_t) -> M_{t+1}
        """
        del pad_token_id
        stop_ids = set(stop_token_ids or [])
        if eos_token_id is not None:
            stop_ids.add(eos_token_id)

        past_key_values = None
        state = self.bank.get_runtime_state()
        generated = input_ids.clone()

        prompt_outputs = self.gpt2.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        hidden = prompt_outputs.last_hidden_state
        past_key_values = prompt_outputs.past_key_values

        next_token_logits = None
        for t in range(hidden.size(1)):
            h_t = hidden[:, t, :]
            # 1. READ
            r_t, alpha_t = self.bank.read(h_t, state)
            # 2. FUSE
            z_t, g_t = self.bank.fuse(h_t, r_t)
            # 3. PREDICT
            next_token_logits = self.gpt2.lm_head(z_t)
            # 4. WRITE
            state, _ = self.bank.write(h_t, state, alpha_read=alpha_t, fusion_gate=g_t)

        for _ in range(max_new_tokens):
            logits = next_token_logits.clone()

            if repetition_penalty != 1.0:
                for b in range(generated.size(0)):
                    for prev_token in set(generated[b].tolist()):
                        if logits[b, prev_token] < 0:
                            logits[b, prev_token] *= repetition_penalty
                        else:
                            logits[b, prev_token] /= repetition_penalty

            if temperature > 0:
                logits = logits / max(temperature, 1e-5)
                if top_k > 0:
                    kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
                    logits = logits.masked_fill(logits < kth, -float("inf"))

                if 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits = logits.masked_fill(indices_to_remove, -float("inf"))

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)
            if stop_ids and next_token.item() in stop_ids:
                break

            step_out = self.gpt2.transformer(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = step_out.past_key_values

            h_t = step_out.last_hidden_state[:, -1, :]
            # 1. READ
            r_t, alpha_t = self.bank.read(h_t, state)
            # 2. FUSE
            z_t, g_t = self.bank.fuse(h_t, r_t)
            # 3. PREDICT
            next_token_logits = self.gpt2.lm_head(z_t)
            # 4. WRITE
            state, _ = self.bank.write(h_t, state, alpha_read=alpha_t, fusion_gate=g_t)

        self.bank.persist_runtime_state(state)
        return generated
