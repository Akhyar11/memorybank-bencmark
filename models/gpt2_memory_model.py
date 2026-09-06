from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from models.tiny_memory_bank import MemoryState, TinyMemoryBank, TinyMemoryConfig


class GPT2MemoryModel(nn.Module):
    """
    GPT-2 with Turn-Level Semantic Memory Bank.
    
    Architecture:
      1. Backbone GPT-2 is frozen.
      2. Trainable Parameters:
         - W_c (Query Projection): R^(768 -> 768)
         - W_f (Fusion Projection): R^(1536 -> 768)
      3. Turn Lifecycle:
         - Prompt arrival: 1 forward pass -> last hidden state (768) saved to memory (Memory 1).
         - Read: Top-K cosine similarity between C = h_prompt * W_c and Memory Bank.
         - Generation: token-by-token generation with fused states z = [h_t ; M_bar] * W_f -> LM Head.
         - Generation complete: final AI token hidden state (768) saved to memory (Memory 2).
         - Lifecycle Eviction: memories with reads < 5% of mean(reads) for age >= min_age are evicted.
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

        # Freeze backbone parameters
        if freeze_backbone:
            for p in self.gpt2.parameters():
                p.requires_grad = False

        if memory_config is None:
            memory_config = TinyMemoryConfig(memory_dim=embed_dim, hidden_size=embed_dim)
        else:
            memory_config.hidden_size = embed_dim
            memory_config.memory_dim = embed_dim

        self.memory_config = memory_config
        self.bank = TinyMemoryBank(memory_config)

        # 1. Trainable Query Projection: C = h * W_c
        self.c_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.eye_(self.c_proj.weight)
        self.c_proj.weight.requires_grad = True

        # 2. Trainable Fusion Projection: z = [h ; M_bar] * W_f
        self.fusion_proj = nn.Linear(embed_dim * 2, embed_dim, bias=False)
        with torch.no_grad():
            self.fusion_proj.weight[:, :embed_dim].copy_(torch.eye(embed_dim))
            nn.init.normal_(self.fusion_proj.weight[:, embed_dim:], mean=0.0, std=0.02)
        self.fusion_proj.weight.requires_grad = True

        self.last_diagnostics: Dict[str, Any] = {}

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
        Forward pass for training with Next Token Prediction (NTP) loss or feature extraction.
        """
        transformer_outputs = self.gpt2.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = transformer_outputs.last_hidden_state  # (B, T, D)
        bsz, seqlen, d = hidden.shape

        h_prompt = hidden[:, -1, :]  # Last token representation (B, D)

        if use_memory:
            # Query C and read from memory bank for every position in sequence
            C = self.c_proj(hidden)
            M_bar, top_indices = self.bank.read(C, top_k=self.memory_config.top_k)

            # Ingest to bank if persist_memory requested
            if persist_memory:
                for b in range(bsz):
                    self.bank.add_memory(h_prompt[b].detach())

            # Fuse across all token positions in the sequence
            fused_input = torch.cat([hidden, M_bar], dim=-1)   # (B, T, 2D)
            z = self.fusion_proj(fused_input)                  # (B, T, D)
        else:
            z = hidden

        logits = self.gpt2.lm_head(z)  # (B, T, V)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        self.last_diagnostics = {
            "num_memories": float(self.bank.num_memories),
            "mean_reads": float(sum(self.bank.read_counts) / max(1, len(self.bank.read_counts))),
        }

        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": z,
            "memory_state": self.bank.get_runtime_state(),
            "diagnostics": self.last_diagnostics,
        }

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
        use_memory: bool = True,
    ) -> torch.Tensor:
        """
        Turn-level Autoregressive Generation:
          1. 1x forward prompt through transformer -> extract h_prompt (768).
          2. Read Top-K memory using Query C = h_prompt * W_c.
          3. Save Memory 1 (h_prompt) to Memory Bank.
          4. Decode tokens: z_t = [h_t ; M_bar] * W_f -> LM Head.
          5. Save Memory 2 (final AI token hidden state 768) to Memory Bank.
          6. Step turn & execute lifecycle eviction.
        """
        del pad_token_id
        stop_ids = set(stop_token_ids or [])
        if eos_token_id is not None:
            stop_ids.add(eos_token_id)

        # 1. Forward prompt through transformer (1x forward)
        prompt_outputs = self.gpt2.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = prompt_outputs.past_key_values
        hidden = prompt_outputs.last_hidden_state
        h_prompt = hidden[:, -1, :]  # (B, 768)

        if use_memory:
            # Query C and Read from existing memory
            C = self.c_proj(h_prompt)
            M_bar, _ = self.bank.read(C, top_k=self.memory_config.top_k)

            # SIMPAN MEMORY 1: Prompt user langsung disimpan ke Memory Bank
            for b in range(input_ids.size(0)):
                self.bank.add_memory(h_prompt[b].detach())

            # Fuse prompt hidden state for first token prediction
            fused_prompt = torch.cat([h_prompt, M_bar], dim=-1)
            z_prompt = self.fusion_proj(fused_prompt)
            next_token_logits = self.gpt2.lm_head(z_prompt)
        else:
            next_token_logits = self.gpt2.lm_head(h_prompt)
            M_bar = None

        generated = input_ids.clone()
        last_ai_hidden = None

        # 2. Token-by-token decoding loop
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
            last_ai_hidden = h_t

            if use_memory:
                # Membaca memory di SETIAP forward pass decoding
                C_t = self.c_proj(h_t)
                M_bar_t, _ = self.bank.read(C_t, top_k=self.memory_config.top_k)
                fused_t = torch.cat([h_t, M_bar_t], dim=-1)
                z_t = self.fusion_proj(fused_t)
                next_token_logits = self.gpt2.lm_head(z_t)
            else:
                next_token_logits = self.gpt2.lm_head(h_t)

        # 3. SIMPAN MEMORY 2: Forward paling terakhir dari respon AI
        if use_memory and last_ai_hidden is not None:
            for b in range(input_ids.size(0)):
                self.bank.add_memory(last_ai_hidden[b].detach())

        # 4. Lifecycle turn step & eviction check
        if use_memory:
            self.bank.step_turn()
            self.bank.evict_lifecycle()

        return generated

    def reset_memory(self):
        """Clears all stored memories in the bank."""
        self.bank.reset_memory()

    def detach_memory_state(self, memory_state: Optional[MemoryState] = None) -> MemoryState:
        """Compatibility method for TBPTT boundaries."""
        if memory_state is not None:
            return memory_state.detach()
        return self.bank.get_runtime_state().detach()

    def print_trainable_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        ratio = (trainable_params / total_params) * 100

        print("=" * 55)
        print("   PARAMETER AUDIT: GPT-2 + TURN SEMANTIC MEMORY")
        print("=" * 55)
        print(f"Total Parameters     : {total_params:,}")
        print(f"Frozen Parameters    : {frozen_params:,}")
        print(f"Trainable Parameters : {trainable_params:,}")
        print(f"Trainable Ratio      : {ratio:.2f}%")
        print("=" * 55)
        return trainable_params, total_params
