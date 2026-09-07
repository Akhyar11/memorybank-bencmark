"""
models/gpt2_matrix_memory_model.py
==================================
GPT-2 with Differentiable Memory Matrix Bank:
  - Frozen GPT-2 Backbone (100% frozen).
  - Trainable Query Encoder: W_q in R^(768 -> 768).
  - Non-Trainable Memory Matrix State: M in R^(128 x 768) (requires_grad = False).
  - Continuous Memory Read: s = q @ M^T, m = (1 / sqrt(d)) * s @ M.
  - Trainable Fusion Layer: W_f in R^(1536 -> 768).
  - Frozen LM Head: logits = W_lm @ z.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from models.matrix_memory_bank import DifferentiableMemoryMatrix


class GPT2MatrixMemoryModel(nn.Module):
    """
    GPT-2 with Differentiable Memory Matrix.
    
    Architecture:
      1. Backbone GPT-2 and LM Head are frozen.
      2. Memory Matrix M in R^(128 x 768) is a non-trainable state buffer.
      3. Trainable Parameters:
         - W_q (Query Encoder): R^(768 -> 768)
         - W_f (Fusion Projection): R^(1536 -> 768)
      4. Fully continuous, linear differentiable read:
         q = W_q(h)
         s = q @ M^T          (128 activations across slots)
         m = (1/sqrt(d)) * s @ M  (768 reconstructed memory)
         z = W_f [h ; m] + b_f
         logits = W_lm @ z
    """

    def __init__(
        self,
        model_name_or_path: str,
        capacity: int = 128,
        scaling: bool = True,
        freeze_backbone: bool = True,
        semantic_extractor: Optional[Any] = None,
    ):
        super().__init__()

        self.gpt2 = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.config = self.gpt2.config
        embed_dim = self.config.n_embd
        self.semantic_extractor = semantic_extractor

        # 1. Freeze backbone parameters
        if freeze_backbone:
            for p in self.gpt2.parameters():
                p.requires_grad = False

        # 2. Non-trainable Memory Matrix (128 slots x 768-D)
        self.matrix_bank = DifferentiableMemoryMatrix(
            capacity=capacity,
            memory_dim=embed_dim,
            scaling=scaling,
        )

        # 3. Trainable Query Encoder: q = W_q(h)
        self.query_encoder = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.eye_(self.query_encoder.weight)
        self.query_encoder.weight.requires_grad = True

        # 4. Trainable Fusion Layer: z = W_f [h ; m] + b_f
        self.fusion_proj = nn.Linear(embed_dim * 2, embed_dim, bias=True)
        with torch.no_grad():
            self.fusion_proj.weight[:, :embed_dim].copy_(torch.eye(embed_dim))
            nn.init.normal_(self.fusion_proj.weight[:, embed_dim:], mean=0.0, std=0.02)
            self.fusion_proj.bias.zero_()
        self.fusion_proj.weight.requires_grad = True
        self.fusion_proj.bias.requires_grad = True

        self.last_diagnostics: Dict[str, Any] = {}

    def set_semantic_extractor(self, extractor: Any):
        """Attaches or updates the semantic extractor module."""
        self.semantic_extractor = extractor

    def write_semantic_text(self, text: str):
        """Encodes text using semantic extractor and writes 768-D representation to matrix bank."""
        if self.semantic_extractor is None:
            raise ValueError("No semantic_extractor configured in model!")
        vec = self.semantic_extractor.encode(text)
        self.matrix_bank.write(vec)

    def print_trainable_parameters(self) -> Tuple[int, int]:
        """Prints and returns (trainable_params, total_params)."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(
            f"Trainable params: {trainable:,} / {total:,} "
            f"({100.0 * trainable / max(1, total):.4f}% trainable)"
        )
        return trainable, total

    def get_adapter_state_dict(self) -> Dict[str, torch.Tensor]:
        """Extracts only the trainable adapter parameters (query_encoder & fusion_proj). Size: ~7 MB."""
        return {
            k: v.cpu().clone()
            for k, v in self.state_dict().items()
            if any(k.startswith(pfx) for pfx in ["query_encoder", "fusion_proj"])
        }

    def load_adapter(self, checkpoint_path_or_dict: Union[str, Dict[str, Any]]):
        """Loads lightweight adapter weights (~7 MB) onto the frozen backbone."""
        if isinstance(checkpoint_path_or_dict, str):
            st = torch.load(checkpoint_path_or_dict, map_location="cpu", weights_only=False)
        else:
            st = checkpoint_path_or_dict

        if "adapter_state_dict" in st:
            sd = st["adapter_state_dict"]
        elif "model_state_dict" in st:
            sd = st["model_state_dict"]
        else:
            sd = st

        msg = self.load_state_dict(sd, strict=False)
        print(f"✓ MemoryBank Adapter loaded ({len(sd)} tensors): {msg}")

    def reset_memory(self):
        """Clears memory matrix state."""
        self.matrix_bank.reset_memory()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_memory: bool = True,
        prompt_len: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass with continuous linear memory read.
        
        Args:
            input_ids: Tensor of shape (B, T).
            attention_mask: Optional attention mask of shape (B, T).
            labels: Optional labels of shape (B, T) for NTP loss.
            use_memory: Whether to fuse differentiable memory.
            prompt_len: Boundary index where user prompt ends and assistant response starts.
        """
        transformer_outputs = self.gpt2.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = transformer_outputs.last_hidden_state  # (B, T, D)
        bsz, seqlen, d = hidden.shape

        s_activations = None
        if use_memory:
            if prompt_len is not None and 0 < prompt_len < seqlen:
                # Turn-level prompt-conditioned query:
                h_prompt = hidden[:, prompt_len - 1, :]  # (B, D)
                q_prompt = self.query_encoder(h_prompt)   # (B, D)
                m_prompt, s_activations = self.matrix_bank.read(q_prompt)  # (B, D), (B, 128)
                m = m_prompt.unsqueeze(1).expand(bsz, seqlen, d)
            else:
                # Full sequence query
                q = self.query_encoder(hidden)  # (B, T, D)
                m, s_activations = self.matrix_bank.read(q)  # (B, T, D), (B, T, 128)

            fused_input = torch.cat([hidden, m], dim=-1)  # (B, T, 2D)
            z = self.fusion_proj(fused_input)             # (B, T, D)
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
            "num_memories": float(self.matrix_bank.num_memories),
            "activations": s_activations.detach() if s_activations is not None else None,
        }

        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": z,
            "activations": s_activations,
            "diagnostics": self.last_diagnostics,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        query_text: Optional[str] = None,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        stop_token_ids: Optional[list] = None,
        use_memory: bool = True,
        write_after_gen: bool = False,
    ) -> torch.Tensor:
        """
        Turn-level Autoregressive Generation with Differentiable Memory Matrix:
          1. Forward prompt -> extract h_prompt.
          2. Form query q (semantic query via query_text or projection of h_prompt).
          3. Read continuous memory m_turn = (1/sqrt(d)) * (q @ M^T) @ M.
          4. Decode tokens conditioned on [h_t ; m_turn].
        """
        del pad_token_id
        stop_ids = set(stop_token_ids or [])
        if eos_token_id is not None:
            stop_ids.add(eos_token_id)

        # 1. Forward prompt through transformer
        prompt_outputs = self.gpt2.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = prompt_outputs.past_key_values
        hidden = prompt_outputs.last_hidden_state
        h_prompt = hidden[:, -1, :]  # (B, D)

        if use_memory:
            # Check if semantic query from raw text is available
            if query_text is not None and self.semantic_extractor is not None:
                q_sem = self.semantic_extractor.encode(query_text).to(self.gpt2.device)
                q = self.query_encoder(q_sem)
            else:
                # Query encoder forms query q from last prompt token
                q = self.query_encoder(h_prompt)

            # Continuous memory read (no softmax, no top-k)
            m_turn, _ = self.matrix_bank.read(q)

            fused_prompt = torch.cat([h_prompt, m_turn], dim=-1)
            z_prompt = self.fusion_proj(fused_prompt)
            next_token_logits = self.gpt2.lm_head(z_prompt)
        else:
            next_token_logits = self.gpt2.lm_head(h_prompt)
            m_turn = None

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
                # Fused with turn-level memory representation m_turn
                fused_t = torch.cat([h_t, m_turn], dim=-1)
                z_t = self.fusion_proj(fused_t)
                next_token_logits = self.gpt2.lm_head(z_t)
            else:
                next_token_logits = self.gpt2.lm_head(h_t)

        # 3. WRITE TO MEMORY BANK: After turn generation completes (optional)
        if use_memory and write_after_gen:
            for b in range(input_ids.size(0)):
                self.matrix_bank.write(h_prompt[b])
            if last_ai_hidden is not None:
                for b in range(input_ids.size(0)):
                    self.matrix_bank.write(last_ai_hidden[b])

        return generated
