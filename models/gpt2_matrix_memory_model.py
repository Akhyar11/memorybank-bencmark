"""
models/gpt2_matrix_memory_model.py
==================================
GPT-2 with Differentiable Memory Matrix Bank & LoRA Conditioning:
  - Frozen GPT-2 Backbone with LoRA Adaptation on Self-Attention (c_attn).
  - Trainable Query Encoder: W_q in R^(768 -> 768).
  - Non-Trainable Memory Matrix State: M in R^(128 x 768) (requires_grad = False).
  - Softmax Attention Memory Read: attn = Softmax((q @ W_q) @ M^T / scaling), m = attn @ M.
  - Trainable Virtual Memory Token Projection: v_mem = W_mem(m) prepended to sequence embeddings.
  - Deep Attention Conditioning: All 12 transformer layers attend to v_mem via LoRA-adapted c_attn.
  - Trainable Residual Fusion Layer: W_f in R^(1536 -> 768).
  - Frozen LM Head: logits = W_lm @ z.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from models.matrix_memory_bank import DifferentiableMemoryMatrix
from models.seed import set_seed


class LoRAConv1D(nn.Module):
    """
    Zero-dependency Native PyTorch LoRA wrapper for HuggingFace Conv1D layers (e.g. GPT-2 c_attn).
    Original Conv1D computes: x @ weight + bias, where weight is (in_features, out_features).
    LoRA computes: original_conv1d(x) + (dropout(x) @ lora_A @ lora_B) * (lora_alpha / r).
    """

    def __init__(
        self,
        original_conv1d: nn.Module,
        r: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.conv1d = original_conv1d
        self.conv1d.weight.requires_grad = False
        if hasattr(self.conv1d, "bias") and self.conv1d.bias is not None:
            self.conv1d.bias.requires_grad = False

        in_features = original_conv1d.weight.size(0)   # 768
        out_features = original_conv1d.weight.size(1)  # 2304
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0

        self.lora_A = nn.Parameter(torch.zeros(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        # Kaiming initialization for A, zero initialization for B
        # Guarantees that initial forward pass exactly matches original pretrained GPT-2!
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.conv1d(x)
        lora_out = (self.dropout(x) @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out


class GPT2MatrixMemoryModel(nn.Module):
    """
    GPT-2 with Differentiable Memory Matrix and LoRA Attention Conditioning.
    
    Architecture:
      1. Backbone GPT-2 base weights and LM Head are frozen.
      2. LoRA modules attached to c_attn in all 12 transformer layers.
      3. Non-trainable Memory Matrix M in R^(128 x 768) is a continuous state buffer.
      4. Trainable Parameters:
         - W_q (Query Encoder): R^(768 -> 768)
         - W_mem (Virtual Memory Token Projection): R^(768 -> 768)
         - W_f (Residual Fusion Projection): R^(1536 -> 768)
         - LoRA matrices (lora_A in R^(768 x r), lora_B in R^(r x 2304)) across 12 layers
      5. Forward Pass:
         - Query q retrieved from W_q(q_sem) or W_q(h_prompt).
         - Memory vector m = Softmax(q @ M^T / scaling) @ M.
         - Memory vector projected to virtual token v_mem = W_mem(m).
         - v_mem prepended to token embeddings: [v_mem ; token_embeds].
         - Transformer with LoRA processes sequence; all attention heads attend to v_mem.
         - Output tokens passed through residual fusion layer: z = h + GeLU(W_f [h ; m] + b_f).
         - Logits computed via frozen LM head: logits = W_lm @ z.
    """

    def __init__(
        self,
        model_name_or_path: str,
        capacity: int = 128,
        scaling: Union[bool, str] = "none",
        freeze_backbone: bool = True,
        semantic_extractor: Optional[Any] = None,
        use_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.0,
        seed: Optional[int] = 42,
    ):
        super().__init__()
        if seed is not None:
            set_seed(seed)

        self.gpt2 = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.config = self.gpt2.config
        embed_dim = self.config.n_embd
        self.semantic_extractor = semantic_extractor

        # 1. Freeze backbone parameters
        if freeze_backbone:
            for p in self.gpt2.parameters():
                p.requires_grad = False

        # 2. Attach LoRA to GPT-2 Attention blocks (c_attn)
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        if use_lora and lora_rank > 0:
            for block in self.gpt2.transformer.h:
                block.attn.c_attn = LoRAConv1D(
                    block.attn.c_attn,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                )

        # 3. Non-trainable Memory Matrix (128 slots x 768-D)
        self.matrix_bank = DifferentiableMemoryMatrix(
            capacity=capacity,
            memory_dim=embed_dim,
            scaling=scaling,
        )

        # 4. Trainable Query Encoder: q = W_q(h)
        self.query_encoder = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.eye_(self.query_encoder.weight)
        self.query_encoder.weight.requires_grad = True

        # 5. Trainable Virtual Memory Token Projection: v_mem = W_mem(m)
        self.memory_token_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        nn.init.eye_(self.memory_token_proj.weight)
        nn.init.zeros_(self.memory_token_proj.bias)
        self.memory_token_proj.weight.requires_grad = True
        self.memory_token_proj.bias.requires_grad = True

        # 6. Trainable Cross-GeLU Fusion Layer: Delta = GeLU(W_f [h ; m] + b_f), z = h + Delta
        self.fusion_proj = nn.Linear(embed_dim * 2, embed_dim, bias=True)
        self.fusion_act = nn.GELU()
        with torch.no_grad():
            nn.init.normal_(self.fusion_proj.weight, mean=0.0, std=0.01)
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
        """Extracts only trainable adapter & LoRA parameters. Size: ~11-12 MB."""
        target_prefixes = ["query_encoder", "fusion_proj", "memory_token_proj", "lora_A", "lora_B"]
        return {
            k: v.cpu().clone()
            for k, v in self.state_dict().items()
            if any(pfx in k for pfx in target_prefixes)
        }

    def load_adapter(self, checkpoint_path_or_dict: Union[str, Dict[str, Any]]):
        """Loads lightweight adapter weights (~7-12 MB) onto the backbone."""
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
        print(f"✓ MemoryBank + LoRA Adapter loaded ({len(sd)} tensors): {msg}")

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
        query_text: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass with continuous linear memory read & virtual memory token conditioning.
        """
        bsz, seqlen = input_ids.shape
        device = input_ids.device
        s_activations = None

        if use_memory:
            # 1. Read memory m via semantic query or prompt hidden state
            if query_text is not None and self.semantic_extractor is not None:
                q_sem = self.semantic_extractor.encode(query_text, normalize=True).to(device)
                q = self.query_encoder(q_sem)
                m_prompt, s_activations = self.matrix_bank.read(q)
            elif prompt_len is not None and 0 < prompt_len < seqlen:
                with torch.no_grad():
                    prompt_out = self.gpt2.transformer(
                        input_ids=input_ids[:, :prompt_len],
                        use_cache=False,
                        return_dict=True,
                    )
                    h_prompt = prompt_out.last_hidden_state[:, -1, :]
                q_prompt = self.query_encoder(h_prompt)
                m_prompt, s_activations = self.matrix_bank.read(q_prompt)
            else:
                with torch.no_grad():
                    h_all = self.gpt2.transformer.wte(input_ids).mean(dim=1)
                q = self.query_encoder(h_all)
                m_prompt, s_activations = self.matrix_bank.read(q)

            # 2. Virtual Memory Token Embedding: v_mem in R^(B x 1 x D)
            v_mem = self.memory_token_proj(m_prompt).unsqueeze(1)  # (B, 1, D)

            # 3. Prepend Virtual Memory Token to Token Embeddings
            token_embeds = self.gpt2.transformer.wte(input_ids)     # (B, T, D)
            inputs_embeds = torch.cat([v_mem, token_embeds], dim=1) # (B, T + 1, D)

            if attention_mask is not None:
                mem_mask = torch.ones((bsz, 1), device=device, dtype=attention_mask.dtype)
                comb_mask = torch.cat([mem_mask, attention_mask], dim=1)
            else:
                comb_mask = None

            # 4. Forward through GPT-2 with LoRA attention
            transformer_outputs = self.gpt2.transformer(
                inputs_embeds=inputs_embeds,
                attention_mask=comb_mask,
                use_cache=False,
                return_dict=True,
            )
            # Hidden states corresponding to input_ids (drop the memory token at index 0)
            hidden = transformer_outputs.last_hidden_state[:, 1:, :]  # (B, T, D)

            # 5. Residual Fusion Layer at LM Head
            m = m_prompt.unsqueeze(1).expand(bsz, seqlen, hidden.size(-1))
            fused_input = torch.cat([hidden, m], dim=-1)  # (B, T, 2D)
            delta = self.fusion_act(self.fusion_proj(fused_input))  # (B, T, D)
            z = hidden + delta
        else:
            transformer_outputs = self.gpt2.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            z = transformer_outputs.last_hidden_state

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
        write_after_gen: bool = True,
    ) -> torch.Tensor:
        """
        Turn-level Autoregressive Generation with Deep Memory Token & LoRA Conditioning:
          1. Retrieve memory vector m_turn from Memory Matrix Bank.
          2. Project m_turn to virtual memory token v_mem = W_mem(m_turn).
          3. Prepend v_mem to input_ids embeddings and forward prompt into KV-cache.
          4. Decode tokens conditioned on [h_t ; m_turn] and past KV-cache attending to v_mem.
        """
        del pad_token_id
        stop_ids = set(stop_token_ids or [])
        if eos_token_id is not None:
            stop_ids.add(eos_token_id)

        bsz = input_ids.size(0)
        device = input_ids.device

        if use_memory:
            # 1. Read memory vector m
            if query_text is not None and self.semantic_extractor is not None:
                q_sem = self.semantic_extractor.encode(query_text, normalize=True).to(device)
                q = self.query_encoder(q_sem)
            else:
                with torch.no_grad():
                    temp_out = self.gpt2.transformer(input_ids=input_ids, use_cache=False)
                    h_last = temp_out.last_hidden_state[:, -1, :]
                q = self.query_encoder(h_last)

            m_turn, _ = self.matrix_bank.read(q)  # (B, D)

            # 2. Virtual Memory Token Embedding
            v_mem = self.memory_token_proj(m_turn).unsqueeze(1)  # (B, 1, D)
            token_embeds = self.gpt2.transformer.wte(input_ids)
            inputs_embeds = torch.cat([v_mem, token_embeds], dim=1)  # (B, T + 1, D)

            if attention_mask is not None:
                mem_mask = torch.ones((bsz, 1), device=device, dtype=attention_mask.dtype)
                comb_mask = torch.cat([mem_mask, attention_mask], dim=1)
            else:
                comb_mask = None

            prompt_outputs = self.gpt2.transformer(
                inputs_embeds=inputs_embeds,
                attention_mask=comb_mask,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = prompt_outputs.past_key_values
            hidden = prompt_outputs.last_hidden_state
            h_prompt = hidden[:, -1, :]  # (B, D)

            fused_prompt = torch.cat([h_prompt, m_turn], dim=-1)
            delta_prompt = self.fusion_act(self.fusion_proj(fused_prompt))
            z_prompt = h_prompt + delta_prompt
            next_token_logits = self.gpt2.lm_head(z_prompt)
        else:
            prompt_outputs = self.gpt2.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = prompt_outputs.past_key_values
            hidden = prompt_outputs.last_hidden_state
            h_prompt = hidden[:, -1, :]
            next_token_logits = self.gpt2.lm_head(h_prompt)
            m_turn = None

        generated = input_ids.clone()
        last_ai_hidden = None

        # 3. Token-by-token decoding loop
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

            if use_memory and m_turn is not None:
                fused_t = torch.cat([h_t, m_turn], dim=-1)
                delta_t = self.fusion_act(self.fusion_proj(fused_t))
                z_t = h_t + delta_t
                next_token_logits = self.gpt2.lm_head(z_t)
            else:
                next_token_logits = self.gpt2.lm_head(h_t)

        # 4. WRITE TO MEMORY BANK: After turn generation completes (optional)
        if use_memory and write_after_gen:
            for b in range(input_ids.size(0)):
                self.matrix_bank.write(h_prompt[b])
            if last_ai_hidden is not None:
                for b in range(input_ids.size(0)):
                    self.matrix_bank.write(last_ai_hidden[b])

        return generated
