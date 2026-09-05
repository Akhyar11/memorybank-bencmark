"""
GPT2MemoryModel – Integration of Frozen Pretrained GPT-2 with Mature Locked TinyMemoryBank
========================================================================================
Repo: MemoryBank-bencmark/models/gpt2_memory_model.py
Menghubungkan Pretrained GPT-2 (Frozen Backbone) dengan modul TinyMemoryBank yang matang dan teraudit.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.tiny_memory_bank import TinyMemoryBank, TinyMemoryConfig, STATE_ACTIVE, STATE_EXPIRED


class GPT2MemoryModel(nn.Module):
    """
    Model GPT-2 yang diaugmentasi dengan mature TinyMemoryBank.
    - GPT-2 Backbone: 100% frozen (tidak diupdate saat training).
    - TinyMemoryBank + Write/Read Heads: Trainable (hanya ~0.99% total parameter).
    """
    def __init__(
        self,
        model_name_or_path: str,
        memory_config: Optional[TinyMemoryConfig] = None,
        freeze_backbone: bool = True
    ):
        super().__init__()
        
        # 1. Muat Pretrained GPT-2
        self.gpt2 = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.config = self.gpt2.config
        embed_dim = self.config.n_embd  # 768 untuk gpt2-small

        # 2. Bekukan parameter GPT-2
        if freeze_backbone:
            for param in self.gpt2.parameters():
                param.requires_grad = False

        # 3. Konfigurasi Memory Bank
        if memory_config is None:
            memory_config = TinyMemoryConfig(
                memory_capacity=128,
                memory_dim=embed_dim,
                hidden_size=embed_dim
            )
        else:
            memory_config.hidden_size = embed_dim
            if memory_config.memory_dim is None:
                memory_config.memory_dim = embed_dim

        self.memory_config = memory_config

        # 4. Inisialisasi Locked Mature TinyMemoryBank
        self.bank = TinyMemoryBank(memory_config)

        # Proyeksi jika memory_dim != embed_dim
        if memory_config.memory_dim != embed_dim:
            self.memory_proj_in = nn.Linear(embed_dim, memory_config.memory_dim, bias=False)
            self.memory_proj_out = nn.Linear(memory_config.memory_dim, embed_dim, bias=False)
        else:
            self.memory_proj_in = nn.Identity()
            self.memory_proj_out = nn.Identity()

        # 5. Host-level Write & Read Heads (Trainable)
        self.write_head = nn.Linear(embed_dim, 1)
        self.read_head  = nn.Linear(embed_dim, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        write_targets: Optional[torch.Tensor] = None,
        read_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass melalui Backbone GPT-2 (frozen) + TinyMemoryBank (trainable).
        """
        with torch.no_grad():
            transformer_outputs = self.gpt2.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            hidden_states = transformer_outputs[0]  # (B, T, embed_dim)

        # Prediksi gate (write & read)
        write_logits = self.write_head(hidden_states).squeeze(-1)  # (B, T)
        write_probs = torch.sigmoid(write_logits)

        read_logits = self.read_head(hidden_states).squeeze(-1)    # (B, T)
        read_probs = torch.sigmoid(read_logits)

        B, T, H = hidden_states.size()
        h_flat = hidden_states.reshape(B * T, H)
        r_flat = read_probs.reshape(B * T)

        # Read dari TinyMemoryBank (100% transmisi sinyal memori tanpa diredam gate)
        query_rep = self.memory_proj_in(h_flat)
        m_read_flat = self.bank.read(query_rep)
        m_read = self.memory_proj_out(m_read_flat).view(B, T, -1)

        # Fuse hidden state dengan m_read menggunakan fusion_proj milik TinyMemoryBank
        fused_hidden = self.bank.fuse(hidden_states, m_read)

        # Logits prediksi kata berikutnya
        logits = self.gpt2.lm_head(fused_hidden)

        # Kalkulasi Loss
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
            loss = lm_loss

            if write_targets is not None:
                mask = (write_targets != -100)
                if mask.sum() > 0:
                    gate_loss = F.binary_cross_entropy(
                        write_probs[mask], write_targets[mask].float()
                    )
                    loss = loss + 0.5 * gate_loss

            if read_targets is not None:
                mask = (read_targets != -100)
                if mask.sum() > 0:
                    gate_loss = F.binary_cross_entropy(
                        read_probs[mask], read_targets[mask].float()
                    )
                    loss = loss + 0.5 * gate_loss

        return {
            "loss": loss,
            "logits": logits,
            "write_probs": write_probs,
            "read_probs": read_probs,
            "hidden_states": fused_hidden
        }

    def print_trainable_parameters(self):
        """Menampilkan perbandingan parameter beku vs dilatih."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        ratio = (trainable_params / total_params) * 100

        print("="*55)
        print("   PARAMETER AUDIT: GPT-2 + MATURE MEMORY BANK")
        print("="*55)
        print(f"Total Parameters     : {total_params:,}")
        print(f"Frozen Parameters    : {frozen_params:,} (GPT-2 Indo Backbone)")
        print(f"Trainable Parameters : {trainable_params:,} (TinyMemoryBank & Gates)")
        print(f"Trainable Ratio      : {ratio:.2f}%")
        print("="*55)
        return trainable_params, total_params

    def reset_memory(self):
        """Mereset seluruh memori TinyMemoryBank."""
        self.bank.reset_memory()

    def write_memory(self, h_t: torch.Tensor, write_prob: torch.Tensor):
        """
        Menulis representasi hidden state ke dalam TinyMemoryBank.
        """
        if h_t.dim() == 1:
            h_t = h_t.unsqueeze(0)
        if write_prob.dim() == 0:
            write_prob = write_prob.unsqueeze(0)
        is_eos = torch.ones(h_t.size(0), dtype=torch.bool, device=h_t.device)
        return self.bank.write(self.memory_proj_in(h_t), is_eos=is_eos, write_prob=write_prob)

    def pool_n_tokens(
        self,
        token_hiddens: torch.Tensor,
        write_probs: torch.Tensor,
        write_logits: Optional[torch.Tensor] = None,
        chunk_size: int = 8,
        temperature: float = 1.0,
    ):
        """
        Mengelompokkan sequence token menjadi segmen berukuran N (default chunk_size=8)
        dan melakukan Attention-Weighted Soft Pooling (Opsi 1) secara sepenuhnya terdiferensiasi.
        """
        if token_hiddens.dim() == 1:
            return token_hiddens.unsqueeze(0), write_probs.unsqueeze(0) if write_probs.dim() == 0 else write_probs

        L = token_hiddens.size(0)
        if L == 0:
            return token_hiddens, write_probs

        scores = write_logits if write_logits is not None else write_probs

        chunk_h_list = []
        chunk_w_list = []

        for i in range(0, L, chunk_size):
            h_c = token_hiddens[i : i + chunk_size]
            s_c = scores[i : i + chunk_size]
            w_c = write_probs[i : i + chunk_size]

            # Softmax Attention over the N tokens in the chunk
            attn = F.softmax(s_c / temperature, dim=-1)
            
            # Convex combination of token vectors (768-dim)
            v_pooled = torch.sum(attn.unsqueeze(-1) * h_c, dim=0)
            
            # Weighted write probability for the chunk
            w_pooled = torch.sum(attn * w_c)
            
            chunk_h_list.append(v_pooled)
            chunk_w_list.append(w_pooled)

        chunk_hiddens = torch.stack(chunk_h_list, dim=0)
        chunk_w_probs = torch.stack(chunk_w_list, dim=0)
        return chunk_hiddens, chunk_w_probs

    def write_memory_chunked(
        self,
        token_hiddens: torch.Tensor,
        write_probs: torch.Tensor,
        write_logits: Optional[torch.Tensor] = None,
        chunk_size: int = 8,
        temperature: float = 1.0,
    ):
        """
        Menulis sequence token ke dalam slot memori per segmen N-Token (Opsi 1).
        """
        pooled_h, pooled_w = self.pool_n_tokens(
            token_hiddens=token_hiddens,
            write_probs=write_probs,
            write_logits=write_logits,
            chunk_size=chunk_size,
            temperature=temperature
        )
        return self.write_memory(pooled_h, pooled_w)

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
        Autoregressive generation menggunakan TinyMemoryBank Fused Representations.
        Memastikan setiap token baru dihasilkan dari output fusi memory adapter, bukan raw backbone.
        """
        curr_ids = input_ids.clone()
        stop_ids = set()
        if eos_token_id is not None:
            stop_ids.add(eos_token_id)
        if stop_token_ids is not None:
            for s in stop_token_ids:
                stop_ids.add(s)

        for _ in range(max_new_tokens):
            curr_mask = torch.ones_like(curr_ids) if attention_mask is not None else None
            out = self.forward(input_ids=curr_ids, attention_mask=curr_mask)
            next_token_logits = out["logits"][:, -1, :].clone()

            # Repetition penalty
            if repetition_penalty != 1.0:
                for b in range(curr_ids.size(0)):
                    for prev_token in set(curr_ids[b].tolist()):
                        if next_token_logits[b, prev_token] < 0:
                            next_token_logits[b, prev_token] *= repetition_penalty
                        else:
                            next_token_logits[b, prev_token] /= repetition_penalty

            # Sampling / Greedy
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float("Inf")

                if 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float("Inf")

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            curr_ids = torch.cat([curr_ids, next_token], dim=-1)

            if len(stop_ids) > 0 and next_token.item() in stop_ids:
                break

        return curr_ids
