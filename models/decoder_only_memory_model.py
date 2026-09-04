"""
models/decoder_only_memory_model.py – Pure Decoder-Only Autoregressive LM with Locked Memory Bank.

Architecture:
- Token Embedding + Sinusoidal Positional Encoding (max_len=2048)
- Stack of CausalDecoderBlock (Self-Attention with upper-triangular causal mask, FFN, LayerNorm)
- NO Encoders, NO cross-attention, NO separate answer decoder
- Host Decoder Write Head: write_head = nn.Linear(embed_dim, 1) -> write_prob = sigmoid(write_head(h))
- Locked Memory Bank: TinyMemoryBank(config) with locked q_proj, k_proj, v_proj, i_proj, fusion_proj
- LM Head: maps decoder hidden states to vocabulary logits P(x_t | x_{<t})
- Supports three clean, fair baselines:
    1. 'bank' : Full episodic Memory Bank (write, read, decay, fuse)
    2. 'none' : No Memory baseline (fair causal LM, memory contribution = 0)
    3. 'nn'   : Independent Key-Value NN Memory (pure similarity, no MB metadata/decay/reinforcement)
"""
import math
from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tiny_memory_bank import TinyMemoryBank, TinyMemoryConfig


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 2048):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        
        pe_sin = torch.sin(position * div_term)
        pe_cos = torch.cos(position * div_term)
        
        pe = torch.stack([pe_sin, pe_cos], dim=-1).view(max_len, dim)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class CausalDecoderBlock(nn.Module):
    """
    Standard Transformer Causal Decoder Block:
    - Multi-Head Self-Attention with Causal Mask (strictly prevents future token leakage)
    - Pre/Post LayerNorm
    - Feed-Forward Network with GELU activation
    - Residual Connections
    """
    def __init__(self, embed_dim: int = 32, num_heads: int = 2, ff_dim: int = 216, dropout_rate: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, causal_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN Self-Attention with causal mask
        norm_x = self.ln1(x)
        attn_out, _ = self.self_attn(
            norm_x, norm_x, norm_x,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask
        )
        x = x + self.dropout1(attn_out)
        
        # Pre-LN FFN
        x = x + self.dropout2(self.ffn(self.ln2(x)))
        return x


class DecoderOnlyMemoryLM(nn.Module):
    """
    Pure Decoder-Only Language Model with Locked Memory Bank integration.
    """
    def __init__(
        self,
        config: TinyMemoryConfig,
        vocab_size: int = 2002,
        embed_dim: int = 32,
        num_layers: int = 1,
        num_heads: int = 2,
        ff_dim: int = 216,
        dropout_rate: float = 0.1,
        pad_id: int = 0,
        bos_id: int = 2,
        eos_id: int = 3,
        max_seq_len: int = 2048
    ):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.pos_encoding = SinusoidalPositionalEncoding(dim=embed_dim, max_len=max_seq_len)
        self.dropout = nn.Dropout(dropout_rate)

        # Causal Decoder Stack (No Encoders, No Cross-Attention)
        self.blocks = nn.ModuleList([
            CausalDecoderBlock(embed_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(embed_dim)

        # Host-level Write Head (does NOT alter locked Memory Bank i_proj)
        self.write_head = nn.Linear(embed_dim, 1)

        # Memory Projections (adapt between embed_dim and MemoryBank hidden_size if needed)
        self.memory_proj_in = nn.Linear(embed_dim, config.hidden_size) if embed_dim != config.hidden_size else nn.Identity()
        self.memory_proj_out = nn.Linear(config.hidden_size, embed_dim) if embed_dim != config.hidden_size else nn.Identity()

        # The LOCKED Memory Bank
        self.bank = TinyMemoryBank(config)

        # Independent Projections for True Key-Value NN Baseline
        self.nn_k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.nn_v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.nn_proj_out = nn.Linear(embed_dim, embed_dim)

        # LM Head: projects to vocabulary logits P(x_t | x_{<t})
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Standard upper-triangular boolean causal mask (True where masked)."""
        return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)

    def decode_step(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Runs purely the causal transformer decoder backbone over input_ids.
        Returns: hidden states h of shape (batch_size, seq_len, embed_dim)
        """
        seq_len = input_ids.size(1)
        device = input_ids.device

        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        causal_mask = self._generate_causal_mask(seq_len, device)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None

        for block in self.blocks:
            x = block(x, causal_mask=causal_mask, key_padding_mask=key_padding_mask)

        h = self.final_ln(x)
        return h

    def compute_write_prob(self, h: torch.Tensor) -> torch.Tensor:
        """Computes learned write decision probability from host decoder hidden states."""
        return torch.sigmoid(self.write_head(h).squeeze(-1))

    def write_representation(
        self,
        h_rep: torch.Tensor,
        write_prob: Optional[torch.Tensor] = None,
        memory_mode: str = 'bank'
    ) -> Tuple[Any, Optional[torch.Tensor]]:
        """
        Writes representation into episodic memory.
        h_rep: (batch_size, embed_dim)
        """
        b_size = h_rep.size(0)
        device = h_rep.device

        if write_prob is None:
            write_prob = self.compute_write_prob(h_rep)

        # is_eos is causally inert in Memory Bank, passed as ones for compatibility
        is_eos_inert = torch.ones(b_size, device=device)

        if memory_mode == 'bank':
            h_proj = self.memory_proj_in(h_rep)
            written_indices = self.bank.write(h_proj, is_eos_inert, write_prob)
            return h_proj, written_indices
        elif memory_mode == 'nn':
            k_proj = self.nn_k_proj(h_rep)
            v_proj = self.nn_v_proj(h_rep)
            return (k_proj, v_proj), None
        else:
            return h_rep, None

    def read_and_fuse(
        self,
        h_rep: torch.Tensor,
        read_prob: Optional[torch.Tensor] = None,
        write_prob: Optional[torch.Tensor] = None,
        memory_mode: str = 'bank',
        fact_store: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_scores: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Reads from episodic memory and fuses with current hidden state.
        h_rep: (batch_size, embed_dim)
        """
        b_size = h_rep.size(0)
        device = h_rep.device

        if read_prob is None:
            read_prob = torch.ones(b_size, device=device)
        if write_prob is None:
            write_prob = torch.zeros(b_size, device=device)

        if memory_mode == 'none':
            # Fair baseline: zero memory contribution, h_rep untouched
            return h_rep, None

        elif memory_mode == 'bank':
            h_proj = self.memory_proj_in(h_rep)
            if return_scores:
                fused_proj, scores = self.bank(
                    h_proj, read_prob, write_prob,
                    deterministic=True, return_scores=True
                )
            else:
                fused_proj = self.bank(
                    h_proj, read_prob, write_prob,
                    deterministic=True, return_scores=False
                )
                scores = None
            fused_h = self.memory_proj_out(fused_proj)
            return fused_h, scores

        elif memory_mode == 'nn':
            # Independent NN Memory baseline (pure Key-Value cosine similarity)
            if fact_store is not None:
                nn_keys, nn_vals = fact_store
                k_q = self.nn_k_proj(h_rep)
                k_q_norm = F.normalize(k_q, dim=-1)
                nn_keys_norm = F.normalize(nn_keys, dim=-1)

                scores = torch.matmul(k_q_norm, nn_keys_norm.T)  # (batch, capacity)
                weights = F.softmax(scores, dim=-1)
                m = torch.matmul(weights, nn_vals)  # (batch, embed_dim)
                m_eff = m * read_prob.unsqueeze(-1)
                fused_h = h_rep + self.nn_proj_out(m_eff)
                return fused_h, scores
            return h_rep, None

        return h_rep, None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_ids: Optional[torch.Tensor] = None,
        memory_mode: str = 'bank',
        read_prob: Optional[torch.Tensor] = None,
        write_prob: Optional[torch.Tensor] = None,
        fact_store: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_scores: bool = False,
        query_token_idx: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Forward pass for Pure Decoder-Only Next Token Prediction.
        """
        b_size, seq_len = input_ids.shape
        device = input_ids.device

        # 1. Causal Decoder Pass over entire input sequence
        h = self.decode_step(input_ids, attention_mask=attention_mask)

        scores = None
        # 2. Memory Bank integration at query position if provided, else on final hidden state
        if memory_mode != 'none':
            if query_token_idx is not None:
                # Target memory retrieval on specific query positions (e.g. before answer generation)
                batch_indices = torch.arange(b_size, device=device)
                h_query = h[batch_indices, query_token_idx]
                fused_h_query, scores = self.read_and_fuse(
                    h_query, read_prob=read_prob, write_prob=write_prob,
                    memory_mode=memory_mode, fact_store=fact_store,
                    return_scores=return_scores
                )
                h = h.clone()
                h[batch_indices, query_token_idx] = fused_h_query
            else:
                # Default: apply memory read & fuse to the latest hidden state
                h_last = h[:, -1, :]
                fused_last, scores = self.read_and_fuse(
                    h_last, read_prob=read_prob, write_prob=write_prob,
                    memory_mode=memory_mode, fact_store=fact_store,
                    return_scores=return_scores
                )
                h = h.clone()
                h[:, -1, :] = fused_last

        # 3. LM Head -> Logits over Vocabulary
        logits = self.lm_head(h)

        # 4. NTP Loss Computation if target_ids is supplied
        loss = None
        if target_ids is not None:
            # Shift target tokens for Next Token Prediction: P(x_t | x_{<t})
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                target_ids.view(-1),
                ignore_index=self.pad_id
            )

        return {
            "logits": logits,
            "hidden_states": h,
            "loss": loss,
            "scores": scores
        }
