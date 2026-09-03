"""
TransformerQAModel – Fixed end-to-end QA model with locked Memory Bank (PyTorch Version).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.tiny_memory_bank import TinyMemoryBank, TinyMemoryConfig


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        
        pe_sin = torch.sin(position * div_term)
        pe_cos = torch.cos(position * div_term)
        
        pe = torch.stack([pe_sin, pe_cos], dim=-1).view(max_len, dim)
        self.register_buffer('pe', pe.unsqueeze(0)) # (1, max_len, dim)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, ff_dim: int = 256, dropout_rate: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask)
        x = self.ln1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        x = self.ln2(x + self.dropout2(ffn_out))
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, ff_dim: int = 256, dropout_rate: float = 0.1):
        super().__init__()
        self.self_mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.cross_mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

    def forward(self, x, mem, tgt_mask=None):
        attn1, _ = self.self_mha(x, x, x, attn_mask=tgt_mask)
        x = self.ln1(x + self.dropout1(attn1))
        
        attn2, _ = self.cross_mha(x, mem, mem)
        x = self.ln2(x + self.dropout2(attn2))
        
        ffn_out = self.ffn(x)
        x = self.ln3(x + self.dropout3(ffn_out))
        return x


class TransformerQAModel(nn.Module):
    def __init__(self, config: TinyMemoryConfig, vocab_size: int = 2000, embed_dim: int = 128, 
                 num_layers: int = 2, num_heads: int = 4, ff_dim: int = 256, max_target_len: int = 16,
                 dropout_rate: float = 0.1, pad_id: int = 0, bos_id: int = 2, eos_id: int = 3):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.max_target_len = max_target_len
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.pos_encoding = SinusoidalPositionalEncoding(dim=embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fact_encoders = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout_rate) for _ in range(num_layers)
        ])
        
        self.query_encoders = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout_rate) for _ in range(num_layers)
        ])
        
        self.memory_proj_in = nn.Linear(embed_dim, config.hidden_size)
        self.memory_proj_out = nn.Linear(config.hidden_size, embed_dim)
        
        # Independent Projections for True Key-Value NN Baseline (P0-08)
        self.nn_k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.nn_v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.nn_proj_out = nn.Linear(embed_dim, embed_dim)
        # Backwards compatibility alias
        self.nn_proj_in = self.nn_k_proj
        
        self.bank = TinyMemoryBank(config)
        
        self.decoders = nn.ModuleList([
            TransformerDecoderBlock(embed_dim, num_heads, ff_dim, dropout_rate) for _ in range(num_layers)
        ])
        
        self.decoder_out = nn.Linear(embed_dim, vocab_size)

    def encode_fact(self, input_ids, mask=None):
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        key_padding_mask = (mask == 0) if mask is not None else None
        
        for enc in self.fact_encoders:
            x = enc(x, key_padding_mask=key_padding_mask)
            
        if mask is not None:
            mask_f = mask.unsqueeze(-1).float()
            h_eos = torch.sum(x * mask_f, dim=1) / torch.clamp(torch.sum(mask_f, dim=1), min=1.0)
        else:
            h_eos = torch.mean(x, dim=1)
            
        return h_eos

    def encode_query(self, input_ids, mask=None):
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        key_padding_mask = (mask == 0) if mask is not None else None
        
        for enc in self.query_encoders:
            x = enc(x, key_padding_mask=key_padding_mask)
            
        if mask is not None:
            mask_f = mask.unsqueeze(-1).float()
            h_eos = torch.sum(x * mask_f, dim=1) / torch.clamp(torch.sum(mask_f, dim=1), min=1.0)
        else:
            h_eos = torch.mean(x, dim=1)
            
        return h_eos

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _decode_from_context(self, context, target_ids):
        batch_size, seq_len = target_ids.shape
        device = target_ids.device
        
        bos = torch.full((batch_size, 1), self.bos_id, dtype=torch.long, device=device)
        dec_in = torch.cat([bos, target_ids[:, :-1]], dim=1)
        
        x = self.embedding(dec_in)
        x = self.pos_encoding(x)
        
        causal_mask = self._generate_square_subsequent_mask(seq_len).to(device)
        
        y = x
        for dec in self.decoders:
            y = dec(y, context, tgt_mask=causal_mask)
            
        logits = self.decoder_out(y)
        return logits

    def write_only(self, input_ids, mask, is_eos, write_prob, memory_mode='bank'):
        h_eos = self.encode_fact(input_ids, mask)
        written_indices = None
        if memory_mode == 'bank':
            h_eos_proj = self.memory_proj_in(h_eos)
            written_indices = self.bank.write(h_eos_proj, is_eos, write_prob)
            return h_eos_proj, written_indices
        elif memory_mode == 'nn':
            k_proj = self.nn_k_proj(h_eos)
            v_proj = self.nn_v_proj(h_eos)
            return (k_proj, v_proj), written_indices
        else:
            return h_eos, written_indices

    def greedy_decode(self, input_ids, mask, read_prob, write_prob, max_len=16, memory_mode='bank', fact_store=None):
        device = input_ids.device
        h_eos = self.encode_query(input_ids, mask)
        
        if memory_mode == 'none':
            # P0-06: No Memory clean baseline
            h_fused_proj = h_eos
        elif memory_mode == 'nn':
            # P0-08: True Key-Value NN Memory
            q = self.nn_k_proj(h_eos)
            if fact_store is None:
                fact_keys = torch.zeros(self.config.memory_capacity, self.embed_dim, device=device)
                fact_vals = torch.zeros(self.config.memory_capacity, self.embed_dim, device=device)
            elif isinstance(fact_store, tuple):
                fact_keys, fact_vals = fact_store
            else:
                fact_keys, fact_vals = fact_store, fact_store

            q_n = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)
            k_n = fact_keys / (torch.norm(fact_keys, dim=-1, keepdim=True) + 1e-8)
            sim = torch.matmul(q_n, k_n.T)
            active_mask = torch.any(fact_keys != 0, dim=-1).unsqueeze(0)
            masked_sim = torch.where(active_mask, sim, torch.tensor(-1e9, device=device))
            attn = F.softmax(masked_sim, dim=-1)
            retrieved = torch.matmul(attn, fact_vals)
            h_fused = self.nn_proj_out(h_eos + retrieved)
            h_fused_proj = h_fused
        else:
            h_eos_proj = self.memory_proj_in(h_eos)
            h_fused = self.bank(h_eos_proj, read_prob, write_prob, deterministic=True)
            h_fused_proj = self.memory_proj_out(h_fused)
            
        context = h_fused_proj.unsqueeze(1)
        batch_size = h_eos.size(0)
        next_token = torch.full((batch_size, 1), self.bos_id, dtype=torch.long, device=device)
        output_tokens = []
        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        dec_in = next_token
        
        for t in range(max_len):
            if t > 0:
                dec_in = torch.cat([dec_in, next_token], dim=1)
                
            x = self.embedding(dec_in)
            x = self.pos_encoding(x)
            cur_len = dec_in.size(1)
            causal_mask = self._generate_square_subsequent_mask(cur_len).to(device)
            
            y = x
            for dec in self.decoders:
                y = dec(y, context, tgt_mask=causal_mask)
                
            y_t = y[:, -1, :]
            logit_t = self.decoder_out(y_t)
            next_token_raw = torch.argmax(logit_t, dim=-1, keepdim=True)
            
            is_finished = is_finished | (next_token_raw[:, 0] == self.eos_id)
            masked_tok = torch.where(is_finished, torch.tensor(self.pad_id, device=device), next_token_raw[:, 0])
            output_tokens.append(masked_tok)
            next_token = torch.where(is_finished.unsqueeze(-1), torch.tensor(self.pad_id, device=device), next_token_raw)
            
        return torch.stack(output_tokens, dim=1)

    def forward(self, input_ids, mask, read_prob, write_prob, target_ids, memory_mode='bank', fact_store=None):
        h_eos = self.encode_query(input_ids, mask)
        device = h_eos.device
        
        sim = torch.zeros(h_eos.size(0), self.config.memory_capacity, device=device)
        
        if memory_mode == 'none':
            # P0-06: No Memory clean baseline
            h_eos_proj = h_eos
            h_fused = h_eos
            h_fused_proj = h_eos
        elif memory_mode == 'nn':
            # P0-08: True Key-Value NN Memory
            q = self.nn_k_proj(h_eos)
            h_eos_proj = q
            if fact_store is None:
                fact_keys = torch.zeros(self.config.memory_capacity, self.embed_dim, device=device)
                fact_vals = torch.zeros(self.config.memory_capacity, self.embed_dim, device=device)
            elif isinstance(fact_store, tuple):
                fact_keys, fact_vals = fact_store
            else:
                fact_keys, fact_vals = fact_store, fact_store

            q_n = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)
            k_n = fact_keys / (torch.norm(fact_keys, dim=-1, keepdim=True) + 1e-8)
            sim = torch.matmul(q_n, k_n.T)
            active_mask = torch.any(fact_keys != 0, dim=-1).unsqueeze(0)
            masked_sim = torch.where(active_mask, sim, torch.tensor(-1e9, device=device))
            attn = F.softmax(masked_sim, dim=-1)
            retrieved = torch.matmul(attn, fact_vals)
            h_fused = self.nn_proj_out(h_eos + retrieved)
            h_fused_proj = h_fused
        else:
            h_eos_proj = self.memory_proj_in(h_eos)
            # P1-02: Use actual composite scores directly from TinyMemoryBank
            h_fused, actual_scores = self.bank(
                h_eos_proj, read_prob, write_prob,
                deterministic=not self.training, return_scores=True
            )
            h_fused_proj = self.memory_proj_out(h_fused)
            sim = actual_scores

        context = h_fused_proj.unsqueeze(1)
        logits = self._decode_from_context(context, target_ids)
        
        return logits, sim, h_eos_proj, h_fused
