import jax
import jax.numpy as jnp
import flax.linen as nn
from models.tiny_memory_bank import TinyMemoryBank, TinyMemoryConfig

class SinusoidalPositionalEncoding(nn.Module):
    dim: int
    max_len: int = 512

    @nn.compact
    def __call__(self, x):
        seq_len = x.shape[1]
        position = jnp.arange(seq_len)[:, None]
        div_term = jnp.exp(jnp.arange(0, self.dim, 2) * (-jnp.log(10000.0) / self.dim))
        pe = jnp.zeros((seq_len, self.dim))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        return x + jnp.expand_dims(pe, axis=0)

class TransformerEncoderBlock(nn.Module):
    embed_dim: int = 128
    num_heads: int = 4
    ff_dim: int = 256
    dropout_rate: float = 0.1

    def setup(self):
        self.mha = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)
        self.ln1 = nn.LayerNorm()
        self.ln2 = nn.LayerNorm()
        self.ffn = nn.Sequential([
            nn.Dense(self.ff_dim),
            nn.gelu,
            nn.Dense(self.embed_dim)
        ])
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, x, mask=None, deterministic=False):
        attn_mask = None
        if mask is not None:
            attn_mask = mask[:, None, None, :]
            
        attn_out = self.mha(x, x, mask=attn_mask)
        x = self.ln1(x + self.dropout(attn_out, deterministic=deterministic))
        
        ffn_out = self.ffn(x)
        x = self.ln2(x + self.dropout(ffn_out, deterministic=deterministic))
        return x

class TransformerDecoderBlock(nn.Module):
    embed_dim: int = 128
    num_heads: int = 4
    ff_dim: int = 256
    dropout_rate: float = 0.1

    def setup(self):
        self.self_mha = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)
        self.cross_mha = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)
        self.ln1 = nn.LayerNorm()
        self.ln2 = nn.LayerNorm()
        self.ln3 = nn.LayerNorm()
        self.ffn = nn.Sequential([
            nn.Dense(self.ff_dim),
            nn.gelu,
            nn.Dense(self.embed_dim)
        ])
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, x, mem, causal_mask=None, deterministic=False):
        self_attn = self.self_mha(x, x, mask=causal_mask)
        x = self.ln1(x + self.dropout(self_attn, deterministic=deterministic))
        
        cross_attn = self.cross_mha(x, mem)
        x = self.ln2(x + self.dropout(cross_attn, deterministic=deterministic))
        
        ffn_out = self.ffn(x)
        x = self.ln3(x + self.dropout(ffn_out, deterministic=deterministic))
        return x

class TransformerQAModel(nn.Module):
    config: TinyMemoryConfig
    vocab_size: int = 2000
    embed_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    ff_dim: int = 256
    max_target_len: int = 16
    dropout_rate: float = 0.1

    def setup(self):
        self.embedding = nn.Embed(num_embeddings=self.vocab_size, features=self.embed_dim)
        self.pos_encoding = SinusoidalPositionalEncoding(dim=self.embed_dim)
        
        self.fact_encoders = [
            TransformerEncoderBlock(embed_dim=self.embed_dim, num_heads=self.num_heads, ff_dim=self.ff_dim, dropout_rate=self.dropout_rate)
            for _ in range(self.num_layers)
        ]
        self.query_encoders = [
            TransformerEncoderBlock(embed_dim=self.embed_dim, num_heads=self.num_heads, ff_dim=self.ff_dim, dropout_rate=self.dropout_rate)
            for _ in range(self.num_layers)
        ]
        
        self.bank = TinyMemoryBank(self.config)
        
        self.decoders = [
            TransformerDecoderBlock(embed_dim=self.embed_dim, num_heads=self.num_heads, ff_dim=self.ff_dim, dropout_rate=self.dropout_rate)
            for _ in range(self.num_layers)
        ]
        self.decoder_out = nn.Dense(self.vocab_size)
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def encode_fact(self, input_ids, mask=None, deterministic=False):
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        x = self.dropout(x, deterministic=deterministic)
        for enc in self.fact_encoders:
            x = enc(x, mask=mask, deterministic=deterministic)
        
        # Take the first tokens_per_slot tokens
        n_tok = self.config.tokens_per_slot
        seq_len = x.shape[1]
        
        if seq_len >= n_tok:
            fact_tokens = x[:, :n_tok, :]
        else:
            pad_len = n_tok - seq_len
            fact_tokens = jnp.pad(x, ((0,0), (0, pad_len), (0,0)))
            
        return fact_tokens

    def encode_query(self, input_ids, mask=None, deterministic=False):
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        x = self.dropout(x, deterministic=deterministic)
        for enc in self.query_encoders:
            x = enc(x, mask=mask, deterministic=deterministic)
            
        n_tok = self.config.tokens_per_slot
        seq_len = x.shape[1]
        
        if seq_len >= n_tok:
            query_tokens = x[:, :n_tok, :]
        else:
            pad_len = n_tok - seq_len
            query_tokens = jnp.pad(x, ((0,0), (0, pad_len), (0,0)))
            
        if mask is not None:
            mask_expanded = mask[:, :, None]
            h_eos = jnp.sum(x * mask_expanded, axis=1) / jnp.maximum(jnp.sum(mask_expanded, axis=1), 1.0)
        else:
            h_eos = jnp.mean(x, axis=1)
            
        return query_tokens, h_eos

    def decode_h_fused(self, mem, target_ids, deterministic=False):
        batch_size, seq_len = target_ids.shape
        next_token = jnp.full((batch_size, 1), 2, dtype=jnp.int32)
        all_logits = []
        
        for t in range(seq_len):
            if t == 0:
                dec_in = next_token
            else:
                dec_in = jnp.concatenate([dec_in, next_token], axis=1)
                
            x = self.embedding(dec_in)
            x = self.pos_encoding(x)
            cur_len = dec_in.shape[1]
            causal_mask = jnp.tril(jnp.ones((cur_len, cur_len)))[None, None, :, :]
            
            y = x
            for dec in self.decoders:
                y = dec(y, mem, causal_mask=causal_mask, deterministic=deterministic)
                
            logits_seq = self.decoder_out(y)
            logits_t = logits_seq[:, -1, :]
            all_logits.append(logits_t)
            
            next_token = jax.lax.stop_gradient(
                jnp.argmax(logits_t, axis=-1, keepdims=True)
            )
            
        return jnp.stack(all_logits, axis=1)

    def write_only(self, input_ids, mask, is_eos, write_prob, deterministic=False):
        fact_tokens = self.encode_fact(input_ids, mask, deterministic=deterministic)
        return self.bank.write(fact_tokens, is_eos, write_prob), fact_tokens

    def decode_oracle(self, write_ids, write_mask, query_ids, query_mask, target_ids, deterministic=False):
        fact_tokens = self.encode_fact(write_ids, write_mask, deterministic=deterministic)
        query_tokens, h_query = self.encode_query(query_ids, query_mask, deterministic=deterministic)
        
        # Simulasikan bank fuse oracle jika perlu, atau langsung concatenate
        mem = jnp.concatenate([fact_tokens, query_tokens], axis=1)
        logits = self.decode_h_fused(mem, target_ids, deterministic=deterministic)
        return logits

    def init_all(self, input_ids, mask, is_eos, write_prob, read_prob, target_ids):
        self.write_only(input_ids, mask, is_eos, write_prob, deterministic=True)
        self.decode_oracle(input_ids, mask, input_ids, mask, target_ids, deterministic=True)
        return self.__call__(input_ids, mask, read_prob, write_prob, target_ids, deterministic=True)

    def greedy_decode(self, input_ids, mask, read_prob, write_prob, max_len=16, bos_id=2, pad_id=0, eos_id=3, deterministic=False):
        query_tokens, h_eos = self.encode_query(input_ids, mask, deterministic=True)
        h_fused, sim = self.bank(h_eos, read_prob, write_prob, deterministic=True)
        
        mem = jnp.concatenate([h_fused, query_tokens], axis=1)
        
        batch_size = h_fused.shape[0]
        next_token = jnp.full((batch_size, 1), bos_id, dtype=jnp.int32)
        
        output_tokens = []
        is_finished = jnp.zeros((batch_size,), dtype=jnp.bool_)
        dec_in = next_token
        
        for t in range(max_len):
            if t > 0:
                dec_in = jnp.concatenate([dec_in, next_token], axis=1)
                
            x = self.embedding(dec_in)
            x = self.pos_encoding(x)
            cur_len = dec_in.shape[1]
            causal_mask = jnp.tril(jnp.ones((cur_len, cur_len)))[None, None, :, :]
            
            y = x
            for dec in self.decoders:
                y = dec(y, mem, causal_mask=causal_mask, deterministic=True)
                
            logits_seq = self.decoder_out(y)
            logits_t = logits_seq[:, -1, :]
            next_token_raw = jnp.argmax(logits_t, axis=-1, keepdims=True)
            
            is_finished = is_finished | (next_token_raw[:, 0] == eos_id)
            next_token_masked = jnp.where(is_finished, pad_id, next_token_raw[:, 0])
            output_tokens.append(next_token_masked)
            
            next_token = jnp.where(is_finished[:, None], pad_id, next_token_raw)
            
        return jnp.stack(output_tokens, axis=1)

    def __call__(self, input_ids, mask, read_prob, write_prob, target_ids, deterministic=False):
        query_tokens, h_eos = self.encode_query(input_ids, mask, deterministic=deterministic)
        h_fused, sim = self.bank(h_eos, read_prob, write_prob, deterministic=deterministic)
        
        mem = jnp.concatenate([h_fused, query_tokens], axis=1)
        logits = self.decode_h_fused(mem, target_ids, deterministic=deterministic)
        return logits, sim, h_eos, h_fused
