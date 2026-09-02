"""
TransformerQAModel – Fixed end-to-end QA model with locked Memory Bank.

Fixes applied:
- encode_fact  now outputs h_eos (batch, dim) not token sequences
- encode_query now outputs h_eos (batch, dim)
- write_only calls bank.write(h_eos, is_eos, write_prob) correctly
- __call__ calls bank(h_eos, read_prob, write_prob) which runs full pipeline
- fuse via bank.fuse is used in the forward pass
- decode_oracle is kept as a DIAGNOSTIC tool only (NOT mixed into main loss)
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from models.tiny_memory_bank import TinyMemoryBank, TinyMemoryConfig


class SinusoidalPositionalEncoding(nn.Module):
    dim: int
    max_len: int = 512

    @nn.compact
    def __call__(self, x):
        seq_len  = x.shape[1]
        position = jnp.arange(seq_len)[:, None]
        div_term = jnp.exp(
            jnp.arange(0, self.dim, 2) * (-jnp.log(10000.0) / self.dim)
        )
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
        self.mha     = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)
        self.ln1     = nn.LayerNorm()
        self.ln2     = nn.LayerNorm()
        self.ffn     = nn.Sequential([
            nn.Dense(self.ff_dim), nn.gelu, nn.Dense(self.embed_dim)
        ])
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, x, mask=None, deterministic=False):
        attn_mask = mask[:, None, None, :] if mask is not None else None
        attn_out  = self.mha(x, x, mask=attn_mask)
        x = self.ln1(x + self.dropout(attn_out, deterministic=deterministic))
        x = self.ln2(x + self.dropout(self.ffn(x), deterministic=deterministic))
        return x


class TransformerDecoderBlock(nn.Module):
    embed_dim: int = 128
    num_heads: int = 4
    ff_dim: int = 256
    dropout_rate: float = 0.1

    def setup(self):
        self.self_mha  = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)
        self.cross_mha = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)
        self.ln1       = nn.LayerNorm()
        self.ln2       = nn.LayerNorm()
        self.ln3       = nn.LayerNorm()
        self.ffn       = nn.Sequential([
            nn.Dense(self.ff_dim), nn.gelu, nn.Dense(self.embed_dim)
        ])
        self.dropout   = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, x, mem, causal_mask=None, deterministic=False):
        x = self.ln1(x + self.dropout(
            self.self_mha(x, x, mask=causal_mask), deterministic=deterministic
        ))
        x = self.ln2(x + self.dropout(
            self.cross_mha(x, mem), deterministic=deterministic
        ))
        x = self.ln3(x + self.dropout(self.ffn(x), deterministic=deterministic))
        return x


class TransformerQAModel(nn.Module):
    """
    End-to-end QA model:
        Text → Embedding → Encoder → Memory Bank → Decoder → Logits

    Architecture:
        - Dual-stream Transformer encoder (fact / query)
        - TinyMemoryBank (locked architecture)
        - Cross-attention Transformer decoder
        - Output: logits (NOT probabilities) → use with cross-entropy loss

    Fusion (via locked Memory Bank fuse):
        h_fused = fusion_proj(concat([h_eos, retrieved_memory]))
    """
    config: TinyMemoryConfig
    vocab_size: int = 2000
    embed_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    ff_dim: int = 256
    max_target_len: int = 16
    dropout_rate: float = 0.1
    pad_id: int = 0
    bos_id: int = 2
    eos_id: int = 3

    def setup(self):
        self.embedding    = nn.Embed(num_embeddings=self.vocab_size, features=self.embed_dim)
        self.pos_encoding = SinusoidalPositionalEncoding(dim=self.embed_dim)

        self.fact_encoders = [
            TransformerEncoderBlock(
                embed_dim=self.embed_dim, num_heads=self.num_heads,
                ff_dim=self.ff_dim, dropout_rate=self.dropout_rate
            )
            for _ in range(self.num_layers)
        ]
        self.query_encoders = [
            TransformerEncoderBlock(
                embed_dim=self.embed_dim, num_heads=self.num_heads,
                ff_dim=self.ff_dim, dropout_rate=self.dropout_rate
            )
            for _ in range(self.num_layers)
        ]

        # Locked Memory Bank
        self.bank = TinyMemoryBank(self.config)

        self.decoders = [
            TransformerDecoderBlock(
                embed_dim=self.embed_dim, num_heads=self.num_heads,
                ff_dim=self.ff_dim, dropout_rate=self.dropout_rate
            )
            for _ in range(self.num_layers)
        ]
        self.decoder_out = nn.Dense(self.vocab_size)
        self.dropout     = nn.Dropout(rate=self.dropout_rate)

    # ------------------------------------------------------------------
    # Encoders – output h_eos (batch, hidden_size)
    # ------------------------------------------------------------------
    def encode_fact(self, input_ids, mask=None, deterministic=False):
        """Encode fact tokens → h_eos (batch, hidden_size) via mean-pooling."""
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        x = self.dropout(x, deterministic=deterministic)
        for enc in self.fact_encoders:
            x = enc(x, mask=mask, deterministic=deterministic)

        # Mean-pool over sequence → h_eos: (batch, embed_dim)
        if mask is not None:
            mask_f = mask[:, :, None].astype(jnp.float32)
            h_eos  = jnp.sum(x * mask_f, axis=1) / jnp.maximum(
                jnp.sum(mask_f, axis=1), 1.0
            )
        else:
            h_eos = jnp.mean(x, axis=1)
        return h_eos  # (batch, embed_dim)

    def encode_query(self, input_ids, mask=None, deterministic=False):
        """Encode query tokens → h_eos (batch, hidden_size)."""
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        x = self.dropout(x, deterministic=deterministic)
        for enc in self.query_encoders:
            x = enc(x, mask=mask, deterministic=deterministic)

        if mask is not None:
            mask_f = mask[:, :, None].astype(jnp.float32)
            h_eos  = jnp.sum(x * mask_f, axis=1) / jnp.maximum(
                jnp.sum(mask_f, axis=1), 1.0
            )
        else:
            h_eos = jnp.mean(x, axis=1)
        return h_eos  # (batch, embed_dim)

    # ------------------------------------------------------------------
    # Decoder – autoregressive, free-running (NOT teacher forcing)
    # ------------------------------------------------------------------
    def _decode_from_context(self, context, target_ids, deterministic=False):
        """
        Autoregressive decoder using model's own predictions as next input.
        This is FREE-RUNNING (not teacher forcing): input_t = argmax(logit_{t-1}).

        context: (batch, context_len, embed_dim) – cross-attention memory
        target_ids: (batch, tgt_seq_len) – used for length only during training

        Returns:
            logits: (batch, tgt_seq_len, vocab_size)
        """
        batch_size, seq_len = target_ids.shape
        next_token = jnp.full((batch_size, 1), self.bos_id, dtype=jnp.int32)
        all_logits = []
        dec_in     = next_token

        for t in range(seq_len):
            if t > 0:
                dec_in = jnp.concatenate([dec_in, next_token], axis=1)

            x = self.embedding(dec_in)
            x = self.pos_encoding(x)
            cur_len    = dec_in.shape[1]
            causal_mask= jnp.tril(jnp.ones((cur_len, cur_len)))[None, None, :, :]

            y = x
            for dec in self.decoders:
                y = dec(y, context, causal_mask=causal_mask, deterministic=deterministic)

            y_t     = y[:, -1, :]           # (batch, dim)
            logit_t = self.decoder_out(y_t)  # (batch, vocab_size) → logits (NOT probs)
            all_logits.append(logit_t)

            # Free-running: feed own prediction as next input
            next_token = jax.lax.stop_gradient(
                jnp.argmax(logit_t, axis=-1, keepdims=True)
            )

        return jnp.stack(all_logits, axis=1)  # (batch, seq_len, vocab_size)

    # ------------------------------------------------------------------
    # Memory operations
    # ------------------------------------------------------------------
    def write_only(self, input_ids, mask, is_eos, write_prob, deterministic=False):
        """Encode fact and write to memory bank."""
        h_eos = self.encode_fact(input_ids, mask, deterministic=deterministic)
        self.bank.write(h_eos, is_eos, write_prob)
        return h_eos

    def decode_oracle(self, write_ids, write_mask, query_ids, query_mask,
                      target_ids, deterministic=False):
        """
        DIAGNOSTIC ONLY – Oracle decoder.
        Directly fuses h_fact + h_query without going through memory retrieval.
        Used as oracle_upper_bound only.  DO NOT mix loss into main training.
        """
        h_fact  = self.encode_fact(write_ids,  write_mask,  deterministic=deterministic)
        h_query = self.encode_query(query_ids, query_mask,  deterministic=deterministic)
        # Simulate perfect retrieval: memory = h_fact
        h_fused = self.bank.fuse(h_query, h_fact)  # (batch, hidden_size)
        # Build context for decoder: (batch, 1, hidden_size) for cross-attention
        context = h_fused[:, None, :]
        logits  = self._decode_from_context(context, target_ids, deterministic=deterministic)
        return logits  # logits, NOT probabilities

    def init_all(self, input_ids, mask, is_eos, write_prob, read_prob, target_ids):
        """Initialize all sub-module parameters."""
        self.write_only(input_ids, mask, is_eos, write_prob, deterministic=True)
        self.decode_oracle(input_ids, mask, input_ids, mask, target_ids, deterministic=True)
        return self.__call__(input_ids, mask, read_prob, write_prob, target_ids, deterministic=True)

    def greedy_decode(self, input_ids, mask, read_prob, write_prob,
                      max_len=16, deterministic=True):
        """Greedy decoding for inference."""
        bos_id = self.bos_id
        pad_id = self.pad_id
        eos_id = self.eos_id
        h_eos   = self.encode_query(input_ids, mask, deterministic=True)
        h_fused = self.bank(h_eos, read_prob, write_prob, deterministic=True)

        context     = h_fused[:, None, :]  # (batch, 1, dim)
        batch_size  = h_fused.shape[0]
        next_token  = jnp.full((batch_size, 1), bos_id, dtype=jnp.int32)
        output_tokens = []
        is_finished   = jnp.zeros((batch_size,), dtype=jnp.bool_)
        dec_in        = next_token

        for t in range(max_len):
            if t > 0:
                dec_in = jnp.concatenate([dec_in, next_token], axis=1)

            x = self.embedding(dec_in)
            x = self.pos_encoding(x)
            cur_len     = dec_in.shape[1]
            causal_mask = jnp.tril(jnp.ones((cur_len, cur_len)))[None, None, :, :]

            y = x
            for dec in self.decoders:
                y = dec(y, context, causal_mask=causal_mask, deterministic=True)

            y_t          = y[:, -1, :]
            logit_t      = self.decoder_out(y_t)
            next_token_raw = jnp.argmax(logit_t, axis=-1, keepdims=True)

            is_finished = is_finished | (next_token_raw[:, 0] == eos_id)
            masked_tok  = jnp.where(is_finished, pad_id, next_token_raw[:, 0])
            output_tokens.append(masked_tok)
            next_token  = jnp.where(is_finished[:, None], pad_id, next_token_raw)

        return jnp.stack(output_tokens, axis=1)

    def __call__(self, input_ids, mask, read_prob, write_prob, target_ids,
                 deterministic=False):
        """
        Main forward pass (QA inference).

        Pipeline:
            1. encode_query → h_eos
            2. bank(h_eos, read_prob, write_prob) → decay → read → fuse → h_fused
            3. decode h_fused → logits

        Returns:
            logits:  (batch, tgt_len, vocab_size)   ← use with cross_entropy
            sim:     (batch, capacity)                ← retrieval similarity scores
            h_eos:   (batch, hidden_size)
            h_fused: (batch, hidden_size)
        """
        h_eos   = self.encode_query(input_ids, mask, deterministic=deterministic)
        h_fused = self.bank(h_eos, read_prob, write_prob, deterministic=deterministic)

        context = h_fused[:, None, :]  # (batch, 1, hidden_size) for cross-attention
        logits  = self._decode_from_context(context, target_ids, deterministic=deterministic)

        # Retrieve sim scores from internal read (for logging purposes)
        # We cannot call read again here; return zeros as placeholder for sim
        sim = jnp.zeros((h_eos.shape[0], self.config.memory_capacity))

        return logits, sim, h_eos, h_fused
