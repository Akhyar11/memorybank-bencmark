import jax
import jax.numpy as jnp
import flax.linen as nn
from models.tiny_memory_bank import TinyMemoryBank, TinyMemoryConfig
from models.text_encoder import TextEmbedding

class TextQAModel(nn.Module):
    """
    End-to-End model: Text -> Embedding -> Encoder -> Memory Bank -> Decoder -> Text.
    Autoregressive GRU Decoder dengan Dropout dan LayerNorm untuk stabilisasi.
    """
    config: TinyMemoryConfig
    vocab_size: int = 2000
    embed_dim: int = 32
    hidden_size: int = 32
    memory_capacity: int = 128
    max_target_len: int = 16
    dropout_rate: float = 0.2

    def setup(self):
        # 1. Teks ke Vektor (Embedding)
        self.embedding = nn.Embed(num_embeddings=self.vocab_size, features=self.embed_dim)
        
        # Dual Encoder: Terpisah untuk Fakta dan Query
        self.encoder_cell = nn.RNN(nn.GRUCell(self.config.hidden_size), return_carry=True) # Encoder Fakta
        self.query_encoder_cell = nn.RNN(nn.GRUCell(self.config.hidden_size), return_carry=True) # Encoder Query
        
        self.encoder_norm = nn.LayerNorm()
        self.query_encoder_norm = nn.LayerNorm()
        
        # 2. Vektor diproses di Memory Bank
        self.bank = TinyMemoryBank(self.config)
        
        # 3. Vektor dikembalikan ke Teks (Decoder Seq2Seq)
        self.decoder_cell = nn.RNN(nn.GRUCell(self.config.hidden_size), return_carry=True)
        self.decoder_norm = nn.LayerNorm()
        self.decoder_out = nn.Dense(self.vocab_size)
        self.dropout = nn.Dropout(rate=self.dropout_rate)
        
    def encode_fact(self, input_ids, mask=None, deterministic=False):
        """Menerjemahkan fakta ke vektor h_eos untuk disimpan di Memory Bank"""
        embs = self.embedding(input_ids)
        embs = self.dropout(embs, deterministic=deterministic)
        
        if mask is not None:
            lengths = jnp.sum(mask, axis=1).astype(jnp.int32)
            carry, _ = self.encoder_cell(embs, seq_lengths=lengths)
        else:
            carry, _ = self.encoder_cell(embs)
            
        h_eos = carry
        h_eos = self.encoder_norm(h_eos)
        h_eos = self.dropout(h_eos, deterministic=deterministic)
        return h_eos

    def encode_query(self, input_ids, mask=None, deterministic=False):
        """Menerjemahkan query ke vektor h_query untuk retrieval dari Memory Bank"""
        embs = self.embedding(input_ids)
        embs = self.dropout(embs, deterministic=deterministic)
        
        if mask is not None:
            lengths = jnp.sum(mask, axis=1).astype(jnp.int32)
            carry, _ = self.query_encoder_cell(embs, seq_lengths=lengths)
        else:
            carry, _ = self.query_encoder_cell(embs)
            
        h_eos = carry
        h_eos = self.query_encoder_norm(h_eos)
        h_eos = self.dropout(h_eos, deterministic=deterministic)
        return h_eos

    def encode_text(self, input_ids, mask=None, deterministic=False):
        """Fallback / default alias ke encode_fact"""
        return self.encode_fact(input_ids, mask, deterministic=deterministic)

        
    def decode_h_fused(self, h_fused, target_ids, deterministic=False):
        """Decode autoregresif: setiap langkah pakai prediksi model sendiri (bukan token target).
        Ini membuat training loss jujur — kondisi training = kondisi inferensi.
        
        Gradient tetap mengalir melalui logits setiap langkah, tapi tidak melalui
        operasi argmax (pakai stop_gradient agar tetap differentiable).
        """
        batch_size, seq_len = target_ids.shape
        
        carry = h_fused  # Initial GRU state dari memory retrieval
        next_token = jnp.full((batch_size, 1), 2, dtype=jnp.int32)  # BOS token
        all_logits = []
        
        for _ in range(seq_len):
            emb = self.embedding(next_token)  # (batch, 1, embed_dim)
            # Tambahkan h_fused di setiap langkah agar memori selalu tersedia
            h_fused_step = jnp.expand_dims(h_fused, axis=1)  # (batch, 1, hidden)
            emb_concat = jnp.concatenate([emb, h_fused_step], axis=-1)
            
            carry, out = self.decoder_cell(emb_concat, initial_carry=carry)
            out_ln = self.decoder_norm(out[:, 0, :])
            out_drop = self.dropout(out_ln, deterministic=deterministic)
            logits_t = self.decoder_out(out_drop)  # (batch, vocab_size)
            all_logits.append(logits_t)
            
            # Gunakan prediksi model sendiri sebagai input berikutnya
            # stop_gradient: gradient tidak mengalir melalui argmax (non-differentiable),
            # tapi tetap mengalir melalui logits_t di iterasi ini
            next_token = jax.lax.stop_gradient(
                jnp.argmax(logits_t, axis=-1, keepdims=True)
            )
        
        return jnp.stack(all_logits, axis=1)  # (batch, seq_len, vocab_size)

        
    def write_only(self, input_ids, mask, is_eos, write_prob, deterministic=False):
        """Fase Menulis Fakta (Belajar)"""
        h_eos = self.encode_fact(input_ids, mask, deterministic=deterministic)
        return self.bank.write(h_eos, is_eos, write_prob), h_eos
        
    def decode_oracle(self, write_ids, write_mask, query_ids, query_mask, target_ids, deterministic=False):
        """Oracle decode: gunakan h_query + h_fact langsung (tanpa retrieval dari memori).
        Ini melatih decoder untuk skenario yang sama dengan inference asli:
        fuse(h_query, h_fact) → target tokens."""
        h_fact = self.encode_fact(write_ids, write_mask, deterministic=deterministic)
        h_query = self.encode_query(query_ids, query_mask, deterministic=deterministic)
        # Oracle: anggap retrieval sempurna → fuse query dengan fakta yang benar
        h_fused_oracle = self.bank.fuse(h_query, h_fact)
        logits = self.decode_h_fused(h_fused_oracle, target_ids, deterministic=deterministic)
        return logits
        
    def init_all(self, input_ids, mask, is_eos, write_prob, read_prob, target_ids):
        """Helper untuk menginisialisasi semua parameter dengan deterministic=True agar tidak error PRNG"""
        self.write_only(input_ids, mask, is_eos, write_prob, deterministic=True)
        self.decode_oracle(input_ids, mask, input_ids, mask, target_ids, deterministic=True)
        return self.__call__(input_ids, mask, read_prob, write_prob, target_ids, deterministic=True)
        
    def greedy_decode(self, input_ids, mask, read_prob, write_prob, max_len=16, bos_id=2, pad_id=0, eos_id=3, deterministic=False):
        """Fase Inference: Menghasilkan teks token-per-token tanpa Teacher Forcing"""
        h_eos = self.encode_query(input_ids, mask, deterministic=True)
        h_fused, sim = self.bank(h_eos, read_prob, write_prob, deterministic=True)
        
        carry = h_fused
        next_token = jnp.full((h_fused.shape[0], 1), bos_id, dtype=jnp.int32)
        
        output_tokens = []
        is_finished = jnp.zeros((h_fused.shape[0],), dtype=jnp.bool_)
        
        for _ in range(max_len):
            emb = self.embedding(next_token)
            # Concatenate h_fused to input
            emb_concat = jnp.concatenate([emb, jnp.expand_dims(h_fused, axis=1)], axis=-1)
            
            carry, out = self.decoder_cell(emb_concat, initial_carry=carry)
            
            out_ln = self.decoder_norm(out[:, 0, :])
            out_drop = self.dropout(out_ln, deterministic=True)
            
            logits = self.decoder_out(out_drop)
            next_token_raw = jnp.argmax(logits, axis=-1, keepdims=True)
            
            # Update finished status BEFORE masking
            is_finished = is_finished | (next_token_raw[:, 0] == eos_id)
            
            # Mask output: jika sudah selesai, output PAD
            next_token_masked = jnp.where(is_finished, pad_id, next_token_raw[:, 0])
            output_tokens.append(next_token_masked)
            
            # PENTING: update next_token yang akan diinput ke GRU di iterasi berikutnya
            # Jika sudah selesai, feed PAD agar tidak menghasilkan repetisi
            next_token = jnp.where(is_finished[:, None], pad_id, next_token_raw)
            
        return jnp.stack(output_tokens, axis=1)
        
    def __call__(self, input_ids, mask, read_prob, write_prob, target_ids, deterministic=False):
        """Fase Query/Mengingat (Q&A)"""
        # 1. Konversi kalimat query jadi vektor (Encoding Query)
        h_eos = self.encode_query(input_ids, mask, deterministic=deterministic)
        
        # 2. Cari di Memory Bank (Retrieval)
        h_fused, sim = self.bank(h_eos, read_prob, write_prob, deterministic=deterministic)
        
        # 3. Ubah hasil pencarian jadi teks (Decoding Autoregressive)
        logits = self.decode_h_fused(h_fused, target_ids, deterministic=deterministic)
        return logits, sim, h_eos, h_fused

