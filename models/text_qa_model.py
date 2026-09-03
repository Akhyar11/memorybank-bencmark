"""
models/text_qa_model.py (Pure PyTorch Version)
GRU-based Text QA Model wrapping TinyMemoryBank.
"""
import torch
import torch.nn as nn
from models.tiny_memory_bank import TinyMemoryBank, TinyMemoryConfig


class TextQAModel(nn.Module):
    """
    End-to-End GRU Seq2Seq model with Memory Bank (PyTorch Version).
    """
    def __init__(
        self,
        config: TinyMemoryConfig = None,
        vocab_size: int = 2000,
        embed_dim: int = 32,
        hidden_size: int = 32,
        memory_capacity: int = 128,
        max_target_len: int = 16,
        dropout_rate: float = 0.2,
        pad_id: int = 0,
        bos_id: int = 2,
        eos_id: int = 3
    ):
        super().__init__()
        if config is None:
            config = TinyMemoryConfig()
        self.config = config
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.memory_capacity = memory_capacity
        self.max_target_len = max_target_len
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder_gru = nn.GRU(embed_dim, hidden_size, batch_first=True)
        self.query_encoder_gru = nn.GRU(embed_dim, hidden_size, batch_first=True)

        self.encoder_norm = nn.LayerNorm(hidden_size)
        self.query_encoder_norm = nn.LayerNorm(hidden_size)

        self.bank = TinyMemoryBank(config)

        self.decoder_gru = nn.GRU(embed_dim, hidden_size, batch_first=True)
        self.decoder_norm = nn.LayerNorm(hidden_size)
        self.decoder_out = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)

    def encode_fact(self, input_ids: torch.Tensor):
        embs = self.dropout(self.embedding(input_ids))
        _, h_n = self.encoder_gru(embs)
        h_eos = self.dropout(self.encoder_norm(h_n.squeeze(0)))
        return h_eos

    def encode_query(self, input_ids: torch.Tensor):
        embs = self.dropout(self.embedding(input_ids))
        _, h_n = self.query_encoder_gru(embs)
        h_query = self.dropout(self.query_encoder_norm(h_n.squeeze(0)))
        return h_query

    def forward(self, query_ids, target_ids=None):
        h_query = self.encode_query(query_ids)
        read_val = self.bank.read(h_query)
        fused = self.bank.fuse(h_query, read_val)

        if target_ids is not None:
            tgt_embs = self.dropout(self.embedding(target_ids))
            out, _ = self.decoder_gru(tgt_embs, fused.unsqueeze(0))
            logits = self.decoder_out(self.decoder_norm(out))
            return logits
        return fused
