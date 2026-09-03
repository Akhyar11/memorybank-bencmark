"""
models/text_encoder.py (Pure PyTorch Version)
"""
import torch
import torch.nn as nn


class TextEmbedding(nn.Module):
    """
    Embedding layer for token inputs (PyTorch Version).
    """
    def __init__(self, vocab_size: int = 2000, embed_dim: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, input_ids: torch.Tensor):
        return self.embedding(input_ids)
