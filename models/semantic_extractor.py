"""
models/semantic_extractor.py
============================
Modular Semantic Vector Extractor for Episodic Memory Systems.

Replaces the lossy last-token pooling (hidden[:, -1, :]) with comprehensive
bidirectional sequence pooling:
  - "indobert": Pre-trained bidirectional encoder (indolem/indobert-base-uncased)
    with attention-mask weighted mean pooling.
  - "gpt2_pooling": Attention-weighted self-pooling over GPT-2 representations
    for zero-extra-model VRAM efficiency.
"""

from typing import List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class SemanticSentenceExtractor(nn.Module):
    """
    Extracts high-fidelity 768-dimensional semantic sentence vectors.
    """

    def __init__(
        self,
        extractor_type: str = "indobert",
        model_name_or_path: Optional[str] = None,
        embed_dim: int = 768,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.extractor_type = extractor_type.lower()
        self.embed_dim = embed_dim
        self._target_device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.extractor_type == "indobert":
            model_path = model_name_or_path or "indolem/indobert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.encoder = AutoModel.from_pretrained(model_path)
            # Freeze encoder to keep semantic space stable and save compute
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()
            self.encoder.to(self._target_device)
        elif self.extractor_type == "gpt2_pooling":
            # Uses attention-weighted self-pooling
            self.tokenizer = None
            self.encoder = None
            self.attn_pool = nn.Sequential(
                nn.Linear(embed_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1),
            ).to(self._target_device)
        else:
            raise ValueError(f"Unknown extractor_type: '{extractor_type}'. Choose 'indobert' or 'gpt2_pooling'.")

    @property
    def device(self) -> torch.device:
        return self._target_device

    def to(self, *args, **kwargs):
        device = kwargs.get("device", None)
        if args and isinstance(args[0], (str, torch.device)):
            device = args[0]
        if device is not None:
            self._target_device = torch.device(device)
        return super().to(*args, **kwargs)

    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        max_length: int = 128,
    ) -> torch.Tensor:
        """
        Encodes one or multiple text strings into a (Batch, 768) semantic vector.
        """
        if isinstance(texts, str):
            texts = [texts]

        if self.extractor_type == "indobert":
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self._target_device)

            with torch.no_grad():
                outputs = self.encoder(**encoded)
                token_embeddings = outputs.last_hidden_state  # (B, T, 768)
                attention_mask = encoded["attention_mask"]   # (B, T)

                # Mean pooling with attention mask
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
                sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
                pooled = sum_embeddings / sum_mask  # (B, 768)

                if normalize:
                    pooled = F.normalize(pooled, p=2, dim=-1)

            return pooled

        else:
            raise NotImplementedError("For 'gpt2_pooling', use encode_from_hidden_states().")

    def encode_from_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Encodes sequence of hidden states (B, T, 768) using weighted attention pooling.
        """
        # (B, T, 1)
        weights = self.attn_pool(hidden_states)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)
            weights = weights.masked_fill(mask == 0, -1e9)

        attn_scores = F.softmax(weights, dim=1)  # (B, T, 1)
        pooled = torch.sum(hidden_states * attn_scores, dim=1)  # (B, 768)

        if normalize:
            pooled = F.normalize(pooled, p=2, dim=-1)

        return pooled
