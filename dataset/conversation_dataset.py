"""
conversation_dataset.py – Lazy/Streaming Conversational Dataset Loader for Pure Decoder-Only LM.

Loads ChatML multi-turn conversations from JSONL files (conversations_100M_*.jsonl):
- Tokenizes ChatML text with standard special tokens (<|im_start|>, <|im_end|>).
- Generates causal input_ids (x_0 ... x_{T-1}) and target_ids (x_1 ... x_T).
- Computes loss_mask for Next-Token Prediction with padding ignored.
- Extracts turn boundaries for memory read/write scheduling.
- Strictly isolates evaluation metadata (facts, target_recall) so no data leakage occurs.
"""
import os
import json
from typing import Optional, List, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer


def get_or_create_tokenizer(tokenizer_path: str = "dataset/tokenizer.json") -> Tokenizer:
    tok = Tokenizer.from_file(tokenizer_path)
    # Ensure special tokens exist
    special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "<|im_start|>", "<|im_end|>"]
    vocab = tok.get_vocab()
    missing = [t for t in special_tokens if t not in vocab]
    if missing:
        tok.add_special_tokens(missing)
    return tok


class ConversationDataset(Dataset):
    """
    Lazy / Index-based Dataset for Multi-Turn Conversational JSONL files.
    Reads file line-by-line using byte offsets for minimal RAM footprint.
    """
    def __init__(
        self,
        jsonl_path: str,
        tokenizer_path: str = "dataset/tokenizer.json",
        max_seq_len: int = 1024,
        max_samples: Optional[int] = None
    ):
        self.jsonl_path = jsonl_path
        self.max_seq_len = max_seq_len
        self.tokenizer = get_or_create_tokenizer(tokenizer_path)
        
        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        if self.pad_id is None:
            self.pad_id = 0
        self.bos_id = self.tokenizer.token_to_id("[BOS]")
        self.eos_id = self.tokenizer.token_to_id("[EOS]")
        self.im_start_id = self.tokenizer.token_to_id("<|im_start|>")
        self.im_end_id = self.tokenizer.token_to_id("<|im_end|>")
        self.vocab_size = self.tokenizer.get_vocab_size()

        # Build line byte offsets for fast random access without loading 400MB into memory
        self.offsets = []
        with open(jsonl_path, 'rb') as f:
            offset = 0
            for line in f:
                self.offsets.append(offset)
                offset += len(line)
                if max_samples and len(self.offsets) >= max_samples:
                    break

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        with open(self.jsonl_path, 'rb') as f:
            f.seek(self.offsets[idx])
            line = f.readline().decode('utf-8')
        item = json.loads(line)

        # Tokenize entire ChatML string
        chatml_text = item["chatml"]
        enc = self.tokenizer.encode(chatml_text)
        token_ids = enc.ids[:self.max_seq_len]

        # In standard NTP: input is token_ids[:-1], target is token_ids[1:]
        # If seq length is N:
        seq_len = len(token_ids)
        if seq_len < 2:
            token_ids = token_ids + [self.pad_id] * (2 - seq_len)
            seq_len = 2

        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(token_ids[1:], dtype=torch.long)
        
        # Track turns and turn token spans for episodic read/write scheduling
        # Extract turn offsets by tokenizing turns progressively
        turn_end_indices = []
        running_text = ""
        for t in item.get("turns", []):
            running_text += f"<|im_start|>{t['role']}\n{t['content']}<|im_end|>\n"
            t_enc = self.tokenizer.encode(running_text.strip())
            turn_end_indices.append(min(len(t_enc.ids) - 1, self.max_seq_len - 2))

        return {
            "id": item["id"],
            "entity_id": item["entity_id"],
            "topic": item["topic"],
            "input_ids": input_ids,
            "target_ids": target_ids,
            "seq_len": input_ids.size(0),
            "turn_end_indices": turn_end_indices,
            "facts": item.get("facts", []),
            "target_recall": item.get("target_recall", {}),
            "turns": item.get("turns", [])
        }


def conversation_collate_fn(batch: List[Dict[str, Any]], pad_id: int = 0) -> Dict[str, Any]:
    max_len = max(item["seq_len"] for item in batch)
    b_size = len(batch)

    padded_inputs = torch.full((b_size, max_len), pad_id, dtype=torch.long)
    padded_targets = torch.full((b_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((b_size, max_len), dtype=torch.float32)
    loss_mask = torch.zeros((b_size, max_len), dtype=torch.float32)

    for i, item in enumerate(batch):
        l = item["seq_len"]
        padded_inputs[i, :l] = item["input_ids"]
        padded_targets[i, :l] = item["target_ids"]
        attention_mask[i, :l] = 1.0
        loss_mask[i, :l] = 1.0  # calculate loss over all non-padding tokens

    return {
        "input_ids": padded_inputs,
        "target_ids": padded_targets,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "seq_lens": [item["seq_len"] for item in batch],
        "turn_end_indices": [item["turn_end_indices"] for item in batch],
        "facts": [item["facts"] for item in batch],
        "target_recall": [item["target_recall"] for item in batch],
        "entity_ids": [item["entity_id"] for item in batch],
        "ids": [item["id"] for item in batch],
        "turns": [item["turns"] for item in batch],
    }
