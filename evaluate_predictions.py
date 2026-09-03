"""
evaluate_predictions.py (Pure PyTorch Version)
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

from models.tiny_memory_bank import TinyMemoryConfig
from models.transformer_qa_model import TransformerQAModel
from dataset.text_dataset_loader import TextDataset
from torch.utils.data import DataLoader


def main():
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
    train_csv = os.path.join(dataset_dir, "train.csv")
    tokenizer_path = os.path.join(dataset_dir, "tokenizer.json")

    if not os.path.exists(train_csv) or not os.path.exists(tokenizer_path):
        print("Dataset or tokenizer not found.")
        return

    batch_size = 32
    max_input_len = 16
    max_target_len = 16

    dataset = TextDataset(train_csv, tokenizer_path, max_input_len=max_input_len, max_target_len=max_target_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    config = TinyMemoryConfig(
        memory_capacity=128,
        memory_dim=32,
        hidden_size=32,
        memory_threshold=-1.0,
        memory_read_threshold=-1.0,
        memory_write_threshold=-1.0,
        memory_top_k=8
    )

    model = TransformerQAModel(
        config=config,
        vocab_size=dataset.vocab_size,
        embed_dim=32,
        num_layers=2,
        num_heads=2,
        pad_id=dataset.pad_id,
        bos_id=dataset.bos_id,
        eos_id=dataset.eos_id
    )

    batch = next(iter(loader))
    model.eval()
    with torch.no_grad():
        write_ids = batch['write_ids']
        write_mask = batch['write_mask']
        query_ids = batch['query_ids']
        query_mask = batch['query_mask']
        target_ids = batch['target_ids']
        b_size = write_ids.size(0)

        # Write then Query
        model.write_only(write_ids, write_mask, torch.ones(b_size), torch.ones(b_size))
        logits, sim, _, _ = model(query_ids, query_mask, torch.ones(b_size), torch.zeros(b_size), target_ids)
        preds = torch.argmax(logits, dim=-1)

        print(f"Sample evaluated successfully. Logits shape: {logits.shape}, Preds shape: {preds.shape}")


if __name__ == '__main__':
    main()
