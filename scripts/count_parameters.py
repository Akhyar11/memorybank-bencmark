"""
scripts/count_parameters.py (Pure PyTorch Version)

Reports:
- Embedding parameters
- Memory Bank parameters (learned projections only, not memory state)
- Encoder parameters
- Decoder parameters
- Total trainable parameters
- Memory state buffer tensors (runtime episodic state, not trainable)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.tiny_memory_bank import TinyMemoryConfig
from models.transformer_qa_model import TransformerQAModel


def report(config=None, vocab_size=2000, embed_dim=32, num_layers=1, num_heads=2, max_target_len=16):
    if config is None:
        config = TinyMemoryConfig(
            memory_capacity=128, memory_dim=32, hidden_size=32,
            memory_top_k=8,
        )

    model = TransformerQAModel(
        config=config, vocab_size=vocab_size, embed_dim=embed_dim,
        num_layers=num_layers, num_heads=num_heads,
    )

    print("=" * 60)
    print("  PARAMETER COUNT REPORT (PyTorch)")
    print("=" * 60)

    total_params = 0
    for name, param in model.named_parameters():
        num = param.numel()
        total_params += num
        print(f"  {name:40s}: {num:>8d} params (requires_grad={param.requires_grad})")

    print("-" * 60)
    print(f"  TOTAL TRAINABLE PARAMETERS: {total_params:>10d}")
    print("=" * 60)

    print("\n  MEMORY STATE BUFFERS (Non-trainable Episodic State):")
    total_buffer_elements = 0
    for name, buf in model.bank.named_buffers():
        num = buf.numel()
        total_buffer_elements += num
        print(f"  bank.{name:35s}: {num:>8d} elements ({str(buf.dtype):>15s})")

    print("-" * 60)
    print(f"  TOTAL BUFFER ELEMENTS:      {total_buffer_elements:>10d}")
    print("=" * 60)


if __name__ == '__main__':
    report()
