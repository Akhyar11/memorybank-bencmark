"""
TinyModel – Standalone PyTorch wrapper around TinyMemoryBank for benchmark experiments.
"""
import torch
import torch.nn as nn
from models.tiny_memory_bank import TinyMemoryBank, TinyMemoryConfig


class TinyModel(nn.Module):
    """
    Standalone tiny model for benchmark testing (PyTorch Version).
    Inputs: continuous h_eos vectors  (batch, hidden_size)
    Wraps TinyMemoryBank with a simple auto-encoding decoder.
    """
    def __init__(self, config: TinyMemoryConfig = None):
        super().__init__()
        if config is None:
            config = TinyMemoryConfig()
        self.config = config
        self.bank = TinyMemoryBank(config=config)

        # Simple decoder to reconstruct h_eos from fused representation
        self.decoder = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size * 2, self.config.hidden_size),
        )

    def get_v_proj(self, h_eos):
        """Return v_proj(h_eos) – expected value representation."""
        return self.bank.v_proj(h_eos)

    def get_h_eos(self, inputs):
        """Identity – inputs are already h_eos vectors in this model."""
        return inputs

    def decay_memory(self):
        return self.bank.decay_memory()

    def read_only(self, h_eos, read_prob=None):
        """Read from memory without decay or fuse."""
        return self.bank.read(h_eos, read_prob=read_prob)

    def write_only(self, h_eos, is_eos, write_prob):
        """Write to memory only."""
        return self.bank.write(h_eos, is_eos, write_prob)

    def forward(self, h_eos, read_prob=None, write_prob=None, deterministic=False):
        """Full pipeline: decay → read → fuse → decode."""
        if read_prob is None:
            read_prob = torch.ones(h_eos.shape[0], device=h_eos.device)
        if write_prob is None:
            write_prob = torch.ones(h_eos.shape[0], device=h_eos.device)
        fused_h = self.bank(h_eos, read_prob, write_prob, deterministic=deterministic)
        reconstructed = self.decoder(fused_h)
        return reconstructed
