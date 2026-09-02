"""
TinyModel – Standalone wrapper around TinyMemoryBank for benchmark experiments.

Fixes applied:
- BUG-P0-017: __call__ now calls full pipeline (decay→read→fuse) not just read
- BUG-P0-016: get_v_proj works correctly since v_proj now exists in bank
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from models.tiny_memory_bank import TinyMemoryBank, TinyMemoryConfig


class TinyModel(nn.Module):
    """
    Standalone tiny model for benchmark testing.
    Inputs: continuous h_eos vectors  (batch, hidden_size)
    Wraps TinyMemoryBank with a simple auto-encoding decoder.
    """
    config: TinyMemoryConfig

    def setup(self):
        self.bank = TinyMemoryBank(config=self.config)

        # Simple decoder to reconstruct h_eos from fused representation
        self.decoder = nn.Sequential([
            nn.Dense(self.config.hidden_size * 2),
            nn.relu,
            nn.Dense(self.config.hidden_size),
        ])

    # ------------------------------------------------------------------
    # Convenience helpers used by the adapter
    # ------------------------------------------------------------------
    def get_v_proj(self, h_eos):
        """Return v_proj(h_eos) – expected value representation."""
        return self.bank.v_proj(h_eos)

    def get_h_eos(self, inputs):
        """Identity – inputs are already h_eos vectors in this model."""
        return inputs

    # ------------------------------------------------------------------
    # Memory operations (delegated to bank)
    # ------------------------------------------------------------------
    def decay_memory(self):
        return self.bank.decay_memory()

    def read_only(self, h_eos):
        """Read from memory without decay or fuse."""
        return self.bank.read(h_eos)

    def write_only(self, h_eos, is_eos, write_prob):
        """Write to memory only."""
        return self.bank.write(h_eos, is_eos, write_prob)

    # ------------------------------------------------------------------
    # Full forward pass
    # ------------------------------------------------------------------
    def __call__(self, h_eos, read_prob, write_prob, deterministic=False):
        """
        Full pipeline: decay → read → fuse → decode.
        Write must be called separately before __call__.
        """
        fused_h = self.bank(h_eos, read_prob, write_prob, deterministic=deterministic)
        reconstructed = self.decoder(fused_h)
        return reconstructed
