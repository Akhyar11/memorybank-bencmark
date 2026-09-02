import jax
import jax.numpy as jnp
import flax.linen as nn
from models.tiny_memory_bank import TinyMemoryBank, TinyMemoryConfig

class TinyModel(nn.Module):
    """
    Standalone tiny model for benchmark testing.
    Works natively on continuous h_eos vectors.
    """
    config: TinyMemoryConfig
    
    def setup(self):
        # Memory Bank
        self.bank = TinyMemoryBank(config=self.config)
        
        # Simple Decoder to reconstruct h_eos
        self.decoder = nn.Sequential([
            nn.Dense(self.config.hidden_size * 2),
            nn.relu,
            nn.Dense(self.config.hidden_size) # Outputs reconstructed h_eos
        ])

    def get_v_proj(self, h_eos):
        return self.bank.v_proj(h_eos)
        
    def get_h_eos(self, inputs):
        # In this synthetic benchmark, inputs are already h_eos
        return inputs

    def decay_memory(self):
        return self.bank.decay_memory()

    def read_only(self, h_eos):
        return self.bank.read(h_eos)

    def write_only(self, h_eos, is_eos, write_prob):
        return self.bank.write(h_eos, is_eos, write_prob)

    def __call__(self, h_eos, read_prob, write_prob, deterministic=False):
        # Memory interaction
        h_fused = self.bank(h_eos, read_prob, write_prob, deterministic=deterministic)
        
        # Decode (auto-encode)
        reconstructed = self.decoder(h_fused)
        return reconstructed
