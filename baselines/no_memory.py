import jax
import jax.numpy as jnp
import flax.linen as nn

class NoMemory(nn.Module):
    """
    Baseline with absolutely no memory. It just passes the input through.
    """
    dim: int

    def setup(self):
        self.fusion_proj = nn.Dense(self.dim, use_bias=False)
        
    @nn.compact
    def __call__(self, h_eos):
        # Pretend it retrieved zeros
        read_val = jnp.zeros_like(h_eos)
        fused = self.fusion_proj(jnp.concatenate([h_eos, read_val], axis=-1))
        return fused
