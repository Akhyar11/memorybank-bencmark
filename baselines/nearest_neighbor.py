import jax
import jax.numpy as jnp
import flax.linen as nn

class NearestNeighborMemory(nn.Module):
    """
    Baseline that simply retrieves from an external fixed dataset using Cosine Similarity.
    It doesn't 'write' online, it just assumes a pre-populated key-value store.
    """
    capacity: int
    dim: int
    top_k: int = 4

    def setup(self):
        # We pretend it's pre-populated or we can populate it manually
        self.mem_keys = self.variable('memory', 'keys', jnp.zeros, (self.capacity, self.dim))
        self.mem_vals = self.variable('memory', 'vals', jnp.zeros, (self.capacity, self.dim))
        
        self.q_proj = nn.Dense(self.dim, use_bias=False)
        self.fusion_proj = nn.Dense(self.dim, use_bias=False)
        
    def read(self, h_eos):
        q = self.q_proj(h_eos)
        keys = self.mem_keys.value
        vals = self.mem_vals.value
        
        q_norm = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)
        k_norm = keys / (jnp.linalg.norm(keys, axis=-1, keepdims=True) + 1e-8)
        
        sim = jnp.matmul(q_norm, k_norm.T) # (batch, capacity)
        
        topk_sim, topk_indices = jax.lax.top_k(sim, self.top_k)
        
        attn_weights = jax.nn.softmax(topk_sim, axis=-1)
        
        def gather(idx): return vals[idx]
        topk_vals = jax.vmap(gather)(topk_indices) # (batch, k, dim)
        
        read_val = jnp.sum(attn_weights[..., None] * topk_vals, axis=1)
        return read_val

    @nn.compact
    def __call__(self, h_eos):
        read_val = self.read(h_eos)
        fused = self.fusion_proj(jnp.concatenate([h_eos, read_val], axis=-1))
        return fused
