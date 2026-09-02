import jax
import jax.numpy as jnp
import flax.linen as nn

class SimpleMemory(nn.Module):
    """
    A simple baseline memory that stores K-V pairs sequentially.
    No decay, no importance, no update mechanism (just overwrite/FIFO).
    """
    capacity: int
    dim: int

    def setup(self):
        self.mem_keys = self.variable('memory', 'keys', jnp.zeros, (self.capacity, self.dim))
        self.mem_vals = self.variable('memory', 'vals', jnp.zeros, (self.capacity, self.dim))
        self.head = self.variable('memory', 'head', jnp.zeros, (), jnp.int32)
        
        self.q_proj = nn.Dense(self.dim, use_bias=False)
        self.k_proj = nn.Dense(self.dim, use_bias=False)
        self.v_proj = nn.Dense(self.dim, use_bias=False)
        
        self.fusion_proj = nn.Dense(self.dim, use_bias=False)

    def write(self, h_eos):
        k_new = self.k_proj(h_eos)
        v_new = self.v_proj(h_eos)
        
        head = self.head.value
        keys = self.mem_keys.value
        vals = self.mem_vals.value
        
        def update_single(state, inputs):
            k, v, head_idx = state
            k_n, v_n = inputs
            k = k.at[head_idx].set(k_n)
            v = v.at[head_idx].set(v_n)
            next_head = (head_idx + 1) % self.capacity
            return (k, v, next_head), None

        (new_keys, new_vals, new_head), _ = jax.lax.scan(
            update_single, (keys, vals, head), (k_new, v_new)
        )
        
        self.mem_keys.value = new_keys
        self.mem_vals.value = new_vals
        self.head.value = new_head
        
    def read(self, h_eos):
        q = self.q_proj(h_eos)
        keys = self.mem_keys.value
        vals = self.mem_vals.value
        
        q_norm = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)
        k_norm = keys / (jnp.linalg.norm(keys, axis=-1, keepdims=True) + 1e-8)
        
        sim = jnp.matmul(q_norm, k_norm.T) # (batch, capacity)
        
        # Simple top-1 read
        best_idx = jnp.argmax(sim, axis=-1)
        
        def gather(idx): return vals[idx]
        read_val = jax.vmap(gather)(best_idx)
        return read_val

    @nn.compact
    def __call__(self, h_eos):
        self.write(h_eos)
        read_val = self.read(h_eos)
        fused = self.fusion_proj(jnp.concatenate([h_eos, read_val], axis=-1))
        return fused
