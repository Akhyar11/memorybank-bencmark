import jax
import jax.numpy as jnp
import flax.linen as nn
from dataclasses import dataclass

# State Constants
STATE_EXPIRED = 0
STATE_ACTIVE = 1
STATE_DORMANT = 2
STATE_EMPTY = 3

@dataclass
class TinyMemoryConfig:
    memory_capacity: int = 128
    memory_dim: int = 32
    hidden_size: int = 32
    tokens_per_slot: int = 4  # N token representasi per fakta
    mem_decay_rate: float = 0.001
    mem_importance_protection: float = 0.5
    mem_alpha: float = 1.0
    mem_beta: float = 0.5
    mem_gamma: float = 0.1
    mem_delta: float = 0.2
    memory_top_k: int = 4     # N slot yang dibaca
    memory_write_threshold: float = 0.7
    memory_threshold: float = 0.0
    memory_read_threshold: float = 0.0
    mem_reinforcement_rate: float = 0.05

class TinyMemoryBank(nn.Module):
    """
    Persistent Neural Memory Bank yang menyimpan N token representasi per slot fakta.
    Fase Write: Menyimpan [N_fact_tokens, dim] ke dalam slot.
    Fase Read : Membaca K Slot -> Output [K * N_fact_tokens, dim].
    """
    config: TinyMemoryConfig
    
    def setup(self):
        capacity = self.config.memory_capacity
        dim = self.config.memory_dim
        n_tok = self.config.tokens_per_slot
        
        # Tensors untuk N token per slot
        self.sim_weight = self.param('sim_weight', nn.initializers.constant(5.0), ())
        self.mem_keys = self.variable('memory', 'keys', jnp.zeros, (capacity, n_tok, dim))
        self.mem_vals = self.variable('memory', 'vals', jnp.zeros, (capacity, n_tok, dim))
        self.mem_token_ids = self.variable('memory', 'token_ids', jnp.zeros, (capacity, n_tok), jnp.int32)
        
        # Meta-data
        self.mem_importance = self.variable('memory', 'importance', jnp.zeros, (capacity,))
        self.mem_confidence = self.variable('memory', 'confidence', jnp.zeros, (capacity,))
        
        self.mem_created_at = self.variable('memory', 'created_at', jnp.zeros, (capacity,), jnp.int32)
        self.mem_last_access = self.variable('memory', 'last_access', jnp.zeros, (capacity,), jnp.int32)
        self.mem_access_count = self.variable('memory', 'access_count', jnp.zeros, (capacity,), jnp.int32)
        self.mem_state = self.variable('memory', 'state', lambda s: jnp.full(s, STATE_EMPTY, dtype=jnp.int32), (capacity,))
        self.step = self.variable('memory', 'step', jnp.zeros, (), jnp.int32)
        
    def write(self, fact_tokens, fact_token_ids, is_eos, write_prob):
        """
        Menulis fakta [batch, N_fact_tokens, dim] dan ID aslinya ke memori.
        Setiap slot menyimpan N_fact_tokens.
        """
        batch_size, n_tok, dim = fact_tokens.shape
        step = self.step.value + 1
        self.step.value = step
        
        keys = self.mem_keys.value  # (capacity, n_tok, dim)
        vals = self.mem_vals.value
        token_ids = self.mem_token_ids.value
        state = self.mem_state.value
        
        # Pool vector per slot untuk perbandingan similarity (capacity, dim)
        slot_keys_summary = jnp.mean(keys, axis=1) # (capacity, dim)
        slot_keys_norm = slot_keys_summary / jnp.sqrt(jnp.sum(slot_keys_summary**2, axis=-1, keepdims=True) + 1e-8)
        
        # Fact summary for matching (batch, dim)
        fact_summary = jnp.mean(fact_tokens, axis=1)
        fact_summary_norm = fact_summary / jnp.sqrt(jnp.sum(fact_summary**2, axis=-1, keepdims=True) + 1e-8)
        
        sim = jnp.matmul(fact_summary_norm, slot_keys_norm.T) # (batch, capacity)
        
        def write_single(args):
            f_tokens, f_ids, single_sim = args
            max_sim_idx = jnp.argmax(single_sim)
            max_sim = single_sim[max_sim_idx]
            
            is_update = max_sim >= self.config.memory_write_threshold
            
            empty_mask = (state == STATE_EMPTY) | (state == STATE_EXPIRED) | (state == STATE_DORMANT)
            first_empty = jnp.argmax(empty_mask)
            has_empty = jnp.any(empty_mask)
            
            target_idx = jnp.where(is_update, max_sim_idx, jnp.where(has_empty, first_empty, jnp.argmin(self.mem_importance.value)))
            
            # Write N tokens and IDs to slot
            new_keys = keys.at[target_idx].set(f_tokens)
            new_vals = vals.at[target_idx].set(f_tokens)
            new_token_ids = token_ids.at[target_idx].set(f_ids)
            new_state = state.at[target_idx].set(STATE_ACTIVE)
            
            return new_keys, new_vals, new_token_ids, new_state
            
        for b in range(batch_size):
            new_k, new_v, new_tids, new_s = write_single((fact_tokens[b], fact_token_ids[b], sim[b]))
            keys, vals, token_ids, state = new_k, new_v, new_tids, new_s
            
        self.mem_keys.value = keys
        self.mem_vals.value = vals
        self.mem_token_ids.value = token_ids
        self.mem_state.value = state
        
        return sim

    def read(self, query_summary):
        """
        Input Read: query_summary [batch, dim]
        Output Read: Top-K Slot * N_fact_tokens -> [batch, K * N_fact_tokens, dim]
        """
        batch_size, dim = query_summary.shape
        keys = self.mem_keys.value # (capacity, n_tok, dim)
        vals = self.mem_vals.value # (capacity, n_tok, dim)
        token_ids = self.mem_token_ids.value # (capacity, n_tok)
        state = self.mem_state.value
        top_k = self.config.memory_top_k
        n_tok = self.config.tokens_per_slot
        
        slot_keys_summary = jnp.mean(keys, axis=1) # (capacity, dim)
        slot_norm_sq = jnp.sum(slot_keys_summary**2, axis=-1, keepdims=True)
        q_norm_sq = jnp.sum(query_summary**2, axis=-1, keepdims=True)
        
        slot_keys_norm = slot_keys_summary / jnp.sqrt(slot_norm_sq + 1e-8)
        q_norm = query_summary / jnp.sqrt(q_norm_sq + 1e-8)
        
        sim = jnp.matmul(q_norm, slot_keys_norm.T) * self.sim_weight # (batch, capacity)
        
        valid_mask = (state == STATE_ACTIVE)
        masked_sim = jnp.where(valid_mask[None, :], sim, -1e9)
        
        # Ambil Top-K Slot indices
        top_k_indices = jnp.argsort(masked_sim, axis=-1)[:, -top_k:] # (batch, top_k)
        
        # Ambil token representasi dari Top-K Slot
        def get_top_k_tokens_and_ids(batch_indices):
            # batch_indices: (top_k,)
            selected_vals = vals[batch_indices] # (top_k, n_tok, dim)
            selected_ids = token_ids[batch_indices] # (top_k, n_tok)
            return selected_vals.reshape((top_k * n_tok, dim)), selected_ids.reshape((top_k * n_tok,))
            
        retrieved_memory, retrieved_ids = jax.vmap(get_top_k_tokens_and_ids)(top_k_indices) # (batch, top_k * n_tok, dim), (batch, top_k * n_tok)
        
        return retrieved_memory, retrieved_ids, sim

    def __call__(self, query_summary, read_prob, write_prob, deterministic=False):
        return self.read(query_summary)
