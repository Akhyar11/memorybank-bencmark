"""
TinyMemoryBank – Locked Architecture Implementation
=====================================================
ARCHITECTURE LOCK: DO NOT MODIFY COMPONENT NAMES OR LOGIC.
Source reference: /home/akhyar/Dokumen/Code/python/MemoryBank/mamoe/memory/bank.py
Locked components: q_proj, k_proj, v_proj, i_proj, fusion_proj,
                   mem_keys, mem_vals, mem_importance, mem_confidence,
                   mem_created_at, mem_last_access, mem_access_count,
                   mem_state, global_step
Locked flow: decay_memory → read → write → fuse → __call__
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# State Constants (LOCKED – do not add or rename)
# ---------------------------------------------------------------------------
STATE_EXPIRED = 0   # Memory slot yang sudah kadaluarsa
STATE_ACTIVE  = 1   # Memory slot aktif/fresh
STATE_DORMANT = 2   # Memory slot dormant (belum expired tapi tidak aktif)


@dataclass
class TinyMemoryConfig:
    """Configuration for TinyMemoryBank.

    All fields with LOCKED suffix correspond to architecture-locked parameters.
    allowed_changes: memory_capacity, memory_dim, hidden_size (per architecture_lock.json)
    """
    # Allowed to change
    memory_capacity: int = 128
    memory_dim: int = 32
    hidden_size: int = 32

    # Decay parameters
    mem_decay_rate: float = 0.001           # lambda – decay rate
    mem_importance_protection: float = 0.5  # rho – importance protection factor

    # Score weights (alpha, beta, gamma, delta)
    mem_alpha: float = 1.0    # weight for cosine similarity
    mem_beta: float = 0.5     # weight for importance
    mem_gamma: float = 0.1    # weight for recency
    mem_delta: float = 0.2    # weight for confidence

    # Retrieval params
    memory_top_k: int = 4
    memory_threshold: float = -1e9      # tau – relevance threshold for read (allow all by default)
    memory_read_threshold: float = 0.0  # gate threshold for read_prob
    memory_write_threshold: float = 0.0 # tau – threshold for write gate

    # Reinforcement
    mem_reinforcement_rate: float = 0.05  # eta_a – importance boost on access


class TinyMemoryBank(nn.Module):
    """
    Locked Neural Memory Bank.

    Pipeline (LOCKED):
        decay_memory → read → fuse   (in __call__)
        write is called externally before __call__

    Stores 1D vector representations per slot (capacity × dim).
    Uses learned projections: q_proj, k_proj, v_proj, i_proj, fusion_proj.

    Write semantics:
        k = k_proj(h_eos)
        v = v_proj(h_eos)
        i = sigmoid(i_proj(h_eos))
        similarity search → UPDATE existing or INSERT new
        update importance/confidence/state/timestamps

    Read semantics:
        q = q_proj(h_eos)
        score = alpha*cos(q,k) + beta*importance + gamma*recency + delta*confidence
        mask EXPIRED slots
        top-k selection
        threshold filter
        weighted softmax aggregation
        access reinforcement (count, last_access, importance boost)

    Fuse semantics:
        fused = fusion_proj(concat([h_eos, retrieved_memory]))
    """
    config: TinyMemoryConfig

    def setup(self):
        capacity = self.config.memory_capacity
        dim      = self.config.memory_dim

        # --- Locked Memory State Tensors ---
        self.mem_keys        = self.variable('memory', 'keys',         jnp.zeros, (capacity, dim))
        self.mem_vals        = self.variable('memory', 'vals',         jnp.zeros, (capacity, dim))
        self.mem_importance  = self.variable('memory', 'importance',   jnp.zeros, (capacity,))
        self.mem_confidence  = self.variable('memory', 'confidence',   jnp.zeros, (capacity,))
        self.mem_created_at  = self.variable('memory', 'created_at',   jnp.zeros, (capacity,), jnp.int32)
        self.mem_last_access = self.variable('memory', 'last_access',  jnp.zeros, (capacity,), jnp.int32)
        self.mem_access_count= self.variable('memory', 'access_count', jnp.zeros, (capacity,), jnp.int32)
        # state: 0=EXPIRED, 1=ACTIVE, 2=DORMANT  (initial=EXPIRED → all slots empty)
        self.mem_state       = self.variable('memory', 'state',        jnp.zeros, (capacity,), jnp.int32)
        self.global_step     = self.variable('memory', 'global_step',  jnp.zeros, (), jnp.int32)

        # --- Locked Learned Projections ---
        self.q_proj      = nn.Dense(dim, use_bias=False, name='q_proj')
        self.k_proj      = nn.Dense(dim, use_bias=False, name='k_proj')
        self.v_proj      = nn.Dense(dim, use_bias=False, name='v_proj')
        self.i_proj      = nn.Dense(1,                  name='i_proj')          # importance logit
        # fusion: W_f[h ; m] → hidden_size
        self.fusion_proj = nn.Dense(self.config.hidden_size, use_bias=False, name='fusion_proj')

    # -----------------------------------------------------------------------
    # LOCKED OPERATION 1: decay_memory
    # -----------------------------------------------------------------------
    def decay_memory(self):
        """
        Calculates effective decay and transitions memory states.

        Formula (LOCKED):
            R = exp(-lambda * dt)
            effective_R = R * (1 + rho * importance)

            effective_R < 0.1  → STATE_EXPIRED
            effective_R < 0.5  → STATE_DORMANT
            otherwise          → unchanged (ACTIVE stays ACTIVE)
        """
        step        = self.global_step.value
        last_access = self.mem_last_access.value
        importance  = self.mem_importance.value
        state       = self.mem_state.value

        dt = jnp.maximum(step - last_access, 0)

        lam = self.config.mem_decay_rate
        R   = jnp.exp(-lam * dt.astype(jnp.float32))

        rho         = self.config.mem_importance_protection
        effective_R = R * (1.0 + rho * importance)

        is_expired = effective_R < 0.1
        is_dormant = (effective_R < 0.5) & (~is_expired)

        new_state = jnp.where(is_expired, STATE_EXPIRED, state)
        new_state = jnp.where(is_dormant, STATE_DORMANT, new_state)

        self.mem_state.value = new_state
        return effective_R

    def empty_memory_state(self):
        """
        Creates an explicitly empty memory state.
        This guarantees that no dummy data from init() leaks into the runtime state.
        """
        cap = self.config.memory_capacity
        dim = self.config.memory_dim
        return {
            'keys':         jnp.zeros((cap, dim),  dtype=jnp.float32),
            'vals':         jnp.zeros((cap, dim),  dtype=jnp.float32),
            'importance':   jnp.zeros((cap,),      dtype=jnp.float32),
            'confidence':   jnp.zeros((cap,),      dtype=jnp.float32),
            'created_at':   jnp.zeros((cap,),      dtype=jnp.int32),
            'last_access':  jnp.zeros((cap,),      dtype=jnp.int32),
            'access_count': jnp.zeros((cap,),      dtype=jnp.int32),
            'state':        jnp.full((cap,), STATE_EXPIRED, dtype=jnp.int32),
            'global_step':  jnp.zeros((),          dtype=jnp.int32),
        }

    # -----------------------------------------------------------------------
    # LOCKED OPERATION 2: read
    # -----------------------------------------------------------------------
    def read(self, h_eos: jax.Array, read_prob: jax.Array = None) -> jax.Array:
        """
        Read from memory.

        Args:
            h_eos: (batch_size, hidden_size)
            read_prob: (batch_size,) optional gating probability. 
                       If provided, reinforcement side-effects are disabled when read_prob=0.

        Returns:
            read_result: (batch_size, memory_dim)
        """
        step = self.global_step.value
        cfg  = self.config

        # 1. Query projection (LOCKED: must use q_proj)
        q    = self.q_proj(h_eos)       # (batch, dim)
        keys = self.mem_keys.value       # (capacity, dim)
        vals = self.mem_vals.value       # (capacity, dim)

        importance  = self.mem_importance.value   # (capacity,)
        confidence  = self.mem_confidence.value   # (capacity,)
        last_access = self.mem_last_access.value  # (capacity,)
        state       = self.mem_state.value         # (capacity,)

        # 2. Cosine similarity (LOCKED)
        q_norm = q    / (jnp.linalg.norm(q,    axis=-1, keepdims=True) + 1e-8)
        k_norm = keys / (jnp.linalg.norm(keys, axis=-1, keepdims=True) + 1e-8)
        sim = jnp.matmul(q_norm, k_norm.T)  # (batch, capacity)

        # 3. Recency (LOCKED)
        dt      = jnp.maximum(step - last_access, 0).astype(jnp.float32)
        recency = jnp.exp(-cfg.mem_decay_rate * dt)  # (capacity,)

        # 4. Broadcast metadata to (batch, capacity)
        importance_bd = jnp.broadcast_to(importance[None, :], sim.shape)
        confidence_bd = jnp.broadcast_to(confidence[None, :], sim.shape)
        recency_bd    = jnp.broadcast_to(recency[None, :],    sim.shape)

        # 5. Score formula (LOCKED)
        score = (cfg.mem_alpha * sim
               + cfg.mem_beta  * importance_bd
               + cfg.mem_gamma * recency_bd
               + cfg.mem_delta * confidence_bd)

        # 6. Mask EXPIRED slots
        active_mask = (state != STATE_EXPIRED)[None, :]  # (1, capacity)
        score = jnp.where(active_mask, score, -1e9)

        # 7. Top-K selection (LOCKED)
        k = cfg.memory_top_k
        topk_scores, topk_indices = jax.lax.top_k(score, k)  # (batch, k)

        # 8. Threshold filter (LOCKED: tau = memory_threshold)
        tau        = cfg.memory_threshold
        valid_mask = topk_scores > tau  # (batch, k)
        
        # Also ensure that EXPIRED slots are strictly invalid (score=-1e9)
        valid_mask = jnp.logical_and(valid_mask, topk_scores > -1e8)

        # Gate read effects if read_prob is provided and 0
        if read_prob is not None:
            is_read = read_prob > cfg.memory_read_threshold
            valid_mask = jnp.logical_and(valid_mask, is_read[:, None])

        # 9. Softmax aggregation (set invalid to -1e9 before softmax)
        filtered_scores = jnp.where(valid_mask, topk_scores, -1e9)
        attn_weights    = jax.nn.softmax(filtered_scores, axis=-1)

        # Zero out completely masked contributions
        attn_weights = attn_weights * valid_mask.astype(attn_weights.dtype)

        # 10. Weighted aggregation
        def gather_vals(indices):
            return vals[indices]  # (k, dim)
        topk_vals = jax.vmap(gather_vals)(topk_indices)  # (batch, k, dim)

        attn_sum    = jnp.sum(attn_weights, axis=-1, keepdims=True)  # (batch, 1)
        read_result = jnp.sum(attn_weights[..., None] * topk_vals, axis=1)  # (batch, dim)
        # If no valid memories exist, result must be zero vector
        read_result = jnp.where(attn_sum > 0, read_result, jnp.zeros_like(read_result))

        # 11. Access Reinforcement (LOCKED)
        # Update last_access, access_count, importance for valid retrieved slots
        eta_a = cfg.mem_reinforcement_rate

        def update_reinforcement(state_tuple, inputs):
            last_acc, acc_count, imp = state_tuple
            indices, valid = inputs  # indices: (k,), valid: (k,) bool

            def update_slot(carry, x):
                la, ac, im = carry
                idx, is_valid = x
                la = la.at[idx].set(jnp.where(is_valid, step, la[idx]))
                ac = ac.at[idx].add(jnp.where(is_valid, 1, 0).astype(jnp.int32))
                new_i = jnp.clip(im[idx] + eta_a, 0.0, 1.0)
                im = im.at[idx].set(jnp.where(is_valid, new_i, im[idx]))
                return (la, ac, im), None

            (la, ac, im), _ = jax.lax.scan(
                update_slot, (last_acc, acc_count, imp),
                (indices, valid.astype(jnp.bool_))
            )
            return (la, ac, im), None

        init_state = (
            self.mem_last_access.value,
            self.mem_access_count.value,
            self.mem_importance.value,
        )
        (new_last_acc, new_acc_cnt, new_imp), _ = jax.lax.scan(
            update_reinforcement, init_state, (topk_indices, valid_mask)
        )

        self.mem_last_access.value  = new_last_acc
        self.mem_access_count.value = new_acc_cnt
        self.mem_importance.value   = new_imp

        return read_result

    # -----------------------------------------------------------------------
    # LOCKED OPERATION 3: write
    # -----------------------------------------------------------------------
    def write(self, h_eos: jax.Array, is_eos: jax.Array, write_prob: jax.Array):
        """
        Write h_eos representations to memory.

        Args:
            h_eos:      (batch_size, hidden_size)  – encoded representations
            is_eos:     (batch_size,)               – EOS gate (1=end-of-sequence)
            write_prob: (batch_size,)               – write probability gate

        Write logic per sample (LOCKED):
            k_new = k_proj(h)
            v_new = v_proj(h)
            i_new = sigmoid(i_proj(h))
            c_new = 0.5  (initial confidence)

            sim = cosine(k_new, stored_keys)  → restrict to ACTIVE/DORMANT
            if sim >= threshold:
                UPDATE existing slot (interpolate value, boost confidence)
            else:
                INSERT to EXPIRED/DORMANT slot (or lowest-importance ACTIVE)
                reset created_at, access_count = 1

        Sequential writes via lax.scan ensure each write sees current state.
        """
        step = self.global_step.value
        cfg  = self.config

        # Project inputs (LOCKED: must use k_proj, v_proj, i_proj)
        k_new = self.k_proj(h_eos)                                    # (batch, dim)
        v_new = self.v_proj(h_eos)                                    # (batch, dim)
        i_new = jax.nn.sigmoid(jnp.squeeze(self.i_proj(h_eos), -1))  # (batch,)
        c_new = jnp.ones_like(i_new) * 0.5                           # (batch,) default confidence

        # Normalize new keys for similarity search
        k_new_norm = k_new / (jnp.linalg.norm(k_new, axis=-1, keepdims=True) + 1e-8)

        # Load current memory state
        keys      = self.mem_keys.value        # (capacity, dim)
        vals      = self.mem_vals.value
        state     = self.mem_state.value
        imp       = self.mem_importance.value
        conf      = self.mem_confidence.value
        created   = self.mem_created_at.value
        last_acc  = self.mem_last_access.value
        acc_cnt   = self.mem_access_count.value

        k_norm = keys / (jnp.linalg.norm(keys, axis=-1, keepdims=True) + 1e-8)

        def update_single_write(state_tuple, inputs):
            """Process one sample from the batch sequentially."""
            (keys_, vals_, state_, imp_, conf_, created_, last_acc_, acc_cnt_, k_norm_) = state_tuple
            k_n, v_n, i_n, c_n, is_e, w_p = inputs

            # Gate: only write if is_eos>0 AND write_prob >= threshold
            do_write = jnp.logical_and(is_e > 0.5, w_p >= cfg.memory_write_threshold)

            # Cosine similarity against stored keys (LOCKED)
            sim_ = jnp.dot(k_norm_, k_n)  # (capacity,)

            # Restrict search to ACTIVE or DORMANT slots
            valid_search = (state_ != STATE_EXPIRED)
            sim_masked   = jnp.where(valid_search, sim_, -1.0)

            max_sim     = jnp.max(sim_masked)
            nearest_idx = jnp.argmax(sim_masked)

            # UPDATE vs INSERT decision:
            # tau is the similarity threshold to update instead of insert.
            tau = cfg.memory_write_threshold
            is_update = jnp.logical_and(max_sim >= tau, valid_search[nearest_idx])

            # --- UPDATE branch ---
            # v_new = (1 - conf_old) * v_old + conf_old * v_candidate  (LOCKED)
            eta        = conf_[nearest_idx]
            updated_v  = (1.0 - eta) * vals_[nearest_idx] + eta * v_n
            updated_c  = jnp.clip(conf_[nearest_idx] + 0.1, 0.0, 1.0)

            # --- INSERT branch ---
            # Priority: EXPIRED (sort_key=0) < DORMANT (sort_key=1) < ACTIVE (sort_key=2)
            sort_keys = jnp.where(
                state_ == STATE_EXPIRED, 0,
                jnp.where(state_ == STATE_DORMANT, 1, 2)
            )
            # Tie-break by lowest importance for ACTIVE slots
            tiebreak   = imp_ / (jnp.max(imp_) + 1e-8) * 0.5  # small fraction
            sort_score = sort_keys.astype(jnp.float32) + tiebreak
            insert_idx = jnp.argmin(sort_score)

            target_idx = jnp.where(is_update, nearest_idx, insert_idx)

            # Apply update conditionally using do_write gate
            # Keys: always write k_new (LOCKED: key projection stored)
            keys_ = keys_.at[target_idx].set(
                jnp.where(do_write, k_n, keys_[target_idx])
            )
            # Values
            new_v  = jnp.where(is_update, updated_v, v_n)
            vals_  = vals_.at[target_idx].set(
                jnp.where(do_write, new_v, vals_[target_idx])
            )
            # Importance: on update → max(old, new); on insert → i_new
            new_i  = jnp.where(is_update, jnp.maximum(imp_[nearest_idx], i_n), i_n)
            imp_   = imp_.at[target_idx].set(
                jnp.where(do_write, new_i, imp_[target_idx])
            )
            # Confidence
            new_c  = jnp.where(is_update, updated_c, c_n)
            conf_  = conf_.at[target_idx].set(
                jnp.where(do_write, new_c, conf_[target_idx])
            )
            # State → ACTIVE
            state_ = state_.at[target_idx].set(
                jnp.where(do_write, STATE_ACTIVE, state_[target_idx])
            )
            # last_access = current step
            last_acc_ = last_acc_.at[target_idx].set(
                jnp.where(do_write, step, last_acc_[target_idx])
            )
            # created_at: reset only on INSERT
            new_created = jnp.where(is_update, created_[target_idx], step)
            created_    = created_.at[target_idx].set(
                jnp.where(do_write, new_created, created_[target_idx])
            )
            # access_count: increment on update, set=1 on insert
            new_acc = jnp.where(is_update, acc_cnt_[target_idx] + 1, 1)
            acc_cnt_ = acc_cnt_.at[target_idx].set(
                jnp.where(do_write, new_acc.astype(jnp.int32), acc_cnt_[target_idx])
            )

            # Recompute k_norm after write so next iteration sees fresh state
            k_norm_updated = keys_ / (jnp.linalg.norm(keys_, axis=-1, keepdims=True) + 1e-8)

            new_state = (keys_, vals_, state_, imp_, conf_, created_, last_acc_, acc_cnt_, k_norm_updated)
            return new_state, None

        init_state = (keys, vals, state, imp, conf, created, last_acc, acc_cnt, k_norm)
        (new_keys, new_vals, new_state, new_imp, new_conf,
         new_created, new_last_acc, new_acc_cnt, _), _ = jax.lax.scan(
            update_single_write, init_state,
            (k_new_norm, v_new, i_new, c_new, is_eos, write_prob)
        )

        self.mem_keys.value        = new_keys
        self.mem_vals.value        = new_vals
        self.mem_state.value       = new_state
        self.mem_importance.value  = new_imp
        self.mem_confidence.value  = new_conf
        self.mem_created_at.value  = new_created
        self.mem_last_access.value = new_last_acc
        self.mem_access_count.value= new_acc_cnt

    # -----------------------------------------------------------------------
    # LOCKED OPERATION 4: fuse
    # -----------------------------------------------------------------------
    def fuse(self, h: jax.Array, m: jax.Array) -> jax.Array:
        """
        Fuse h_eos with retrieved memory (LOCKED).

        Formula: fused = fusion_proj(concat([h, m]))
        Args:
            h: (batch, hidden_size)
            m: (batch, memory_dim)
        Returns:
            fused: (batch, hidden_size)
        """
        concatenated = jnp.concatenate([h, m], axis=-1)
        fused        = self.fusion_proj(concatenated)
        return fused

    def init_all(self, h_eos: jax.Array):
        _ = self.k_proj(h_eos)
        _ = self.v_proj(h_eos)
        _ = self.i_proj(h_eos)

    # -----------------------------------------------------------------------
    # LOCKED OPERATION 5: __call__ (decay → read → fuse)
    # -----------------------------------------------------------------------
    def __call__(self, h_eos: jax.Array, read_prob: jax.Array,
                 write_prob: jax.Array, deterministic: bool = False) -> jax.Array:
        """
        Full memory pipeline (LOCKED order: decay → read → fuse).

        NOTE: This also touches k_proj/v_proj/i_proj with zero-gated outputs
        to ensure all parameters are registered during init().

        Args:
            h_eos:       (batch, hidden_size)
            read_prob:   (batch,)  – gate for read operation
            write_prob:  (batch,)  – unused here (write is external)
            deterministic: if True, use hard threshold gate; else soft gate

        Returns:
            fused_h: (batch, hidden_size)
        """
        # Step the global clock
        self.global_step.value = self.global_step.value + 1

        # 1. Decay
        self.decay_memory()

        # 2. Read
        read_val = self.read(h_eos, read_prob=read_prob)

        # 3. Gating
        if deterministic:
            is_read = read_prob > self.config.memory_read_threshold
            m_eff   = jnp.where(is_read[:, None], read_val, jnp.zeros_like(read_val))
        else:
            m_eff = read_val * read_prob[:, None]

        # 4. Fuse (LOCKED)
        fused_h = self.fuse(h_eos, m_eff)

        # 5. Ensure k_proj/v_proj/i_proj parameters are registered during init.
        # These are multiplied by 0.0 so they have no effect on output.
        # Without this, Flax does not register them because write() is not in __call__.
        _k = self.k_proj(h_eos) * 0.0
        _v = self.v_proj(h_eos) * 0.0
        _i = self.i_proj(h_eos) * 0.0
        fused_h = fused_h + _k[:, :fused_h.shape[-1]] * 0.0 + _v[:, :fused_h.shape[-1]] * 0.0

        return fused_h
