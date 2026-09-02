Memory Bank Architecture
========================

Source:
`/home/akhyar/Dokumen/Code/python/MemoryBank/mamoe/memory/bank.py`

Main class:
`MemoryBank` (flax.linen.Module)

Files:
- `mamoe/memory/bank.py`
- `mamoe/configuration_mamoe.py`

Components:
- Key embeddings (`mem_keys`: capacity x dim)
- Value embeddings (`mem_vals`: capacity x dim)
- Importance score (`mem_importance`: capacity)
- Confidence score (`mem_confidence`: capacity)
- Creation step (`mem_created_at`: capacity)
- Last access step (`mem_last_access`: capacity)
- Access count (`mem_access_count`: capacity)
- State (`mem_state`: capacity, 0=EXPIRED, 1=ACTIVE, 2=DORMANT)
- Global step tracker (`global_step`: scalar)
- Projections (`q_proj`, `k_proj`, `v_proj`, `i_proj`, `fusion_proj`)

Input:
`h_eos`: representations at end-of-sequence (batch_size, hidden_size)
`is_eos`: eos mask (batch_size)
`read_prob`: probability gating read
`write_prob`: probability gating write

Output:
`fused_h`: memory-fused representation `W_f[h_eos; m]`

Write:
1. Generate `k_new`, `v_new`, `i_new` using dense projections on `h_eos`.
2. Confidence is initialized to 0.5.
3. Search for nearest memory using cosine similarity.
4. If similarity >= threshold (update):
   - interpolate value based on confidence: `v_new = (1-conf)*v_old + conf*v_new`
   - update confidence `clip(conf + 0.1, 0, 1)`
5. If similarity < threshold (insert):
   - find an EXPIRED slot, or the DORMANT slot with lowest score if no EXPIRED.
   - insert new memory in slot.
   - reset `created_at` and `access_count`.
6. Update `keys`, `vals`, `importance`, `confidence`, `state`, `last_access` for target index.

Read:
1. Query projection `q_proj(h_eos)`.
2. Cosine similarity between `q` and `keys`.
3. Compute recency `R = e^(-lambda * dt)`.
4. Score = `alpha * sim + beta * importance + gamma * recency + delta * confidence`.
5. Mask out EXPIRED slots.
6. Top-K Selection based on score.
7. Filter by `memory_threshold` (tau).
8. Compute weighted aggregation (Softmax).
9. Reinforcement: boost access count, update last access step, boost importance `clip(I + eta_a, 0, 1)`.

Update:
Implemented implicitly as part of the Write operation (see Write step 4).

Retrieval:
Implemented as part of Read operation using Cosine similarity + Top-K + Relevance thresholding.

State:
Maintained via time tracking (step difference). 
Decay calculation is explicitly invoked before Read/Write:
- `effective_R = R * (1 + rho * I)`
- If `effective_R < 0.1` -> EXPIRED
- If `effective_R < 0.5` -> DORMANT
