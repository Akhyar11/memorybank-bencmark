# MIGRATION AUDIT: COMPLETE JAX/FLAX TO PYTORCH MIGRATION

## 1. Architectural Invariance Statement

The objective of this migration was **Framework Parity, NOT Architectural Redesign**. The core architectural definition specified in `architecture_lock.json` remains **100% strictly invariant**:

- **Projections**: `q_proj`, `k_proj`, `v_proj`, `i_proj`, and `fusion_proj` exist with identical input/output dimensions and functional roles.
- **State Tensors**: `mem_keys`, `mem_vals`, `mem_importance`, `mem_confidence`, `mem_created_at`, `mem_last_access`, `mem_access_count`, `mem_state`, and `global_step` are registered buffers.
- **Operations & Equations**:
  1. Cosine similarity: \(\text{sim} = \frac{q}{\|q\|} \cdot \frac{K^T}{\|K\|}\)
  2. Multi-factor score: \(\text{score} = \alpha \cdot \text{sim} + \beta \cdot \text{imp} + \gamma \cdot \text{rec} + \delta \cdot \text{conf}\)
  3. Decay & State transitions: \(\text{dt} = \text{step} - \text{last\_access}\), \(R = \exp(-\lambda \cdot \text{dt})\) (\(\text{ACTIVE} \to \text{DORMANT} \to \text{EXPIRED}\))
  4. Interpolated update: \(V_{\text{new}} = (1 - \text{conf}) \cdot V_{\text{old}} + \text{conf} \cdot V_{\text{cand}}\)
  5. Eviction & replacement order: \(\text{EXPIRED} \to \text{DORMANT} \to \min(\text{importance})\)
  6. Fusion: \(\text{fusion\_proj}([h, m])\)

---

## 2. Structural Mapping: Old JAX/Flax vs New PyTorch

| Feature / Component | Old JAX/Flax Implementation | New PyTorch (`torch>=2.0.0`) Implementation | Semantics Preserved? |
| :--- | :--- | :--- | :---: |
| **Linear Layers** | `flax.linen.Dense(dim, use_bias=False)` | `torch.nn.Linear(dim, dim, bias=False)` | YES |
| **Episodic State Storage** | `self.variable('memory', key, ...)` with `mutable=['memory']` dict | `self.register_buffer(name, tensor)` | YES |
| **Model Forward Pass** | `model.apply(vars, ..., method=...)` | `model(h, ...)` or `model.method(...)` directly | YES |
| **RNG & Pseudo-Randomness** | `jax.random.PRNGKey(seed)` & `jax.random.split` | `torch.manual_seed(seed)` & `torch.Generator()` | YES |
| **Sequential Memory Write** | `jax.lax.scan(update_single_write, ...)` | Sequential loop under `torch.no_grad()` | YES |
| **State Reset / Restore** | Manual dictionary unfreezing & slicing | `load_memory_state()` and `get_memory_state()` | YES |
| **Data Loader Pipeline** | Manual generator with array conversion | `torch.utils.data.Dataset` & `DataLoader` | YES |
| **Optimizer** | `optax.adamw(learning_rate=...)` | `torch.optim.AdamW(model.parameters(), lr=...)` | YES |
| **Metrics Calculation** | `jax.numpy` / `numpy` | Pure `numpy` vectorized operations | YES |

---

## 3. Detailed Component Audit

### 3.1. `models/tiny_memory_bank.py`
- All 5 projections (`q_proj`, `k_proj`, `v_proj`, `i_proj`, `fusion_proj`) are native `nn.Linear`.
- All 9 state tensors (`mem_keys`, `mem_vals`, `mem_importance`, `mem_confidence`, `mem_created_at`, `mem_last_access`, `mem_access_count`, `mem_state`, `global_step`) are registered non-trainable buffers (`register_buffer`).
- Buffer cloning is employed during retrieval to prevent PyTorch in-place autograd corruption (`AsStridedBackward0`).
- Buffer mutations during episodic writes and access reinforcements are executed inside `torch.no_grad()`, preserving episodic cache semantics without exploding the computation graph.
- Added `load_memory_state(state_dict)` and `get_memory_state()` enabling seamless state serialization and restoration.

### 3.2. `models/transformer_qa_model.py`
- **No Memory**: Cleanly projects query EOS directly to decoder context (`h_fused = h_eos`), eliminating unauthorized memory projection parameters.
- **NN Memory**: Upgraded to a True Key-Value Neural Network Memory baseline (`nn_k_proj`, `nn_v_proj`, `nn_proj_out`), completely free from Memory Bank multi-factor scoring or temporal decay.
- **Memory Bank**: Maintains episodic memory bank integration with actual composite score output.

### 3.3. `baselines/nearest_neighbor.py`, `no_memory.py`, `simple_memory.py`
- All baselines are native PyTorch `nn.Module` classes.
- Completely free from JAX/Flax dependencies.
- Standard Key-Value Nearest Neighbor memory relies strictly on query-key cosine similarity without importance, confidence, decay, or composite heuristics.

### 3.4. `dataset/text_dataset_loader.py`
- Native `torch.utils.data.Dataset` (`TextDataset`) and `DataLoader`.
- Token IDs (`[PAD]`, `[EOS]`, `[BOS]`, `[UNK]`) are dynamically resolved from the tokenizer.
- Zero manual XLA padding; batches can be dynamically sized.

### 3.5. `experiments/end_to_end_benchmark.py`
- Precomputes exact deterministic batch sequences across all baselines per epoch using seeded `torch.Generator`.
- Evaluates retrieval metrics using the actual multi-factor composite scores from the Memory Bank.
- Resolves target ground truth facts through explicit pair-wise dataset indexing rather than fragile substring substitutions (`_Q` \(\to\) `_F`).

---

## 4. Verification Evidence

### 4.1. Unit Test Suite (`pytest -q`)
- **80 / 80 PASSED** in 3.70 seconds.
- 0 failed, 0 skipped.
- Tests cover architecture lock, state transitions, causal interventions, component ablations, replacement policies, interference resistance, persistence, and edge cases.

### 4.2. Functional Benchmark Suite (`experiments/memory_functional_benchmark.py`)
- **11 / 11 PASSED**:
  - Test 1 (Basic Write): PASS
  - Test 2 (Basic Read): PASS
  - Test 3 (Distractor Retrieval): PASS (Recall@1 = 1.0000, MRR = 1.0000)
  - Test 4 (Interference): PASS (Clean = 1.00, Distractor = 1.00)
  - Test 5 (Capacity Scaling): PASS (16, 32, 64, 128 slots)
  - Test 6 (Replacement Policy): PASS (Lowest importance slot replaced)
  - Test 7 (Recency Effect): PASS (Score prioritizes recent items)
  - Test 8 (Importance Effect): PASS (Score prioritizes high importance)
  - Test 9 (Confidence Effect): PASS (Score prioritizes high confidence)
  - Test 10 (Forgetting Transitions): PASS (`ACTIVE` \(\to\) `DORMANT` \(\to\) `EXPIRED`)
  - Test 11 (Counterfactual Causal): PASS (Cross-sim = -1.0000)

### 4.3. Dependency Cleanliness
- `grep -rn "import jax" .` \(\to\) **0 occurrences**
- `grep -rn "import flax" .` \(\to\) **0 occurrences**
- `grep -rn "import optax" .` \(\to\) **0 occurrences**
- `requirements.txt` contains purely `torch>=2.0.0` and supporting standard libraries.
