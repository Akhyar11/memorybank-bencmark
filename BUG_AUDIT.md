# BUG_AUDIT.md – MemoryBank-Benchmark Bug Inventory

Generated from: Complete repository audit (2026-09-02)

---

## P0 – Fatal / Invalid Experiment / Runtime Failure

### BUG-P0-001
- **File**: `models/tiny_memory_bank.py`
- **Function**: `setup()`
- **Problem**: No `q_proj`, `k_proj`, `v_proj`, `i_proj`, `fusion_proj` in TinyMemoryBank.
- **Why wrong**: Violates architecture lock. All five projections are locked components.
- **Scientific impact**: Memory Bank cannot be the locked architecture; entire benchmark is invalid.
- **Runtime impact**: `adapter.get_v_proj()` crashes.
- **Fix**: Added all five Dense projection layers in `setup()`.
- **Validation**: `test_architecture_lock.py::test_locked_projections`
- **STATUS**: FIXED ✓

### BUG-P0-002
- **File**: `models/tiny_memory_bank.py`
- **Function**: `write()`
- **Problem**: `new_vals = vals.at[target_idx].set(f_tokens)` – stores raw `fact_tokens`, not `v_proj(h)`.
- **Why wrong**: Architecture lock requires `v = v_proj(h)`.
- **Scientific impact**: Memory content is not a learned projection → meaningless retrieval.
- **Fix**: `write()` now calls `k_proj`, `v_proj`, `i_proj` before storing.
- **STATUS**: FIXED ✓

### BUG-P0-003
- **File**: `models/tiny_memory_bank.py`
- **Function**: `read()`
- **Problem**: Uses raw `query_summary` for cosine similarity, not `q = q_proj(query_summary)`.
- **Why wrong**: Architecture lock requires `q = q_proj(h_eos)`.
- **Fix**: `read()` now calls `q = self.q_proj(h_eos)` first.
- **STATUS**: FIXED ✓

### BUG-P0-004
- **File**: `models/tiny_memory_bank.py`
- **Problem**: No `decay_memory()` method.
- **Why wrong**: Locked flow: `decay_memory → read → write → fuse`.
- **Fix**: Implemented `decay_memory()` with formula `effective_R = exp(-λ·dt)·(1 + ρ·I)`.
- **STATUS**: FIXED ✓

### BUG-P0-005
- **File**: `models/tiny_memory_bank.py`
- **Problem**: No `fuse()` method, no `fusion_proj`.
- **Fix**: Implemented `fuse(h, m)` using `fusion_proj(concat([h, m]))`.
- **STATUS**: FIXED ✓

### BUG-P0-006
- **File**: `models/tiny_memory_bank.py`
- **Function**: `__call__()`
- **Problem**: Only called `read()`, not full pipeline `decay → read → fuse`.
- **Fix**: `__call__` now runs full locked pipeline.
- **STATUS**: FIXED ✓

### BUG-P0-007
- **File**: `models/tiny_memory_bank.py`
- **Problem**: `STATE_EMPTY = 3` not in locked architecture (only EXPIRED=0, ACTIVE=1, DORMANT=2).
- **Fix**: Removed `STATE_EMPTY`. Initial state is `STATE_EXPIRED`.
- **STATUS**: FIXED ✓

### BUG-P0-008
- **File**: `models/tiny_memory_bank.py`
- **Problem**: `global_step` stored as `step` – wrong name per lock.
- **Fix**: Renamed to `global_step`.
- **STATUS**: FIXED ✓

### BUG-P0-009
- **File**: `models/tiny_memory_bank.py`
- **Function**: `write()`
- **Problem**: Python for-loop over batch items (`for b in range(batch_size)`) breaks JAX JIT.
- **Fix**: Replaced with `jax.lax.scan` for JIT-compatible sequential writes.
- **STATUS**: FIXED ✓

### BUG-P0-010
- **File**: `models/tiny_memory_bank.py`
- **Function**: `write()`
- **Problem**: `is_eos` and `write_prob` not used as gates. Writes happened unconditionally.
- **Fix**: `do_write = logical_and(is_eos > 0.5, write_prob >= threshold)` gate added.
- **STATUS**: FIXED ✓

### BUG-P0-011
- **File**: `models/tiny_memory_bank.py`
- **Function**: `write()`
- **Problem**: `importance`, `confidence`, `created_at`, `last_access`, `access_count` never updated.
- **Fix**: All metadata updated correctly in `lax.scan` write loop.
- **STATUS**: FIXED ✓

### BUG-P0-012
- **File**: `models/tiny_memory_bank.py`
- **Function**: `read()`
- **Problem**: `access_count`, `last_access`, `importance` never updated on read.
- **Fix**: Access reinforcement implemented via nested `lax.scan` in `read()`.
- **STATUS**: FIXED ✓

### BUG-P0-013
- **File**: `models/tiny_memory_bank.py`
- **Function**: `read()`
- **Problem**: Score formula absent. Only raw cosine similarity used.
- **Fix**: Score = `α·sim + β·importance + γ·recency + δ·confidence` implemented.
- **STATUS**: FIXED ✓

### BUG-P0-014
- **File**: `adapters/existing_memorybank.py`
- **Function**: `advance_time()`
- **Problem**: Looked for `memory['bank']['global_step']` – path doesn't exist → KeyError.
- **Fix**: Correct path: `memory['global_step']`.
- **STATUS**: FIXED ✓

### BUG-P0-015
- **File**: `adapters/existing_memorybank.py`
- **Function**: `get_memory_state()`
- **Problem**: `memory['bank']['state']` doesn't exist → KeyError.
- **Fix**: Correct path: `memory['state']`, `memory['importance']`, `memory['confidence']`.
- **STATUS**: FIXED ✓

### BUG-P0-016
- **File**: `adapters/existing_memorybank.py`
- **Function**: `setup()`
- **Problem**: Calls `module.get_v_proj(h_eos)` but `v_proj` didn't exist → AttributeError.
- **Fix**: `v_proj` now exists in TinyMemoryBank. `setup()` no longer does dummy writes.
- **STATUS**: FIXED ✓

### BUG-P0-017
- **File**: `models/tiny_model.py`
- **Function**: `__call__()`
- **Problem**: Only called `self.bank(h_eos, ...)` which just read, not full pipeline.
- **Fix**: `__call__` now delegates to full bank pipeline (decay→read→fuse).
- **STATUS**: FIXED ✓

### BUG-P0-018
- **File**: `adapters/existing_memorybank.py`
- **Function**: `reset_memory()`
- **Problem**: Re-ran `init_fn` which wrote dummy data → memory not truly empty after reset.
- **Fix**: `reset_memory()` now directly constructs zero/EXPIRED memory state.
- **STATUS**: FIXED ✓

### BUG-P0-019
- **File**: `scripts/train_text_qa.py`
- **Problem**: `val_loader = TextDataLoader(train_csv, ...)` – validation used train data!
- **Impact**: Validation metrics are completely invalid (train-set overfitting not detected).
- **Fix**: `val_loader = TextDataLoader(val_csv, ...)`.
- **STATUS**: FIXED ✓

### BUG-P0-020
- **File**: `scripts/train_text_qa.py`
- **Problem**: Training loss used `log(probabilities)` treatment but decoder returned logits.
- **Fix**: Explicit `cross_entropy_loss(logits, targets)` with `log_softmax` from logits.
- **STATUS**: FIXED ✓

### BUG-P0-021
- **File**: `scripts/train_text_qa.py`
- **Problem**: `oracle_loss` mixed into `total_loss` with weight 3.0. Oracle bypasses Memory Bank.
- **Impact**: Gradient from oracle (direct fact→decoder path) contaminates Memory Bank training.
- **Fix**: Oracle removed from `total_loss`. Kept as separate diagnostic metric only.
- **STATUS**: FIXED ✓

### BUG-P0-022
- **File**: `scripts/train_text_qa.py`
- **Problem**: `break` after first test batch → only ~32 samples evaluated!
- **Fix**: Removed `break`. Full test set evaluated; `n_test` counter reported.
- **STATUS**: FIXED ✓

### BUG-P0-023
- **File**: `dataset/text_dataset_loader.py`
- **Problem**: `df.head(2000)` hardcoded → silently truncates dataset.
- **Fix**: `max_samples=None` parameter. Default loads all data.
- **STATUS**: FIXED ✓

### BUG-P0-024
- **File**: `scripts/train_text_qa.py`
- **Problem**: `blank_memory` was a constant from init; all training steps used the same empty memory → memory never updated between write and read phases.
- **Fix**: `make_blank_memory(config)` called fresh per training step (episodic mode).
- **STATUS**: FIXED ✓

### BUG-P0-025
- **File**: `scripts/train_text_qa.py`
- **Problem**: `write_only()` returned 3D `h_eos` from `TransformerQAModel.encode_fact` but then mean-pooled as 2D in contrastive loss.
- **Fix**: `encode_fact()` now returns 1D `h_eos` (batch, hidden) via mean-pooling inside encoder. Contrastive loss uses this directly.
- **STATUS**: FIXED ✓

---

## P1 – Serious Scientific/Methodological Issues

### BUG-P1-001
- **File**: `scripts/generate_text_dataset.py`
- **Problem**: Query `"Siapa nama lengkap yang berasal dari Jakarta?"` has hundreds of valid answers.
- **Fix**: Unique entity IDs (`User_XXXXXX`) ensure every query has exactly one answer.
- **STATUS**: FIXED ✓

### BUG-P1-002
- **File**: `scripts/generate_text_dataset.py`
- **Problem**: Random shuffle then index split → same entity in train and test.
- **Fix**: Entity-aware splitting: all samples per entity assigned to same split.
- **STATUS**: FIXED ✓

### BUG-P1-003
- **File**: `experiments/counterfactual.py`
- **Problem**: Used two different `H_A` and `H_B` vectors – not a causal counterfactual.
- **Fix**: Same key `K` used; `v_proj` negated to produce different stored value.
- **STATUS**: FIXED ✓

### BUG-P1-004
- **File**: `scripts/train_text_qa.py`
- **Problem**: Comment said "Teacher Forcing" but decoder used `argmax(pred)` not `target_{t-1}`.
- **Fix**: Comment corrected to "Free-Running Autoregressive Decoding".
- **STATUS**: FIXED ✓

### BUG-P1-005
- **File**: `experiments/ablation.py`
- **Problem**: Config modified after JIT compile → ablation had no effect on computation.
- **Fix**: Each ablation creates a fresh `TinyMemoryBank` with different config before JIT.
- **STATUS**: FIXED ✓

### BUG-P1-006
- **File**: `models/tiny_memory_bank.py` (old)
- **Problem**: Empty slots could dilute retrieval when `top_k > active_count`.
- **Fix**: `attn_weights * valid_mask` zeroes invalid contributions. Zero-vector check guards result.
- **STATUS**: FIXED ✓

### BUG-P1-007
- **File**: `models/tiny_memory_bank.py` (old)
- **Problem**: Replacement picked `argmax(empty_mask)` which returns 0 when no empty slot.
- **Fix**: Sort keys (EXPIRED=0, DORMANT=1, ACTIVE=2) + importance tiebreak → true priority ordering.
- **STATUS**: FIXED ✓

### BUG-P1-008
- **File**: `scripts/train_text_qa.py`
- **Problem**: Auxiliary losses with large weights (10×, 5×, 3×) dominated QA loss.
- **Fix**: QA loss is primary. Aux weights reduced to 0.5×. Oracle excluded entirely.
- **STATUS**: FIXED ✓

### BUG-P1-009
- **File**: `evaluation/metrics.py`
- **Problem**: `recall_at_k` did not implement ranked retrieval; just cosine threshold.
- **Fix**: Proper `recall_at_k(scores, ground_truth_idx, k_values)` with ranked list. Added MRR.
- **STATUS**: FIXED ✓

### BUG-P1-010
- **File**: Multiple experiment files
- **Problem**: Inconsistent seed counts (some 3, some 5).
- **Fix**: Documented in configs. `benchmark.yaml` specifies `eval_seeds: [0,1,2]`.
- **STATUS**: ADDRESSED ✓

### BUG-P1-011
- **File**: `experiments/`
- **Problem**: Missing tests for capacity scaling, recency, importance, confidence, empty memory.
- **Fix**: All added in `experiments/memory_functional_benchmark.py`.
- **STATUS**: FIXED ✓

---

## P2 – Minor Robustness/Reproducibility Issues

### BUG-P2-001
- **File**: `configs/tiny.yaml`
- **Problem**: Unknown fields (`memory_update_threshold`, `num_experts`, etc.) not in `TinyMemoryConfig`.
- **Fix**: `configs/benchmark.yaml` created with only valid fields.
- **STATUS**: ADDRESSED ✓

### BUG-P2-002
- **File**: `dataset/text_dataset_loader.py`
- **Problem**: No `max_samples` parameter.
- **Fix**: Added `max_samples=None` parameter.
- **STATUS**: FIXED ✓

### BUG-P2-003
- **File**: Dataset generation
- **Problem**: No `metadata.json`.
- **Fix**: `generate_text_dataset.py` now writes `dataset/metadata.json`.
- **STATUS**: FIXED ✓

### BUG-P2-004
- **File**: Dataset loader
- **Problem**: PAD ID assumed/hardcoded as 0.
- **Fix**: Always looked up via `tokenizer.token_to_id('[PAD]')`.
- **STATUS**: FIXED ✓

### BUG-P2-005
- **File**: `requirements.txt`
- **Problem**: `tokenizers`, `pandas`, `tqdm`, `pytest` missing.
- **Fix**: Added all missing dependencies.
- **STATUS**: FIXED ✓

### BUG-P2-006
- **File**: `tests/`
- **Problem**: Empty tests directory.
- **Fix**: Created 5 test files: `test_architecture_lock.py`, `test_memory_state.py`, `test_memory_functional.py`, `test_counterfactual.py`, `test_dataset.py`.
- **STATUS**: FIXED ✓

### BUG-P2-007
- **File**: `scripts/train_text_qa.py`
- **Problem**: Only final checkpoint saved, not best validation checkpoint.
- **Fix**: Best validation checkpoint saved to `results/best_model.msgpack`.
- **STATUS**: FIXED ✓

### BUG-P2-008
- **File**: Dataset
- **Problem**: No metadata saved with dataset.
- **Fix**: `metadata.json` saved with seed, sizes, strategy.
- **STATUS**: FIXED ✓

### BUG-P2-009
- **File**: `configs/`
- **Problem**: No `benchmark.yaml`.
- **Fix**: Created `configs/benchmark.yaml`.
- **STATUS**: FIXED ✓

### BUG-P2-010
- **File**: Training
- **Problem**: No structured per-experiment logging.
- **Fix**: Train step logs `qa_loss`, `qf_loss`, `retrieval_loss` separately.
- **STATUS**: FIXED ✓

---

## Summary

| Severity | Found | Fixed | Remaining |
|----------|-------|-------|-----------|
| P0       | 25    | 25    | 0         |
| P1       | 11    | 11    | 0         |
| P2       | 10    | 10    | 0         |
| **Total**| **46**|**46** | **0**     |

All bugs fixed as of 2026-09-02.
