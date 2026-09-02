"""
Skrip Ablation Test untuk Membuktikan Kontribusi Nyata Memory Bank:
1. Normal Memory (Real Facts)
2. Zero Memory (Zeros)
3. Random Memory (Wrong/Random Facts)
"""
import os
import sys
import jax
import jax.numpy as jnp
from collections import defaultdict
import numpy as np

sys.path.insert(0, '/home/akhyar/Dokumen/Code/python/MemoryBank-bencmark')
from models.transformer_qa_model import TransformerQAModel
from models.tiny_memory_bank import TinyMemoryConfig
from dataset.text_dataset_loader import TextDataLoader
from flax import serialization

tokenizer_path = "dataset/tokenizer.json"
checkpoint_path = "results/model.msgpack"

batch_size = 32
max_input_len = 32
max_target_len = 16
vocab_size = 2000
embed_dim = 32
hidden_size = 32
pad_id = 0

config = TinyMemoryConfig(
    hidden_size=hidden_size, memory_dim=32, memory_capacity=128, memory_top_k=8,
    memory_threshold=0.0, memory_read_threshold=0.0, memory_write_threshold=0.7
)
model = TransformerQAModel(config=config, vocab_size=vocab_size, embed_dim=embed_dim, num_layers=1, num_heads=2, ff_dim=64, max_target_len=max_target_len, dropout_rate=0.0)

# Init variables & load checkpoint
rng = jax.random.PRNGKey(0)
dummy_input = jnp.ones((1, max_input_len), dtype=jnp.int32)
dummy_mask = jnp.ones((1, max_input_len), dtype=jnp.int32)
dummy_target = jnp.ones((1, max_target_len), dtype=jnp.int32)
dummy_p = jnp.ones((1,))

variables = model.init(rng, dummy_input, dummy_mask, dummy_p, dummy_p, dummy_p, dummy_target, method=model.init_all)

print(f"Memuat checkpoint dari {checkpoint_path}...")
with open(checkpoint_path, "rb") as f:
    params = serialization.from_bytes(variables['params'], f.read())

blank_memory = variables['memory']

test_loader = TextDataLoader("dataset/test.csv", tokenizer_path, batch_size, max_input_len, max_target_len)
tok = test_loader.tokenizer

# Functions for the 3 conditions
@jax.jit
def infer_normal(params, batch):
    vars = {'params': params, 'memory': blank_memory}
    is_eos = jnp.ones((batch_size,))
    write_p = jnp.ones((batch_size,))
    
    # Write real facts
    (_, h_eos), updated_memory = model.apply(vars, batch['write_ids'], batch['write_mask'],
                                             is_eos, write_p, deterministic=True,
                                             method=model.write_only, mutable=['memory'])
    new_vars = {'params': params, 'memory': updated_memory['memory']}
    read_p = jnp.ones((batch_size,))
    write_p_zero = jnp.zeros((batch_size,))

    preds, _ = model.apply(new_vars, batch['query_ids'], batch['query_mask'],
                           read_p, write_p_zero, max_target_len, 2, pad_id, 3,
                           deterministic=True, method=model.greedy_decode, mutable=['memory'])
    return preds

@jax.jit
def infer_zero_memory(params, batch):
    # Pass zeroed memory (no facts written at all)
    vars = {'params': params, 'memory': blank_memory}
    read_p = jnp.ones((batch_size,))
    write_p_zero = jnp.zeros((batch_size,))

    preds, _ = model.apply(vars, batch['query_ids'], batch['query_mask'],
                           read_p, write_p_zero, max_target_len, 2, pad_id, 3,
                           deterministic=True, method=model.greedy_decode, mutable=['memory'])
    return preds

@jax.jit
def infer_random_memory(params, batch):
    # Pass permuted/random write_ids (wrong facts written)
    vars = {'params': params, 'memory': blank_memory}
    is_eos = jnp.ones((batch_size,))
    write_p = jnp.ones((batch_size,))
    
    # Shift write_ids by 1 to write WRONG facts for queries
    shuffled_write_ids = jnp.roll(batch['write_ids'], shift=5, axis=0)
    shuffled_write_mask = jnp.roll(batch['write_mask'], shift=5, axis=0)
    
    (_, h_eos), updated_memory = model.apply(vars, shuffled_write_ids, shuffled_write_mask,
                                             is_eos, write_p, deterministic=True,
                                             method=model.write_only, mutable=['memory'])
    new_vars = {'params': params, 'memory': updated_memory['memory']}
    read_p = jnp.ones((batch_size,))
    write_p_zero = jnp.zeros((batch_size,))

    preds, _ = model.apply(new_vars, batch['query_ids'], batch['query_mask'],
                           read_p, write_p_zero, max_target_len, 2, pad_id, 3,
                           deterministic=True, method=model.greedy_decode, mutable=['memory'])
    return preds

print("=" * 70)
print("HASIL ABLATION TEST: KONTRIBUSI MEMORY BANK NYATA vs TEBAKAN STATISTIK")
print("=" * 70)

total_tokens = 0
correct_normal = 0
correct_zero = 0
correct_random = 0

sample_comparisons = []

for b_idx, batch in enumerate(test_loader.iter_batches(shuffle=False)):
    if len(batch['write_ids']) < batch_size:
        continue
    if b_idx >= 5: # 5 batch = 160 sampel
        break
        
    p_norm = np.array(infer_normal(params, batch))
    p_zero = np.array(infer_zero_memory(params, batch))
    p_rand = np.array(infer_random_memory(params, batch))
    
    targets = np.array(batch['target_ids'])
    queries = np.array(batch['query_ids'])
    
    for i in range(batch_size):
        target = targets[i]
        mask = (target != pad_id)
        n_tok = int(mask.sum())
        total_tokens += n_tok
        
        c_norm = int(((p_norm[i] == target) * mask).sum())
        c_zero = int(((p_zero[i] == target) * mask).sum())
        c_rand = int(((p_rand[i] == target) * mask).sum())
        
        correct_normal += c_norm
        correct_zero += c_zero
        correct_random += c_rand
        
        if b_idx == 0 and i < 5:
            q_str = tok.decode(queries[i].tolist(), skip_special_tokens=True)
            t_str = tok.decode(target.tolist(), skip_special_tokens=True).strip()
            pred_norm_str = tok.decode(p_norm[i].tolist(), skip_special_tokens=True).strip()
            pred_zero_str = tok.decode(p_zero[i].tolist(), skip_special_tokens=True).strip()
            pred_rand_str = tok.decode(p_rand[i].tolist(), skip_special_tokens=True).strip()
            
            sample_comparisons.append({
                'query': q_str,
                'target': t_str,
                'normal': pred_norm_str,
                'zero': pred_zero_str,
                'random': pred_rand_str
            })

print("\n[ DOKUMENTASI CONTOH PREDIKSI DALAM 3 KONDISI ]\n")
for idx, sc in enumerate(sample_comparisons, 1):
    print(f"Sampel #{idx}:")
    print(f"  Query        : {sc['query']}")
    print(f"  Target Real  : {sc['target']}")
    print(f"  Memori Normal: {sc['normal']}")
    print(f"  Memori NOL   : {sc['zero']}")
    print(f"  Memori Acak  : {sc['random']}")
    print("-" * 50)

print("\n[ HASIL AKURASI KESELURUHAN (160 SAMPEL TEST) ]\n")
acc_norm = (correct_normal / total_tokens) * 100
acc_zero = (correct_zero / total_tokens) * 100
acc_rand = (correct_random / total_tokens) * 100

print(f"1. Memori NORMAL (Fakta Benar) : {acc_norm:.2f}% Token Accuracy")
print(f"2. Memori NOL    (Tanpa Fakta) : {acc_zero:.2f}% Token Accuracy")
print(f"3. Memori ACAK   (Fakta Salah) : {acc_rand:.2f}% Token Accuracy")

diff = abs(acc_norm - acc_zero)
print("\n" + "=" * 70)
if diff < 1.5:
    print("KESIMPULAN EMIRIK: HIPOTESIS USER BENAR 100%!")
    print("Model TIDAK menggunakan memorinya (Selisih Akurasi Memori Normal vs Nol hanya {:.2f}%).".format(diff))
    print("Model hanya menebak berdasarkan statistik weights internal.")
else:
    print("KESIMPULAN EMPIRIK: Memory Bank memberikan kontribusi sebesar {:.2f}%.".format(diff))
print("=" * 70)
