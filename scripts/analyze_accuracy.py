"""
Skrip analisis detail: apa yang benar dan salah di 50% Val Acc
"""
import os
import sys
import jax
import jax.numpy as jnp
from collections import defaultdict

sys.path.insert(0, '/home/akhyar/Dokumen/Code/python/MemoryBank-bencmark')
from models.transformer_qa_model import TransformerQAModel
from models.tiny_memory_bank import TinyMemoryConfig
from dataset.text_dataset_loader import TextDataLoader
import msgpack
import numpy as np

dataset_dir = "dataset"
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


# Load test data
test_loader = TextDataLoader("dataset/test.csv", tokenizer_path, batch_size, max_input_len, max_target_len)
tok = test_loader.tokenizer

@jax.jit
def run_inference(params, batch):
    vars = {'params': params, 'memory': blank_memory}
    is_eos = jnp.ones((batch_size,))
    write_p = jnp.ones((batch_size,))
    
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

rng = jax.random.PRNGKey(0)
dummy_input = jnp.ones((1, max_input_len), dtype=jnp.int32)
dummy_mask  = jnp.ones((1, max_input_len), dtype=jnp.int32)
dummy_target = jnp.ones((1, max_target_len), dtype=jnp.int32)
dummy_p = jnp.ones((1,))
variables = model.init(rng, dummy_input, dummy_mask, dummy_p, dummy_p, dummy_p, dummy_target,
                       method=model.init_all)

from flax import serialization
print(f"Memuat checkpoint dari {checkpoint_path}...")
with open(checkpoint_path, "rb") as f:
    params = serialization.from_bytes(variables['params'], f.read())

blank_memory = variables['memory']

# Statistik per jenis query
stats = defaultdict(lambda: {'total_tokens': 0, 'correct_tokens': 0, 'total_samples': 0, 'exact_match': 0})
position_stats = defaultdict(lambda: {'total': 0, 'correct': 0})  # per posisi token

all_details = []
num_batches = 0
max_batches = 5  # analisis 5 batch = 160 sampel

print("=" * 70)
print("ANALISIS DETAIL: TOKEN MANA YANG BENAR vs SALAH")
print("=" * 70)

for batch in test_loader.iter_batches(shuffle=False):
    if len(batch['write_ids']) < batch_size:
        continue
    if num_batches >= max_batches:
        break
    num_batches += 1
    
    preds = run_inference(params, batch)
    preds = np.array(preds)
    target_ids = np.array(batch['target_ids'])
    query_ids = np.array(batch['query_ids'])
    
    for i in range(batch_size):
        target = target_ids[i]
        pred = preds[i]
        query_str = tok.decode(query_ids[i].tolist(), skip_special_tokens=True)
        target_str = tok.decode(target.tolist(), skip_special_tokens=True).strip()
        pred_str = tok.decode(pred.tolist(), skip_special_tokens=True).strip()
        
        # Deteksi jenis query
        if 'kode rahasia' in query_str:
            qtype = 'kode_rahasia'
        elif 'pekerjaan' in query_str:
            qtype = 'pekerjaan'
        elif 'lahir' in query_str or 'dibesarkan' in query_str:
            qtype = 'kota_lahir'
        elif 'warna' in query_str and 'merek' in query_str:
            qtype = 'merek_warna'
        elif 'warna' in query_str:
            qtype = 'warna'
        elif 'nama lengkap' in query_str:
            qtype = 'nama_orang'
        elif 'nama' in query_str and 'peliharaan' in query_str:
            qtype = 'nama_hewan'
        else:
            qtype = 'lainnya'
        
        # Hitung token-level accuracy
        mask = (target != pad_id)
        n_target_tokens = int(mask.sum())
        
        # Token yang benar per posisi
        token_match = []
        for pos in range(n_target_tokens):
            is_correct = (pred[pos] == target[pos])
            token_match.append(is_correct)
            position_stats[pos]['total'] += 1
            if is_correct:
                position_stats[pos]['correct'] += 1
        
        n_correct = sum(token_match)
        exact_match = (n_correct == n_target_tokens) and (n_target_tokens > 0)
        
        stats[qtype]['total_tokens'] += n_target_tokens
        stats[qtype]['correct_tokens'] += n_correct
        stats[qtype]['total_samples'] += 1
        stats[qtype]['exact_match'] += int(exact_match)
        
        if num_batches == 1 and i < 8:
            all_details.append({
                'query': query_str, 'target': target_str, 'pred': pred_str,
                'qtype': qtype, 'correct': n_correct, 'total': n_target_tokens,
                'token_match': token_match
            })

# Tampilkan detail 8 sampel pertama
print("\n[ CONTOH DETAIL PREDIKSI PER TOKEN ]\n")
for d in all_details:
    pct = d['correct']/d['total']*100 if d['total'] > 0 else 0
    print(f"Query   : {d['query']}")
    print(f"Target  : {d['target']}")
    print(f"Prediksi: {d['pred']}")
    print(f"Akurasi : {d['correct']}/{d['total']} token benar ({pct:.0f}%)")
    match_str = ' '.join(['✅' if m else '❌' for m in d['token_match']])
    print(f"Per-tok : {match_str}")
    print()

# Statistik per jenis query
print("=" * 70)
print("[ AKURASI PER JENIS QUERY ]\n")
total_tokens_all = 0
correct_tokens_all = 0
for qtype, s in sorted(stats.items()):
    tok_acc = s['correct_tokens']/s['total_tokens']*100 if s['total_tokens'] > 0 else 0
    exact_acc = s['exact_match']/s['total_samples']*100 if s['total_samples'] > 0 else 0
    total_tokens_all += s['total_tokens']
    correct_tokens_all += s['correct_tokens']
    print(f"{qtype:15s}: Token Acc={tok_acc:5.1f}% | Exact Match={exact_acc:5.1f}% | ({s['total_samples']} sampel)")

print(f"\n{'TOTAL':15s}: Token Acc={correct_tokens_all/total_tokens_all*100:.1f}% | ({sum(s['total_samples'] for s in stats.values())} sampel)")

# Statistik per posisi token
print("\n[ AKURASI PER POSISI TOKEN ]\n")
for pos in sorted(position_stats.keys()):
    s = position_stats[pos]
    acc = s['correct']/s['total']*100 if s['total'] > 0 else 0
    bar = '█' * int(acc/5)
    print(f"Posisi {pos:2d}: {acc:5.1f}% {bar}")
