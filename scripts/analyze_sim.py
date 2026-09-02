import os
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.text_qa_model import TextQAModel
from dataset.text_dataset_loader import TextDataLoader

def main():
    dataset_dir = "dataset"
    test_csv = os.path.join(dataset_dir, "test.csv")
    tokenizer_path = os.path.join(dataset_dir, "tokenizer.json")
    ckpt_dir = os.path.abspath(os.path.join(dataset_dir, "..", "results", "weights"))
    
    batch_size = 32
    max_input_len = 32
    max_target_len = 16
    vocab_size = 2000
    embed_dim = 64
    hidden_size = 64
    memory_capacity = 128
    
    from models.tiny_memory_bank import TinyMemoryConfig
    config = TinyMemoryConfig(
        memory_capacity=memory_capacity,
        memory_dim=embed_dim,
        hidden_size=hidden_size,
        memory_threshold=0.5,
        memory_write_threshold=0.9
    )
    
    model = TextQAModel(config=config, vocab_size=vocab_size, embed_dim=embed_dim, 
                        hidden_size=hidden_size, max_target_len=max_target_len, dropout_rate=0.0)
    
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, max_input_len), dtype=jnp.int32)
    dummy_mask = jnp.ones((1, max_input_len), dtype=jnp.int32)
    dummy_target = jnp.ones((1, max_target_len), dtype=jnp.int32)
    dummy_p = jnp.ones((1,))
    
    variables = model.init(rng, dummy_input, dummy_mask, dummy_p, dummy_p, dummy_p, dummy_target, method=model.init_all)
    
    from flax import serialization
    ckpt_path = os.path.abspath(os.path.join(dataset_dir, "..", "results", "model.msgpack"))
    print(f"Memuat checkpoint dari {ckpt_path}...")
    if not os.path.exists(ckpt_path):
        print("Checkpoint tidak ditemukan!")
        return
        
    with open(ckpt_path, "rb") as f:
        params = serialization.from_bytes(variables['params'], f.read())
    
    loader = TextDataLoader(test_csv, tokenizer_path, batch_size, max_input_len, max_target_len)
    batch = next(loader.iter_batches(shuffle=False))
    
    # 1. Encode fakta
    print("\n[ ANALISIS KEMIRIPAN VEKTOR FAKTA ]")
    # Dapatkan h_eos menggunakan apply
    vars_with_blank_mem = {'params': params, 'memory': variables['memory']}
    is_eos = jnp.ones((batch_size,))
    write_p = jnp.ones((batch_size,))
    (_, h_eos), updated_vars = model.apply(vars_with_blank_mem, batch['write_ids'], batch['write_mask'], is_eos, write_p, 
                                  deterministic=True, method=model.write_only, mutable=['memory'])
    
    # Hitung k_proj secara manual menggunakan bobot dari params
    # Bobot k_proj ada di params['bank']['k_proj']['kernel'] dan (opsional) 'bias'
    k_kernel = params['bank']['k_proj']['kernel']
    
    # Dense layer di Flax melakukan x @ kernel + bias
    keys = jnp.dot(h_eos, k_kernel)
    if 'bias' in params['bank']['k_proj']:
        keys += params['bank']['k_proj']['bias']
        
    
    # Normalisasi vektor
    keys_norm_sq = jnp.sum(keys**2, axis=-1, keepdims=True)
    keys_norm = keys / jnp.sqrt(keys_norm_sq + 1e-8)
    
    # Hitung matriks cosine similarity (pairwise)
    sim_matrix = jnp.matmul(keys_norm, keys_norm.T) # shape: (32, 32)
    
    # Ambil nilai kemiripan antar fakta yang berbeda (non-diagonal)
    mask = ~jnp.eye(batch_size, dtype=bool)
    off_diagonal_sims = sim_matrix[mask]
    
    avg_sim = jnp.mean(off_diagonal_sims)
    max_sim = jnp.max(off_diagonal_sims)
    min_sim = jnp.min(off_diagonal_sims)
    
    print(f"Rata-rata Cosine Similarity antar fakta : {avg_sim:.4f}")
    print(f"Similarity Tertinggi antar fakta        : {max_sim:.4f}")
    print(f"Similarity Terendah antar fakta         : {min_sim:.4f}")
    
    print("\n[ SIMULASI PENGISIAN SLOT ]")
    print(f"Jika memory_threshold = 0.50 (Konfigurasi Saat Ini):")
    # Hitung berapa fakta yang memiliki kemiripan > 0.5 dengan fakta sebelumnya
    # Fakta ke-i dicek apakah ada kemiripan > threshold dengan fakta ke-0 sampai i-1
    slots_used = 0
    for threshold in [0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]:
        slots = 0
        for i in range(batch_size):
            if i == 0:
                slots += 1 # Fakta pertama pasti buat slot baru
            else:
                # Cek kemiripan fakta i dengan fakta 0 s/d i-1
                past_sims = sim_matrix[i, :i]
                max_past_sim = jnp.max(past_sims)
                if max_past_sim < threshold:
                    slots += 1 # Buat slot baru
        print(f"- Threshold {threshold:.2f} -> Akan menggunakan {slots} slot (dari {batch_size} fakta)")
        
    print("\nContoh beberapa kalimat fakta pertama:")
    for i in range(3):
        fakta = loader.tokenizer.decode(batch['write_ids'][i].tolist(), skip_special_tokens=True)
        print(f"{i+1}. {fakta}")

if __name__ == "__main__":
    main()
