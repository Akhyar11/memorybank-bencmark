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
        memory_threshold=0.0,
        memory_read_threshold=0.0,
        memory_write_threshold=0.7
    )
    
    model = TextQAModel(config=config, vocab_size=vocab_size, embed_dim=embed_dim, 
                        hidden_size=hidden_size, max_target_len=max_target_len, dropout_rate=0.0)
    
    # Init dummy vars
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, max_input_len), dtype=jnp.int32)
    dummy_mask = jnp.ones((1, max_input_len), dtype=jnp.int32)
    dummy_target = jnp.ones((1, max_target_len), dtype=jnp.int32)
    dummy_p = jnp.ones((1,))
    
    variables = model.init(rng, dummy_input, dummy_mask, dummy_p, dummy_p, dummy_p, dummy_target, method=model.init_all)
    
    # Load weights
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
    
    # 1. Tulis ke Memory Bank
    print("\n[ FASE MENULIS FAKTA (WRITE PHASE) ]")
    blank_memory = variables['memory']
    vars_with_blank_mem = {'params': params, 'memory': blank_memory}
    
    is_eos = jnp.ones((batch_size,))
    write_p = jnp.ones((batch_size,))
    
    (_, h_eos), updated_vars = model.apply(vars_with_blank_mem, batch['write_ids'], batch['write_mask'], is_eos, write_p,
                                  deterministic=True, method=model.write_only, mutable=['memory'])
                                  
    memory_state = updated_vars['memory']['bank']
    
    states = memory_state['state']
    access_cnt = memory_state['access_count']
    created_at = memory_state['created_at']
    
    # states: 0=EXPIRED, 1=ACTIVE, 2=DORMANT, 3=EMPTY
    active_slots = jnp.sum(states == 1)
    empty_slots = jnp.sum(states == 3)
    
    print(f"Total kapasitas   : {memory_capacity}")
    print(f"Slot terpakai     : {active_slots}")
    print(f"Slot kosong       : {empty_slots}")
    
    # 2. Ambil dari Memory Bank & Prediksi
    print("\n[ FASE BERTANYA (READ PHASE) ]")
    vars_with_filled_mem = {'params': params, 'memory': updated_vars['memory']}
    read_p = jnp.ones((batch_size,))
    write_p_zero = jnp.zeros((batch_size,))
    
    (logits, sim, _, _), _ = model.apply(vars_with_filled_mem, batch['query_ids'], batch['query_mask'], 
                           read_p, write_p_zero, batch['target_ids'], 
                           deterministic=True, method=model.__call__, mutable=['memory'])
    
    preds = jnp.argmax(logits, axis=-1)
    
    # sim memiliki dimensi (batch_size, memory_capacity)
    # Temukan slot yang paling mirip (dipilih) untuk setiap query
    selected_slots = jnp.argmax(sim, axis=-1)
    max_sims = jnp.max(sim, axis=-1)
    
    for i in range(5):
        print(f"\n--- Analisis Query {i+1} ---")
        q_str = loader.tokenizer.decode(batch['query_ids'][i].tolist(), skip_special_tokens=True)
        t_str = loader.tokenizer.decode(batch['target_ids'][i].tolist(), skip_special_tokens=True)
        pred_ids = preds[i]
        pred_ids = pred_ids[(pred_ids != 2) & (pred_ids != 0) & (pred_ids != 3)]
        p_str = loader.tokenizer.decode(pred_ids.tolist(), skip_special_tokens=True)
        
        slot_idx = selected_slots[i]
        slot_sim = max_sims[i]
        
        print(f"Query         : {q_str}")
        print(f"Fakta Input   : {loader.tokenizer.decode(batch['write_ids'][i].tolist(), skip_special_tokens=True)}")
        print(f"Target        : {t_str}")
        print(f"Prediksi      : {p_str}")
        print(f"-> Slot Memory yang dipilih : Slot {slot_idx}")
        print(f"-> Tingkat Kemiripan (Sim)  : {slot_sim:.4f}")
        
    # Cek statistik slot mana yang paling sering digunakan (dari access_count)
    print("\n[ STATISTIK SLOT TERAKTIF ]")
    final_mem = preds[1]['memory'] if isinstance(preds, tuple) else None
    
    # We can just look at access_cnt from earlier if we don't have the updated one
    # But let's get the updated memory from apply if we returned it, but greedy_decode does not return updated memory.
    # We will just print the slots chosen in this batch.
    
    # 3. Analisis Slot Overwrite
    print("\n[ ANALISIS OVERWRITE ]")
    # Jika ada fakta baru yang masuk di iterasi batch berikutnya, apakah dia menimpa slot yang sudah ada?
    print("Simulasi memasukkan fakta yang sama ke model...")
    (_, _), overwritten_vars = model.apply(vars_with_filled_mem, batch['write_ids'], batch['write_mask'], is_eos, write_p, 
                                  deterministic=True, method=model.write_only, mutable=['memory'])
    
    # new_active = jnp.sum(overwritten_vars['memory']['bank']['state'] == 1) # Depends on flax nesting
    # print(f"Slot terpakai awal : {active_slots}")
    # print(f"Slot terpakai baru : {new_active} (seharusnya sama karena fakta identik akan me-replace slot lamanya)")

if __name__ == "__main__":
    main()
