import os
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.tiny_memory_bank import TinyMemoryConfig
from models.text_qa_model import TextQAModel
from dataset.text_dataset_loader import TextDataLoader

def main():
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
    train_csv = os.path.join(dataset_dir, "train.csv")
    tokenizer_path = os.path.join(dataset_dir, "tokenizer.json")
    
    batch_size = 32
    max_input_len = 16
    max_target_len = 16
    vocab_size = 2000
    
    loader = TextDataLoader(train_csv, tokenizer_path, batch_size, max_input_len, max_target_len)
    pad_id = loader.pad_id
    eos_id = loader.eos_id
    
    config = TinyMemoryConfig(
        memory_capacity=128,
        memory_dim=32,
        hidden_size=32,
        memory_threshold=-1.0,
        memory_read_threshold=-1.0,
        memory_write_threshold=-1.0,
        memory_top_k=8
    )
    model = TextQAModel(config=config, vocab_size=vocab_size, embed_dim=32, hidden_size=32, max_target_len=max_target_len, dropout_rate=0.0)
    
    # Init dummy
    init_rng = jax.random.PRNGKey(0)
    dummy_input = jnp.zeros((batch_size, max_input_len), dtype=jnp.int32)
    dummy_mask = jnp.ones((batch_size, max_input_len), dtype=jnp.int32)
    dummy_p = jnp.ones((batch_size,))
    dummy_target = jnp.zeros((batch_size, max_target_len), dtype=jnp.int32)
    
    variables = model.init(init_rng, dummy_input, dummy_mask, dummy_p, dummy_p, dummy_p, dummy_target, method=model.init_all)
    
    # Just take 1 batch
    batch = next(loader.iter_batches(shuffle=False))
    
    # Forward pass without training just to see output shape and tokens
    is_eos = jnp.ones((batch_size,))
    write_p = jnp.ones((batch_size,))
    read_p = jnp.ones((batch_size,))
    
    _, updated_memory = model.apply(variables, batch['write_ids'], batch['write_mask'], is_eos, write_p, 
                                    deterministic=True, method=model.write_only, mutable=['memory'])
                                    
    new_vars = {'params': variables['params'], 'memory': updated_memory['memory']}
    
    logits, sim = model.apply(new_vars, batch['query_ids'], batch['query_mask'], read_p, write_p, batch['target_ids'],
                         deterministic=True)
                         
    preds = jnp.argmax(logits, axis=-1)
    
    print("Example Targets vs Preds (Untrained):")
    for i in range(3):
        print(f"Target: {batch['target_ids'][i]}")
        print(f"Pred:   {preds[i]}")
        mask = (batch['target_ids'][i] != pad_id)
        print(f"Valid tokens: {jnp.sum(mask)}")
        correct = jnp.sum((preds[i] == batch['target_ids'][i]) * mask)
        print(f"Correct: {correct}/{jnp.sum(mask)} ({correct/jnp.maximum(jnp.sum(mask), 1)*100:.1f}%)")
        print("---")

if __name__ == "__main__":
    main()
