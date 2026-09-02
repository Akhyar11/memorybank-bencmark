import jax
import jax.numpy as jnp
from dataset.generator import generate_orthogonal_dataset, create_synthetic_batch
from evaluation.metrics import compute_cosine_similarity
import numpy as np
from adapters.existing_memorybank import MemoryBankAdapter

def run_ablation(config_path, num_memories, key, ablation_type="none"):
    """
    Test 12: Ablation
    Temporarily disables features to see impact on basic retrieval or persistence.
    """
    adapter = MemoryBankAdapter(config_path=config_path)
    adapter.setup()      # initialize variables first
    adapter.reset_memory()  # then clear memory state
    
    # Apply ablation by manipulating config values
    if ablation_type == "no_recency":
        adapter.config.mem_gamma = 0.0
    elif ablation_type == "no_importance":
        adapter.config.mem_beta = 0.0
    elif ablation_type == "no_confidence":
        adapter.config.mem_delta = 0.0
    elif ablation_type == "no_decay":
        adapter.config.mem_decay_rate = 0.0
        
    dim = adapter.config.memory_dim
    key, subkey = jax.random.split(key)
    dataset = generate_orthogonal_dataset(subkey, num_memories, dim)
    
    h_eos, is_eos, write_prob, _ = create_synthetic_batch(dataset)
    for i in range(num_memories):
        adapter.write_only(h_eos[i:i+1], is_eos[i:i+1], write_prob[i:i+1])
        
    target_idx = num_memories // 2
    query_h_eos = h_eos[target_idx:target_idx+1]
    
    # Optional: simulate time passing to test decay ablations better
    adapter.advance_time(100)
    
    retrieved_v = adapter.read_only(query_h_eos)
    expected_v = adapter.get_v_proj(query_h_eos)
    
    sim = compute_cosine_similarity(retrieved_v, expected_v)
    return np.mean(np.array(sim))

def run_experiment(config_path, config, seeds=3):
    print("Running Ablation Test...")
    sizes = 50
    ablations = ["none", "no_recency", "no_importance", "no_confidence", "no_decay"]
    results = {}
    
    for ab in ablations:
        scores = []
        for seed in range(seeds):
            key = jax.random.PRNGKey(seed)
            score = run_ablation(config_path, sizes, key, ablation_type=ab)
            scores.append(score)
        results[ab] = {'mean': np.mean(scores), 'std': np.std(scores)}
        print(f"  Ablation {ab}: Sim = {results[ab]['mean']:.4f} ± {results[ab]['std']:.4f}")
        
    return results
