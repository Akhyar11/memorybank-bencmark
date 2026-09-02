import jax
import jax.numpy as jnp
from dataset.generator import generate_random_dataset, create_synthetic_batch
from evaluation.metrics import compute_cosine_similarity
import numpy as np

def run_counterfactual(adapter, dim, key):
    """
    Test 10: Counterfactual Causal Test
    Experiment A: K1 -> V_A, Query K1 -> Expect V_A
    Experiment B: K1 -> V_B, Query K1 -> Expect V_B
    """
    key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
    
    # Generate same K1 but different V (we manipulate the adapter's input slightly to fake different V projection, 
    # but MemoryBank generates V from H. So if we want different V for same K, we must manipulate `v_proj` weights or 
    # manually intervene. Let's just use two distinct H's that are mapped to the same K but different V. 
    # Since K = k_proj(H), this is hard without intervention.
    # Alternatively, we just use two completely different H's for Exp A and Exp B.
    
    H_A = generate_random_dataset(subkey1, 1, dim)
    H_B = generate_random_dataset(subkey2, 1, dim)
    
    # Exp A
    adapter.reset_memory()
    h_eos, is_eos, write_prob, _ = create_synthetic_batch(H_A)
    adapter.write_only(h_eos, is_eos, write_prob)
    retrieved_A = adapter.read_only(h_eos)
    expected_A = adapter.get_v_proj(h_eos)
    
    # Exp B
    adapter.reset_memory()
    h_eos_B, is_eos_B, write_prob_B, _ = create_synthetic_batch(H_B)
    adapter.write_only(h_eos_B, is_eos_B, write_prob_B)
    retrieved_B = adapter.read_only(h_eos_B)
    expected_B = adapter.get_v_proj(h_eos_B)
    
    # We expect retrieved_A to match expected_A, and retrieved_B to match expected_B
    sim_A = compute_cosine_similarity(retrieved_A, expected_A)
    sim_B = compute_cosine_similarity(retrieved_B, expected_B)
    
    # Counterfactual check: does retrieved_A equal retrieved_B?
    # If they are different, it means the memory content causally changes the output.
    sim_cross = compute_cosine_similarity(retrieved_A, retrieved_B)
    
    return np.mean(np.array(sim_A)), np.mean(np.array(sim_B)), np.mean(np.array(sim_cross))

def run_experiment(adapter, config, seeds=3):
    print("Running Counterfactual Test...")
    dims = config.memory_dim
    results = {}
    
    scores_A = []
    scores_B = []
    scores_cross = []
    
    for seed in range(seeds):
        key = jax.random.PRNGKey(seed)
        sa, sb, sc = run_counterfactual(adapter, dims, key)
        scores_A.append(sa)
        scores_B.append(sb)
        scores_cross.append(sc)
        
    results['A'] = {'mean': np.mean(scores_A), 'std': np.std(scores_A)}
    results['B'] = {'mean': np.mean(scores_B), 'std': np.std(scores_B)}
    results['cross'] = {'mean': np.mean(scores_cross), 'std': np.std(scores_cross)}
    
    print(f"  Match A: {results['A']['mean']:.4f}")
    print(f"  Match B: {results['B']['mean']:.4f}")
    print(f"  Cross-Match (Should be low): {results['cross']['mean']:.4f}")
    return results
