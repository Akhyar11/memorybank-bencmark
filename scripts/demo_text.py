import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
from adapters.existing_memorybank import MemoryBankAdapter

def cosine_sim(a, b):
    a = np.array(a).flatten()
    b = np.array(b).flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

print("=" * 60)
print("  DEMO MEMORY BANK: TANYA JAWAB (Q&A)")
print("=" * 60)

adapter = MemoryBankAdapter()
adapter.setup()
adapter.load_weights("results/weights/small_trained.msgpack")
hidden_size = adapter.config.hidden_size
key = jax.random.PRNGKey(42)

# --- SIMULASI EMBEDDING & LANGUAGE MODEL ---
# Di dunia nyata: 
# 1. Text -> Sentence-BERT (Embedding) -> MemoryBank
# 2. MemoryBank -> Vektor -> LLM Decoder -> Text (Jawaban)
# Di sini kita simulasikan proses tersebut.

fact_db = {
    "kucing": {
        "fakta": "Saya punya kucing bernama Mochi.",
        "jawaban": "Mochi"
    },
    "ibukota": {
        "fakta": "Ibu kota Indonesia adalah Jakarta.",
        "jawaban": "Jakarta"
    },
    "resep": {
        "fakta": "Resep nasi goreng membutuhkan bawang dan kecap.",
        "jawaban": "Bawang dan kecap"
    }
}

embedding_space = {}

# Buat vektor memori untuk masing-masing fakta
for topic, data in fact_db.items():
    key, subkey = jax.random.split(key)
    base_vec = jax.random.normal(subkey, (1, hidden_size))
    base_vec = base_vec / jnp.linalg.norm(base_vec)
    embedding_space[data["fakta"]] = base_vec

def get_query_vector(query_text):
    """Mengubah pertanyaan menjadi vektor (Simulasi Embedding)"""
    if "kucing" in query_text.lower():
        base = embedding_space[fact_db["kucing"]["fakta"]]
    elif "ibu kota" in query_text.lower():
        base = embedding_space[fact_db["ibukota"]["fakta"]]
    elif "resep" in query_text.lower():
        base = embedding_space[fact_db["resep"]["fakta"]]
    else:
        base = jax.random.normal(jax.random.PRNGKey(99), (1, hidden_size))
    
    # Tambahkan noise karena kalimat pertanyaan TIDAK identik dengan kalimat memori
    noise = jax.random.normal(jax.random.PRNGKey(len(query_text)), (1, hidden_size)) * 0.2
    noisy_vec = base + noise
    return noisy_vec / jnp.linalg.norm(noisy_vec)

def decode_ke_jawaban(retrieved_vec):
    """Menerjemahkan ingatan dari MemoryBank menjadi teks (Simulasi LLM Decoder)"""
    best_sim = -1.0
    best_answer = "Maaf, saya tidak ingat / tidak ada di memori."
    
    for topic, data in fact_db.items():
        expected_v = adapter.get_v_proj(embedding_space[data["fakta"]])
        sim = cosine_sim(retrieved_vec, expected_v)
        
        if sim > best_sim:
            best_sim = sim
            if sim > 0.4:  # Threshold ingatan
                best_answer = data["jawaban"]
            else:
                best_answer = "Maaf, saya tidak ingat / tidak ada di memori."
                
    return best_answer, best_sim


print("\n[LANGKAH 1: BELAJAR FAKTA BARU]")
print("-" * 60)
for topic, data in fact_db.items():
    fakta = data["fakta"]
    vec = embedding_space[fakta]
    adapter.write_only(vec, jnp.ones((1,)), jnp.ones((1,)))
    print(f"Menyimpan ke memori: '{fakta}'")


print("\n[LANGKAH 2: GANGGUAN / DISTRACTOR]")
print("-" * 60)
print("Memasukkan 20 fakta acak (distractor) ke dalam memori...")
key, subkey = jax.random.split(key)
noise_vectors = jax.random.normal(subkey, (20, hidden_size))
noise_vectors = noise_vectors / jnp.linalg.norm(noise_vectors, axis=1, keepdims=True)
for i in range(20):
    adapter.write_only(noise_vectors[i:i+1], jnp.ones((1,)), jnp.ones((1,)))
print("✓ Selesai. Memori inti sekarang terkubur di bawah informasi lain.")


print("\n[LANGKAH 3: TANYA JAWAB (Q&A)]")
print("-" * 60)

queries = [
    "Siapa nama kucing saya?",
    "Apa ibu kota Indonesia?",
    "Apa saja bahan utama resep nasi goreng?",
    "Kapan perang dunia kedua terjadi?"
]

for q in queries:
    print(f"User (Query): {q}")
    
    # 1. Pertanyaan diubah jadi Vektor
    q_vec = get_query_vector(q)
    
    # 2. Vektor Pertanyaan dikirim ke Memory Bank -> Dapat Ingatan (Retrieved Vector)
    retrieved_vec = adapter.read_only(q_vec)
    
    # 3. Ingatan diterjemahkan ke teks oleh Decoder
    jawaban, conf = decode_ke_jawaban(retrieved_vec)
    
    print(f"Model (Ans) : {jawaban}")
    print(f"(Keyakinan Memori: {conf:.2f})\n")

print("=" * 60)
