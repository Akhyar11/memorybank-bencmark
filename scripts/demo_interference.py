"""
Demo langsung interference test.
Menunjukkan secara eksplisit:
- Input yang masuk ke Memory Bank
- Output yang dihasilkan
- Ekspektasi output yang seharusnya
"""
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

# ─────────────────────────────────────────
print("=" * 60)
print("  DEMO INTERFERENCE TEST — STEP BY STEP")
print("=" * 60)

# Setup adapter
adapter = MemoryBankAdapter()
adapter.setup()
adapter.load_weights("results/weights/small_trained.msgpack")

key = jax.random.PRNGKey(7)

# ─────────────────────────────────────────
# LANGKAH 1: Buat memori TARGET yang ingin kita ingat
# ─────────────────────────────────────────
print("\n[STEP 1] BUAT MEMORI TARGET")
print("-" * 60)

key, k1 = jax.random.split(key)
target = jax.random.normal(k1, (1, adapter.config.hidden_size))
target = target / jnp.linalg.norm(target)

print(f"  Input TARGET (vektor h_eos, dim={adapter.config.hidden_size}):")
print(f"  {np.array(target[0, :8]).round(3)} ... (8 dari {adapter.config.hidden_size} dimensi)")
print(f"  Ini representasi sebuah 'fakta' atau 'informasi penting'")

# Tulis target ke memori
is_eos   = jnp.ones((1,))
write_p  = jnp.ones((1,))
adapter.write_only(target, is_eos, write_p)
print(f"\n  ✓ Target DITULIS ke Memory Bank")

# ─────────────────────────────────────────
# LANGKAH 2: Baca SEBELUM distractor
# ─────────────────────────────────────────
print("\n[STEP 2] BACA MEMORI (sebelum ada gangguan)")
print("-" * 60)

retrieved_before = adapter.read_only(target)
sim_before = cosine_sim(retrieved_before, adapter.get_v_proj(target))

print(f"  Query (input baca) = TARGET itu sendiri")
print(f"  Output Memory Bank  = {np.array(retrieved_before[0, :8]).round(3)} ...")
print(f"  Ekspektasi output   = semirip mungkin dengan target")
print(f"  Cosine Similarity   = {sim_before:.4f}  (1.0 = sempurna, 0.0 = tidak ada)")
print(f"  → {'✓ BERHASIL diambil' if sim_before > 0.3 else '✗ GAGAL diambil'}")

# ─────────────────────────────────────────
# LANGKAH 3: Injeksi DISTRACTOR (gangguan)
# ─────────────────────────────────────────
n_distractor = 20
noise_level  = 0.3

print(f"\n[STEP 3] INJEKSI {n_distractor} DISTRACTOR (noise={noise_level})")
print("-" * 60)
print(f"  Distractor = vektor BERBEDA yang ditulis ke memori setelah target")
print(f"  Ini mensimulasikan: model menerima {n_distractor} informasi lain")
print(f"  setelah menyimpan target, dan kita lihat apakah target masih bisa diambil")

key, k2 = jax.random.split(key)
noise      = jax.random.normal(k2, (n_distractor, adapter.config.hidden_size)) * noise_level
distractors = target + noise
distractors = distractors / (jnp.linalg.norm(distractors, axis=-1, keepdims=True) + 1e-8)

for i in range(n_distractor):
    d = distractors[i:i+1]
    adapter.write_only(d, jnp.ones((1,)), jnp.ones((1,)))

print(f"\n  Contoh distractor[0]: {np.array(distractors[0, :8]).round(3)} ...")
print(f"  Contoh distractor[1]: {np.array(distractors[1, :8]).round(3)} ...")
print(f"  (berbeda dari target, tapi berada di 'wilayah' yang sama)")
print(f"\n  ✓ {n_distractor} distractor DITULIS ke Memory Bank")

# ─────────────────────────────────────────
# LANGKAH 4: Baca SETELAH distractor
# ─────────────────────────────────────────
print(f"\n[STEP 4] BACA MEMORI (setelah ada {n_distractor} gangguan)")
print("-" * 60)

retrieved_after = adapter.read_only(target)
sim_after = cosine_sim(retrieved_after, adapter.get_v_proj(target))

print(f"  Query (input baca)  = TARGET (sama persis seperti STEP 2)")
print(f"  Output Memory Bank  = {np.array(retrieved_after[0, :8]).round(3)} ...")
print(f"  Ekspektasi output   = TETAP mirip dengan target, meskipun ada {n_distractor} distractor")
print(f"  Cosine Similarity   = {sim_after:.4f}")
print(f"  → {'✓ TAHAN GANGGUAN' if sim_after > 0.3 else '✗ TERPENGARUH GANGGUAN'}")

# ─────────────────────────────────────────
# LANGKAH 5: Ringkasan
# ─────────────────────────────────────────
print("\n[RINGKASAN]")
print("=" * 60)
print(f"  Sim SEBELUM distractor : {sim_before:.4f}")
print(f"  Sim SETELAH  distractor: {sim_after:.4f}")
drop = sim_before - sim_after
print(f"  Penurunan              : {drop:.4f}  ({'tinggi — memory terpengaruh' if drop > 0.3 else 'rendah — memory kuat'})")
print()
if sim_after > 0.3:
    print("  ✅ KESIMPULAN: Memory Bank berhasil MEMPERTAHANKAN memori target")
    print("     meskipun ada gangguan. Mekanisme scoring (cosine similarity +")
    print("     importance + recency) berhasil membedakan target dari distractor.")
else:
    print("  ❌ KESIMPULAN: Memory Bank GAGAL mempertahankan memori target.")
    print("     Target tertimpa atau terkubur oleh distractor.")
