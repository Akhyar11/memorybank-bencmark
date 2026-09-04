# Catatan Arsitektur: Memory Bank Layer

> **Status**: LOCKED — arsitektur tidak boleh dimodifikasi.
> **Sumber Kebenaran**: [`/home/akhyar/Dokumen/Code/python/MemoryBank/mamoe/memory/bank.py`](file:///home/akhyar/Dokumen/Code/python/MemoryBank/mamoe/memory/bank.py)
> **Implementasi PyTorch**: [`/home/akhyar/Dokumen/Code/python/MemoryBank-bencmark/models/tiny_memory_bank.py`](file:///home/akhyar/Dokumen/Code/python/MemoryBank-bencmark/models/tiny_memory_bank.py)

---

## 1. Gambaran Umum

Memory Bank adalah sebuah **modul memori persisten neural** yang dirancang untuk menyimpan, membaca, dan memperbarui representasi faktual dalam ruang vektor. Berbeda dengan attention biasa yang bersifat *in-context*, Memory Bank mempertahankan state memori lintas batch dan lintas waktu (episodic), mirip dengan cara kerja working memory pada kognisi manusia.

```
Input Konteks (h_eos)
        │
        ▼
┌───────────────────────────────────────────────────┐
│                   MEMORY BANK                     │
│                                                   │
│  ┌─────────┐   ┌─────────┐   ┌───────────────┐   │
│  │  DECAY  │──▶│  READ   │──▶│     FUSE      │──▶ Output
│  └─────────┘   └────┬────┘   └───────────────┘   │
│                     │                             │
│              ┌──────▼──────┐                      │
│              │   WRITE     │ (saat training)       │
│              └─────────────┘                      │
└───────────────────────────────────────────────────┘
              ▲
       Episodic Memory Slots
    [keys | vals | metadata]
```

---

## 2. Konfigurasi Hyperparameter (`TinyMemoryConfig`)

| Parameter | Default | Simbol | Deskripsi |
| :--- | :---: | :---: | :--- |
| `memory_capacity` | 128 | $C$ | Jumlah slot memori yang tersedia |
| `memory_dim` | 32 | $d$ | Dimensi vektor per slot memori |
| `hidden_size` | 32 | $h$ | Dimensi vektor input/output dari backbone |
| `mem_decay_rate` | 0.001 | $\lambda$ | Laju peluruhan memori per langkah waktu |
| `mem_importance_protection` | 0.5 | $\rho$ | Faktor perlindungan slot penting dari peluruhan |
| `mem_alpha` | 1.0 | $\alpha$ | Bobot kemiripan cosine pada skor retrieval |
| `mem_beta` | 0.5 | $\beta$ | Bobot importance pada skor retrieval |
| `mem_gamma` | 0.1 | $\gamma$ | Bobot recency (kebaruan akses) pada skor retrieval |
| `mem_delta` | 0.2 | $\delta$ | Bobot confidence pada skor retrieval |
| `memory_top_k` | 4 | $k$ | Jumlah kandidat teratas yang diambil saat retrieval |
| `memory_threshold` | -1e9 | $\tau$ | Ambang batas minimum skor untuk dianggap relevan |
| `memory_read_threshold` | 0.0 | — | Ambang batas gating `read_prob` |
| `memory_write_threshold` | 0.0 | — | Ambang batas gating `write_prob` |
| `mem_reinforcement_rate` | 0.05 | $\eta_a$ | Boost importance saat slot berhasil dibaca |

---

## 3. Komponen Terlatih (Trainable Parameters)

Total parameter terlatih: **5,153 parameter** (gradient flow aktif).

| Layer | Bentuk Tensor | Jumlah Param | Deskripsi |
| :--- | :---: | :---: | :--- |
| `q_proj` | `[32 × 32]` | 1,024 | Proyeksi kueri untuk retrieval cosine similarity |
| `k_proj` | `[32 × 32]` | 1,024 | Proyeksi kunci untuk penulisan slot memori |
| `v_proj` | `[32 × 32]` | 1,024 | Proyeksi nilai yang disimpan di slot |
| `i_proj` | `[1 × 32] + bias[1]` | 33 | Proyeksi logit importance (scalar) |
| `fusion_proj` | `[32 × 64]` | 2,048 | Proyeksi fusi `[h; m]` → output |

> **Catatan**: Semua proyeksi menggunakan `nn.Linear(..., bias=False)` kecuali `i_proj` yang memiliki bias. Gradien autograd hanya mengalir melalui kelima proyeksi ini.

### Formula Proyeksi

$$q = W_q \cdot h, \quad k = W_k \cdot h, \quad v = W_v \cdot h$$

$$i = \sigma(W_i \cdot h + b_i) \in [0, 1]$$

---

## 4. Buffer Episodic (Non-Trainable State)

Total buffer: **8,961 elemen** (tidak memiliki gradient, mirip `'memory'` variable collection di Flax asli).

| Buffer | Bentuk | Dtype | Deskripsi |
| :--- | :---: | :---: | :--- |
| `mem_keys` | `[128, 32]` | float32 | Kunci yang disimpan tiap slot (hasil `k_proj`) |
| `mem_vals` | `[128, 32]` | float32 | Nilai yang disimpan tiap slot (hasil `v_proj`) |
| `mem_importance` | `[128]` | float32 | Skor kepentingan $I \in [0,1]$ tiap slot |
| `mem_confidence` | `[128]` | float32 | Skor keyakinan $C \in [0,1]$ tiap slot |
| `mem_created_at` | `[128]` | int32 | Langkah waktu saat slot dibuat |
| `mem_last_access` | `[128]` | int32 | Langkah waktu terakhir slot dibaca |
| `mem_access_count` | `[128]` | int32 | Jumlah total akses baca pada slot |
| `mem_state` | `[128]` | int32 | Status slot: `EXPIRED=0`, `ACTIVE=1`, `DORMANT=2` |
| `global_step` | `[1]` | int32 | Penghitung global langkah waktu |

> **Implementasi Parity dengan Flax**: Di Flax asli, semua buffer di atas disimpan dalam koleksi `self.variable('memory', ...)` yang bersifat *mutable runtime variable* (bukan parameter). Di PyTorch, diimplementasikan menggunakan `register_buffer()` dan setiap mutasi dilakukan dalam blok `with torch.no_grad():` untuk mempertahankan semantik yang identik.

---

## 5. Operasi Inti (Locked Operations)

### Operasi 1: `decay_memory()`

Meluruhkan signifikansi slot berdasarkan waktu. Dijalankan otomatis setiap `forward()`.

**Formula**:

$$R_i = \exp(-\lambda \cdot \Delta t_i)$$

$$R^{\text{eff}}_i = R_i \cdot (1 + \rho \cdot I_i)$$

Di mana $\Delta t_i = \text{global\_step} - \text{last\_access}_i$.

**Transisi Status**:

| Kondisi | Transisi |
| :--- | :--- |
| $R^{\text{eff}}_i < 0.1$ | `ACTIVE/DORMANT` → `EXPIRED` |
| $R^{\text{eff}}_i < 0.5$ | `ACTIVE` → `DORMANT` |
| Selainnya | Status tidak berubah |

> Slot dengan `importance` tinggi terlindungi dari peluruhan berkat faktor $\rho \cdot I_i$.

---

### Operasi 2: `read(h_eos, read_prob)`

Mengambil informasi dari slot memori berdasarkan relevansi query.

**Alur lengkap (10 langkah)**:

**Langkah 1** — Query Projection: $q = W_q \cdot h_{\text{eos}}$

**Langkah 2** — Cosine Similarity:
$$\text{sim}_{ij} = \frac{q_i \cdot k_j}{\|q_i\| \cdot \|k_j\|}$$

**Langkah 3** — Recency:
$$\text{rec}_j = \exp(-\lambda \cdot \Delta t_j)$$

**Langkah 4** — Broadcast Metadata: Ekspansi $I_j$, $C_j$, $\text{rec}_j$ ke dimensi batch

**Langkah 5** — Skor Komposit:
$$S_{ij} = \alpha \cdot \text{sim}_{ij} + \beta \cdot I_j + \gamma \cdot \text{rec}_j + \delta \cdot C_j$$

**Langkah 6** — Masking EXPIRED: Slot `EXPIRED` diberi skor $-10^9$

**Langkah 7** — Top-K Selection: Ambil $k$ slot dengan skor tertinggi

**Langkah 8** — Threshold Filter: Abaikan slot dengan skor $< \tau$, terapkan `read_prob` gate

**Langkah 9** — Softmax Aggregation:
$$a_{ij} = \text{softmax}(S_{ij}) \text{ atas top-k yang valid}$$

**Langkah 10** — Weighted Sum + Reinforcement:
$$m_i = \sum_{j \in \text{top-k}} a_{ij} \cdot v_j$$

Untuk setiap slot yang berhasil dibaca: $I_j \mathrel{+}= \eta_a$ (clamped ke $[0,1]$), `last_access` dan `access_count` diperbarui.

---

### Operasi 3: `write(h_eos, is_eos, write_prob)`

Menyimpan representasi fakta baru ke slot memori.

**Gate Check**:

> **WRITE SEMANTICS UPDATE (v2)**
>
> **Old behavior** (before this change):
> ```python
> do_write = (is_eos > 0.5) AND (write_prob >= τ_write)
> ```
> `is_eos` was a mandatory prerequisite — writing could only occur at EOS token boundaries.
>
> **New behavior** (current, after source-of-truth audit):
> ```python
> do_write = (write_prob >= τ_write)
> ```
> `is_eos` is **no longer a write prerequisite**. It is retained as an API parameter for backward-compatibility but has no effect on the write decision.
>
> **Rationale**: Source-of-truth audit of the Flax implementation confirmed that `is_eos` was a caller-level signal, not a locked architectural component. The Memory Bank itself never detected EOS — the caller decided when to invoke `write()`. Removing `is_eos` from the gate restores correct semantics where `write_prob` is the sole write-decision variable.
>
> `is_eos` is described as a modification of the write invocation/gating semantics, subject to source-of-truth architecture verification. The Memory Bank architecture (projections, state schema, retrieval, replacement) is **unchanged**.

```
do_write = (write_prob >= τ_write)
```

**Proyeksi Masukan**:
$$k_{\text{new}} = W_k \cdot h, \quad v_{\text{new}} = W_v \cdot h, \quad i_{\text{new}} = \sigma(W_i \cdot h)$$

**Decision Logic — UPDATE vs INSERT**:

Cari slot aktif yang paling mirip dengan kunci baru:
$$\hat{j} = \arg\max_{j: \text{state}_j \neq \text{EXPIRED}} \text{cosim}(k_{\text{new}}, k_j)$$

| Kondisi | Aksi |
| :--- | :--- |
| $\text{sim}(\hat{j}) \geq \tau_{\text{write}}$ | **UPDATE slot** $\hat{j}$ (refinement) |
| Selainnya | **INSERT** ke slot prioritas terendah |

**Prioritas Slot untuk INSERT** (argmin dari):
$$\text{sort\_score}_j = \text{state\_rank}_j + \frac{I_j}{\max(I) + \epsilon} \cdot 0.5$$

Di mana state_rank: `EXPIRED=0`, `DORMANT=1`, `ACTIVE=2`.

**UPDATE — Refinement Rule**:
$$v_j \leftarrow (1 - C_j) \cdot v_j + C_j \cdot v_{\text{new}}$$
$$I_j \leftarrow \max(I_j, i_{\text{new}})$$
$$C_j \leftarrow \text{clamp}(C_j + 0.1, 0, 1)$$

**INSERT — Fresh Rule**:
$$k_j = k_{\text{new}}, \quad v_j = v_{\text{new}}, \quad I_j = i_{\text{new}}, \quad C_j = 0.5$$
$$\text{state}_j = \text{ACTIVE}, \quad \text{created\_at}_j = t, \quad \text{access\_count}_j = 1$$

---

### Operasi 4: `fuse(h, m)`

Menggabungkan representasi query dengan hasil retrieval memori.

$$\text{fused} = W_f \cdot [h \,\|\, m]$$

Di mana $[\,\cdot\,\|\,\cdot\,]$ adalah concatenation, dan $W_f \in \mathbb{R}^{h \times 2h}$.

---

### Operasi 5: `forward(h_eos, read_prob, write_prob)`

Orkestrasi urutan operasi lengkap:

```
global_step += 1
        ↓
decay_memory()          ← transisi status slot
        ↓
read(h, read_prob)      ← retrieval komposit
        ↓
fuse(h, m)              ← integrasi informasi
        ↓
      output
```

> **Catatan penting**: `write()` **tidak** dipanggil di dalam `forward()`. Penulisan dilakukan secara eksplisit dari luar oleh `TransformerQAModel.write_only()` sebelum forward pass kueri.

---

## 6. Alur Gradient dan Non-Gradient

```
h_eos (input dari encoder)
  │
  ├── q_proj ────────────────────────────────▶ q  (retrieval sim)
  │                                               ▲ gradient mengalir
  ├── k_proj ──▶ k_new ──▶ [buffer, detached]
  │
  ├── v_proj ──▶ v_new ──▶ [buffer, detached]
  │
  ├── i_proj ──▶ i_new ──▶ [buffer, detached]
  │
  └── (read result m) ──▶ fusion_proj ──▶ output
                                              ▲ gradient mengalir
```

**Gradient mengalir melalui**: `q_proj`, `fusion_proj`.

**Tidak ada gradient melalui**: `mem_keys`, `mem_vals`, `mem_importance`, `mem_confidence`, `mem_state`, dll. (semua mutasi dalam `torch.no_grad()`).

---

## 7. Karakteristik Empiris (Dari Eksperimen Nyata, 100 Epoch)

### 7.1 Distribusi Slot Memori (Setelah 150 Sampel Inferensi)

| Status | Jumlah Slot | Persentase |
| :--- | :---: | :---: |
| ACTIVE | 19 | 14.8% |
| DORMANT | 2 | 1.6% |
| EXPIRED | 107 | 83.6% |

### 7.2 Metadata Slot Aktif

| Metrik | Importance | Confidence | Umur Fakta |
| :--- | :---: | :---: | :---: |
| Mean | 0.773 | 0.842 | 505.3 step |
| Min | 0.463 | 0.500 | 2 step |
| Max | 1.000 | 1.000 | 3,220 step |
| Std | 0.241 | 0.214 | — |

### 7.3 Pola Akses Read (448,182 Total Read Ops)

- **Rata-rata per slot**: 3,501 kali
- **Slot terpanas (max)**: 136,180 kali akses (Slot 0)
- **Hot Slots (≥20x)**: 12 slot pertama (slot 0–11)
- **Cold/Unaccessed**: 107 slot (slot EXPIRED, belum pernah ditulis)

> **Fenomena Hot Slot**: Slot-slot awal menerima mayoritas akses karena menyimpan fakta yang paling sering ditanyakan, sehingga mendapatkan skor recency dan importance tertinggi yang terus membuat mereka terpilih dalam Top-K (self-reinforcing loop).

### 7.4 Ketahanan Interferensi

| Kondisi Gangguan | Cosine Similarity Retrieval |
| :--- | :---: |
| 10 distractor noise rendah (0.1) | **0.8320 ± 0.069** |
| 50 distractor noise rendah (0.1) | **0.9235 ± 0.014** |
| 10 distractor noise tinggi (0.5) | 0.4322 ± 0.278 |
| 50 distractor noise tinggi (0.5) | 0.7201 ± 0.053 |

### 7.5 Benchmark End-to-End (100 Epoch, Seed 42)

| Baseline | EM | Token F1 | Recall@1 | Recall@5 | MRR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Memory Bank | 7.52% | 12.57% | **4.71%** | 6.17% | 0.0544 |
| NN Memory | 7.48% | 12.46% | 2.35% | 8.37% | **0.0704** |
| No Memory | **7.80%** | **12.61%** | — | — | — |

---

## 8. Perbandingan Implementasi: Flax vs PyTorch

| Aspek | Flax Asli | PyTorch (Locked) |
| :--- | :--- | :--- |
| **Storage Buffer** | `self.variable('memory', ...)` | `register_buffer(...)` |
| **Mutasi Buffer** | Mutable variable collection | `with torch.no_grad():` |
| **Projections** | `nn.Dense(dim, use_bias=False)` | `nn.Linear(dim, dim, bias=False)` |
| **Operasi Sequential** | `jax.lax.scan(...)` | Python `for` loop |
| **RNG** | `jax.random.PRNGKey(seed)` | `torch.manual_seed(seed)` |
| **Optimizer** | `optax.adamw(...)` | `torch.optim.AdamW(...)` |
| **Parity** | ✅ Identik secara matematis | ✅ Diverifikasi via `test_architecture_equivalence.py` |

---

## 9. Keterbatasan yang Diketahui

1. **Kapasitas Terbatas**: Dengan kapasitas 128 slot dan fakta lebih dari itu, terjadi eviction berdasarkan prioritas terendah.
2. **Sensitivity Threshold**: `memory_write_threshold` sangat mempengaruhi apakah fakta baru masuk sebagai UPDATE atau INSERT. Kalibrasi manual diperlukan per domain.
3. **Sequential Write**: Penulisan per-sample dalam loop Python (tidak di-vectorize), ~3x lebih lambat dari No Memory per epoch.
4. **Non-Differentiable Memory**: Gradien tidak mengalir melalui buffer slot. Pembelajaran "apa yang layak disimpan" bersifat implisit melalui `q_proj` dan `fusion_proj`.
5. **EOS-Dependent Write**: Penulisan hanya terjadi ketika `is_eos > 0.5`, artinya yang ditulis ke memori adalah pooled encoding seluruh urutan fakta, bukan token individual.
6. **Hot Slot Bias**: Slot yang sudah pernah dibaca banyak terus mendapat skor tinggi (importance reinforcement loop), berpotensi mengabaikan fakta yang lebih relevan tapi baru ditulis.

---

## 10. File Referensi

| File | Deskripsi |
| :--- | :--- |
| [`models/tiny_memory_bank.py`](file:///home/akhyar/Dokumen/Code/python/MemoryBank-bencmark/models/tiny_memory_bank.py) | Implementasi PyTorch (LOCKED) |
| [`mamoe/memory/bank.py`](file:///home/akhyar/Dokumen/Code/python/MemoryBank/mamoe/memory/bank.py) | Sumber kebenaran Flax asli |
| [`tests/test_architecture_equivalence.py`](file:///home/akhyar/Dokumen/Code/python/MemoryBank-bencmark/tests/test_architecture_equivalence.py) | Verifikasi parity matematis |
| [`tests/test_lifecycle.py`](file:///home/akhyar/Dokumen/Code/python/MemoryBank-bencmark/tests/test_lifecycle.py) | Uji siklus hidup method-driven |
| [`tests/test_memory_state.py`](file:///home/akhyar/Dokumen/Code/python/MemoryBank-bencmark/tests/test_memory_state.py) | Validasi state buffer |
| [`tests/test_causal.py`](file:///home/akhyar/Dokumen/Code/python/MemoryBank-bencmark/tests/test_causal.py) | Verifikasi gradient dan kausalitas |
| [`scripts/analyze_memory.py`](file:///home/akhyar/Dokumen/Code/python/MemoryBank-bencmark/scripts/analyze_memory.py) | Analisis distribusi slot & akses |
| [`experiments/long_term_memory_benchmark.py`](file:///home/akhyar/Dokumen/Code/python/MemoryBank-bencmark/experiments/long_term_memory_benchmark.py) | Evaluasi retensi jangka panjang |
| [`architecture_lock.json`](file:///home/akhyar/Dokumen/Code/python/MemoryBank-bencmark/architecture_lock.json) | Daftar komponen yang dikunci |
