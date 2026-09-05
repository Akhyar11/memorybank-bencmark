# MemoryBank-bencmark: GPT-2 Indo + Mature TinyMemoryBank

Repositori ini adalah rumah utama untuk integrasi dan pengujian **Backbone GPT-2 Indo (Frozen)** dengan arsitektur **TinyMemoryBank (Trainable Adapter)** yang telah terkunci (*architecture-locked*) dan teraudit secara matematis.

---

## 📁 Struktur Direktori

```text
MemoryBank-bencmark/
├── dataset/
│   ├── conversations_100M_train.jsonl    # Dataset percakapan latih multi-turn
│   ├── conversations_100M_val.jsonl      # Dataset validasi
│   ├── conversations_100M_test.jsonl     # Dataset evaluasi
│   └── conversation_dataset.py           # DataLoader
├── gpt2-indo-instruct-tuned/             # Bobot Pretrained GPT-2 Indo
├── models/
│   ├── tiny_memory_bank.py               # Locked & Audited Neural Memory Bank
│   └── gpt2_memory_model.py              # Wrapper GPT-2 Frozen + TinyMemoryBank
├── scripts/
│   ├── chat_gpt2_mature_memory.py        # Chat CLI interaktif dengan status slot live
│   ├── train_gpt2_mature_memory.py       # Script training efisien khusus Memory Bank
│   ├── test_gpt2_indo.py                 # Pengujian inferensi dasar GPT-2
│   ├── generate_100m_tokens_dataset.py   # Generator dataset percakapan 100M token
│   └── generate_conversation_dataset.py  # Generator percakapan berbasis fakta & slot
├── tests/
│   ├── test_architecture_lock.py         # Verifikasi kontrak arsitektur locked
│   ├── test_gpt2_mature_memory.py        # Verifikasi pembekuan parameter & gradien
│   └── conftest.py
├── architecture_lock.json
├── requirements.txt
└── README.md
```

---

## 🚀 Penggunaan Cepat

### 1. Chat Interaktif dengan Memory Bank
```bash
python3 scripts/chat_gpt2_mature_memory.py
```
Perintah interaktif dalam chat:
- `/slots` : Melihat status slot memori yang aktif, importance, dan confidence.
- `/decay` : Mengaplikasikan peluruhan eksponensial pada memori.
- `/reset` : Mengosongkan memori.
- `/exit`  : Keluar dari chat.

### 2. Menjalankan Unit Tests (18 Tests)
```bash
pytest -v
```

### 3. Pelatihan Memory Bank
```bash
# Dry run uji coba forward pass:
python3 scripts/train_gpt2_mature_memory.py --dry-run

# Pelatihan penuh:
python3 scripts/train_gpt2_mature_memory.py --epochs 3 --batch_size 4
```

### 4. Membuat Dataset Percakapan Tambahan
```bash
python3 scripts/generate_conversation_dataset.py
```
