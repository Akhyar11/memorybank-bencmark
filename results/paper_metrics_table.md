# Rekomendasi Format Perbandingan untuk Laporan / Paper

Evaluasi dilakukan pada 15 sampel dialog percakapan test.

| Model / Pendekatan | Hit@1 Retrieval | Entity Recall | BERTScore F1 | Sentence Cosine | Avg. Input Tokens | Latency (ms/turn) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pure GPT-2 (Zero-shot)** | - | 0.00% | 36.48% | 55.55% | 16.7 | 375.4 ms |
| **Full-Context GPT-2** | - | 13.33% | 37.21% | 55.14% | 279.4 | 465.2 ms |
| **MemoryBank W_q (scaling='dim')** | 0.00% | 0.00% | 43.34% | 57.71% | 16.7 | 441.7 ms |
| **MemoryBank W_q (scaling='sqrt')** | 0.00% | 0.00% | 42.91% | 58.55% | 16.7 | 440.0 ms |

### Detail Metrik Tambahan (Retrieval & Factual Alignment)

- **Hit@3 Retrieval**: `dim`=0.00%, `sqrt`=0.00%
- **MRR**: `dim`=0.1511, `sqrt`=0.1511
- **Entity Token Overlap**: `Pure`=0.00%, `Full-Context`=13.33%, `MemoryBank(sqrt)`=1.67%
