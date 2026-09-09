# Rekomendasi Format Perbandingan untuk Laporan / Paper

Evaluasi dilakukan pada 2 sampel dialog percakapan test.

| Model / Pendekatan | Hit@1 Retrieval | Entity Recall | BERTScore F1 | Sentence Cosine | Avg. Input Tokens | Latency (ms/turn) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pure GPT-2 (Zero-shot)** | - | 0.00% | 35.63% | 55.97% | 13.5 | 400.8 ms |
| **Full-Context GPT-2** | - | 0.00% | 30.77% | 45.01% | 306.0 | 469.6 ms |
| **MemoryBank W_q (scaling='none')** | 0.00% | 0.00% | 44.52% | 71.11% | 13.5 | 443.6 ms |
| **MemoryBank W_q (scaling='sqrt')** | 0.00% | 0.00% | 40.33% | 61.81% | 13.5 | 437.3 ms |

### Detail Metrik Tambahan (Retrieval & Factual Alignment)

- **Hit@3 Retrieval**: `none`=0.00%, `sqrt`=0.00%
- **MRR**: `none`=0.2250, `sqrt`=0.2250
- **Entity Token Overlap**: `Pure`=0.00%, `Full-Context`=0.00%, `MemoryBank(none)`=10.00%
