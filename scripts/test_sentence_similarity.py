import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

def test_similarity():
    model_dir = "gpt2-indo-instruct-tuned" if os.path.exists("gpt2-indo-instruct-tuned") else "izzulgod/gpt2-indo-instruct-tuned"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)
    model.eval()

    s1 = "Aku makan ayam"
    s2 = "Ayam makan aku"

    inputs1 = tokenizer(s1, return_tensors="pt")
    inputs2 = tokenizer(s2, return_tensors="pt")

    print(f"Sentence 1: '{s1}'")
    print(f"Tokens 1  : {[tokenizer.decode([t]) for t in inputs1['input_ids'][0]]}")
    print(f"IDs 1     : {inputs1['input_ids'][0].tolist()}")

    print(f"\nSentence 2: '{s2}'")
    print(f"Tokens 2  : {[tokenizer.decode([t]) for t in inputs2['input_ids'][0]]}")
    print(f"IDs 2     : {inputs2['input_ids'][0].tolist()}")

    with torch.no_grad():
        out1 = model(**inputs1)
        out2 = model(**inputs2)

    # 1. Last token hidden state (metode yang dipakai memory bank saat ini: h[:, -1, :])
    last1 = out1.last_hidden_state[:, -1, :]
    last2 = out2.last_hidden_state[:, -1, :]
    cos_last = F.cosine_similarity(last1, last2).item()
    l2_last = torch.norm(last1 - last2).item()

    # 2. Mean pooling (rata-rata semua token)
    mean1 = out1.last_hidden_state.mean(dim=1)
    mean2 = out2.last_hidden_state.mean(dim=1)
    cos_mean = F.cosine_similarity(mean1, mean2).item()
    l2_mean = torch.norm(mean1 - mean2).item()

    print("\n" + "=" * 50)
    print("HASIL PENGUKURAN SIMILARITAS (768-D):")
    print("=" * 50)
    print(f"1. Last-Token Pooling (Metode Memory Bank saat ini):")
    print(f"   - Cosine Similarity : {cos_last:.4f} ({cos_last * 100:.2f}%)")
    print(f"   - Euclidean Dist (L2): {l2_last:.4f}")
    print(f"\n2. Mean Pooling (Rata-rata representasi kalimat):")
    print(f"   - Cosine Similarity : {cos_mean:.4f} ({cos_mean * 100:.2f}%)")
    print(f"   - Euclidean Dist (L2): {l2_mean:.4f}")
    print("=" * 50)

    # Cek juga baseline kontrol: kalimat acak lain yang tidak ada hubungannya
    s_control = "Hari ini cuaca cerah sekali di langit"
    inputs_c = tokenizer(s_control, return_tensors="pt")
    with torch.no_grad():
        out_c = model(**inputs_c)
    last_c = out_c.last_hidden_state[:, -1, :]
    cos_control = F.cosine_similarity(last1, last_c).item()
    print(f"Baseline Kontrol ('Aku makan ayam' vs 'Hari ini cuaca cerah...'):")
    print(f"   - Cosine Similarity : {cos_control:.4f} ({cos_control * 100:.2f}%)")
    print("=" * 50)

if __name__ == "__main__":
    test_similarity()
