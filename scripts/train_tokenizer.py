import os
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def train_tokenizer():
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")
    train_file = os.path.join(dataset_path, "train.csv")
    
    # Baca semua kolom dataset
    df = pd.read_csv(train_file)
    text_data = df['write_fact_A'].tolist() + df['query_B'].tolist() + df['expected_output_A'].tolist()
    
    # Inisialisasi Tokenizer BPE (Byte-Pair Encoding)
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    # Setting Vocab Size 2000 sesuai permintaan
    trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"], vocab_size=2000)
    
    print("Mulai training tokenizer dari dataset...")
    tokenizer.train_from_iterator(text_data, trainer)
    
    # Simpan Tokenizer
    out_path = os.path.join(dataset_path, "tokenizer.json")
    tokenizer.save(out_path)
    
    print(f"✓ Tokenizer berhasil di-train!")
    print(f"  Vocab Size: {tokenizer.get_vocab_size()} (Target: 2000)")
    print(f"  Disimpan di: {out_path}")
    
    # Test encoding
    test_str = "Budi memiliki kucing bernama Mochi."
    output = tokenizer.encode(test_str)
    print(f"\n[CONTOH ENCODING]")
    print(f"Teks input  : {test_str}")
    print(f"Tokens      : {output.tokens}")
    print(f"Token IDs   : {output.ids}")

if __name__ == "__main__":
    train_tokenizer()
