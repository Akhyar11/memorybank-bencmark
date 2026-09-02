import os
import pandas as pd
import numpy as np
from tokenizers import Tokenizer

class TextDataLoader:
    def __init__(self, csv_file, tokenizer_path, batch_size=32, max_input_len=32, max_target_len=16):
        self.df = pd.read_csv(csv_file).head(2000)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # Pastikan ada PAD dan EOS token
        if "[PAD]" not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens(["[PAD]", "[EOS]"])
        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        self.eos_id = self.tokenizer.token_to_id("[EOS]")
        
        self.batch_size = batch_size
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.num_batches = int(np.ceil(len(self.df) / batch_size))
        
        # Konfigurasi tokenizer untuk otomasi padding/truncation
        self.tokenizer.enable_padding(pad_id=self.pad_id, length=max_input_len)
        self.tokenizer.enable_truncation(max_length=max_input_len)
        
    def pad_or_truncate(self, encoding, max_len):
        ids = encoding.ids
        mask = encoding.attention_mask
        
        if len(ids) > max_len:
            ids = ids[:max_len]
            mask = mask[:max_len]
        elif len(ids) < max_len:
            pad_len = max_len - len(ids)
            ids = ids + [self.pad_id] * pad_len
            mask = mask + [0] * pad_len
            
        return ids, mask
        
    def iter_batches(self, shuffle=True):
        indices = np.arange(len(self.df))
        if shuffle:
            np.random.shuffle(indices)
            
        for i in range(0, len(self.df), self.batch_size):
            batch_indices = indices[i:i+self.batch_size]
            
            write_texts = self.df.iloc[batch_indices]['write_fact_A'].tolist()
            query_texts = self.df.iloc[batch_indices]['query_B'].tolist()
            target_texts = self.df.iloc[batch_indices]['expected_output_A'].tolist()
            
            # Encode Write Facts (Fakta yang akan dimasukkan ke memori)
            write_encodings = self.tokenizer.encode_batch(write_texts)
            write_ids = np.array([self.pad_or_truncate(e, self.max_input_len)[0] for e in write_encodings])
            write_mask = np.array([self.pad_or_truncate(e, self.max_input_len)[1] for e in write_encodings])
            
            # Encode Queries (Pertanyaan Q&A)
            query_encodings = self.tokenizer.encode_batch(query_texts)
            query_ids = np.array([self.pad_or_truncate(e, self.max_input_len)[0] for e in query_encodings])
            query_mask = np.array([self.pad_or_truncate(e, self.max_input_len)[1] for e in query_encodings])
            
            # Encode Targets (Jawaban yang diharapkan)
            target_encodings = self.tokenizer.encode_batch(target_texts)
            target_ids = []
            for e in target_encodings:
                ids = e.ids
                if len(ids) >= self.max_target_len:
                    ids = ids[:self.max_target_len - 1]
                ids.append(self.eos_id)
                # manual pad
                pad_len = self.max_target_len - len(ids)
                ids = ids + [self.pad_id] * pad_len
                target_ids.append(ids)
            target_ids = np.array(target_ids)
            
            yield {
                'write_ids': write_ids,
                'write_mask': write_mask,
                'query_ids': query_ids,
                'query_mask': query_mask,
                'target_ids': target_ids
            }
