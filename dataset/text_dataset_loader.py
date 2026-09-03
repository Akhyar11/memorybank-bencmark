"""
TextDataLoader – PyTorch dataset loader.
"""
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer


class TextDataset(Dataset):
    """
    PyTorch Dataset for text QA (write_fact_A, query_B, expected_output_A).
    """
    def __init__(self, csv_file, tokenizer_path, max_input_len=32, max_target_len=16, max_samples=None):
        df = pd.read_csv(csv_file)
        if max_samples is not None:
            df = df.head(max_samples)
        self.df = df

        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        # Always look up special token IDs from tokenizer
        vocab = self.tokenizer.get_vocab()
        for token in ['[PAD]', '[EOS]', '[BOS]', '[UNK]']:
            if token not in vocab:
                self.tokenizer.add_special_tokens([token])

        self.pad_id = self.tokenizer.token_to_id('[PAD]')
        self.eos_id = self.tokenizer.token_to_id('[EOS]')
        self.bos_id = self.tokenizer.token_to_id('[BOS]')
        self.unk_id = self.tokenizer.token_to_id('[UNK]')

        self.max_input_len  = max_input_len
        self.max_target_len = max_target_len

        self.tokenizer.enable_padding(pad_id=self.pad_id, length=max_input_len)
        self.tokenizer.enable_truncation(max_length=max_input_len)
        
        self._pre_tokenize_dataset()

    def _pre_tokenize_dataset(self):
        self.fact_ids    = self.df['fact_id'].tolist()
        self.query_ids   = self.df['query_id'].tolist()
        write_texts  = self.df['write_fact_A'].tolist()
        query_texts  = self.df['query_B'].tolist()
        target_texts = self.df['expected_output_A'].tolist()

        write_encs = self.tokenizer.encode_batch(write_texts)
        write_ids_list  = [self.pad_or_truncate(e, self.max_input_len)[0] for e in write_encs]
        write_mask_list = [self.pad_or_truncate(e, self.max_input_len)[1] for e in write_encs]

        query_encs = self.tokenizer.encode_batch(query_texts)
        query_ids_list  = [self.pad_or_truncate(e, self.max_input_len)[0] for e in query_encs]
        query_mask_list = [self.pad_or_truncate(e, self.max_input_len)[1] for e in query_encs]

        target_encs = self.tokenizer.encode_batch(target_texts)
        target_ids_list  = []
        for e in target_encs:
            ids = e.ids
            if len(ids) >= self.max_target_len:
                ids = ids[:self.max_target_len - 1]
            ids.append(self.eos_id)
            pad_len = self.max_target_len - len(ids)
            ids = ids + [self.pad_id] * pad_len
            target_ids_list.append(ids)

        self.all_write_ids = torch.tensor(write_ids_list, dtype=torch.long)
        self.all_write_mask = torch.tensor(write_mask_list, dtype=torch.long)
        self.all_query_ids = torch.tensor(query_ids_list, dtype=torch.long)
        self.all_query_mask = torch.tensor(query_mask_list, dtype=torch.long)
        self.all_target_ids = torch.tensor(target_ids_list, dtype=torch.long)

    def pad_or_truncate(self, encoding, max_len):
        ids  = encoding.ids
        mask = encoding.attention_mask
        if len(ids) > max_len:
            ids  = ids[:max_len]
            mask = mask[:max_len]
        elif len(ids) < max_len:
            pad_len = max_len - len(ids)
            ids     = ids  + [self.pad_id] * pad_len
            mask    = mask + [0]           * pad_len
        return ids, mask

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            'write_ids': self.all_write_ids[idx],
            'write_mask': self.all_write_mask[idx],
            'query_ids': self.all_query_ids[idx],
            'query_mask': self.all_query_mask[idx],
            'target_ids': self.all_target_ids[idx],
            'fact_str_id': self.fact_ids[idx],
            'query_str_id': self.query_ids[idx]
        }

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

def get_dataloader(csv_file, tokenizer_path, batch_size=32, max_input_len=32, max_target_len=16, max_samples=None, shuffle=True):
    dataset = TextDataset(csv_file, tokenizer_path, max_input_len, max_target_len, max_samples)
    
    def collate_fn(batch):
        # Default PyTorch collate will batch tensors. 
        # For strings (fact_str_id), it will create a tuple of strings.
        collated = torch.utils.data.dataloader.default_collate(batch)
        # In PyTorch dataloader, if batch is partial (at the end), it just returns smaller tensors.
        # We don't need to manually pad the batch dimension to a fixed size unlike in JAX/TPU XLA!
        return collated
        
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    loader.vocab_size = dataset.vocab_size
    loader.pad_id = dataset.pad_id
    loader.bos_id = dataset.bos_id
    loader.eos_id = dataset.eos_id
    
    return loader
