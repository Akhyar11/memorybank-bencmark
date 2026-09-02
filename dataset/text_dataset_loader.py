"""
TextDataLoader – Fixed dataset loader.

Fixes applied:
- BUG-P0-023: Removed hardcoded .head(2000) – replaced with max_samples=None parameter
- BUG-P2-004: PAD ID always looked up from tokenizer, never assumed 0
"""
import os
import pandas as pd
import numpy as np
from tokenizers import Tokenizer


class TextDataLoader:
    """
    Loads and iterates over text QA dataset (write_fact_A, query_B, expected_output_A).

    Args:
        csv_file:       Path to CSV file.
        tokenizer_path: Path to tokenizer JSON.
        batch_size:     Number of samples per batch.
        max_input_len:  Max token length for input sequences.
        max_target_len: Max token length for target sequences.
        max_samples:    Maximum number of samples to load.
                        None = load all samples (default).
                        Set explicitly when a specific limit is needed.
    """

    def __init__(self, csv_file, tokenizer_path, batch_size=32,
                 max_input_len=32, max_target_len=16, max_samples=None):
        df = pd.read_csv(csv_file)
        if max_samples is not None:
            df = df.head(max_samples)
        self.df = df

        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        # Always look up special token IDs from tokenizer (never assume 0)
        vocab = self.tokenizer.get_vocab()
        if '[PAD]' not in vocab:
            self.tokenizer.add_special_tokens(['[PAD]'])
        if '[EOS]' not in vocab:
            self.tokenizer.add_special_tokens(['[EOS]'])
        if '[BOS]' not in vocab:
            self.tokenizer.add_special_tokens(['[BOS]'])
        if '[UNK]' not in vocab:
            self.tokenizer.add_special_tokens(['[UNK]'])

        self.pad_id = self.tokenizer.token_to_id('[PAD]')
        self.eos_id = self.tokenizer.token_to_id('[EOS]')
        self.bos_id = self.tokenizer.token_to_id('[BOS]')
        self.unk_id = self.tokenizer.token_to_id('[UNK]')

        assert self.pad_id is not None, "Tokenizer missing [PAD] token"
        assert self.eos_id is not None, "Tokenizer missing [EOS] token"

        self.batch_size     = batch_size
        self.max_input_len  = max_input_len
        self.max_target_len = max_target_len
        self.num_batches    = int(np.ceil(len(self.df) / batch_size))

        self.tokenizer.enable_padding(pad_id=self.pad_id, length=max_input_len)
        self.tokenizer.enable_truncation(max_length=max_input_len)

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

    def iter_batches(self, shuffle=True):
        indices = np.arange(len(self.df))
        if shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(self.df), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]

            write_texts  = self.df.iloc[batch_indices]['write_fact_A'].tolist()
            query_texts  = self.df.iloc[batch_indices]['query_B'].tolist()
            target_texts = self.df.iloc[batch_indices]['expected_output_A'].tolist()

            # Encode write facts
            write_encs = self.tokenizer.encode_batch(write_texts)
            write_ids_list  = [self.pad_or_truncate(e, self.max_input_len)[0] for e in write_encs]
            write_mask_list = [self.pad_or_truncate(e, self.max_input_len)[1] for e in write_encs]

            # Encode queries
            query_encs = self.tokenizer.encode_batch(query_texts)
            query_ids_list  = [self.pad_or_truncate(e, self.max_input_len)[0] for e in query_encs]
            query_mask_list = [self.pad_or_truncate(e, self.max_input_len)[1] for e in query_encs]

            # Encode targets: truncate, append EOS, pad
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
                
            valid_count = len(write_texts)
            
            # Pad batch to batch_size if it's a partial batch
            if valid_count < self.batch_size:
                pad_rows = self.batch_size - valid_count
                write_ids_list.extend([[self.pad_id] * self.max_input_len] * pad_rows)
                write_mask_list.extend([[0] * self.max_input_len] * pad_rows)
                query_ids_list.extend([[self.pad_id] * self.max_input_len] * pad_rows)
                query_mask_list.extend([[0] * self.max_input_len] * pad_rows)
                target_ids_list.extend([[self.pad_id] * self.max_target_len] * pad_rows)

            yield {
                'write_ids':   np.array(write_ids_list),
                'write_mask':  np.array(write_mask_list),
                'query_ids':   np.array(query_ids_list),
                'query_mask':  np.array(query_mask_list),
                'target_ids':  np.array(target_ids_list),
                'valid_count': valid_count,
            }
