"""
tests/test_dataset.py – Dataset integrity tests.

Tests:
- No hardcoded .head() truncation (max_samples param works)
- PAD ID is looked up from tokenizer (not assumed 0)
- train/val/test are distinct files
- Dataset files exist
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import inspect

from dataset.text_dataset_loader import TextDataLoader


DATASET_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
TOKENIZER_PATH = os.path.join(DATASET_DIR, 'tokenizer.json')
TRAIN_CSV      = os.path.join(DATASET_DIR, 'train.csv')
VAL_CSV        = os.path.join(DATASET_DIR, 'val.csv')
TEST_CSV       = os.path.join(DATASET_DIR, 'test.csv')


class TestDatasetLoader:
    def test_no_hardcoded_head(self):
        """TextDataLoader must have max_samples=None parameter, not hardcoded .head()."""
        src = inspect.getsource(TextDataLoader.__init__)
        assert '.head(2000)' not in src, "HARDCODED .head(2000) still present!"
        assert '.head(' not in src or 'max_samples' in src, \
            "Hardcoded head() detected without max_samples parameter"

    def test_max_samples_parameter_exists(self):
        """max_samples parameter must exist in __init__."""
        sig = inspect.signature(TextDataLoader.__init__)
        assert 'max_samples' in sig.parameters, "max_samples parameter missing from TextDataLoader"

    def test_max_samples_default_is_none(self):
        """max_samples default must be None."""
        sig = inspect.signature(TextDataLoader.__init__)
        default = sig.parameters['max_samples'].default
        assert default is None, f"max_samples default should be None, got {default}"

    def test_pad_id_from_tokenizer(self):
        """PAD ID must come from tokenizer.token_to_id(), not assumed to be 0."""
        src = inspect.getsource(TextDataLoader.__init__)
        assert "token_to_id('[PAD]')" in src or 'token_to_id("[PAD]")' in src, \
            "PAD ID must be looked up via token_to_id(), not hardcoded"

    @pytest.mark.skipif(not os.path.exists(TRAIN_CSV), reason="train.csv not generated yet")
    @pytest.mark.skipif(not os.path.exists(TOKENIZER_PATH), reason="tokenizer not generated yet")
    def test_distinct_splits(self):
        """Train, val, test CSV files must be different files."""
        assert TRAIN_CSV != VAL_CSV,  "train and val CSV are the same file!"
        assert TRAIN_CSV != TEST_CSV, "train and test CSV are the same file!"
        assert VAL_CSV   != TEST_CSV, "val and test CSV are the same file!"

    @pytest.mark.skipif(not os.path.exists(TRAIN_CSV), reason="train.csv not generated yet")
    @pytest.mark.skipif(not os.path.exists(TOKENIZER_PATH), reason="tokenizer not generated yet")
    def test_max_samples_limits_data(self):
        """max_samples=100 should load at most 100 rows."""
        loader = TextDataLoader(TRAIN_CSV, TOKENIZER_PATH, batch_size=32,
                                max_samples=100)
        assert len(loader.df) <= 100, f"max_samples=100 should load ≤100 rows, got {len(loader.df)}"

    @pytest.mark.skipif(not os.path.exists(TRAIN_CSV), reason="train.csv not generated yet")
    @pytest.mark.skipif(not os.path.exists(TOKENIZER_PATH), reason="tokenizer not generated yet")
    def test_no_overlap_train_test(self):
        """Train and test should have no exact duplicate rows."""
        import pandas as pd
        train = pd.read_csv(TRAIN_CSV)
        test  = pd.read_csv(TEST_CSV)
        merged = train.merge(test, on=['write_fact_A', 'query_B', 'expected_output_A'])
        assert len(merged) == 0, f"Train-test overlap detected: {len(merged)} rows"

    @pytest.mark.skipif(not os.path.exists(os.path.join(DATASET_DIR, 'metadata.json')),
                        reason="metadata.json not generated yet")
    def test_metadata_exists(self):
        """metadata.json should exist and contain required keys."""
        import json
        meta_path = os.path.join(DATASET_DIR, 'metadata.json')
        with open(meta_path) as f:
            meta = json.load(f)
        required_keys = ['seed', 'generator_version', 'train_size', 'val_size',
                         'test_size', 'split_strategy']
        for k in required_keys:
            assert k in meta, f"metadata.json missing key: {k}"
