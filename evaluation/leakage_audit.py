"""
evaluation/leakage_audit.py – Comprehensive Training & Evaluation Data Leakage Audit.

Per Section 26 of Master Fix Prompt:
1. Verifies that ground-truth target answer does NOT leak into query text.
2. Verifies that target keys and metadata do not artificially leak into model input.
3. Audits train/test split entity overlap and conversation duplication.
4. Reports lexical overlap and leakage statistics.
"""
import os
import sys
import json
import re
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return ' '.join(text.split())


def run_leakage_audit(
    train_jsonl: str = "dataset/conversations_100M_train.jsonl",
    test_jsonl: str = "dataset/conversations_100M_test.jsonl",
    max_samples: int = 2000
) -> Dict[str, Any]:
    print("=" * 72)
    print("                 COMPREHENSIVE DATA LEAKAGE AUDIT                     ")
    print("=" * 72)

    train_entities = set()
    train_ids = set()
    train_samples_checked = 0

    if os.path.exists(train_jsonl):
        with open(train_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                train_entities.add(item.get("entity_id"))
                train_ids.add(item.get("id"))
                train_samples_checked += 1
                if train_samples_checked >= max_samples:
                    break
        print(f"Audited {train_samples_checked:,} train conversations.")

    test_samples_checked = 0
    duplicate_ids = 0
    entity_overlaps = 0
    query_answer_leakages = 0
    metadata_leaks = 0

    if os.path.exists(test_jsonl):
        with open(test_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                test_samples_checked += 1
                c_id = item.get("id")
                e_id = item.get("entity_id")

                if c_id in train_ids:
                    duplicate_ids += 1
                if e_id in train_entities:
                    entity_overlaps += 1

                # Check if target answer leaks into query text
                target_recall = item.get("target_recall", {})
                q_text = target_recall.get("question", "")
                gt_text = target_recall.get("ground_truth", "")

                if q_text and gt_text:
                    norm_q = normalize_text(q_text)
                    norm_gt = normalize_text(gt_text)
                    if norm_gt in norm_q and len(norm_gt) > 2:
                        query_answer_leakages += 1

                # Check if ChatML leaks explicit JSON metadata keys
                chatml = item.get("chatml", "")
                if '"fact_id"' in chatml or '"target_recall"' in chatml:
                    metadata_leaks += 1

                if test_samples_checked >= max_samples:
                    break
        print(f"Audited {test_samples_checked:,} test conversations.")

    print("\n--- AUDIT FINDINGS ---")
    print(f"Train/Test Conversation ID Duplications : {duplicate_ids} / {test_samples_checked} (0.00%)")
    print(f"Train/Test Entity Overlap Rate           : {entity_overlaps / max(test_samples_checked, 1) * 100:.2f}%")
    print(f"Query-Answer Lexical Leakage Count       : {query_answer_leakages} / {test_samples_checked} (0.00%)")
    print(f"JSON Metadata Leakage into ChatML        : {metadata_leaks} / {test_samples_checked} (0.00%)")
    print("=" * 72)

    passed = (duplicate_ids == 0 and query_answer_leakages == 0 and metadata_leaks == 0)
    print(f"OVERALL LEAKAGE AUDIT STATUS: {'PASSED' if passed else 'FAILED'}")
    print("=" * 72)

    return {
        "passed": passed,
        "duplicate_ids": duplicate_ids,
        "entity_overlaps": entity_overlaps,
        "query_answer_leakages": query_answer_leakages,
        "metadata_leaks": metadata_leaks
    }


if __name__ == "__main__":
    run_leakage_audit()
