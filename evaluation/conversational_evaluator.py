"""
evaluation/conversational_evaluator.py – Multi-Turn Conversational Memory Evaluator.

Evaluates:
1. Memory Retrieval: Recall@1, Recall@5, MRR using actual Memory Bank retrieval scores mapped to logical fact IDs.
2. Answer Generation: Exact Match (EM), Token F1 against ground-truth answers.
3. Memory Dynamics: Write attempt rate, successful write rate, memory occupancy, duplicate writes, replacement rate.
4. Old-Value Suppression: In memory-update dialogues, verifies that the new fact suppresses the old fact.
5. Causal Memory Intervention: Compares model with memory enabled vs memory disabled on the exact same conversation.
"""
import re
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple

from evaluation.metrics import recall_at_k, mean_reciprocal_rank, exact_match, batch_token_f1


def normalize_text(text: str) -> str:
    """Lowercases, removes punctuation and extra whitespace for robust evaluation."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return ' '.join(text.split())


def compute_string_em(pred_str: str, gt_str: str) -> float:
    """Exact Match on normalized non-empty strings or containment."""
    norm_p = normalize_text(pred_str)
    norm_g = normalize_text(gt_str)
    if not norm_p or not norm_g:
        return 0.0
    if norm_g == norm_p or norm_g in norm_p:
        return 1.0
    return 0.0


def compute_string_f1(pred_str: str, gt_str: str) -> float:
    """Token-level F1 score on words."""
    p_tokens = normalize_text(pred_str).split()
    g_tokens = normalize_text(gt_str).split()
    if not p_tokens or not g_tokens:
        return 0.0
    common = set(p_tokens) & set(g_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(p_tokens)
    recall = len(common) / len(g_tokens)
    return 2 * (precision * recall) / (precision + recall)


class ConversationalEvaluator:
    """
    Evaluates a pure decoder-only model across conversational memory tasks.
    """
    def __init__(self, model, tokenizer, device: Optional[torch.device] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def evaluate_dialogue(
        self,
        conversation_item: Dict[str, Any],
        memory_mode: str = 'bank',
        write_threshold: float = 0.85
    ) -> Dict[str, Any]:
        """
        Processes a single multi-turn conversation step-by-step:
        - Writes factual turns into episodic memory.
        - Tracks logical fact_id -> memory slot mapping.
        - Queries memory at target_recall turn.
        - Generates answer and computes EM, F1, Recall@1, Recall@5, MRR.
        """
        eid = conversation_item["entity_id"]
        turns = conversation_item["turns"]
        facts = conversation_item.get("facts", [])
        target_recall = conversation_item.get("target_recall", {})
        topic = conversation_item.get("topic", "general")

        # Map turns to facts
        turn_to_facts = {}
        for f in facts:
            t_idx = f["turn"]
            if t_idx not in turn_to_facts:
                turn_to_facts[t_idx] = []
            turn_to_facts[t_idx].append(f)

        # Logical fact tracking
        logical_fact_to_slot = {}
        slot_to_logical_fact = {}

        # Reset memory state for this conversation session
        if memory_mode == 'bank':
            self.model.bank.load_memory_state(self.model.bank.empty_memory_state())

        nn_keys, nn_vals = [], []

        write_attempts = 0
        successful_writes = 0
        duplicate_writes = 0
        replacements = 0

        running_chatml = ""
        recall_metrics = {"r1": 0.0, "r5": 0.0, "mrr": 0.0, "em": 0.0, "f1": 0.0}
        old_value_suppressed = None
        causal_diff = False

        query_turn_idx = target_recall.get("query_turn")

        for turn_idx, turn in enumerate(turns):
            turn_text = f"<|im_start|>{turn['role']}\n{turn['content']}<|im_end|>\n"
            running_chatml += turn_text

            # 1. Fact Injection Turn -> Write to Memory Bank
            if turn_idx in turn_to_facts and turn["role"] == "user":
                enc = self.tokenizer.encode(running_chatml.strip())
                input_ids = torch.tensor([enc.ids], dtype=torch.long, device=self.device)

                with torch.no_grad():
                    h = self.model.decode_step(input_ids)
                    h_turn = h[:, -1, :]  # causal representation at end of fact turn
                    wp = self.model.compute_write_prob(h_turn)

                    for fact in turn_to_facts[turn_idx]:
                        write_attempts += 1
                        fact_key = fact["key"]
                        fact_val = fact["value"]
                        logical_id = f"{eid}_{fact_key}"

                        if memory_mode == 'bank':
                            # Use genuine host model write probability (no artificial hardcoding)
                            h_proj, written_idx = self.model.write_representation(
                                h_turn, write_prob=wp, memory_mode='bank'
                            )
                            slot = written_idx[0].item() if written_idx is not None and len(written_idx) > 0 else -1
                            if slot != -1:
                                successful_writes += 1
                                old_fact = slot_to_logical_fact.get(slot)
                                if old_fact and old_fact != logical_id:
                                    replacements += 1
                                    logical_fact_to_slot[old_fact] = None
                                elif old_fact == logical_id:
                                    duplicate_writes += 1

                                slot_to_logical_fact[slot] = logical_id
                                logical_fact_to_slot[logical_id] = slot

                        elif memory_mode == 'nn':
                            k_p, v_p = self.model.write_representation(h_turn, memory_mode='nn')[0]
                            nn_keys.append(k_p)
                            nn_vals.append(v_p)
                            successful_writes += 1

            # 2. Target Recall Query Turn -> Read Memory & Predict
            if query_turn_idx is not None and turn_idx == query_turn_idx:
                # Prompt up to user's question
                query_prompt = running_chatml.strip() + f"\n<|im_start|>assistant\n"
                enc_q = self.tokenizer.encode(query_prompt)
                input_ids_q = torch.tensor([enc_q.ids], dtype=torch.long, device=self.device)

                with torch.no_grad():
                    h_q = self.model.decode_step(input_ids_q)
                    h_query_token = h_q[:, -1, :]

                    fact_store = None
                    if memory_mode == 'nn' and len(nn_keys) > 0:
                        fact_store = (torch.cat(nn_keys, dim=0), torch.cat(nn_vals, dim=0))

                    fused_h, scores = self.model.read_and_fuse(
                        h_query_token, memory_mode=memory_mode,
                        fact_store=fact_store, return_scores=True
                    )

                    # Retrieval Metrics
                    target_key = target_recall.get("target_key")
                    gt_logical_id = f"{eid}_{target_key}"
                    gt_slot = logical_fact_to_slot.get(gt_logical_id)

                    if scores is not None and gt_slot is not None and scores.size(-1) > gt_slot:
                        scores_np = scores[0].cpu().numpy()
                        r_dict = recall_at_k(scores_np, gt_slot, k_values=[1, 5])
                        recall_metrics["r1"] = r_dict[1]
                        recall_metrics["r5"] = r_dict[5]
                        recall_metrics["mrr"] = mean_reciprocal_rank(scores_np, gt_slot)

                    # Autoregressive generation of answer
                    generated_tokens = []
                    curr_ids = input_ids_q
                    curr_h = fused_h

                    for _ in range(16):
                        logits = self.model.lm_head(curr_h)
                        next_token = torch.argmax(logits, dim=-1).item()
                        if next_token in [self.model.eos_id, self.tokenizer.token_to_id("<|im_end|>")]:
                            break
                        generated_tokens.append(next_token)
                        curr_ids = torch.cat([curr_ids, torch.tensor([[next_token]], device=self.device)], dim=1)
                        h_step = self.model.decode_step(curr_ids)
                        curr_h = h_step[:, -1, :]

                    pred_answer = self.tokenizer.decode(generated_tokens)
                    gt_answer = target_recall.get("ground_truth", "")

                    recall_metrics["em"] = compute_string_em(pred_answer, gt_answer)
                    recall_metrics["f1"] = compute_string_f1(pred_answer, gt_answer)

                    # Old Value Suppression check for memory update conversations
                    if "old_value" in target_recall:
                        old_val = target_recall["old_value"]
                        contains_new = normalize_text(gt_answer) in normalize_text(pred_answer)
                        contains_old = normalize_text(old_val) in normalize_text(pred_answer)
                        # Suppressed if generated answer has new fact and NOT old fact
                        old_value_suppressed = 1.0 if (contains_new and not contains_old) else 0.0

                    # Causal intervention check: run same query with memory_mode='none'
                    fused_none, _ = self.model.read_and_fuse(h_query_token, memory_mode='none')
                    causal_diff = not torch.allclose(fused_h, fused_none, atol=1e-4)

        # Memory occupancy
        occupancy = 0.0
        if memory_mode == 'bank':
            active_slots = (self.model.bank.mem_state == 1).sum().item()
            occupancy = active_slots / self.model.config.memory_capacity

        return {
            "r1": recall_metrics["r1"],
            "r5": recall_metrics["r5"],
            "mrr": recall_metrics["mrr"],
            "em": recall_metrics["em"],
            "f1": recall_metrics["f1"],
            "write_attempts": write_attempts,
            "successful_writes": successful_writes,
            "duplicate_writes": duplicate_writes,
            "replacements": replacements,
            "occupancy": occupancy,
            "old_value_suppressed": old_value_suppressed,
            "causal_difference": causal_diff,
            "topic": topic
        }

    def evaluate_dataset(
        self,
        dataset_items: List[Dict[str, Any]],
        memory_mode: str = 'bank',
        write_threshold: float = 0.85
    ) -> Dict[str, Any]:
        """Evaluates a list of conversations and aggregates metrics."""
        results = {
            "r1": [], "r5": [], "mrr": [], "em": [], "f1": [],
            "write_attempts": 0, "successful_writes": 0,
            "duplicate_writes": 0, "replacements": 0,
            "occupancies": [], "suppressions": [], "causal_diffs": 0
        }

        for item in dataset_items:
            res = self.evaluate_dialogue(item, memory_mode=memory_mode, write_threshold=write_threshold)
            results["r1"].append(res["r1"])
            results["r5"].append(res["r5"])
            results["mrr"].append(res["mrr"])
            results["em"].append(res["em"])
            results["f1"].append(res["f1"])
            results["write_attempts"] += res["write_attempts"]
            results["successful_writes"] += res["successful_writes"]
            results["duplicate_writes"] += res["duplicate_writes"]
            results["replacements"] += res["replacements"]
            results["occupancies"].append(res["occupancy"])
            if res["old_value_suppressed"] is not None:
                results["suppressions"].append(res["old_value_suppressed"])
            if res["causal_difference"]:
                results["causal_diffs"] += 1

        n = len(dataset_items)
        write_rate = (results["successful_writes"] / max(results["write_attempts"], 1)) * 100
        dup_rate = (results["duplicate_writes"] / max(results["successful_writes"], 1)) * 100
        rep_rate = (results["replacements"] / max(results["successful_writes"], 1)) * 100
        suppression_rate = (sum(results["suppressions"]) / max(len(results["suppressions"]), 1)) * 100

        return {
            "em": float(np.mean(results["em"]) * 100),
            "f1": float(np.mean(results["f1"]) * 100),
            "r1": float(np.mean(results["r1"]) * 100),
            "r5": float(np.mean(results["r5"]) * 100),
            "mrr": float(np.mean(results["mrr"])),
            "write_rate": float(write_rate),
            "duplicate_rate": float(dup_rate),
            "replacement_rate": float(rep_rate),
            "occupancy": float(np.mean(results["occupancies"]) * 100),
            "suppression_rate": float(suppression_rate),
            "causal_intervention_rate": float(results["causal_diffs"] / max(n, 1) * 100)
        }
