from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from answer_extraction import completion_stats


def compute_metrics(records: list[dict[str, Any]], long_threshold_words: int = 256) -> dict[str, Any]:
    """Compute core metrics from prediction records."""
    if not records:
        return {}
    stats = []
    for r in records:
        completion = r.get("completion") or r.get("prediction") or r.get("output") or ""
        gold = r.get("answer") or r.get("gold_answer") or r.get("label") or ""
        s = completion_stats(completion, gold)
        stats.append(s)
    n = len(stats)
    strict_correct = np.array([s["strict_correct"] for s in stats], dtype=float)
    lenient_correct = np.array([s["lenient_correct"] for s in stats], dtype=float)
    has_final = np.array([s["has_final_answer"] for s in stats], dtype=float)
    strict_extracted = np.array([s["answer_extracted_strict"] for s in stats], dtype=float)
    lenient_extracted = np.array([s["answer_extracted_lenient"] for s in stats], dtype=float)
    lengths = np.array([s["word_length"] for s in stats], dtype=float)
    long_wrong = np.array([(s["word_length"] > long_threshold_words) and (not s["strict_correct"]) for s in stats], dtype=float)
    if np.std(lengths) > 0 and np.std(strict_correct) > 0:
        length_acc_corr = float(np.corrcoef(lengths, strict_correct)[0, 1])
    else:
        length_acc_corr = 0.0
    return {
        "num_examples": n,
        "accuracy_strict": float(strict_correct.mean()),
        "accuracy_lenient": float(lenient_correct.mean()),
        "format_success_rate": float(has_final.mean()),
        "answer_extraction_rate_strict": float(strict_extracted.mean()),
        "answer_extraction_rate_lenient": float(lenient_extracted.mean()),
        "avg_response_length_words": float(lengths.mean()),
        "median_response_length_words": float(np.median(lengths)),
        "overthinking_rate": float(long_wrong.mean()),
        "length_accuracy_correlation": length_acc_corr,
    }


def add_prediction_stats(record: dict[str, Any]) -> dict[str, Any]:
    completion = record.get("completion") or record.get("prediction") or record.get("output") or ""
    gold = record.get("answer") or record.get("gold_answer") or record.get("label") or ""
    out = dict(record)
    out.update(completion_stats(completion, gold))
    return out
