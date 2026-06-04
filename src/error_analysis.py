from __future__ import annotations

from typing import Any

from metrics import add_prediction_stats

ERROR_TYPES = [
    "算术错误",
    "推理路径错误",
    "条件理解错误",
    "答案格式错误",
    "过度推理",
    "幻觉条件",
    "奖励交叉检查",
]


def build_error_case_rows(records: list[dict[str, Any]], limit: int = 50) -> list[dict[str, Any]]:
    """Build a CSV-friendly list of wrong cases for manual annotation."""
    rows = []
    for r in records:
        s = add_prediction_stats(r)
        if not s.get("strict_correct", False):
            rows.append(
                {
                    "question": s.get("question", ""),
                    "gold_answer": s.get("answer", s.get("gold_answer", "")),
                    "pred_strict": s.get("pred_strict"),
                    "pred_lenient": s.get("pred_lenient"),
                    "word_length": s.get("word_length"),
                    "has_final_answer": s.get("has_final_answer"),
                    "completion": s.get("completion", "")[:1500],
                    "manual_error_type": "待标注",
                    "notes": "",
                }
            )
        if len(rows) >= limit:
            break
    return rows
