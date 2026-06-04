"""Robust answer extraction for GSM8K/GRPO.

The module provides two extraction modes:
- strict: only accepts explicit final-answer markers such as "Final Answer:".
- lenient: falls back to the last numeric expression in the completion.

GRPO rewards should prefer strict extraction when format compliance matters.
Final analysis can report both strict and lenient metrics.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction
from typing import Any

FINAL_MARKERS = [
    r"final\s*answer\s*[:：]",
    r"answer\s*[:：]",
    r"答案\s*[:：是为]",
    r"最终答案\s*[:：是为]",
    r"####\s*",
]

# Supports: -1, 1,234, 3.14, -2/3, 1,000.5
NUMBER_PATTERN = re.compile(
    r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d+\.\d+|[-+]?\d+\s*/\s*[-+]?\d+|[-+]?\d+"
)


def _clean_numeric_text(s: str) -> str:
    s = str(s).strip()
    s = s.replace("$", "").replace("￥", "").replace("€", "").replace("£", "")
    s = s.replace(",", "")
    s = s.replace("％", "%")
    # Keep the numeric value, not the percent ratio, because GSM8K answers usually compare literal numbers.
    s = s.replace("%", "")
    # Remove common trailing units/words, but keep signs, decimal points and slash.
    s = re.sub(r"[^0-9+\-./]", "", s)
    return s


def parse_number(s: Any) -> float | None:
    """Parse an extracted numeric string into float.

    Returns None for invalid values.
    """
    if s is None:
        return None
    raw = _clean_numeric_text(str(s))
    if not raw:
        return None
    try:
        if "/" in raw:
            return float(Fraction(raw))
        return float(raw)
    except Exception:
        return None


def _first_number_after_marker(text: str) -> str | None:
    lowered = text.lower()
    candidates: list[tuple[int, str]] = []
    for marker in FINAL_MARKERS:
        for m in re.finditer(marker, lowered, flags=re.IGNORECASE):
            tail = text[m.end() : m.end() + 120]
            num = NUMBER_PATTERN.search(tail)
            if num:
                candidates.append((m.start(), num.group(0)))
    if not candidates:
        return None
    # Use the last explicit marker; models sometimes mention an earlier tentative answer.
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _last_number(text: str) -> str | None:
    matches = list(NUMBER_PATTERN.finditer(str(text)))
    if not matches:
        return None
    return matches[-1].group(0)


def extract_answer_text(text: str, strict: bool = False) -> str | None:
    """Extract answer string.

    strict=True only accepts explicit markers. strict=False falls back to last number.
    """
    if text is None:
        return None
    text = str(text)
    marked = _first_number_after_marker(text)
    if marked is not None:
        return marked
    if strict:
        return None
    return _last_number(text)


def extract_answer_value(text: str, strict: bool = False) -> float | None:
    return parse_number(extract_answer_text(text, strict=strict))


def extract_gold_answer_text(gold: str) -> str | None:
    """Extract gold answer from GSM8K official answer or a plain answer field."""
    if gold is None:
        return None
    gold = str(gold)
    if "####" in gold:
        after = gold.split("####")[-1]
        num = NUMBER_PATTERN.search(after)
        return num.group(0) if num else after.strip()
    marked = _first_number_after_marker(gold)
    if marked is not None:
        return marked
    return _last_number(gold)


def extract_gold_answer_value(gold: str) -> float | None:
    return parse_number(extract_gold_answer_text(gold))


def numeric_equal(a: Any, b: Any, tol: float = 1e-6) -> bool:
    va = parse_number(a) if not isinstance(a, (int, float)) else float(a)
    vb = parse_number(b) if not isinstance(b, (int, float)) else float(b)
    if va is None or vb is None:
        return False
    if math.isnan(va) or math.isnan(vb):
        return False
    return abs(va - vb) <= tol


def is_correct_completion(completion: str, gold_answer: str, strict: bool = False, tol: float = 1e-6) -> bool:
    pred = extract_answer_value(completion, strict=strict)
    gold = extract_gold_answer_value(gold_answer)
    if pred is None or gold is None:
        return False
    return abs(pred - gold) <= tol


def has_final_answer_marker(text: str) -> bool:
    if text is None:
        return False
    lowered = str(text).lower()
    return any(re.search(marker, lowered, flags=re.IGNORECASE) for marker in FINAL_MARKERS)


def simple_step_count(text: str) -> int:
    """Heuristic reasoning step count for analysis."""
    if not text:
        return 0
    # Count line breaks and sentence separators that commonly indicate reasoning steps.
    return max(1, len(re.findall(r"\n+|\.\s+|;\s+|therefore|so ", str(text), flags=re.IGNORECASE)))


def completion_stats(completion: str, gold_answer: str) -> dict[str, Any]:
    strict_text = extract_answer_text(completion, strict=True)
    lenient_text = extract_answer_text(completion, strict=False)
    gold_val = extract_gold_answer_value(gold_answer)
    strict_val = parse_number(strict_text)
    lenient_val = parse_number(lenient_text)
    return {
        "pred_strict": strict_text,
        "pred_lenient": lenient_text,
        "gold_value": gold_val,
        "strict_correct": strict_val is not None and gold_val is not None and numeric_equal(strict_val, gold_val),
        "lenient_correct": lenient_val is not None and gold_val is not None and numeric_equal(lenient_val, gold_val),
        "has_final_answer": has_final_answer_marker(completion),
        "answer_extracted_strict": strict_text is not None,
        "answer_extracted_lenient": lenient_text is not None,
        "char_length": len(str(completion)),
        "word_length": len(str(completion).split()),
        "step_count": simple_step_count(completion),
    }
