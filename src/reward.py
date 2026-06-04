"""Reward functions for TRL GRPOTrainer.

The reward functions are intentionally simple and rule-based:
- R1: final answer correctness.
- R2: R1 + format reward.
- R3: R2 - length penalty.
"""

from __future__ import annotations

from typing import Any, Callable

from answer_extraction import (
    extract_answer_value,
    extract_gold_answer_value,
    has_final_answer_marker,
    is_correct_completion,
)


def completion_to_text(completion: Any) -> str:
    """Convert TRL completion object to plain text.

    Different TRL versions may pass either strings or chat-style message lists.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        # Chat-style completion: [[{"role":"assistant", "content":"..."}]] or [{...}]
        if completion and isinstance(completion[0], dict):
            return "\n".join(str(m.get("content", "")) for m in completion)
        return "\n".join(completion_to_text(x) for x in completion)
    if isinstance(completion, dict):
        return str(completion.get("content", completion))
    return str(completion)


def _get_answers_from_kwargs(n: int, kwargs: dict[str, Any]) -> list[str]:
    # The training dataset should have an "answer" column.
    ans = kwargs.get("answer") or kwargs.get("gold_answer") or kwargs.get("label")
    if ans is None:
        return [""] * n
    if isinstance(ans, list):
        return [str(x) for x in ans]
    return [str(ans)] * n


def correctness_scores(completions: list[Any], **kwargs: Any) -> list[float]:
    answers = _get_answers_from_kwargs(len(completions), kwargs)
    scores = []
    for comp, gold in zip(completions, answers):
        text = completion_to_text(comp)
        if extract_answer_value(text, strict=True) is None:
            scores.append(-0.5)
        elif is_correct_completion(text, gold, strict=True):
            scores.append(1.0)
        else:
            scores.append(0.0)
    return scores


def format_scores(completions: list[Any], **kwargs: Any) -> list[float]:
    scores = []
    for comp in completions:
        text = completion_to_text(comp)
        score = 0.0
        if has_final_answer_marker(text):
            score += 0.4
        # A rough check for step-by-step reasoning content.
        if any(tok in text.lower() for tok in ["reasoning", "first", "then", "so", "therefore", "because", "=", "步骤"]):
            score += 0.3
        if extract_answer_value(text, strict=True) is not None:
            score += 0.3
        if not text.strip() or len(text.strip()) < 5:
            score -= 0.2
        scores.append(max(-0.2, min(1.0, score)))
    return scores


def length_penalty_scores(completions: list[Any], target_length: int = 256, max_no_penalty_length: int = 256, **kwargs: Any) -> list[float]:
    penalties = []
    for comp in completions:
        text = completion_to_text(comp)
        length = len(text.split())
        if length <= max_no_penalty_length:
            penalties.append(0.0)
        else:
            # Normalized to roughly [0, 1].
            penalties.append(min(1.0, (length - max_no_penalty_length) / max(target_length, 1)))
    return penalties


def build_composite_reward(
    reward_type: str,
    format_weight: float = 0.2,
    length_weight: float = 0.1,
    target_length: int = 256,
    max_no_penalty_length: int = 256,
) -> Callable[..., list[float]]:
    """Return a GRPO-compatible reward function."""

    def reward_func(completions: list[Any], **kwargs: Any) -> list[float]:
        corr = correctness_scores(completions, **kwargs)
        if reward_type == "r1_correct":
            return corr
        fmt = format_scores(completions, **kwargs)
        if reward_type == "r2_correct_format":
            return [c + format_weight * f for c, f in zip(corr, fmt)]
        if reward_type == "r3_correct_format_length":
            length_pen = length_penalty_scores(
                completions,
                target_length=target_length,
                max_no_penalty_length=max_no_penalty_length,
                **kwargs,
            )
            return [c + format_weight * f - length_weight * lp for c, f, lp in zip(corr, fmt, length_pen)]
        raise ValueError(f"Unknown reward_type: {reward_type}")

    reward_func.__name__ = reward_type
    return reward_func
