"""Prompt templates for GSM8K-style mathematical reasoning."""

from __future__ import annotations

SYSTEM_PROMPT = "You are a careful mathematical reasoning assistant."


def build_math_prompt(question: str) -> str:
    """Build the unified prompt used by Base, SFT evaluation and GRPO."""
    question = str(question).strip()
    return (
        f"Question:\n{question}\n\n"
        "Please solve the problem step by step.\n"
        "End your answer with: Final Answer: <number>"
    )


def build_chat_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_math_prompt(question)},
    ]


# Marks the start of the assistant turn in Qwen2.5 ChatML. Used as the
# response template for completion-only loss masking during SFT, so that the
# loss is computed only over the assistant's reasoning + final answer.
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"


def clean_reasoning(reasoning: str, strip_think: bool = True) -> str:
    """Normalize a distilled CoT trace into a concise step-by-step solution.

    The distillation data (camel-ai/gsm8k_distilled, OpenR1) is R1-style:
    a long ``<think>...</think>`` trace followed by a polished, concise
    solution. The polished part after ``</think>`` is what we want to teach:
    it is short (median ~140 tokens), already step-by-step, and ends cleanly,
    so the trailing ``Final Answer:`` always survives the sequence-length cap.
    """
    r = str(reasoning).strip()
    if strip_think and "</think>" in r:
        tail = r.split("</think>")[-1].strip()
        # Fall back to the full (tag-stripped) trace if the tail is too short
        # to be a real solution.
        if len(tail) >= 40:
            r = tail
    return r.replace("<think>", "").replace("</think>", "").strip()


def build_sft_messages(
    question: str,
    reasoning: str,
    answer: str | float | int,
    strip_think: bool = True,
) -> list[dict[str, str]]:
    """Build one CoT-SFT sample as chat messages.

    Reuses :func:`build_chat_messages` so the system + user turns are byte-for-byte
    identical to what :mod:`evaluate` feeds the model. The assistant turn is the
    cleaned reasoning followed by the exact ``Final Answer: <number>`` line the
    metrics/extraction expect. Rendering this with the tokenizer's chat template
    makes the SFT input distribution match evaluation exactly.
    """
    messages = build_chat_messages(question)
    body = clean_reasoning(reasoning, strip_think=strip_think)
    messages.append(
        {"role": "assistant", "content": f"{body}\n\nFinal Answer: {str(answer).strip()}"}
    )
    return messages


def format_sft_text(question: str, reasoning: str, answer: str | float | int) -> str:
    """Legacy raw-text SFT formatting (no chat template).

    Kept for backward compatibility with older data dumps. The current SFT
    pipeline uses :func:`build_sft_messages` + the tokenizer chat template
    instead; training on this raw format caused a train/eval template mismatch.
    """
    return (
        f"Question:\n{str(question).strip()}\n\n"
        f"Reasoning:\n{str(reasoning).strip()}\n\n"
        f"Final Answer: {str(answer).strip()}"
    )
