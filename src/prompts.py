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


def format_sft_text(question: str, reasoning: str, answer: str | float | int) -> str:
    """Format one CoT-SFT sample as a single text field."""
    return (
        f"Question:\n{str(question).strip()}\n\n"
        f"Reasoning:\n{str(reasoning).strip()}\n\n"
        f"Final Answer: {str(answer).strip()}"
    )
