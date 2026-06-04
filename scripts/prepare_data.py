#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from datasets import load_dataset
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from answer_extraction import extract_gold_answer_text, extract_answer_text
from io_utils import set_seed, write_jsonl
from prompts import build_math_prompt, format_sft_text

QUESTION_KEYS = ["question", "problem", "prompt", "input", "instruction"]
REASONING_KEYS = ["reasoning", "solution", "response", "completion", "output", "answer", "generated_solution", "cot", "trace"]
ANSWER_KEYS = ["final_answer", "final", "label", "gold", "gold_answer", "answer"]


def _extract_from_messages(messages: Any) -> tuple[str | None, str | None]:
    if not isinstance(messages, list):
        return None, None
    question = None
    response = None
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content") or msg.get("value")
        if role in {"user", "human"} and question is None:
            question = content
        if role in {"assistant", "gpt"}:
            response = content
    return question, response


def get_field(row: dict[str, Any], keys: list[str]) -> Any:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return None


def parse_cot_row(row: dict[str, Any]) -> dict[str, str] | None:
    # Special case: camel-ai/gsm8k_distilled.
    # Fields: problem, reasoning_solution, groud_truth_solution, boxed_answer_success.
    # This dataset is GSM8K-aligned and is preferred for CoT-SFT.
    if "problem" in row and "reasoning_solution" in row:
        if row.get("boxed_answer_success") is False:
            return None
        question = str(row.get("problem", "")).strip()
        reasoning = str(row.get("reasoning_solution", "")).strip()
        gold_solution = str(row.get("groud_truth_solution", "")).strip()
        answer = extract_gold_answer_text(gold_solution) or extract_answer_text(reasoning, strict=False)
        if not question or not reasoning or answer is None:
            return None
        text = format_sft_text(question, reasoning, answer)
        return {
            "question": question,
            "reasoning": reasoning,
            "answer": str(answer),
            "text": text,
        }

    question = get_field(row, QUESTION_KEYS)
    reasoning = get_field(row, REASONING_KEYS)
    if question is None or reasoning is None:
        q2, r2 = _extract_from_messages(row.get("messages") or row.get("conversations"))
        question = question or q2
        reasoning = reasoning or r2
    if question is None or reasoning is None:
        return None
    answer = get_field(row, ANSWER_KEYS)
    if answer is None:
        answer = extract_answer_text(str(reasoning), strict=False)
    if answer is None:
        return None
    text = format_sft_text(str(question), str(reasoning), str(answer))
    return {"question": str(question), "reasoning": str(reasoning), "answer": str(answer), "text": text}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_dataset", default="camel-ai/gsm8k_distilled")
    parser.add_argument("--sft_split", default="train")
    parser.add_argument("--sft_size", type=int, default=1500)
    parser.add_argument("--grpo_size", type=int, default=800)
    parser.add_argument("--val_size", type=int, default=300)
    parser.add_argument("--test_size", type=int, default=-1, help="-1 means full GSM8K test set")
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SFT CoT dataset: {args.sft_dataset} [{args.sft_split}]")
    sft_ds = load_dataset(args.sft_dataset, split=args.sft_split)
    sft_ds = sft_ds.shuffle(seed=args.seed)
    sft_rows = []
    for row in tqdm(sft_ds, desc="Parsing CoT rows"):
        parsed = parse_cot_row(row)
        if parsed is not None:
            sft_rows.append(parsed)
        if len(sft_rows) >= args.sft_size + args.val_size:
            break
    if len(sft_rows) < args.sft_size:
        raise RuntimeError(f"Only parsed {len(sft_rows)} CoT rows; check dataset field names.")
    write_jsonl(out_dir / "sft_train.jsonl", sft_rows[: args.sft_size])
    write_jsonl(out_dir / "sft_val.jsonl", sft_rows[args.sft_size : args.sft_size + args.val_size])
    print(f"Saved {args.sft_size} SFT train rows and {min(args.val_size, len(sft_rows)-args.sft_size)} val rows.")

    print("Loading GSM8K dataset: openai/gsm8k")
    gsm_train = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=args.seed)
    gsm_test = load_dataset("openai/gsm8k", "main", split="test")

    grpo_rows = []
    for row in gsm_train.select(range(min(args.grpo_size, len(gsm_train)))):
        ans = extract_gold_answer_text(row["answer"])
        grpo_rows.append(
            {
                "question": row["question"],
                "prompt": build_math_prompt(row["question"]),
                "answer": ans,
                "original_answer": row["answer"],
            }
        )
    write_jsonl(out_dir / "grpo_train.jsonl", grpo_rows)

    test_limit = len(gsm_test) if args.test_size < 0 else min(args.test_size, len(gsm_test))
    test_rows = []
    for row in gsm_test.select(range(test_limit)):
        ans = extract_gold_answer_text(row["answer"])
        test_rows.append(
            {
                "question": row["question"],
                "prompt": build_math_prompt(row["question"]),
                "answer": ans,
                "original_answer": row["answer"],
            }
        )
    write_jsonl(out_dir / "gsm8k_test.jsonl", test_rows)
    print(f"Saved {len(grpo_rows)} GRPO train rows and {len(test_rows)} GSM8K test rows to {out_dir}")


if __name__ == "__main__":
    main()
