#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from io_utils import read_jsonl, save_json, write_jsonl
from metrics import compute_metrics, add_prediction_stats
from prompts import build_chat_messages, build_math_prompt


def batch_iter(xs, batch_size):
    for i in range(0, len(xs), batch_size):
        yield xs[i : i + batch_size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--metrics_file", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--use_chat_template", action="store_true", default=True)
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    if args.adapter_path:
        print(f"Loading adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    rows = read_jsonl(args.data_file)
    if args.limit > 0:
        rows = rows[: args.limit]

    prompts = []
    for r in rows:
        q = r.get("question") or r.get("prompt")
        if args.use_chat_template and hasattr(tokenizer, "apply_chat_template"):
            messages = build_chat_messages(q)
            prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        else:
            prompts.append(build_math_prompt(q))

    out_rows = []
    do_sample = args.temperature > 0
    for batch_idx, batch_prompts in enumerate(tqdm(list(batch_iter(prompts, args.batch_size)), desc="Evaluating")):
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                temperature=args.temperature if do_sample else None,
                top_p=args.top_p if do_sample else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = generated[:, enc["input_ids"].shape[1] :]
        completions = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        start = batch_idx * args.batch_size
        for j, comp in enumerate(completions):
            r = dict(rows[start + j])
            r["completion"] = comp.strip()
            r = add_prediction_stats(r)
            out_rows.append(r)

    write_jsonl(args.output_jsonl, out_rows)
    metrics = compute_metrics(out_rows)
    metrics["model_name_or_path"] = args.model_name_or_path
    metrics["adapter_path"] = args.adapter_path
    metrics["data_file"] = args.data_file
    save_json(args.metrics_file, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
