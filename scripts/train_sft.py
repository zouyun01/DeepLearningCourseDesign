#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from io_utils import load_yaml, set_seed
from prompts import RESPONSE_TEMPLATE, build_sft_messages


def render_text(tokenizer, strip_think: bool):
    """Map a raw SFT row -> ChatML text matching the evaluation prompt exactly.

    Rebuilds the training text from the (question, reasoning, answer) fields so
    the input distribution equals what evaluate.py feeds the model. Ignores any
    stale `text` field baked by older prepare_data runs.
    """

    def _fn(row):
        messages = build_sft_messages(
            row["question"], row["reasoning"], row["answer"], strip_think=strip_think
        )
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    return _fn


class NumericOnlyCollator:
    """Drop raw text fields before delegating to TRL's completion-only collator.

    Some TRL versions keep the source `text` column after SFT tokenization. The
    base HF padding collator then tries to convert that string field to a tensor
    and fails with "too many dimensions 'str'". Keeping only numeric/list fields
    makes the batch shape exactly what the completion-only collator expects.
    """

    def __init__(self, collator):
        self.collator = collator

    def __call__(self, features):
        cleaned = []
        for feature in features:
            cleaned.append(
                {
                    k: v
                    for k, v in feature.items()
                    if isinstance(v, (int, float, list, tuple))
                }
            )
        return self.collator(cleaned)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_name_or_path", default=None,
                        help="Override the base model in the config (e.g. a local path).")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 42)))

    model_name = args.model_name_or_path or cfg["model_name_or_path"]
    output_dir = cfg["output_dir"]
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if cfg.get("bf16", True) else torch.float16,
        trust_remote_code=True,
    )
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    strip_think = bool(cfg.get("strip_think", True))
    raw_train = Dataset.from_json(cfg["train_file"])
    raw_eval = Dataset.from_json(cfg["eval_file"]) if cfg.get("eval_file") else None
    render = render_text(tokenizer, strip_think)
    # Re-render `text` with the chat template; drop the stale columns so the
    # collator only ever sees the freshly templated text.
    train_ds = raw_train.map(render, remove_columns=raw_train.column_names)
    eval_ds = raw_eval.map(render, remove_columns=raw_eval.column_names) if raw_eval else None
    print("Example rendered SFT text (first 600 chars):")
    print(train_ds[0]["text"][:600])

    # Completion-only loss: mask everything up to and including the assistant
    # header so the model is trained only on reasoning + "Final Answer:".
    from trl import DataCollatorForCompletionOnlyLM

    collator = NumericOnlyCollator(DataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TEMPLATE, tokenizer=tokenizer
    ))

    lora_cfg = cfg.get("lora", {})
    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=lora_cfg.get("target_modules"),
        bias="none",
        task_type="CAUSAL_LM",
    )

    # TRL APIs changed across versions. Prefer SFTConfig when available, fall back to TrainingArguments.
    try:
        from trl import SFTConfig

        train_args = SFTConfig(
            output_dir=output_dir,
            dataset_text_field="text",
            max_seq_length=int(cfg.get("max_seq_length", 1024)),
            packing=False,
            num_train_epochs=float(cfg.get("num_train_epochs", 2)),
            learning_rate=float(cfg.get("learning_rate", 2e-4)),
            per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 2)),
            gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 8)),
            warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
            logging_steps=int(cfg.get("logging_steps", 10)),
            save_steps=int(cfg.get("save_steps", 100)),
            save_total_limit=int(cfg.get("save_total_limit", 2)),
            bf16=bool(cfg.get("bf16", True)),
            gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
            report_to=cfg.get("report_to", "tensorboard"),
            remove_unused_columns=False,
        )
        trainer = SFTTrainer(
            model=model,
            args=train_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            peft_config=peft_config,
            processing_class=tokenizer,
            data_collator=collator,
        )
    except Exception as e:
        print(f"Falling back to older SFTTrainer API because: {e}")
        train_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=float(cfg.get("num_train_epochs", 2)),
            learning_rate=float(cfg.get("learning_rate", 2e-4)),
            per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 2)),
            gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 8)),
            warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
            logging_steps=int(cfg.get("logging_steps", 10)),
            save_steps=int(cfg.get("save_steps", 100)),
            save_total_limit=int(cfg.get("save_total_limit", 2)),
            bf16=bool(cfg.get("bf16", True)),
            gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
            report_to=cfg.get("report_to", "tensorboard"),
            remove_unused_columns=False,
        )
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=train_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            dataset_text_field="text",
            max_seq_length=int(cfg.get("max_seq_length", 1024)),
            peft_config=peft_config,
            packing=False,
            data_collator=collator,
        )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved SFT adapter to {output_dir}")


if __name__ == "__main__":
    main()
