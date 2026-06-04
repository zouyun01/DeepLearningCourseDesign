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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 42)))

    model_name = cfg["model_name_or_path"]
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

    train_ds = Dataset.from_json(cfg["train_file"])
    eval_ds = Dataset.from_json(cfg["eval_file"]) if cfg.get("eval_file") else None

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
        )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved SFT adapter to {output_dir}")


if __name__ == "__main__":
    main()
