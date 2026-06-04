#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer

import torch
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from io_utils import load_yaml, set_seed
from reward import build_composite_reward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 42)))

    model_name = cfg["model_name_or_path"]
    print(f"Loading GRPO policy base: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Passing model as an instantiated module avoids some ambiguity with local paths.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if cfg.get("bf16", True) else torch.float16,
        trust_remote_code=True,
    )
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    ds = Dataset.from_json(cfg["train_file"])
    # GRPOTrainer expects a prompt column. The prepare script already creates it.
    required_cols = {"prompt", "answer"}
    missing = required_cols - set(ds.column_names)
    if missing:
        raise ValueError(f"GRPO train file missing columns: {missing}. Run scripts/prepare_data.py first.")

    reward_cfg = cfg.get("reward", {}) or {}
    reward_func = build_composite_reward(
        reward_type=cfg.get("reward_type", "r1_correct"),
        format_weight=float(reward_cfg.get("format_weight", 0.2)),
        length_weight=float(reward_cfg.get("length_weight", 0.1)),
        target_length=int(reward_cfg.get("target_length", 256)),
        max_no_penalty_length=int(reward_cfg.get("max_no_penalty_length", 256)),
    )

    lora_cfg = cfg.get("lora", {})
    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=lora_cfg.get("target_modules"),
        bias="none",
        task_type="CAUSAL_LM",
    )

    try:
        from trl import GRPOConfig

        grpo_args = GRPOConfig(
            output_dir=cfg["output_dir"],
            learning_rate=float(cfg.get("learning_rate", 5e-6)),
            per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 4)),
            gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 4)),
            num_generations=int(cfg.get("num_generations", 4)),
            max_prompt_length=int(cfg.get("max_prompt_length", 256)),
            max_completion_length=int(cfg.get("max_completion_length", 512)),
            max_steps=int(cfg.get("max_steps", 200)),
            logging_steps=int(cfg.get("logging_steps", 5)),
            save_steps=int(cfg.get("save_steps", 50)),
            save_total_limit=int(cfg.get("save_total_limit", 2)),
            bf16=bool(cfg.get("bf16", True)),
            gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
            report_to=cfg.get("report_to", "tensorboard"),
            remove_unused_columns=False,
        )
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_func,
            args=grpo_args,
            train_dataset=ds,
            peft_config=peft_config,
            processing_class=tokenizer,
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize GRPOTrainer. Please check your TRL version. "
            "This project expects trl>=0.17.0 with GRPOTrainer support. "
            f"Original error: {e}"
        )

    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"Saved GRPO LoRA adapter to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
