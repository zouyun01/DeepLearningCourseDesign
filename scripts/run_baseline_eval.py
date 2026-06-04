#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from io_utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    cmd = [
        sys.executable,
        "scripts/evaluate.py",
        "--model_name_or_path",
        str(cfg["model_name_or_path"]),
        "--data_file",
        str(cfg["data_file"]),
        "--output_jsonl",
        str(cfg["output_jsonl"]),
        "--metrics_file",
        str(cfg["metrics_file"]),
        "--batch_size",
        str(cfg.get("batch_size", 8)),
        "--max_new_tokens",
        str(cfg.get("max_new_tokens", 512)),
    ]
    if cfg.get("adapter_path"):
        cmd += ["--adapter_path", str(cfg["adapter_path"])]
    if cfg.get("use_chat_template", True):
        cmd += ["--use_chat_template"]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
