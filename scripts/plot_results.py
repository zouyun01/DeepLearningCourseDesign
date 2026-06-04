#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def bar_plot(df: pd.DataFrame, x: str, y: str, title: str, ylabel: str, out_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    plt.bar(df[x], df[y])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.metrics_csv)

    if "method" not in df.columns:
        raise ValueError("metrics_csv must contain a method column")

    plots = [
        ("accuracy_strict", "Accuracy (strict)", "accuracy_strict.png"),
        ("accuracy_lenient", "Accuracy (lenient)", "accuracy_lenient.png"),
        ("format_success_rate", "Format Success Rate", "format_success_rate.png"),
        ("answer_extraction_rate_strict", "Strict Answer Extraction Rate", "answer_extraction_rate.png"),
        ("avg_response_length_words", "Average Response Length (words)", "avg_response_length.png"),
        ("overthinking_rate", "Overthinking Rate", "overthinking_rate.png"),
    ]
    for col, title, name in plots:
        if col in df.columns:
            bar_plot(df, "method", col, title, col, out_dir / name)

    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()
