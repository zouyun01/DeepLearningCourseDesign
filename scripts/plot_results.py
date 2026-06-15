#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_style import apply_style, method_color, add_bar_labels


def bar_plot(df: pd.DataFrame, x: str, y: str, title: str, ylabel: str,
             out_path: Path, as_percent: bool) -> None:
    labels = df[x].tolist()
    values = df[y].tolist()
    colors = [method_color(lbl, i) for i, lbl in enumerate(labels)]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(labels, values, color=colors, width=0.62,
                  edgecolor="white", linewidth=0.8, zorder=3)
    add_bar_labels(ax, bars, values, as_percent=as_percent,
                   fmt=None if as_percent else ".1f")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.margins(y=0.15)
    if as_percent:
        ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    apply_style()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.metrics_csv)

    if "method" not in df.columns:
        raise ValueError("metrics_csv must contain a method column")

    # (column, title, filename, is_rate_in_0_1)
    plots = [
        ("accuracy_strict", "Accuracy (strict)", "accuracy_strict.png", True),
        ("accuracy_lenient", "Accuracy (lenient)", "accuracy_lenient.png", True),
        ("format_success_rate", "Format Success Rate", "format_success_rate.png", True),
        ("answer_extraction_rate_strict", "Strict Answer Extraction Rate", "answer_extraction_rate.png", True),
        ("avg_response_length_words", "Average Response Length (words)", "avg_response_length.png", False),
        ("overthinking_rate", "Overthinking Rate", "overthinking_rate.png", True),
    ]
    for col, title, name, as_pct in plots:
        if col in df.columns:
            bar_plot(df, "method", col, title, title, out_dir / name, as_pct)

    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()
