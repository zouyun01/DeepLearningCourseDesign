#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import (apply_style, method_color, method_fill, add_bar_labels,
                        save_fig, SERIES_FILL, SERIES_EDGE)


def save_figure(fig, out_path: Path) -> None:
    save_fig(fig, out_path)


def _grouped(ax, x, w, vals, idx, label):
    return ax.bar(x, vals, w, label=label, color=SERIES_FILL[idx], alpha=1.0,
                  edgecolor=SERIES_EDGE[idx], linewidth=1.3, zorder=3)


def strict_lenient_plot(df: pd.DataFrame, out_path: Path) -> None:
    labels = df["method"].tolist()
    strict = df["accuracy_strict"].tolist()
    lenient = df["accuracy_lenient"].tolist()
    x = np.arange(len(labels)); width = 0.38

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    b1 = _grouped(ax, x - width / 2, width, strict, 0, "strict")
    b2 = _grouped(ax, x + width / 2, width, lenient, 1, "lenient")
    add_bar_labels(ax, b1, strict, as_percent=True)
    add_bar_labels(ax, b2, lenient, as_percent=True)
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Strict vs. lenient accuracy")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    ax.set_ylim(0, min(1.08, max(max(strict), max(lenient)) * 1.16))
    ax.legend(loc="upper left")
    ax.tick_params(axis="x", length=0)
    plt.tight_layout()
    save_figure(fig, out_path)
    plt.close()


def format_extraction_plot(df: pd.DataFrame, out_path: Path) -> None:
    labels = df["method"].tolist()
    fmt = df["format_success_rate"].tolist()
    extracted = df["answer_extraction_rate_strict"].tolist()
    x = np.arange(len(labels)); width = 0.38

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    b1 = _grouped(ax, x - width / 2, width, fmt, 0, "format success")
    b2 = _grouped(ax, x + width / 2, width, extracted, 1, "strict extraction")
    add_bar_labels(ax, b1, fmt, as_percent=True)
    add_bar_labels(ax, b2, extracted, as_percent=True)
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("Rate")
    ax.set_title("Format success vs. strict answer extraction")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    ax.set_ylim(0, min(1.10, max(max(fmt), max(extracted)) * 1.12))
    ax.legend(loc="lower right")
    ax.tick_params(axis="x", length=0)
    plt.tight_layout()
    save_figure(fig, out_path)
    plt.close()


def length_overthinking_plot(df: pd.DataFrame, out_path: Path) -> None:
    """Dual-axis DOUBLE-BAR chart: avg length (left) + overthinking rate (right)."""
    labels = df["method"].tolist()
    lengths = df["avg_response_length_words"].tolist()
    over = df["overthinking_rate"].tolist()
    x = np.arange(len(labels)); width = 0.38

    fig, ax1 = plt.subplots(figsize=(8.8, 5.0))
    b1 = ax1.bar(x - width / 2, lengths, width, label="average length",
                 color=SERIES_FILL[0], alpha=1.0, edgecolor=SERIES_EDGE[0], linewidth=1.3, zorder=3)
    add_bar_labels(ax1, b1, lengths, fmt=".0f")
    ax1.set_ylabel("Average output length (words)")
    ax1.set_ylim(0, max(lengths) * 1.22)
    ax1.set_xticks(x, labels, rotation=18, ha="right")
    ax1.tick_params(axis="x", length=0)

    ax2 = ax1.twinx()
    b2 = ax2.bar(x + width / 2, over, width, label="overthinking rate",
                 color=SERIES_FILL[1], alpha=1.0, edgecolor=SERIES_EDGE[1], linewidth=1.3, zorder=3)
    for xi, yi in zip(x + width / 2, over):
        ax2.text(xi, yi + max(over) * 0.03 + 1e-3, f"{yi * 100:.1f}%",
                 ha="center", va="bottom", fontsize=8.8, color="#222222")
    ax2.set_ylabel("Overthinking rate")
    ax2.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    ax2.set_ylim(0, (max(over) * 1.45) if max(over) > 0 else 0.05)
    ax2.grid(False)

    ax1.set_title("Output length and overthinking risk")
    ax1.legend([b1, b2], [b1.get_label(), b2.get_label()], loc="upper left")
    plt.tight_layout()
    save_figure(fig, out_path)
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

    if {"accuracy_strict", "accuracy_lenient"}.issubset(df.columns):
        strict_lenient_plot(df, out_dir / "accuracy_strict_lenient.svg")
    if {"format_success_rate", "answer_extraction_rate_strict"}.issubset(df.columns):
        format_extraction_plot(df, out_dir / "format_extraction_rate.svg")
    if {"avg_response_length_words", "overthinking_rate"}.issubset(df.columns):
        length_overthinking_plot(df, out_dir / "length_overthinking.svg")

    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()
