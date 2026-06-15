#!/usr/bin/env python
"""Deeper analysis figures + tables from per-sample prediction files.

Complements plot_results.py (which only draws bar charts of the summary metrics).
Everything here is computed directly from the predictions, so it needs no
training logs or GPU. Produces:

  Figures (results/figures/):
    - accuracy_vs_length.png     scatter: one point per method (len vs acc) ⭐
    - length_distribution.png    box plot of output length per method
    - accuracy_by_length.png     strict accuracy vs output-length bucket ⭐
    - accuracy_grouped.png       strict vs lenient grouped bars
    - radar.png                  multi-metric radar across methods
  Tables (results/):
    - length_stats.csv           per-method length mean/median/p90/max/std

Usage:
  python scripts/plot_analysis.py \
    --prediction_files outputs/base_1.5b/predictions.jsonl outputs/sft/predictions.jsonl ... \
    --labels Base CoT-SFT ... \
    --fig_dir results/figures --table_dir results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_style import apply_style, method_color, add_bar_labels, PALETTE, MARKERS


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_methods(pred_files: list[str], labels: list[str]) -> list[dict]:
    """Return one dict per method with arrays of word_length and strict_correct."""
    methods = []
    for path, label in zip(pred_files, labels):
        rows = read_jsonl(path)
        lengths = np.array(
            [r.get("word_length", len(str(r.get("completion", "")).split())) for r in rows],
            dtype=float,
        )
        correct = np.array([bool(r.get("strict_correct", False)) for r in rows], dtype=float)
        lenient = np.array([bool(r.get("lenient_correct", False)) for r in rows], dtype=float)
        fmt = np.array([bool(r.get("has_final_answer", False)) for r in rows], dtype=float)
        extracted = np.array([bool(r.get("answer_extracted_strict", False)) for r in rows], dtype=float)
        methods.append({
            "label": label, "n": len(rows), "lengths": lengths,
            "correct": correct, "lenient": lenient, "fmt": fmt, "extracted": extracted,
        })
    return methods


def fig_accuracy_vs_length(methods: list[dict], out: Path) -> None:
    plt.figure(figsize=(8, 6))
    for i, m in enumerate(methods):
        x = m["lengths"].mean()
        y = m["correct"].mean()
        plt.scatter(x, y, s=140, color=method_color(m["label"], i), zorder=3,
                    edgecolors="black", linewidths=0.6)
        plt.annotate(m["label"], (x, y), xytext=(6, 6), textcoords="offset points",
                     fontsize=10, fontweight="bold")
    plt.xlabel("Average output length (words)")
    plt.ylabel("Strict accuracy")
    plt.title("Accuracy vs. Output Length (per method)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def fig_length_distribution(methods: list[dict], out: Path) -> None:
    plt.figure(figsize=(10, 6))
    data = [m["lengths"] for m in methods]
    labels = [m["label"] for m in methods]
    # Set tick labels via xticks (not the boxplot `labels` kwarg, which is
    # deprecated in matplotlib 3.9+) so this works across versions.
    bp = plt.boxplot(data, showfliers=False, patch_artist=True,
                     medianprops=dict(color="black"))
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(method_color(methods[i]["label"], i))
        box.set_alpha(0.75)
    plt.ylabel("Output length (words)")
    plt.title("Output Length Distribution per Method")
    plt.xticks(range(1, len(labels) + 1), labels, rotation=25, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def fig_accuracy_by_length(methods: list[dict], out: Path) -> None:
    # Buckets of output length (words). Last bucket is open-ended.
    edges = [0, 50, 100, 150, 200, 250, np.inf]
    centers = ["0-50", "50-100", "100-150", "150-200", "200-250", "250+"]
    plt.figure(figsize=(9, 6))
    for i, m in enumerate(methods):
        # Fixed x positions per bucket so the axis is always left-to-right
        # ordered; sparse buckets become gaps (NaN) instead of reordering it.
        xs = np.arange(len(centers), dtype=float)
        accs = np.full(len(centers), np.nan)
        for b in range(len(edges) - 1):
            mask = (m["lengths"] >= edges[b]) & (m["lengths"] < edges[b + 1])
            if mask.sum() >= 10:  # skip sparse buckets to avoid noisy points
                accs[b] = m["correct"][mask].mean()
        valid = ~np.isnan(accs)
        plt.plot(xs[valid], accs[valid], marker=MARKERS[i % len(MARKERS)],
                 markersize=6, color=method_color(m["label"], i), label=m["label"])
    plt.xticks(np.arange(len(centers)), centers)
    plt.xlabel("Output length bucket (words)")
    plt.ylabel("Strict accuracy")
    plt.title("Accuracy by Output-Length Bucket (over-reasoning analysis)")
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def fig_accuracy_grouped(methods: list[dict], out: Path) -> None:
    labels = [m["label"] for m in methods]
    strict = [m["correct"].mean() for m in methods]
    lenient = [m["lenient"].mean() for m in methods]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - w / 2, strict, w, label="strict", color=PALETTE[1],
                edgecolor="white", linewidth=0.8, zorder=3)
    b2 = ax.bar(x + w / 2, lenient, w, label="lenient", color=PALETTE[4],
                edgecolor="white", linewidth=0.8, zorder=3)
    add_bar_labels(ax, b1, strict, as_percent=True)
    add_bar_labels(ax, b2, lenient, as_percent=True)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Strict vs. Lenient Accuracy")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    ax.margins(y=0.15)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def fig_radar(methods: list[dict], out: Path) -> None:
    # Five comparable axes; each min-max normalized across methods to [0,1].
    raw = {
        "Strict Acc": np.array([m["correct"].mean() for m in methods]),
        "Format": np.array([m["fmt"].mean() for m in methods]),
        "Extraction": np.array([m["extracted"].mean() for m in methods]),
        "Conciseness": -np.array([m["lengths"].mean() for m in methods]),  # shorter is better
        "Low Over-reason": -np.array(  # fewer long-and-wrong is better
            [((m["lengths"] > 256) & (m["correct"] == 0)).mean() for m in methods]
        ),
    }
    axes_names = list(raw.keys())
    norm = []
    for k in axes_names:
        v = raw[k]
        lo, hi = v.min(), v.max()
        norm.append((v - lo) / (hi - lo) if hi > lo else np.ones_like(v))
    norm = np.array(norm)  # shape [n_axes, n_methods]

    angles = np.linspace(0, 2 * np.pi, len(axes_names), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for j, m in enumerate(methods):
        vals = norm[:, j].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=method_color(m["label"], j), linewidth=2, label=m["label"])
        ax.fill(angles, vals, color=method_color(m["label"], j), alpha=0.10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_names, fontsize=10)
    ax.set_yticklabels([])
    ax.set_title("Normalized Multi-metric Comparison", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=9)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def write_length_stats(methods: list[dict], out: Path) -> None:
    lines = ["method,mean,median,p90,max,std,n"]
    for m in methods:
        L = m["lengths"]
        lines.append(
            f"{m['label']},{L.mean():.1f},{np.median(L):.1f},"
            f"{np.percentile(L, 90):.1f},{L.max():.0f},{L.std():.1f},{m['n']}"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_files", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--fig_dir", default="results/figures")
    parser.add_argument("--table_dir", default="results")
    args = parser.parse_args()
    if len(args.prediction_files) != len(args.labels):
        raise ValueError("prediction_files and labels must have the same length")

    apply_style()
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir = Path(args.table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)

    methods = load_methods(args.prediction_files, args.labels)

    fig_accuracy_vs_length(methods, fig_dir / "accuracy_vs_length.png")
    fig_length_distribution(methods, fig_dir / "length_distribution.png")
    fig_accuracy_by_length(methods, fig_dir / "accuracy_by_length.png")
    fig_accuracy_grouped(methods, fig_dir / "accuracy_grouped.png")
    fig_radar(methods, fig_dir / "radar.png")
    write_length_stats(methods, table_dir / "length_stats.csv")

    print(f"Saved analysis figures to {fig_dir}")
    print(f"Saved length stats table to {table_dir / 'length_stats.csv'}")


if __name__ == "__main__":
    main()
