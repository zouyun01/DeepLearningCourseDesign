#!/usr/bin/env python
"""Deeper analysis figures + tables from per-sample prediction files.

Complements plot_results.py (which only draws bar charts of the summary metrics).
Everything here is computed directly from the predictions, so it needs no
training logs or GPU. Produces:

  Figures (results/figures/):
    - accuracy_vs_length.svg     scatter: one point per method (len vs acc) ⭐
    - length_distribution.svg    box plot of output length per method
    - accuracy_by_length.svg     strict accuracy vs output-length bucket ⭐
    - accuracy_grouped.svg       strict vs lenient grouped bars
    - radar.svg                  multi-metric radar across methods
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

from plot_style import (apply_style, method_color, method_fill, add_bar_labels,
                        save_fig, PALETTE, MARKERS, SERIES_FILL, SERIES_EDGE)


def save_figure(fig, out: Path) -> None:
    save_fig(fig, out)


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
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for i, m in enumerate(methods):
        x = m["lengths"].mean()
        y = m["correct"].mean()
        ax.scatter(x, y, s=105, color=method_color(m["label"], i), zorder=3,
                   edgecolors="white", linewidths=0.9, label=m["label"])
    ax.set_xlabel("Average output length (words)")
    ax.set_ylabel("Strict accuracy")
    ax.set_title("Accuracy vs. output length")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    ax.margins(x=0.13, y=0.15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(fig, out)
    plt.close()


def fig_length_distribution(methods: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    data = [m["lengths"] for m in methods]
    labels = [m["label"] for m in methods]
    # Set tick labels via xticks (not the boxplot `labels` kwarg, which is
    # deprecated in matplotlib 3.9+) so this works across versions.
    bp = ax.boxplot(
        data,
        showfliers=False,
        patch_artist=True,
        widths=0.58,
        medianprops=dict(color="#111111", linewidth=1.2),
        whiskerprops=dict(color="#586069", linewidth=0.8),
        capprops=dict(color="#586069", linewidth=0.8),
    )
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(method_color(methods[i]["label"], i))
        box.set_alpha(0.82)
        box.set_edgecolor("#2F3437")
    ax.set_ylabel("Output length (words)")
    ax.set_title("Output length distribution")
    ax.set_xticks(range(1, len(labels) + 1), labels, rotation=20, ha="right")
    ax.tick_params(axis="x", length=0)
    plt.tight_layout()
    save_figure(fig, out)
    plt.close()


def fig_accuracy_by_length(methods: list[dict], out: Path) -> None:
    # Buckets of output length (words). Last bucket is open-ended.
    edges = [0, 50, 100, 150, 200, 250, np.inf]
    centers = ["0-50", "50-100", "100-150", "150-200", "200-250", "250+"]
    fig, ax = plt.subplots(figsize=(8.4, 5.1))
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
        ax.plot(xs[valid], accs[valid], marker=MARKERS[i % len(MARKERS)],
                markersize=5.4, color=method_color(m["label"], i), label=m["label"])
    ax.set_xticks(np.arange(len(centers)), centers)
    ax.set_xlabel("Output length bucket (words)")
    ax.set_ylabel("Strict accuracy")
    ax.set_title("Accuracy by output-length bucket")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    ax.legend(ncol=2, loc="lower left")
    plt.tight_layout()
    save_figure(fig, out)
    plt.close()


def fig_accuracy_grouped(methods: list[dict], out: Path) -> None:
    labels = [m["label"] for m in methods]
    strict = [m["correct"].mean() for m in methods]
    lenient = [m["lenient"].mean() for m in methods]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    b1 = ax.bar(x - w / 2, strict, w, label="strict", color=PALETTE[1],
                edgecolor="white", linewidth=0.7, zorder=3)
    b2 = ax.bar(x + w / 2, lenient, w, label="lenient", color=PALETTE[5],
                edgecolor="white", linewidth=0.7, zorder=3)
    add_bar_labels(ax, b1, strict, as_percent=True)
    add_bar_labels(ax, b2, lenient, as_percent=True)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Strict vs. lenient accuracy")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    ax.set_ylim(0, min(1.08, max(max(strict), max(lenient)) * 1.13))
    ax.legend(loc="upper left")
    plt.tight_layout()
    save_figure(fig, out)
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
    n = len(axes_names)
    # Polygon area per method (proxy for "overall strength"); colours are then
    # assigned by area rank so the LARGEST-area method gets blue (PALETTE[0]).
    area = []
    for j in range(len(methods)):
        v = norm[:, j]
        area.append(0.5 * abs(np.sin(2 * np.pi / n)) * sum(v[i] * v[(i + 1) % n] for i in range(n)))
    rank = {j: r for r, j in enumerate(sorted(range(len(methods)), key=lambda j: -area[j]))}
    radar_color = {j: PALETTE[rank[j] % len(PALETTE)] for j in range(len(methods))}

    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7.2, 7.2), subplot_kw=dict(polar=True))
    for j, m in enumerate(methods):
        vals = norm[:, j].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=radar_color[j], linewidth=1.9, label=m["label"])
        ax.fill(angles, vals, color=radar_color[j], alpha=0.10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_names, fontsize=10)
    ax.set_yticklabels([])
    ax.set_title("Normalized multi-metric comparison", pad=18)
    ax.grid(color="#D6D9DC", linewidth=0.7)
    ax.spines["polar"].set_color("#B8C0C8")
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.10), fontsize=8.6)
    plt.tight_layout()
    save_figure(fig, out)
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

    fig_accuracy_vs_length(methods, fig_dir / "accuracy_vs_length.svg")
    fig_length_distribution(methods, fig_dir / "length_distribution.svg")
    fig_accuracy_by_length(methods, fig_dir / "accuracy_by_length.svg")
    fig_accuracy_grouped(methods, fig_dir / "accuracy_grouped.svg")
    fig_radar(methods, fig_dir / "radar.svg")
    write_length_stats(methods, table_dir / "length_stats.csv")

    print(f"Saved analysis figures to {fig_dir}")
    print(f"Saved length stats table to {table_dir / 'length_stats.csv'}")


if __name__ == "__main__":
    main()
