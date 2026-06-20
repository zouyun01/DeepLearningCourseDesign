#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Two combined (a)(b)(c) group figures for the results section.

  train_curves_combined.{svg,png}     (a) reward  (b) loss  (c) KL divergence
  length_analysis_combined.{svg,png}  (a) accuracy by length-bucket
                                      (b) length vs. accuracy scatter
                                      (c) output-length distribution (box plot)

Reuses the data loaders in plot_training_curves.py and plot_analysis.py.
White-background unified style (plot_style).  No in-figure suptitle.
Run: E:/anaconda3/envs/dl/python.exe scripts/plot_combined_results.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_style import (apply_style, method_color, method_fill, ema,
                        MARKERS, save_fig)
from plot_training_curves import (DEFAULT_RUNS, METRIC_KEYWORDS, find_event_dir,
                                  load_scalars, load_scalars_from_trainer_state, pick_tag)
from plot_analysis import load_methods

FIG = Path("results/figures")

PRED = [
    ("outputs/base_1.5b/predictions.jsonl", "Base"),
    ("outputs/sft/predictions.jsonl", "CoT-SFT"),
    ("outputs/grpo_base_r1/predictions.jsonl", "GRPO"),
    ("outputs/grpo_sft_r1_correct/predictions.jsonl", "SFT-GRPO-R1"),
    ("outputs/grpo_sft_r2_format/predictions.jsonl", "SFT-GRPO-R2"),
    ("outputs/grpo_sft_r3_length/predictions.jsonl", "SFT-GRPO-R3"),
]


# ----------------------------------------------------------- training curves
def load_runs():
    runs = []
    for run_dir, label in DEFAULT_RUNS:
        ev = find_event_dir(run_dir)
        scalars = {}
        if ev is not None:
            try:
                scalars = load_scalars(ev)
            except ModuleNotFoundError:
                pass
        if not scalars:
            scalars = load_scalars_from_trainer_state(run_dir)
        if scalars:
            runs.append((label, scalars))
    return runs


def panel_curve(ax, runs, metric, keywords, title):
    for j, (label, scalars) in enumerate(runs):
        tag = pick_tag(list(scalars.keys()), keywords)
        if tag is None:
            continue
        steps, values = scalars[tag]
        c = method_color(label, j)
        ax.plot(steps, values, color=c, alpha=0.16, linewidth=0.8, zorder=2)
        ax.plot(steps, ema(values, alpha=0.18), color=c, linewidth=2.0,
                label=label, zorder=3)
    ax.set_xlabel("Training step")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title)


def fig_training_combined(runs):
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.5))
    panel_curve(axes[0], runs, "reward", METRIC_KEYWORDS["reward"], "(a)  Training reward")
    panel_curve(axes[1], runs, "loss", METRIC_KEYWORDS["loss"], "(b)  Training loss")
    panel_curve(axes[2], runs, "kl", METRIC_KEYWORDS["kl"], "(c)  KL divergence")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), bbox_to_anchor=(0.5, -0.02),
               frameon=True, fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save_fig(fig, FIG / "train_curves_combined.svg")
    plt.close(fig)
    print("Saved train_curves_combined.svg/.png")


# --------------------------------------------------------- length analysis
def panel_acc_by_length(ax, methods):
    edges = [0, 50, 100, 150, 200, 250, np.inf]
    centers = ["0-50", "50-100", "100-150", "150-200", "200-250", "250+"]
    for i, m in enumerate(methods):
        xs = np.arange(len(centers), dtype=float)
        accs = np.full(len(centers), np.nan)
        for b in range(len(edges) - 1):
            mask = (m["lengths"] >= edges[b]) & (m["lengths"] < edges[b + 1])
            if mask.sum() >= 10:
                accs[b] = m["correct"][mask].mean()
        valid = ~np.isnan(accs)
        ax.plot(xs[valid], accs[valid], marker=MARKERS[i % len(MARKERS)], markersize=5.6,
                color=method_color(m["label"], i), label=m["label"])
    ax.set_xticks(np.arange(len(centers)), centers, rotation=18, ha="right")
    ax.set_xlabel("Output-length bucket (words)")
    ax.set_ylabel("Strict accuracy")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    ax.set_title("(a)  Accuracy by output length")


def panel_len_vs_acc(ax, methods):
    for i, m in enumerate(methods):
        ax.scatter(m["lengths"].mean(), m["correct"].mean(), s=120,
                   color=method_fill(m["label"], i), edgecolors=method_color(m["label"], i),
                   linewidths=1.6, zorder=3, label=m["label"])
    ax.set_xlabel("Average output length (words)")
    ax.set_ylabel("Strict accuracy")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    ax.margins(x=0.16, y=0.16)
    ax.set_title("(b)  Length vs. accuracy")


def panel_length_box(ax, methods):
    data = [m["lengths"] for m in methods]
    labels = [m["label"] for m in methods]
    bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6,
                    medianprops=dict(color="#111111", linewidth=1.3),
                    whiskerprops=dict(color="#586069", linewidth=0.9),
                    capprops=dict(color="#586069", linewidth=0.9))
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(method_fill(methods[i]["label"], i))
        box.set_edgecolor(method_color(methods[i]["label"], i))
        box.set_linewidth(1.3)
    ax.set_xticks(range(1, len(labels) + 1), labels, rotation=18, ha="right")
    ax.set_ylabel("Output length (words)")
    ax.tick_params(axis="x", length=0)
    ax.set_title("(c)  Length distribution")


def fig_length_combined(methods):
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.7))
    panel_acc_by_length(axes[0], methods)
    panel_len_vs_acc(axes[1], methods)
    panel_length_box(axes[2], methods)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), bbox_to_anchor=(0.5, -0.02),
               frameon=True, fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save_fig(fig, FIG / "length_analysis_combined.svg")
    plt.close(fig)
    print("Saved length_analysis_combined.svg/.png")


def main():
    apply_style()
    FIG.mkdir(parents=True, exist_ok=True)

    runs = load_runs()
    if runs:
        fig_training_combined(runs)
    else:
        print("[WARN] no GRPO training logs found; skipped training-curve figure")

    pred = [(p, l) for p, l in PRED if Path(p).exists()]
    if pred:
        methods = load_methods([p for p, _ in pred], [l for _, l in pred])
        fig_length_combined(methods)
    else:
        print("[WARN] no prediction files found; skipped length-analysis figure")


if __name__ == "__main__":
    main()
