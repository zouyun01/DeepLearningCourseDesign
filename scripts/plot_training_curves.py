#!/usr/bin/env python
"""Plot GRPO training curves (reward / loss / kl / length) from tensorboard logs.

Reads the tensorboard event files under each GRPO output dir (e.g.
outputs/grpo_base_r1/runs/<timestamp>/events.out.tfevents.*) and overlays the
four GRPO runs on one figure per metric. Tag names differ across TRL versions,
so metrics are matched by keyword rather than exact name.

Prereq: pip install tensorboard
Sync first: the GRPO `runs/` dirs must be present locally
  (tar czf grpo_runs.tar.gz outputs/grpo_*/runs  on the server, then extract).

Usage:
  python scripts/plot_training_curves.py            # uses the 4 default GRPO dirs
  python scripts/plot_training_curves.py \
    --runs outputs/grpo_base_r1 outputs/grpo_sft_r3_length \
    --labels GRPO SFT-GRPO-R3 --fig_dir results/figures
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt

from plot_style import apply_style, method_color, ema

DEFAULT_RUNS = [
    ("outputs/grpo_base_r1", "GRPO"),
    ("outputs/grpo_sft_r1_correct", "SFT-GRPO-R1"),
    ("outputs/grpo_sft_r2_format", "SFT-GRPO-R2"),
    ("outputs/grpo_sft_r3_length", "SFT-GRPO-R3"),
]
# Each metric figure collects tags whose lowercased name contains a keyword.
METRIC_KEYWORDS = {
    "reward": ["reward"],
    "loss": ["loss"],
    "kl": ["kl"],
    "completion_length": ["completion_length", "completions/mean_length", "length"],
}


def find_event_dir(run_dir: str) -> str | None:
    """Return a directory containing tfevents files (searches run_dir/**)."""
    hits = glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True)
    if not hits:
        return None
    # EventAccumulator wants the directory holding the event file.
    return os.path.dirname(sorted(hits)[-1])


def load_scalars(event_dir: str):
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator, SCALARS,
    )
    ea = EventAccumulator(event_dir, size_guidance={SCALARS: 0})
    ea.Reload()
    out = {}
    for tag in ea.Tags().get("scalars", []):
        evs = ea.Scalars(tag)
        out[tag] = ([e.step for e in evs], [e.value for e in evs])
    return out


def pick_tag(tags: list[str], keywords: list[str]) -> str | None:
    for kw in keywords:
        for t in tags:
            if kw in t.lower():
                return t
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=[r for r, _ in DEFAULT_RUNS])
    parser.add_argument("--labels", nargs="+", default=[l for _, l in DEFAULT_RUNS])
    parser.add_argument("--fig_dir", default="results/figures")
    args = parser.parse_args()
    if len(args.runs) != len(args.labels):
        raise ValueError("runs and labels must have the same length")

    apply_style()
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load scalars for every run that has event files.
    runs = []
    for run_dir, label in zip(args.runs, args.labels):
        ev = find_event_dir(run_dir)
        if ev is None:
            print(f"[WARN] no tensorboard events under {run_dir} -- skipping "
                  f"(did you sync the runs/ dir?)")
            continue
        runs.append((label, load_scalars(ev)))
    if not runs:
        print("[FATAL] no runs with event files found. Sync outputs/grpo_*/runs first.")
        return

    made = 0
    for metric, keywords in METRIC_KEYWORDS.items():
        plotted = False
        plt.figure(figsize=(9, 6))
        for j, (label, scalars) in enumerate(runs):
            tag = pick_tag(list(scalars.keys()), keywords)
            if tag is None:
                continue
            steps, values = scalars[tag]
            color = method_color(label, j)
            # Faint raw trace + bold EMA-smoothed line for a clean, readable look.
            plt.plot(steps, values, color=color, alpha=0.18, linewidth=1.0, zorder=2)
            plt.plot(steps, ema(values, alpha=0.15), color=color, linewidth=2.2,
                     label=label, zorder=3)
            plotted = True
        if plotted:
            plt.xlabel("Training step")
            plt.ylabel(metric.replace("_", " "))
            plt.title(f"GRPO Training — {metric.replace('_', ' ')}")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            out = fig_dir / f"train_{metric}.png"
            plt.savefig(out)
            made += 1
            print(f"Saved {out}")
        plt.close()

    if made == 0:
        # Help the user discover the real tag names if keyword matching missed.
        print("[WARN] no metrics matched. Available scalar tags per run:")
        for label, scalars in runs:
            print(f"  {label}: {sorted(scalars.keys())}")


if __name__ == "__main__":
    main()
