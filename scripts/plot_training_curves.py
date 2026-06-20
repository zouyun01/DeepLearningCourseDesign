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
import json
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


def load_scalars_from_trainer_state(run_dir: str):
    hits = glob.glob(os.path.join(run_dir, "**", "trainer_state.json"), recursive=True)
    if not hits:
        return {}
    # Prefer the latest checkpoint by path name; checkpoint-200 sorts after
    # checkpoint-150 in the current experiment layout.
    path = sorted(hits)[-1]
    with open(path, encoding="utf-8") as f:
        state = json.load(f)
    out = {}
    for row in state.get("log_history", []):
        step = row.get("step")
        if step is None:
            continue
        for k, v in row.items():
            if k in {"step", "epoch"}:
                continue
            if isinstance(v, (int, float)):
                steps, values = out.setdefault(k, ([], []))
                steps.append(step)
                values.append(v)
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
        scalars = {}
        if ev is not None:
            try:
                scalars = load_scalars(ev)
            except ModuleNotFoundError:
                print("[WARN] tensorboard is not installed; falling back to trainer_state.json")
        if not scalars:
            scalars = load_scalars_from_trainer_state(run_dir)
        if not scalars:
            print(f"[WARN] no scalar logs under {run_dir} -- skipping")
            continue
        runs.append((label, scalars))
    if not runs:
        print("[FATAL] no runs with event files found. Sync outputs/grpo_*/runs first.")
        return

    made = 0
    for metric, keywords in METRIC_KEYWORDS.items():
        plotted = False
        fig, ax = plt.subplots(figsize=(8.4, 5.1))
        for j, (label, scalars) in enumerate(runs):
            tag = pick_tag(list(scalars.keys()), keywords)
            if tag is None:
                continue
            steps, values = scalars[tag]
            color = method_color(label, j)
            # Faint raw trace + bold EMA-smoothed line for a clean, readable look.
            ax.plot(steps, values, color=color, alpha=0.15, linewidth=0.8, zorder=2)
            ax.plot(steps, ema(values, alpha=0.18), color=color, linewidth=1.9,
                    label=label, zorder=3)
            plotted = True
        if plotted:
            pretty = metric.replace("_", " ")
            ax.set_xlabel("Training step")
            ax.set_ylabel(pretty)
            ax.set_title(f"GRPO training: {pretty}")
            ax.legend(loc="best", ncol=1)
            plt.tight_layout()
            out = fig_dir / f"train_{metric}.svg"
            fig.savefig(out)
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
