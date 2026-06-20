#!/usr/bin/env python
"""Pre-classify the strict-error cases in error_cases.csv.

error_cases.csv lists every strict-incorrect prediction (one row per wrong
case, all methods). Many of these are NOT reasoning mistakes -- the model got
the number right but failed the strict "Final Answer:" marker, or it rambled
past the token cap. This script auto-labels the mechanically detectable buckets
so a human only has to fine-grain the genuine reasoning errors.

Auto categories (single label per row, priority order):
  1. 格式/抽取问题(答案实际正确)  lenient answer == gold  -> not a reasoning error
  2. 答案格式错误(缺Final Answer)  no "Final Answer:" marker emitted
  3. 过度推理/疑似截断            output length >= --overlong words
  4. 计算或推理错误(待人工细分)   has marker, answer present but wrong

Outputs:
  results/error_cases_classified.csv   original + `auto_error_type` column
  results/error_type_distribution.csv  method x category counts (report table)
  results/figures/error_type_distribution.svg  stacked bar per method
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from answer_extraction import numeric_equal, parse_number
sys.path.append(str(Path(__file__).resolve().parent))
from plot_style import apply_style, save_fig

CAT_CORRECT_FMT = "格式/抽取问题(答案实际正确)"
CAT_FORMAT = "答案格式错误(缺Final Answer)"
CAT_OVERLONG = "过度推理/疑似截断"
CAT_WRONG = "计算或推理错误(待人工细分)"
# Stable column order for the distribution table / stacked bars.
CATEGORIES = [CAT_WRONG, CAT_OVERLONG, CAT_FORMAT, CAT_CORRECT_FMT]
# DEEP solid fills + darker edges (unified with the deep bar palette)
COLORS = {
    CAT_WRONG: "#C23A78",       # deep rose
    CAT_OVERLONG: "#6A4C9C",    # deep purple
    CAT_FORMAT: "#2F6CB0",      # deep blue
    CAT_CORRECT_FMT: "#4F8C3B", # deep green
}
EDGES = {
    CAT_WRONG: "#992D5E",
    CAT_OVERLONG: "#503576",
    CAT_FORMAT: "#21548C",
    CAT_CORRECT_FMT: "#3B6B2C",
}
# English legend labels (the default matplotlib font cannot render CJK).
EN_LABELS = {
    CAT_WRONG: "Calc/Reasoning error",
    CAT_OVERLONG: "Over-reasoning/Truncated",
    CAT_FORMAT: "Format (no Final Answer)",
    CAT_CORRECT_FMT: "Extraction issue (actually correct)",
}


def _truthy(v) -> bool:
    # has_final_answer is written as True/False (csv -> "True"/"False" strings).
    return str(v).strip().lower() in {"true", "1", "yes"}


def classify_row(row, overlong: int) -> str:
    gold = parse_number(row.get("gold_answer"))
    lenient = parse_number(row.get("pred_lenient"))
    if gold is not None and lenient is not None and numeric_equal(lenient, gold):
        return CAT_CORRECT_FMT
    if not _truthy(row.get("has_final_answer")):
        return CAT_FORMAT
    wl = row.get("word_length")
    try:
        if wl is not None and float(wl) >= overlong:
            return CAT_OVERLONG
    except (TypeError, ValueError):
        pass
    return CAT_WRONG


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--error_csv", default="results/error_cases.csv")
    parser.add_argument("--out_csv", default="results/error_cases_classified.csv")
    parser.add_argument("--dist_csv", default="results/error_type_distribution.csv")
    parser.add_argument("--fig", default="results/figures/error_type_distribution.svg")
    parser.add_argument("--overlong", type=int, default=300,
                        help="word_length >= this is flagged as over-reasoning/truncation")
    args = parser.parse_args()

    apply_style()
    df = pd.read_csv(args.error_csv)
    df["auto_error_type"] = df.apply(lambda r: classify_row(r, args.overlong), axis=1)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    # method x category counts
    methods = list(dict.fromkeys(df["method"])) if "method" in df.columns else ["all"]
    dist = pd.crosstab(df["method"], df["auto_error_type"]) if "method" in df.columns \
        else df["auto_error_type"].value_counts().to_frame().T
    for c in CATEGORIES:
        if c not in dist.columns:
            dist[c] = 0
    dist = dist[CATEGORIES].reindex(methods)
    dist.to_csv(args.dist_csv)

    # stacked bar (proportions per method)
    Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
    frac = dist.div(dist.sum(axis=1), axis=0)
    x = np.arange(len(frac.index))
    bottom = np.zeros(len(frac.index))
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.grid(True, axis="y", zorder=0)
    for c in CATEGORIES:
        ax.bar(x, frac[c].values, bottom=bottom, label=EN_LABELS[c], color=COLORS[c],
               width=0.62, edgecolor=EDGES[c], linewidth=1.2, zorder=3)
        bottom += frac[c].values
    ax.set_xticks(x, frac.index, rotation=18, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of strict errors")
    ax.set_title("Error-type composition per method")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    ax.tick_params(axis="x", length=0)
    plt.tight_layout()
    save_fig(fig, Path(args.fig))
    plt.close()

    print("=== auto error-type distribution (counts) ===")
    print(dist.to_string())
    print(f"\nSaved: {args.out_csv}\n       {args.dist_csv}\n       {args.fig}")
    print("\nNote: only `计算或推理错误(待人工细分)` rows need manual fine-graining "
          "(算术/推理路径/条件理解/幻觉条件).")


if __name__ == "__main__":
    main()
