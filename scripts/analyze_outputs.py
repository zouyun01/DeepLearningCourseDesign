#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from error_analysis import build_error_case_rows
from io_utils import read_jsonl
from metrics import compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_files", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--error_csv", required=True)
    parser.add_argument("--error_limit", type=int, default=50)
    args = parser.parse_args()

    if len(args.prediction_files) != len(args.labels):
        raise ValueError("prediction_files and labels must have the same length")

    metric_rows = []
    all_error_rows = []
    for label, path in zip(args.labels, args.prediction_files):
        records = read_jsonl(path)
        m = compute_metrics(records)
        m["method"] = label
        m["prediction_file"] = path
        metric_rows.append(m)
        for row in build_error_case_rows(records, limit=args.error_limit):
            row["method"] = label
            all_error_rows.append(row)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(args.output_csv, index=False)
    pd.DataFrame(all_error_rows).to_csv(args.error_csv, index=False)
    print(f"Saved metrics to {args.output_csv}")
    print(f"Saved error cases to {args.error_csv}")


if __name__ == "__main__":
    main()
