#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_file")
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--tail_chars", type=int, default=200)
    args = parser.parse_args()

    path = Path(args.prediction_file)
    failures = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("answer_extracted_strict"):
                failures.append(row)

    print(f"Strict extraction failures: {len(failures)}")
    for row in failures[: args.limit]:
        completion = str(row.get("completion", ""))
        print("GOLD:", row.get("answer") or row.get("gold_answer") or row.get("label"))
        print("COMP:", completion[-args.tail_chars:])
        print("LENIENT:", row.get("pred_lenient"))
        print("=" * 60)


if __name__ == "__main__":
    main()
