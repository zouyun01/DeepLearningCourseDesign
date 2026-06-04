#!/usr/bin/env python
from __future__ import annotations

import argparse
from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--n", type=int, default=2)
    args = parser.parse_args()
    ds = load_dataset(args.dataset, split=args.split)
    print("Columns:", ds.column_names)
    for i in range(min(args.n, len(ds))):
        print("=" * 80)
        print(ds[i])


if __name__ == "__main__":
    main()
