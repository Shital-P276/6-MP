"""
merge_splits.py
Merges train/val/test JSONs from multiple datasets into combined splits.
Stratified by source so each split contains examples from all datasets.

Usage:
    python merge_splits.py \
        --inputs \
            /kaggle/working/data/processed/cubicasa/splits \
            /kaggle/working/data/processed/pseudo12k/splits \
            /kaggle/working/data/processed/r2v/splits \
            /kaggle/working/data/processed/cvcfp/splits \
        --output /kaggle/working/data/processed/combined/splits
"""

import os
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Paths to split directories (each must have train.json, val.json, test.json)")
    parser.add_argument("--output", required=True, help="Output directory for combined splits")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    combined = defaultdict(list)

    for split_dir in args.inputs:
        split_dir = Path(split_dir)
        for split_name in ("train", "val", "test"):
            json_path = split_dir / f"{split_name}.json"
            if not json_path.exists():
                print(f"  Skipping {json_path} (not found)")
                continue
            with open(json_path) as f:
                records = json.load(f)
            combined[split_name].extend(records)
            print(f"  {split_dir.parent.name}/{split_name}: {len(records)} records")

    for split_name, records in combined.items():
        random.shuffle(records)
        out_path = out_dir / f"{split_name}.json"
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)

        # Source breakdown
        sources = defaultdict(int)
        for r in records:
            sources[r.get("source", "unknown")] += 1
        src_str = " | ".join(f"{k}={v}" for k, v in sorted(sources.items()))
        print(f"\nCombined {split_name}: {len(records)} total  [{src_str}]")
        print(f"  → {out_path}")


if __name__ == "__main__":
    main()
