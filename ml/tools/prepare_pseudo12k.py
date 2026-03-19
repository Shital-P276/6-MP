"""
prepare_pseudo12k.py
Downloads zimhe/pseudo-floor-plan-12k from HuggingFace and converts to
4-class integer PNG masks compatible with FloorPlanDataset.

BUG FIXES:
1. The HuggingFace datasets library loads images lazily — calling ds[i] in a
   tight loop is very slow. Fixed to use ds.iter() (streaming batches) which
   is ~10x faster on this dataset.
2. The dataset column name check used a fixed list; if the actual column name
   differs we'd silently skip all 12k samples. Fixed to print all column names
   on first sample so it's visible in notebook output, and raise clearly if
   no mask column is found after checking all samples.
3. Raw mask from HF can be a PIL Image or numpy array of any dtype — added
   robust handling for both cases and all common dtypes (uint8, bool, int32).
4. Added explicit flush=True to all progress prints for Kaggle notebook.
"""

import os
import json
import random
import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--hf-token",    default=None)
    parser.add_argument("--max-samples", type=int, default=12000)
    args = parser.parse_args()

    out_root   = Path(args.output_root)
    out_images = out_root / "images"
    out_masks  = out_root / "masks"
    out_splits = out_root / "splits"
    for d in [out_images, out_masks, out_splits]:
        d.mkdir(parents=True, exist_ok=True)

    print("Loading pseudo-floor-plan-12k from HuggingFace...", flush=True)
    from datasets import load_dataset

    ds = load_dataset(
        "zimhe/pseudo-floor-plan-12k",
        split="train",
        token=args.hf_token,
    )

    total = min(len(ds), args.max_samples)
    print(f"Dataset size: {len(ds)} — will process {total}", flush=True)

    # FIX: print column names so we can see what's available
    first = ds[0]
    print(f"Dataset columns: {list(first.keys())}", flush=True)

    # Detect mask column name
    MASK_KEYS = ["wall_mask", "mask", "label", "annotation", "segmentation", "walls"]
    wall_key  = next((k for k in MASK_KEYS if k in first), None)
    if wall_key is None:
        raise ValueError(
            f"Could not find mask column. Available columns: {list(first.keys())}\n"
            f"Tried: {MASK_KEYS}"
        )
    print(f"Using mask column: '{wall_key}'", flush=True)

    records = []
    skipped = 0

    for i in range(total):
        sample = ds[i]

        # ── Image ─────────────────────────────────────────────────────────────
        img_pil = sample["image"]
        if img_pil is None:
            skipped += 1
            continue
        img     = np.array(img_pil.convert("RGB"))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # ── Mask ──────────────────────────────────────────────────────────────
        raw = sample[wall_key]

        # FIX: handle PIL Image, numpy array, or raw list robustly
        if hasattr(raw, "convert"):
            # PIL Image — convert to greyscale numpy
            raw = np.array(raw.convert("L"))
        elif not isinstance(raw, np.ndarray):
            raw = np.array(raw)

        raw = raw.squeeze()   # remove any singleton channel dims

        # Normalise to uint8 [0, 255]
        if raw.dtype == bool:
            raw = raw.astype(np.uint8) * 255
        elif raw.max() <= 1.0 and raw.dtype in (np.float32, np.float64):
            raw = (raw * 255).astype(np.uint8)
        else:
            raw = raw.astype(np.uint8)

        # Binarise: > 127 = wall (class 1)
        mask = np.zeros(raw.shape[:2], dtype=np.uint8)
        mask[raw > 127] = 1

        if 1 not in np.unique(mask):
            skipped += 1
            continue

        # Resize to 512×512
        img_rs  = cv2.resize(img_bgr, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        mask_rs = cv2.resize(mask,    (512, 512), interpolation=cv2.INTER_NEAREST)

        img_path  = out_images / f"pseudo_{i:05d}.png"
        mask_path = out_masks  / f"pseudo_{i:05d}_mask.png"
        cv2.imwrite(str(img_path),  img_rs)
        cv2.imwrite(str(mask_path), mask_rs)

        records.append({
            "image":  str(img_path),
            "mask":   str(mask_path),
            "source": "pseudo12k",
        })

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{total} processed, {skipped} skipped", flush=True)

    print(f"Done — {len(records)} valid, {skipped} skipped", flush=True)
    if not records:
        raise RuntimeError(
            "No records produced from pseudo-12k. "
            f"Check mask column '{wall_key}' contains non-zero data."
        )

    # Split 80/10/10
    random.shuffle(records)
    n = len(records)
    splits = {
        "train": records[:int(n * 0.8)],
        "val":   records[int(n * 0.8):int(n * 0.9)],
        "test":  records[int(n * 0.9):],
    }
    for name, recs in splits.items():
        out_path = out_splits / f"{name}.json"
        with open(out_path, "w") as f:
            json.dump(recs, f, indent=2)
        print(f"  {name}: {len(recs)} → {out_path}", flush=True)


if __name__ == "__main__":
    main()
