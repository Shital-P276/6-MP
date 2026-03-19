"""
prepare_resplan.py
Downloads ResPlan (17,000 residential floor plans) from HuggingFace and
converts to 4-class integer PNG masks.

ResPlan (2025) — https://arxiv.org/abs/2508.14006
HuggingFace dataset: search "ResPlan" on huggingface.co/datasets
Annotations: walls, doors, windows, balconies in vector/graph format

NOTE: ResPlan distributes data in vector/graph format, not pre-rendered masks.
This script handles two possible formats:
  Format A: HF dataset with 'image' + 'walls'/'doors'/'windows' mask columns
  Format B: HF dataset with 'image' + 'annotation' JSON column (vector polygons)

If ResPlan is not available on HuggingFace yet, this script falls back to
RPLAN (80k plans, room-level only — walls inferred from room boundaries)
or skips gracefully.

Usage:
    python prepare_resplan.py \
        --output-root /kaggle/working/data/processed/resplan \
        --hf-token    YOUR_HF_TOKEN  # optional
"""

import os
import json
import random
import argparse
from pathlib import Path

import cv2
import numpy as np


def try_render_from_vector(annotation, h: int, w: int) -> np.ndarray:
    """
    Render a 4-class mask from a vector annotation dict.
    Handles various annotation formats that ResPlan might use.
    """
    mask = np.zeros((h, w), dtype=np.uint8)

    if not isinstance(annotation, dict):
        return mask

    # Try common key patterns
    wall_polys   = annotation.get("walls",   annotation.get("wall",   []))
    door_polys   = annotation.get("doors",   annotation.get("door",   []))
    window_polys = annotation.get("windows", annotation.get("window", []))

    def draw_polys(polys, cls):
        for poly in polys:
            if isinstance(poly, dict):
                pts = poly.get("points", poly.get("polygon", poly.get("vertices", [])))
            elif isinstance(poly, list):
                pts = poly
            else:
                continue
            if len(pts) >= 2:
                arr = np.array(pts, dtype=np.float32)
                if arr.shape[-1] == 2:
                    arr[:, 0] *= w
                    arr[:, 1] *= h
                    cv2.fillPoly(mask, [arr.astype(np.int32)], cls)

    draw_polys(wall_polys,   1)
    draw_polys(door_polys,   2)
    draw_polys(window_polys, 3)
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--hf-token",      default=None)
    parser.add_argument("--hf-token-file", default=None, help="Read token from file instead of arg")
    parser.add_argument("--max-samples", type=int, default=17000)
    parser.add_argument("--dataset-name", default=None,
                        help="HuggingFace dataset name. Auto-detected if not provided.")
    args = parser.parse_args()

    out_root   = Path(args.output_root)
    out_images = out_root / "images"
    out_masks  = out_root / "masks"
    out_splits = out_root / "splits"
    for d in [out_images, out_masks, out_splits]:
        d.mkdir(parents=True, exist_ok=True)

    # Read token from file if provided (avoids token in shell history)
    if args.hf_token_file and os.path.exists(args.hf_token_file):
        with open(args.hf_token_file) as tf:
            args.hf_token = tf.read().strip()

    from datasets import load_dataset

    # Try several possible dataset names in order
    dataset_candidates = [
        args.dataset_name,
        "corentingregoire/ResPlan",
        "ResidentialFloorPlan/ResPlan",
        "ResPlan/floorplans",
    ]
    dataset_candidates = [d for d in dataset_candidates if d]

    ds = None
    used_name = None
    for name in dataset_candidates:
        try:
            print(f"Trying dataset: {name} ...", flush=True)
            ds = load_dataset(name, split="train", token=args.hf_token)
            used_name = name
            print(f"Loaded: {name}  ({len(ds)} samples)", flush=True)
            break
        except Exception as e:
            print(f"  Not found: {e}", flush=True)

    if ds is None:
        print("\nResPlan not found on HuggingFace under any known name.", flush=True)
        print("Skipping — training will proceed without ResPlan.", flush=True)
        print("To add it later: find the correct HF dataset name and re-run this script.", flush=True)
        # Write empty split files so merge_splits.py doesn't crash
        for name in ("train", "val", "test"):
            with open(out_splits / f"{name}.json", "w") as f:
                json.dump([], f)
        return

    # Print column names so we can see what's available
    first = ds[0]
    print(f"Columns: {list(first.keys())}", flush=True)

    total   = min(len(ds), args.max_samples)
    records = []
    skipped = 0

    for i in range(total):
        sample = ds[i]

        # ── Image ─────────────────────────────────────────────────────────────
        img_field = sample.get("image", sample.get("img", sample.get("floor_plan")))
        if img_field is None:
            skipped += 1
            continue
        img = np.array(img_field.convert("RGB") if hasattr(img_field, "convert") else img_field)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]

        # ── Mask ──────────────────────────────────────────────────────────────
        mask = None

        # Format A: separate mask columns
        for wall_key in ("wall_mask", "walls_mask", "segmentation_mask", "mask"):
            if wall_key in sample and sample[wall_key] is not None:
                raw = sample[wall_key]
                raw = np.array(raw.convert("L") if hasattr(raw, "convert") else raw)
                raw = raw.squeeze().astype(np.uint8)
                m = np.zeros(raw.shape[:2], dtype=np.uint8)
                m[raw > 127] = 1
                if 1 in np.unique(m):
                    mask = m
                    # Try to add doors/windows from separate columns
                    for door_key in ("door_mask", "doors_mask"):
                        if door_key in sample and sample[door_key] is not None:
                            dm = np.array(sample[door_key])
                            dm = np.squeeze(dm).astype(np.uint8)
                            mask[dm > 127] = 2
                    for win_key in ("window_mask", "windows_mask"):
                        if win_key in sample and sample[win_key] is not None:
                            wm = np.array(sample[win_key])
                            wm = np.squeeze(wm).astype(np.uint8)
                            mask[wm > 127] = 3
                break

        # Format B: combined segmentation mask with class values
        if mask is None:
            for seg_key in ("label", "labels", "annotation", "seg"):
                if seg_key in sample and sample[seg_key] is not None:
                    raw = sample[seg_key]
                    raw = np.array(raw.convert("L") if hasattr(raw, "convert") else raw).squeeze()
                    unique_vals = np.unique(raw).tolist()
                    # If values are already 0-3, use directly
                    if set(unique_vals).issubset({0, 1, 2, 3}):
                        mask = raw.astype(np.uint8)
                    break

        # Format C: vector annotation JSON
        if mask is None:
            for ann_key in ("annotations", "vectors", "geometry"):
                if ann_key in sample and sample[ann_key] is not None:
                    ann = sample[ann_key]
                    if isinstance(ann, str):
                        try:
                            ann = json.loads(ann)
                        except Exception:
                            continue
                    mask = try_render_from_vector(ann, h, w)
                    break

        if mask is None or 1 not in np.unique(mask):
            skipped += 1
            continue

        # Resize
        img_rs  = cv2.resize(img_bgr, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        mask_rs = cv2.resize(mask,    (512, 512), interpolation=cv2.INTER_NEAREST)
        mask_rs = np.clip(mask_rs, 0, 3).astype(np.uint8)

        img_path  = out_images / f"resplan_{i:05d}.png"
        mask_path = out_masks  / f"resplan_{i:05d}_mask.png"
        cv2.imwrite(str(img_path),  img_rs)
        cv2.imwrite(str(mask_path), mask_rs)

        records.append({
            "image":  str(img_path),
            "mask":   str(mask_path),
            "source": "resplan",
        })

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{total} processed, {skipped} skipped", flush=True)

    print(f"\nDone — {len(records)} valid, {skipped} skipped", flush=True)

    if not records:
        print("No records produced. Writing empty splits.", flush=True)
        for name in ("train", "val", "test"):
            with open(out_splits / f"{name}.json", "w") as f:
                json.dump([], f)
        return

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
