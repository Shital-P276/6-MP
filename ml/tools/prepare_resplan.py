"""
prepare_resplan.py
Downloads ResPlan from GitHub and converts to 4-class integer PNG masks.

Source:  https://github.com/m-agour/ResPlan
Dataset: 17,000 residential floor plans with wall/door/window/balcony annotations
Format:  ResPlan.zip → ResPlan.pkl (pickle of list of dicts)

Each sample dict has keys:
  'wall'     → list of polygon point lists  → class 1
  'door'     → list of polygon point lists  → class 2
  'window'   → list of polygon point lists  → class 3
  'balcony'  → ignored (background)
  room keys  → ignored (background)

Usage:
    python prepare_resplan.py \
        --output-root /kaggle/working/data/processed/resplan \
        --pkl-path    /path/to/ResPlan.pkl   # optional, downloads if not given
"""

import os
import sys
import json
import random
import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np

SAVE_SIZE = 512


def draw_polygons(mask, polygons, cls, scale, off_x, off_y):
    for poly in polygons:
        if not poly or len(poly) < 3:
            continue
        try:
            pts = np.array([
                [int((p[0] - off_x) * scale),
                 int((p[1] - off_y) * scale)]
                for p in poly
            ], dtype=np.int32)
            cv2.fillPoly(mask, [pts], cls)
        except Exception:
            continue


def sample_to_mask(sample):
    """Convert a ResPlan sample dict to (rendered_image_bgr, mask_uint8)."""
    if not isinstance(sample, dict):
        return None, None

    # Collect all points to compute bounding box
    all_pts = []
    for key, val in sample.items():
        if not isinstance(val, list):
            continue
        for poly in val:
            if isinstance(poly, list):
                for pt in poly:
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        try:
                            all_pts.append((float(pt[0]), float(pt[1])))
                        except (TypeError, ValueError):
                            pass

    if len(all_pts) < 4:
        return None, None

    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    w = max_x - min_x
    h = max_y - min_y

    if w < 1 or h < 1:
        return None, None

    # Scale to SAVE_SIZE with 5% padding
    pad   = 0.05
    scale = SAVE_SIZE / max(w, h) * (1 - 2 * pad)
    off_x = min_x - max(w, h) * pad
    off_y = min_y - max(w, h) * pad

    mask  = np.zeros((SAVE_SIZE, SAVE_SIZE), dtype=np.uint8)
    image = np.ones( (SAVE_SIZE, SAVE_SIZE, 3), dtype=np.uint8) * 255  # white bg

    # Draw room fills in light grey for image realism
    room_keys = ['bedroom', 'bathroom', 'kitchen', 'living', 'corridor',
                 'balcony', 'storage', 'dining', 'garage', 'hall', 'entry',
                 'study', 'laundry', 'utility', 'toilet', 'wc']
    for key in room_keys:
        if key in sample and isinstance(sample[key], list):
            for poly in sample[key]:
                if isinstance(poly, list) and len(poly) >= 3:
                    try:
                        pts = np.array([
                            [int((p[0] - off_x) * scale),
                             int((p[1] - off_y) * scale)]
                            for p in poly
                        ], dtype=np.int32)
                        cv2.fillPoly(image, [pts], (225, 225, 225))
                    except Exception:
                        pass

    # Walls → class 1, draw dark on image
    wall_keys = ['wall', 'walls', 'exterior_wall', 'interior_wall',
                 'wall_depth', 'neighbor_wall']
    for key in wall_keys:
        if key in sample and isinstance(sample[key], list):
            draw_polygons(mask, sample[key], 1, scale, off_x, off_y)
            for poly in sample[key]:
                if isinstance(poly, list) and len(poly) >= 2:
                    try:
                        pts = np.array([
                            [int((p[0] - off_x) * scale),
                             int((p[1] - off_y) * scale)]
                            for p in poly
                        ], dtype=np.int32)
                        cv2.fillPoly(image, [pts], (60, 60, 60))
                    except Exception:
                        pass

    # Doors → class 2
    for key in ['door', 'doors']:
        if key in sample and isinstance(sample[key], list):
            draw_polygons(mask, sample[key], 2, scale, off_x, off_y)

    # Windows → class 3
    for key in ['window', 'windows']:
        if key in sample and isinstance(sample[key], list):
            draw_polygons(mask, sample[key], 3, scale, off_x, off_y)

    if 1 not in np.unique(mask):
        return None, None

    return image, mask


def get_pkl_path() -> str:
    """Return path to ResPlan.pkl — checks Kaggle attached dataset first, then downloads from GitHub."""

    # 1. Check if uploaded as Kaggle dataset (fastest — no download)
    kaggle_candidates = [
        "/kaggle/input/resplan/ResPlan.pkl",
        "/kaggle/input/resplan-dataset/ResPlan.pkl",
        "/kaggle/input/resplan/resplan/ResPlan.pkl",
    ]
    for p in kaggle_candidates:
        if os.path.exists(p):
            print(f"Found Kaggle-attached ResPlan at: {p}", flush=True)
            return p

    work_dir = "/kaggle/working/resplan_download"
    pkl_path = f"{work_dir}/ResPlan.pkl"

    if os.path.exists(pkl_path):
        return pkl_path

    os.makedirs(work_dir, exist_ok=True)
    zip_path = f"{work_dir}/ResPlan.zip"

    # 2. Try direct zip download from GitHub (most reliable)
    print("Downloading ResPlan.zip from GitHub...", flush=True)
    for url in [
        "https://github.com/m-agour/ResPlan/raw/main/ResPlan.zip",
        "https://raw.githubusercontent.com/m-agour/ResPlan/main/ResPlan.zip",
        "https://github.com/m-agour/ResPlan/releases/latest/download/ResPlan.zip",
    ]:
        ret = os.system(f"wget -q --show-progress '{url}' -O {zip_path}")
        if ret == 0 and os.path.exists(zip_path) and os.path.getsize(zip_path) > 100_000:
            print(f"Downloaded successfully from: {url}", flush=True)
            os.system(f"unzip -q {zip_path} -d {work_dir}")
            pkls = list(Path(work_dir).rglob("*.pkl"))
            if pkls:
                return str(pkls[0])
            break

    # 3. Fallback: git clone
    print("Trying git clone...", flush=True)
    clone_dir = f"{work_dir}/repo"
    ret = os.system(f"git clone --depth 1 https://github.com/m-agour/ResPlan.git {clone_dir}")
    if ret == 0:
        for candidate in [
            f"{clone_dir}/ResPlan.pkl",
            f"{clone_dir}/ResPlan.zip",
            f"{clone_dir}/data/ResPlan.pkl",
            f"{clone_dir}/dataset/ResPlan.pkl",
        ]:
            if os.path.exists(candidate):
                if candidate.endswith('.pkl'):
                    return candidate
                if candidate.endswith('.zip'):
                    os.system(f"unzip -q {candidate} -d {work_dir}")
                    pkls = list(Path(work_dir).rglob("*.pkl"))
                    if pkls:
                        return str(pkls[0])

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--pkl-path",    default=None,
                        help="Path to ResPlan.pkl. If not given, downloads from GitHub.")
    parser.add_argument("--max-samples", type=int, default=17000)
    args = parser.parse_args()

    out_root   = Path(args.output_root)
    out_images = out_root / "images"
    out_masks  = out_root / "masks"
    out_splits = out_root / "splits"
    for d in [out_images, out_masks, out_splits]:
        d.mkdir(parents=True, exist_ok=True)

    # Resolve PKL path
    pkl_path = args.pkl_path
    if not pkl_path or not os.path.exists(pkl_path):
        print("PKL path not provided or not found — downloading from GitHub...", flush=True)
        pkl_path = get_pkl_path()

    if not pkl_path or not os.path.exists(pkl_path):
        print("ERROR: Could not obtain ResPlan.pkl — writing empty splits.", flush=True)
        for name in ("train", "val", "test"):
            with open(out_splits / f"{name}.json", "w") as f:
                json.dump([], f)
        return

    print(f"Loading PKL from: {pkl_path}", flush=True)
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    # Normalise to list of dicts
    if isinstance(raw, list):
        samples = raw
    elif isinstance(raw, dict):
        samples = []
        for v in raw.values():
            if isinstance(v, list):
                samples.extend(v)
            elif isinstance(v, dict):
                samples.append(v)
    else:
        print(f"Unexpected PKL type: {type(raw)}", flush=True)
        for name in ("train", "val", "test"):
            with open(out_splits / f"{name}.json", "w") as f:
                json.dump([], f)
        return

    total = min(len(samples), args.max_samples)
    print(f"Total samples in PKL: {len(samples)} — processing {total}", flush=True)

    # Print keys of first sample so we can verify the format
    if samples and isinstance(samples[0], dict):
        print(f"Sample keys: {list(samples[0].keys())}", flush=True)

    records = []
    skipped = 0

    for i in range(total):
        image, mask = sample_to_mask(samples[i])
        if image is None:
            skipped += 1
            continue

        img_path  = out_images / f"resplan_{i:05d}.png"
        mask_path = out_masks  / f"resplan_{i:05d}_mask.png"
        cv2.imwrite(str(img_path),  image)
        cv2.imwrite(str(mask_path), mask)

        records.append({
            "image":  str(img_path),
            "mask":   str(mask_path),
            "source": "resplan",
        })

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{total} processed, {skipped} skipped", flush=True)

    print(f"Done — {len(records)} valid, {skipped} skipped", flush=True)

    if not records:
        print("No records produced. Printing first sample for debug:", flush=True)
        if samples and isinstance(samples[0], dict):
            print(f"  Keys: {list(samples[0].keys())}", flush=True)
            for k, v in list(samples[0].items())[:5]:
                print(f"  {k}: {str(v)[:100]}", flush=True)
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
        with open(out_splits / f"{name}.json", "w") as f:
            json.dump(recs, f, indent=2)
        print(f"  {name}: {len(recs)}", flush=True)


if __name__ == "__main__":
    main()
