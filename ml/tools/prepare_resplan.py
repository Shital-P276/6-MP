"""
prepare_resplan.py — Convert ResPlan .pkl dataset to FloorViz mask format

ResPlan stores plans as dicts of Shapely geometries:
  plan["wall"]   → Polygon/MultiPolygon  → class 1
  plan["door"]   → Polygon/MultiPolygon  → class 2
  plan["window"] → Polygon/MultiPolygon  → class 3
  everything else → background (class 0)

Usage:
    python prepare_resplan.py \
        --pkl      /kaggle/input/resplan/ResPlan.pkl \
        --output-root /kaggle/working/data/processed/resplan

Output structure:
    processed/resplan/
        images/    ← 512×512 RGB PNGs (rendered from geometry)
        masks/     ← 512×512 grayscale PNGs (values 0-3)
        splits/
            train.json
            val.json
            test.json
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Geometry → mask helpers
# ---------------------------------------------------------------------------

def _poly_to_mask(poly, shape: Tuple[int, int]) -> np.ndarray:
    """Rasterize a single Shapely Polygon to a binary uint8 mask (values 0/1)."""
    h, w = shape
    img = np.zeros((h, w), dtype=np.uint8)
    if poly is None or poly.is_empty:
        return img
    pts = np.array(poly.exterior.coords, dtype=np.int32)
    cv2.fillPoly(img, [pts], color=1)
    for interior in poly.interiors:
        cv2.fillPoly(img, [np.array(interior.coords, dtype=np.int32)], color=0)
    return img


def geom_to_binary(geom: Any, shape: Tuple[int, int]) -> np.ndarray:
    """Convert any Shapely geometry to a binary mask (0/1)."""
    from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, LineString
    out = np.zeros(shape, dtype=np.uint8)
    if geom is None or (hasattr(geom, 'is_empty') and geom.is_empty):
        return out
    if isinstance(geom, Polygon):
        return _poly_to_mask(geom, shape)
    if isinstance(geom, MultiPolygon):
        for p in geom.geoms:
            out = np.maximum(out, _poly_to_mask(p, shape))
        return out
    if isinstance(geom, GeometryCollection):
        for g in geom.geoms:
            out = np.maximum(out, geom_to_binary(g, shape))
        return out
    if isinstance(geom, LineString):
        pts = np.array(list(geom.coords), dtype=np.int32)
        cv2.polylines(out, [pts], isClosed=False, color=1, thickness=2)
        return out
    return out


def _normalise_fn(minx, miny, span, size):
    """Returns a function that normalises Shapely geometries to pixel space."""
    from shapely import affinity
    scale = (size - 4) / span
    def fn(g):
        if g is None:
            return None
        g2 = affinity.translate(g, xoff=-minx, yoff=-miny)
        g2 = affinity.scale(g2, xfact=scale, yfact=scale, origin=(0, 0))
        g2 = affinity.translate(g2, xoff=2, yoff=2)
        return g2
    return fn


def _get_bounds(plan: Dict[str, Any]):
    """Compute bounding box across all geometries in the plan."""
    from shapely.ops import unary_union
    all_geoms = []
    for key in ["wall", "door", "window", "living", "bedroom", "bathroom",
                "kitchen", "balcony", "front_door", "inner"]:
        g = plan.get(key)
        if g is not None and hasattr(g, 'is_empty') and not g.is_empty:
            all_geoms.append(g)
    if not all_geoms:
        return None
    try:
        combined = unary_union(all_geoms)
        minx, miny, maxx, maxy = combined.bounds
        span = max(maxx - minx, maxy - miny)
        if span < 1e-6:
            return None
        return minx, miny, span
    except Exception:
        return None


def plan_to_4class_mask(plan: Dict[str, Any], size: int = 512) -> Optional[np.ndarray]:
    """
    Convert a ResPlan dict to a 4-class integer mask (H×W, values 0-3).
    Priority: wall(1) → door(2) → window(3). Returns None if unusable.
    """
    bounds = _get_bounds(plan)
    if bounds is None:
        return None
    minx, miny, span = bounds
    norm = _normalise_fn(minx, miny, span, size)
    shape = (size, size)
    mask = np.zeros(shape, dtype=np.uint8)

    # Wall — use "wall" key; fall back to "inner" boundary
    wall_geom = plan.get("wall") or plan.get("inner")
    if wall_geom is not None:
        mask[geom_to_binary(norm(wall_geom), shape) > 0] = 1

    # Door overwrites wall
    door_geom = plan.get("door")
    if door_geom is not None:
        mask[geom_to_binary(norm(door_geom), shape) > 0] = 2

    # Window overwrites wall
    win_geom = plan.get("window")
    if win_geom is not None:
        mask[geom_to_binary(norm(win_geom), shape) > 0] = 3

    if (mask == 1).sum() < 50:
        return None

    return mask


def render_plan_image(plan: Dict[str, Any], size: int = 512) -> Optional[np.ndarray]:
    """
    Render a simple RGB floor plan image from Shapely geometries.
    Used as the model input image (replaces a scanned floor plan PNG).
    """
    ROOM_COLORS = {
        "living":     (220, 220, 220),
        "bedroom":    (180, 230, 200),
        "bathroom":   (230, 180, 150),
        "kitchen":    (180, 200, 230),
        "balcony":    (200, 200, 200),
        "front_door": (160, 100,  80),
    }
    WALL_COLOR   = (60,  60,  60)
    DOOR_COLOR   = (230, 150, 180)
    WINDOW_COLOR = (180, 230, 100)

    bounds = _get_bounds(plan)
    if bounds is None:
        return None
    minx, miny, span = bounds
    norm = _normalise_fn(minx, miny, span, size)
    shape = (size, size)

    canvas = np.ones((size, size, 3), dtype=np.uint8) * 255  # white background

    for key, color in ROOM_COLORS.items():
        g = plan.get(key)
        if g is not None:
            canvas[geom_to_binary(norm(g), shape) > 0] = color

    wg = plan.get("wall") or plan.get("inner")
    if wg is not None:
        canvas[geom_to_binary(norm(wg), shape) > 0] = WALL_COLOR

    dg = plan.get("door")
    if dg is not None:
        canvas[geom_to_binary(norm(dg), shape) > 0] = DOOR_COLOR

    wng = plan.get("window")
    if wng is not None:
        canvas[geom_to_binary(norm(wng), shape) > 0] = WINDOW_COLOR

    return canvas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl",         required=True,  help="Path to ResPlan.pkl")
    parser.add_argument("--output-root", required=True,  help="Output directory root")
    parser.add_argument("--size",        type=int, default=512)
    parser.add_argument("--val-frac",    type=float, default=0.1)
    parser.add_argument("--test-frac",   type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap number of plans (for quick testing)")
    args = parser.parse_args()

    pkl_path = Path(args.pkl)
    if not pkl_path.exists():
        print(f"[WARN] ResPlan pkl not found at {pkl_path} — writing empty splits.")
        _write_empty_splits(args.output_root)
        sys.exit(0)

    print(f"Loading ResPlan from {pkl_path} ...", flush=True)
    try:
        with open(pkl_path, "rb") as f:
            dataset = pickle.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load pkl: {e} — writing empty splits.")
        _write_empty_splits(args.output_root)
        sys.exit(0)

    if isinstance(dataset, dict):
        plans = list(dataset.values())
    elif isinstance(dataset, list):
        plans = dataset
    else:
        print(f"[WARN] Unknown pkl format: {type(dataset)} — writing empty splits.")
        _write_empty_splits(args.output_root)
        sys.exit(0)

    print(f"Total plans in pkl: {len(plans)}", flush=True)

    # Print keys from first plan so we can verify
    if plans:
        sample_plan = plans[0] if isinstance(plans[0], dict) else {}
        print(f"Keys in first plan: {sorted(sample_plan.keys())}", flush=True)

    if args.max_samples:
        plans = plans[:args.max_samples]

    out_root  = Path(args.output_root)
    img_dir   = out_root / "images"
    mask_dir  = out_root / "masks"
    split_dir = out_root / "splits"
    for d in [img_dir, mask_dir, split_dir]:
        d.mkdir(parents=True, exist_ok=True)

    records = []
    skipped = 0

    for i, plan in enumerate(plans):
        if i % 500 == 0:
            print(f"  {i}/{len(plans)}  valid={len(records)}  skipped={skipped}", flush=True)

        if not isinstance(plan, dict):
            skipped += 1
            continue

        # Normalise key typo
        if "balacony" in plan and "balcony" not in plan:
            plan["balcony"] = plan.pop("balacony")

        try:
            mask = plan_to_4class_mask(plan, size=args.size)
            if mask is None:
                skipped += 1
                continue
            img = render_plan_image(plan, size=args.size)
            if img is None:
                skipped += 1
                continue
        except Exception as e:
            skipped += 1
            continue

        unique = np.unique(mask).tolist()
        if set(unique) - {0, 1, 2, 3}:
            skipped += 1
            continue

        stem      = f"resplan_{i:05d}"
        img_path  = str(img_dir  / f"{stem}.png")
        mask_path = str(mask_dir / f"{stem}_mask.png")

        cv2.imwrite(img_path,  cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(mask_path, mask)

        records.append({"image": img_path, "mask": mask_path, "source": "resplan"})

    print(f"\nDone. Valid: {len(records)}, Skipped: {skipped}", flush=True)

    if not records:
        print("[WARN] No valid plans extracted — writing empty splits.")
        _write_empty_splits(args.output_root)
        sys.exit(0)

    # Verify a sample mask
    test_mask = cv2.imread(records[0]["mask"], cv2.IMREAD_GRAYSCALE)
    print(f"Sample mask unique: {np.unique(test_mask).tolist()}  "
          f"wall_pixels={(test_mask == 1).sum()}", flush=True)

    import random
    random.seed(42)
    random.shuffle(records)
    n      = len(records)
    n_val  = max(1, int(n * args.val_frac))
    n_test = max(1, int(n * args.test_frac))
    test_set  = records[:n_test]
    val_set   = records[n_test:n_test + n_val]
    train_set = records[n_test + n_val:]

    for name, split in [("train", train_set), ("val", val_set), ("test", test_set)]:
        path = split_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(split, f)
        print(f"  {name}: {len(split)} → {path}", flush=True)


def _write_empty_splits(output_root: str):
    split_dir = Path(output_root) / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for name in ["train", "val", "test"]:
        with open(split_dir / f"{name}.json", "w") as f:
            json.dump([], f)
    print(f"Empty splits written to {split_dir}", flush=True)


if __name__ == "__main__":
    main()
