"""
prepare_cvcfp.py
Converts CVC-FP dataset SVG annotations → 4-class integer PNG masks.

CVC-FP structure (after download and extraction):
  cvc-fp/
    *.png          — floor plan images (4 styles: black, textured1, textured2, parallel)
    *.svg          — annotations with Wall, Door, Window polygons

SVG annotation format (per CVC-FP paper):
  Each SVG contains polygon elements with class attributes:
    class="Wall"   → mask class 1
    class="Door"   → mask class 2
    class="Window" → mask class 3
  Some SVGs use "wall", "door", "window" (lowercase) — handle both.

Download:
  wget https://dag.cvc.uab.es/files/CVC-FP.zip
  or request from: https://dag.cvc.uab.es/dataset/cvc-fp-database/

Usage:
    python prepare_cvcfp.py \
        --cvcfp-root  /kaggle/working/data/raw/cvc-fp \
        --output-root /kaggle/working/data/processed/cvcfp
"""

import os
import sys
import json
import random
import argparse
import re
from pathlib import Path

import cv2
import numpy as np

SAVE_SIZE = 512
MIN_SIZE  = 150   # CVC-FP images are small (~300-500px), lower threshold

# Map SVG class names → our mask class index
CLASS_MAP = {
    "wall":   1, "Wall":   1, "WALL":   1,
    "door":   2, "Door":   2, "DOOR":   2,
    "window": 3, "Window": 3, "WINDOW": 3,
}


def parse_svg_polygons(svg_path: Path, img_h: int, img_w: int) -> np.ndarray:
    """
    Parse CVC-FP SVG annotation and rasterise Wall/Door/Window polygons
    into a 4-class integer mask of shape (img_h, img_w).

    CVC-FP SVGs use a different coordinate system than the image — the SVG
    viewBox encodes the canvas size. We scale polygon coords accordingly.
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(str(svg_path))
        root = tree.getroot()

        # Strip namespace prefixes
        ns_pattern = re.compile(r'\{[^}]+\}')
        def strip_ns(tag):
            return ns_pattern.sub('', tag)

        # Get SVG viewBox for coordinate scaling
        svg_elem = root if strip_ns(root.tag) == 'svg' else root.find('.//{*}svg')
        if svg_elem is None:
            svg_elem = root
        viewbox = svg_elem.get('viewBox', svg_elem.get('viewbox', ''))
        if viewbox:
            parts = viewbox.strip().split()
            if len(parts) == 4:
                vx, vy, vw, vh = [float(p) for p in parts]
                scale_x = img_w / vw if vw > 0 else 1.0
                scale_y = img_h / vh if vh > 0 else 1.0
            else:
                scale_x = scale_y = 1.0
        else:
            # Fallback: use SVG width/height attributes
            svgw = float(svg_elem.get('width',  img_w) or img_w)
            svgh = float(svg_elem.get('height', img_h) or img_h)
            scale_x = img_w / svgw if svgw > 0 else 1.0
            scale_y = img_h / svgh if svgh > 0 else 1.0

        # Draw all polygon/polyline/rect elements with known class
        for elem in root.iter():
            tag = strip_ns(elem.tag)
            cls_name = elem.get('class', elem.get('id', ''))

            # Also check label/type attributes used by some CVC-FP versions
            if not cls_name:
                cls_name = elem.get('label', elem.get('type', ''))

            mask_class = CLASS_MAP.get(cls_name)
            if mask_class is None:
                # Try partial match (e.g. "Wall_exterior")
                for k, v in CLASS_MAP.items():
                    if k.lower() in cls_name.lower():
                        mask_class = v
                        break
            if mask_class is None:
                continue

            if tag == 'polygon' or tag == 'polyline':
                pts_str = elem.get('points', '')
                if not pts_str.strip():
                    continue
                pts = []
                for pair in pts_str.strip().split():
                    xy = pair.split(',')
                    if len(xy) == 2:
                        try:
                            x = float(xy[0]) * scale_x
                            y = float(xy[1]) * scale_y
                            pts.append([int(x), int(y)])
                        except ValueError:
                            continue
                if len(pts) >= 3:
                    cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], mask_class)

            elif tag == 'rect':
                try:
                    rx = float(elem.get('x', 0)) * scale_x
                    ry = float(elem.get('y', 0)) * scale_y
                    rw = float(elem.get('width',  0)) * scale_x
                    rh = float(elem.get('height', 0)) * scale_y
                    pts = np.array([
                        [int(rx),      int(ry)     ],
                        [int(rx + rw), int(ry)     ],
                        [int(rx + rw), int(ry + rh)],
                        [int(rx),      int(ry + rh)],
                    ], dtype=np.int32)
                    cv2.fillPoly(mask, [pts], mask_class)
                except (ValueError, TypeError):
                    continue

            elif tag == 'line':
                # Walls sometimes encoded as thick lines
                try:
                    x1 = int(float(elem.get('x1', 0)) * scale_x)
                    y1 = int(float(elem.get('y1', 0)) * scale_y)
                    x2 = int(float(elem.get('x2', 0)) * scale_x)
                    y2 = int(float(elem.get('y2', 0)) * scale_y)
                    thickness = max(3, int(float(elem.get('stroke-width', 4))))
                    cv2.line(mask, (x1, y1), (x2, y2), mask_class,
                             thickness=int(thickness * min(scale_x, scale_y)))
                except (ValueError, TypeError):
                    continue

    except Exception as e:
        print(f"  [WARN] SVG parse error in {svg_path.name}: {e}", flush=True)

    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cvcfp-root",  required=True, help="Path to extracted CVC-FP folder")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    data_root = Path(args.cvcfp_root)
    out_root  = Path(args.output_root)

    out_images = out_root / "images"
    out_masks  = out_root / "masks"
    out_splits = out_root / "splits"
    for d in [out_images, out_masks, out_splits]:
        d.mkdir(parents=True, exist_ok=True)

    # Find all PNG/JPG files that have a matching SVG
    img_files = sorted(list(data_root.rglob("*.png")) + list(data_root.rglob("*.jpg")))
    print(f"Found {len(img_files)} images in {data_root}", flush=True)

    records = []
    skipped = 0

    for i, img_path in enumerate(img_files):
        svg_path = img_path.with_suffix('.svg')
        if not svg_path.exists():
            # Try same name with .svg in same folder
            svg_path = img_path.parent / (img_path.stem + '.svg')
        if not svg_path.exists():
            skipped += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None or min(img.shape[:2]) < MIN_SIZE:
            skipped += 1
            continue

        h, w = img.shape[:2]
        mask = parse_svg_polygons(svg_path, h, w)

        # Validate — must have at least wall pixels
        if 1 not in np.unique(mask):
            print(f"  [WARN] No wall pixels in {img_path.name} — skipping", flush=True)
            skipped += 1
            continue

        # Resize
        img_rs  = cv2.resize(img,  (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_LANCZOS4)
        mask_rs = cv2.resize(mask, (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_NEAREST)
        mask_rs = np.clip(mask_rs, 0, 3).astype(np.uint8)

        uid       = f"cvcfp_{i:04d}"
        img_out   = out_images / f"{uid}.png"
        mask_out  = out_masks  / f"{uid}_mask.png"
        cv2.imwrite(str(img_out),  img_rs)
        cv2.imwrite(str(mask_out), mask_rs)

        records.append({
            "image":  str(img_out),
            "mask":   str(mask_out),
            "source": "cvcfp",
        })

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(img_files)} processed, {skipped} skipped", flush=True)

    print(f"\nDone — {len(records)} valid, {skipped} skipped", flush=True)

    if not records:
        print("ERROR: no records produced. Check that SVG files exist alongside images.", flush=True)
        return

    # Split 70/15/15 (small dataset — more val/test to be representative)
    random.shuffle(records)
    n  = len(records)
    t  = int(n * 0.70)
    v  = int(n * 0.85)
    splits = {
        "train": records[:t],
        "val":   records[t:v],
        "test":  records[v:],
    }
    for name, recs in splits.items():
        out_path = out_splits / f"{name}.json"
        with open(out_path, "w") as f:
            json.dump(recs, f, indent=2)
        print(f"  {name}: {len(recs)} → {out_path}", flush=True)

    # Sanity check
    sample = random.choice(records)
    m = cv2.imread(sample["mask"], cv2.IMREAD_GRAYSCALE)
    u = np.unique(m).tolist()
    print(f"\nSanity check — mask values: {u}", flush=True)
    if not set(u).issubset({0, 1, 2, 3}):
        print("WARNING: unexpected mask values — check SVG class names in your dataset")
    else:
        print("Sanity check passed ✓")


if __name__ == "__main__":
    main()
