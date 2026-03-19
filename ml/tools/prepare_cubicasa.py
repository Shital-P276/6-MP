"""
prepare_cubicasa.py — CubiCasa5k SVG → 4-class mask converter
Fully self-contained: no dependency on floortrans/House loader.
Parses SVGs directly using Python's built-in xml.etree.ElementTree.

CubiCasa5k SVG structure (confirmed from source):
  Each plan has a model.svg with groups like:
    <g class="Wall"> ... <polygon points="x1,y1 x2,y2 ..."/> </g>
    <g class="Railing"> ... </g>
    <g id="Door" class="..."> ... </g>
    <g class="Window"> ... </g>
  Or individual elements with class attributes directly.

Class mapping:
  Wall, Railing, Column → class 1
  Door, Opening → class 2
  Window → class 3
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
MIN_SIZE  = 150

# CubiCasa SVG class name → our mask class
CLASS_MAP = {}
for name in ['Wall', 'wall', 'Walls', 'walls', 'WallGroup',
             'Railing', 'railing', 'Column', 'column', 'Structure']:
    CLASS_MAP[name] = 1
for name in ['Door', 'door', 'Doors', 'doors', 'Opening', 'opening',
             'DoorGroup', 'SingleSwingDoor', 'DoubleSwingDoor']:
    CLASS_MAP[name] = 2
for name in ['Window', 'window', 'Windows', 'windows', 'WindowGroup',
             'Skylight', 'skylight']:
    CLASS_MAP[name] = 3


def parse_points(pts_str):
    """Parse SVG points string into list of (x,y) floats."""
    pts = []
    for pair in re.split(r'[\s,]+', pts_str.strip()):
        pair = pair.strip()
        if not pair:
            continue
        # Handle "x,y" or separate x y tokens
        if ',' in pair:
            parts = pair.split(',')
            if len(parts) == 2:
                try:
                    pts.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    pass
    return pts


def parse_points_alternating(pts_str):
    """Parse SVG points as alternating x y x y values (no commas)."""
    nums = []
    for tok in re.split(r'\s+', pts_str.strip()):
        try:
            nums.append(float(tok))
        except ValueError:
            pass
    pts = []
    for i in range(0, len(nums) - 1, 2):
        pts.append((nums[i], nums[i+1]))
    return pts


def get_cls_from_elem(elem, ns=''):
    """Extract class name from an element, trying multiple attributes."""
    for attr in ['class', f'{ns}class', 'id', 'type', 'label']:
        val = elem.get(attr, '')
        if val:
            # class can be space-separated list — take first token
            for token in val.strip().split():
                if token in CLASS_MAP:
                    return CLASS_MAP[token]
    return None


def svg_to_mask(svg_path: Path, img_h: int, img_w: int) -> np.ndarray:
    """Parse a CubiCasa SVG and rasterise into a 4-class mask."""
    import xml.etree.ElementTree as ET

    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    try:
        tree = ET.parse(str(svg_path))
        root = tree.getroot()

        # Strip XML namespace from tag names
        def strip_ns(tag):
            return re.sub(r'\{[^}]+\}', '', tag)

        # Get viewBox for coordinate scaling
        svg_tag = root if strip_ns(root.tag) == 'svg' else None
        if svg_tag is None:
            for elem in root.iter():
                if strip_ns(elem.tag) == 'svg':
                    svg_tag = elem
                    break
        if svg_tag is None:
            svg_tag = root

        vb = svg_tag.get('viewBox', svg_tag.get('viewbox', ''))
        if vb:
            parts = vb.strip().split()
            if len(parts) == 4:
                try:
                    vw, vh = float(parts[2]), float(parts[3])
                    sx = img_w / vw if vw > 0 else 1.0
                    sy = img_h / vh if vh > 0 else 1.0
                except ValueError:
                    sx = sy = 1.0
            else:
                sx = sy = 1.0
        else:
            sw = float(svg_tag.get('width',  img_w) or img_w)
            sh = float(svg_tag.get('height', img_h) or img_h)
            sx = img_w / sw if sw > 0 else 1.0
            sy = img_h / sh if sh > 0 else 1.0

        def draw_points(pts, cls, thickness=None):
            if len(pts) < 2:
                return
            arr = np.array([[int(x * sx), int(y * sy)] for x, y in pts], dtype=np.int32)
            if len(pts) >= 3 and thickness is None:
                cv2.fillPoly(mask, [arr], cls)
            else:
                t = thickness or max(3, int(3 * min(sx, sy)))
                for i in range(len(arr) - 1):
                    cv2.line(mask, tuple(arr[i]), tuple(arr[i+1]), cls, t)

        # Walk every element in the SVG
        for elem in root.iter():
            tag = strip_ns(elem.tag)

            # Get class from this element OR its parent group
            cls = get_cls_from_elem(elem)
            if cls is None:
                # Try parent (ET doesn't have parent refs, so check group context below)
                continue

            if tag == 'polygon':
                pts_str = elem.get('points', '')
                pts = parse_points(pts_str)
                if len(pts) < 3:
                    pts = parse_points_alternating(pts_str)
                draw_points(pts, cls)

            elif tag == 'polyline':
                pts_str = elem.get('points', '')
                pts = parse_points(pts_str)
                if len(pts) < 2:
                    pts = parse_points_alternating(pts_str)
                draw_points(pts, cls, thickness=max(3, int(3 * min(sx, sy))))

            elif tag == 'rect':
                try:
                    rx = float(elem.get('x', 0)) * sx
                    ry = float(elem.get('y', 0)) * sy
                    rw = float(elem.get('width',  0)) * sx
                    rh = float(elem.get('height', 0)) * sy
                    if rw > 0 and rh > 0:
                        pts = [(rx, ry), (rx+rw, ry), (rx+rw, ry+rh), (rx, ry+rh)]
                        draw_points(pts, cls)
                except (ValueError, TypeError):
                    pass

            elif tag == 'line':
                try:
                    x1 = float(elem.get('x1', 0)) * sx
                    y1 = float(elem.get('y1', 0)) * sy
                    x2 = float(elem.get('x2', 0)) * sx
                    y2 = float(elem.get('y2', 0)) * sy
                    sw_val = elem.get('stroke-width', '4')
                    try:
                        t = max(3, int(float(sw_val) * min(sx, sy)))
                    except ValueError:
                        t = 4
                    cv2.line(mask,
                             (int(x1), int(y1)),
                             (int(x2), int(y2)),
                             cls, t)
                except (ValueError, TypeError):
                    pass

        # Second pass: handle group-level class inheritance
        # Walk groups, propagate class to children
        for group in root.iter():
            if strip_ns(group.tag) != 'g':
                continue
            cls = get_cls_from_elem(group)
            if cls is None:
                continue
            for child in group:
                tag = strip_ns(child.tag)
                if tag == 'polygon':
                    pts_str = child.get('points', '')
                    pts = parse_points(pts_str)
                    if len(pts) < 3:
                        pts = parse_points_alternating(pts_str)
                    draw_points(pts, cls)
                elif tag == 'polyline':
                    pts_str = child.get('points', '')
                    pts = parse_points(pts_str)
                    draw_points(pts, cls, thickness=max(3, int(3*min(sx,sy))))
                elif tag == 'rect':
                    try:
                        rx = float(child.get('x', 0)) * sx
                        ry = float(child.get('y', 0)) * sy
                        rw = float(child.get('width',  0)) * sx
                        rh = float(child.get('height', 0)) * sy
                        if rw > 0 and rh > 0:
                            pts = [(rx,ry),(rx+rw,ry),(rx+rw,ry+rh),(rx,ry+rh)]
                            draw_points(pts, cls)
                    except (ValueError, TypeError):
                        pass

    except Exception as e:
        pass  # return whatever mask we have

    return mask


def find_all_plan_folders(data_root: Path):
    """Walk data_root and find all folders with both an image and SVG."""
    folders = []
    for root, dirs, files in os.walk(str(data_root)):
        has_svg = any(f.endswith('.svg') for f in files)
        has_img = any(f.endswith(('.png', '.jpg', '.jpeg')) for f in files)
        if has_svg and has_img:
            folders.append(Path(root))
    return folders


def process_folders(folders, out_images, out_masks, split_name):
    records = []
    skipped = 0

    for i, folder in enumerate(folders):
        folder = Path(folder)

        # Find SVG
        svgs = list(folder.glob('*.svg'))
        if not svgs:
            skipped += 1
            continue

        # Find image (prefer F1_original.png)
        imgs = list(folder.glob('F1_original.png'))
        if not imgs:
            imgs = list(folder.glob('*.png')) + list(folder.glob('*.jpg'))
        if not imgs:
            skipped += 1
            continue

        img = cv2.imread(str(imgs[0]))
        if img is None or min(img.shape[:2]) < MIN_SIZE:
            skipped += 1
            continue

        ih, iw = img.shape[:2]
        mask = svg_to_mask(svgs[0], ih, iw)

        if 1 not in np.unique(mask):
            skipped += 1
            continue

        # Resize
        img_rs  = cv2.resize(img,  (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_LANCZOS4)
        mask_rs = cv2.resize(mask, (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_NEAREST)
        mask_rs = np.clip(mask_rs, 0, 3).astype(np.uint8)

        uid = str(folder).replace('/', '_').replace('\\', '_').lstrip('_')[-60:]
        img_p  = out_images / f"cubi_{i:05d}_{uid[:20]}.png"
        mask_p = out_masks  / f"cubi_{i:05d}_{uid[:20]}_mask.png"

        cv2.imwrite(str(img_p),  img_rs)
        cv2.imwrite(str(mask_p), mask_rs)

        records.append({"image": str(img_p), "mask": str(mask_p), "source": "cubicasa"})

        if (i + 1) % 200 == 0:
            print(f"  {split_name}: {i+1}/{len(folders)} done, {skipped} skipped", flush=True)

    print(f"  {split_name}: {len(records)} valid, {skipped} skipped", flush=True)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cubicasa-root",   required=True)
    parser.add_argument("--output-root",     required=True)
    parser.add_argument("--floortrans-repo", default=None,
                        help="Ignored — we no longer use the House loader")
    args = parser.parse_args()

    data_root = Path(args.cubicasa_root)
    out_root  = Path(args.output_root)

    out_images = out_root / "images"
    out_masks  = out_root / "masks"
    out_splits = out_root / "splits"
    for d in [out_images, out_masks, out_splits]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {data_root} for floor plan folders...", flush=True)
    all_folders = find_all_plan_folders(data_root)
    print(f"Found {len(all_folders)} folders", flush=True)

    if not all_folders:
        print("ERROR: no folders found. Check --cubicasa-root path.", flush=True)
        sys.exit(1)

    # Quick test on first folder to verify SVG parsing works
    test_folder = all_folders[0]
    test_svgs = list(test_folder.glob('*.svg'))
    test_imgs = list(test_folder.glob('*.png'))
    if test_svgs and test_imgs:
        test_img = cv2.imread(str(test_imgs[0]))
        if test_img is not None:
            ih, iw = test_img.shape[:2]
            test_mask = svg_to_mask(test_svgs[0], ih, iw)
            unique = np.unique(test_mask).tolist()
            print(f"SVG parse test on {test_folder.name}: mask unique values = {unique}", flush=True)
            if 1 not in unique:
                print("WARNING: no wall pixels in test sample — SVG class names may differ", flush=True)
                # Print first few group class names from SVG to debug
                import xml.etree.ElementTree as ET
                tree = ET.parse(str(test_svgs[0]))
                classes_found = set()
                for elem in tree.getroot().iter():
                    c = elem.get('class', '')
                    if c:
                        for tok in c.strip().split():
                            classes_found.add(tok)
                print(f"  Classes in SVG: {sorted(classes_found)[:30]}", flush=True)

    # Split 80/10/10
    random.shuffle(all_folders)
    n = len(all_folders)
    splits = {
        "train": all_folders[:int(n * 0.8)],
        "val":   all_folders[int(n * 0.8):int(n * 0.9)],
        "test":  all_folders[int(n * 0.9):],
    }

    all_records = {}
    for split_name, folders in splits.items():
        print(f"\nProcessing {split_name} ({len(folders)} folders)...", flush=True)
        records = process_folders(folders, out_images, out_masks, split_name)
        all_records[split_name] = records
        with open(out_splits / f"{split_name}.json", "w") as f:
            json.dump(records, f, indent=2)

    total_train = len(all_records.get("train", []))
    print(f"\nTotal: train={total_train} val={len(all_records.get('val',[]))} test={len(all_records.get('test',[]))}", flush=True)

    if total_train == 0:
        print("ERROR: 0 training records. SVG class names not matching CLASS_MAP.", flush=True)
        sys.exit(1)

    # Sanity check
    samples = random.sample(all_records["train"], min(5, total_train))
    ok = 0
    for s in samples:
        m = cv2.imread(s["mask"], cv2.IMREAD_GRAYSCALE)
        if m is not None and 1 in np.unique(m):
            ok += 1
    print(f"Sanity check: {ok}/{len(samples)} samples have wall pixels {'✓' if ok > 0 else '— check CLASS_MAP'}", flush=True)


if __name__ == "__main__":
    main()
