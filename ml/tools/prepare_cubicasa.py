"""
prepare_cubicasa.py — CubiCasa5k SVG → 4-class mask converter
No dependency on floortrans/House loader. Python 3.12 compatible.

CubiCasa SVG class names (confirmed from actual dataset SVGs):
  Walls   → BoundaryPolygon (room boundary polygons ARE the walls)
  Doors   → Door, Doors
  Windows → Glass (windows tagged as Glass in CubiCasa)
  Ignored → Floor, Floorplan, Bedroom, Bath, etc. (room fills, furniture)
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

# Confirmed from actual CubiCasa5k SVG class names
CLASS_MAP = {
    # Walls — room boundary polygons define the wall geometry
    'BoundaryPolygon': 1,
    # Also handle any alternative wall naming
    'Wall': 1, 'Walls': 1, 'WallSurface': 1,
    'Railing': 1, 'Column': 1, 'Structure': 1,
    # Doors
    'Door': 2, 'Doors': 2,
    'Opening': 2, 'SingleSwingDoor': 2, 'DoubleSwingDoor': 2,
    # Windows — tagged as Glass in CubiCasa
    'Glass': 3, 'Window': 3, 'Windows': 3, 'Skylight': 3,
}


def parse_points(pts_str):
    """Parse SVG points string 'x1,y1 x2,y2 ...' into list of (x,y) floats."""
    pts = []
    # Handle both 'x,y' pairs and alternating 'x y x y' formats
    tokens = re.split(r'[\s]+', pts_str.strip())
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if ',' in tok:
            parts = tok.split(',')
            if len(parts) >= 2:
                try:
                    pts.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    pass
    # If no comma-pairs found, try alternating x y format
    if not pts:
        nums = []
        for tok in tokens:
            try:
                nums.append(float(tok))
            except ValueError:
                pass
        for i in range(0, len(nums) - 1, 2):
            pts.append((nums[i], nums[i+1]))
    return pts


def get_class(elem):
    """Get our mask class from an SVG element's class/id attributes."""
    for attr in ['class', 'id', 'type', 'label']:
        val = elem.get(attr, '')
        if val:
            for token in val.strip().split():
                if token in CLASS_MAP:
                    return CLASS_MAP[token]
    return None


def svg_to_mask(svg_path: Path, img_h: int, img_w: int) -> np.ndarray:
    """Parse CubiCasa SVG and rasterise into 4-class integer mask."""
    import xml.etree.ElementTree as ET

    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    try:
        tree = ET.parse(str(svg_path))
        root = tree.getroot()

        def strip_ns(tag):
            return re.sub(r'\{[^}]+\}', '', tag)

        # Get viewBox for coordinate scaling
        vb = root.get('viewBox', root.get('viewbox', ''))
        if not vb:
            for elem in root.iter():
                vb = elem.get('viewBox', elem.get('viewbox', ''))
                if vb:
                    break

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
            w_attr = float(root.get('width',  img_w) or img_w)
            h_attr = float(root.get('height', img_h) or img_h)
            sx = img_w / w_attr if w_attr > 0 else 1.0
            sy = img_h / h_attr if h_attr > 0 else 1.0

        def draw(pts, cls, as_line=False):
            if len(pts) < 2:
                return
            arr = np.array([[int(x * sx), int(y * sy)] for x, y in pts],
                           dtype=np.int32)
            if as_line or len(pts) < 3:
                t = max(3, int(3 * min(sx, sy)))
                for i in range(len(arr) - 1):
                    cv2.line(mask, tuple(arr[i]), tuple(arr[i+1]), cls, t)
            else:
                cv2.fillPoly(mask, [arr], cls)

        # Two-pass: first pass handles elements with own class attribute,
        # second pass handles elements that inherit class from parent group
        for elem in root.iter():
            cls = get_class(elem)
            if cls is None:
                continue
            tag = strip_ns(elem.tag)
            if tag == 'polygon':
                draw(parse_points(elem.get('points', '')), cls)
            elif tag == 'polyline':
                draw(parse_points(elem.get('points', '')), cls, as_line=True)
            elif tag == 'rect':
                try:
                    rx = float(elem.get('x', 0)) * sx
                    ry = float(elem.get('y', 0)) * sy
                    rw = float(elem.get('width',  0)) * sx
                    rh = float(elem.get('height', 0)) * sy
                    if rw > 0 and rh > 0:
                        draw([(rx,ry),(rx+rw,ry),(rx+rw,ry+rh),(rx,ry+rh)], cls)
                except (ValueError, TypeError):
                    pass
            elif tag == 'line':
                try:
                    x1 = float(elem.get('x1', 0)) * sx
                    y1 = float(elem.get('y1', 0)) * sy
                    x2 = float(elem.get('x2', 0)) * sx
                    y2 = float(elem.get('y2', 0)) * sy
                    t  = max(3, int(float(elem.get('stroke-width', 4)) * min(sx, sy)))
                    cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), cls, t)
                except (ValueError, TypeError):
                    pass

        # Second pass: group-level class → children
        for group in root.iter():
            if strip_ns(group.tag) != 'g':
                continue
            cls = get_class(group)
            if cls is None:
                continue
            for child in group:
                tag = strip_ns(child.tag)
                if tag == 'polygon':
                    draw(parse_points(child.get('points', '')), cls)
                elif tag == 'polyline':
                    draw(parse_points(child.get('points', '')), cls, as_line=True)
                elif tag == 'rect':
                    try:
                        rx = float(child.get('x', 0)) * sx
                        ry = float(child.get('y', 0)) * sy
                        rw = float(child.get('width',  0)) * sx
                        rh = float(child.get('height', 0)) * sy
                        if rw > 0 and rh > 0:
                            draw([(rx,ry),(rx+rw,ry),(rx+rw,ry+rh),(rx,ry+rh)], cls)
                    except (ValueError, TypeError):
                        pass

    except Exception:
        pass

    return mask


def find_all_plan_folders(data_root: Path):
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
        svgs = list(folder.glob('*.svg'))
        if not svgs:
            skipped += 1
            continue

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

        img_rs  = cv2.resize(img,  (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_LANCZOS4)
        mask_rs = cv2.resize(mask, (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_NEAREST)
        mask_rs = np.clip(mask_rs, 0, 3).astype(np.uint8)

        img_p  = out_images / f"cubi_{i:05d}.png"
        mask_p = out_masks  / f"cubi_{i:05d}_mask.png"
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
    parser.add_argument("--floortrans-repo", default=None)  # ignored, kept for compat
    args = parser.parse_args()

    data_root = Path(args.cubicasa_root)
    out_root  = Path(args.output_root)

    out_images = out_root / "images"
    out_masks  = out_root / "masks"
    out_splits = out_root / "splits"
    for d in [out_images, out_masks, out_splits]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {data_root} ...", flush=True)
    all_folders = find_all_plan_folders(data_root)
    print(f"Found {len(all_folders)} plan folders", flush=True)

    if not all_folders:
        print("ERROR: no folders found.", flush=True)
        sys.exit(1)

    # Quick test — show what classes we see and what mask we get
    test_folder = all_folders[0]
    test_svgs   = list(test_folder.glob('*.svg'))
    test_imgs   = list(test_folder.glob('F1_original.png')) or list(test_folder.glob('*.png'))
    if test_svgs and test_imgs:
        import xml.etree.ElementTree as ET
        tree = ET.parse(str(test_svgs[0]))
        all_classes = set()
        for elem in tree.getroot().iter():
            c = elem.get('class', '')
            if c:
                for tok in c.strip().split():
                    all_classes.add(tok)
        print(f"All SVG classes in first sample: {sorted(all_classes)}", flush=True)

        test_img = cv2.imread(str(test_imgs[0]))
        if test_img is not None:
            ih, iw = test_img.shape[:2]
            test_mask = svg_to_mask(test_svgs[0], ih, iw)
            unique = np.unique(test_mask).tolist()
            wall_px = int((test_mask == 1).sum())
            print(f"Test mask unique={unique}  wall_pixels={wall_px}", flush=True)
            if 1 not in unique:
                print("STILL no walls — check CLASS_MAP vs classes above", flush=True)
            else:
                print("Wall pixels found ✓", flush=True)

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

    total = len(all_records.get("train", []))
    print(f"\nTotal: train={total} val={len(all_records.get('val',[]))} test={len(all_records.get('test',[]))}", flush=True)

    if total == 0:
        # Print ALL unique classes across all SVGs to help diagnose
        print("Scanning ALL SVGs for class names...", flush=True)
        import xml.etree.ElementTree as ET
        all_cls = set()
        for folder in all_folders[:50]:
            for svg in folder.glob('*.svg'):
                try:
                    tree = ET.parse(str(svg))
                    for elem in tree.getroot().iter():
                        c = elem.get('class', '')
                        for tok in c.strip().split():
                            if tok:
                                all_cls.add(tok)
                except Exception:
                    pass
        print(f"All classes in first 50 SVGs: {sorted(all_cls)}", flush=True)
        sys.exit(1)

    # Sanity
    samples = random.sample(all_records["train"], min(5, total))
    ok = sum(1 for s in samples
             if (m := cv2.imread(s["mask"], cv2.IMREAD_GRAYSCALE)) is not None
             and 1 in np.unique(m))
    print(f"Sanity: {ok}/{len(samples)} have wall pixels {'✓' if ok > 0 else '✗'}", flush=True)


if __name__ == "__main__":
    main()
