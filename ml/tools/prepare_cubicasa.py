"""
prepare_cubicasa.py
Converts CubiCasa5k SVGs → 4-class integer PNG masks.

CubiCasa5k structure after unzip:
  cubicasa5k/
    high_quality/          (2500 plans)
    high_quality_architectural/ (2500 plans)
    colorful/              (present in some downloads)
    cubicasa5k.csv         (split info — columns: image, split or just image paths)

The txt files (train.txt, val.txt, test.txt) live in the GitHub repo clone,
not in the Zenodo download. This script handles both cases.
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path

import cv2
import numpy as np

SAVE_SIZE = 512
MIN_SIZE  = 200


def find_all_plan_folders(data_root: Path):
    """
    Walk data_root and find all folders that contain both an image and an SVG.
    This is the most robust approach — doesn't rely on txt files or CSV.
    """
    folders = []
    for root, dirs, files in os.walk(str(data_root)):
        root_path = Path(root)
        has_svg = any(f.endswith('.svg') for f in files)
        has_img = any(f.endswith('.png') or f.endswith('.jpg') for f in files)
        if has_svg and has_img:
            folders.append(root_path)
    return folders


def load_splits_from_txt(repo_root: Path):
    """Try to load folder lists from train/val/test txt files."""
    splits = {}
    for name in ("train", "val", "test"):
        # txt files can be in repo root or in a data/ subdirectory
        for candidate in [
            repo_root / f"{name}.txt",
            repo_root / "data" / f"{name}.txt",
            repo_root / "splits" / f"{name}.txt",
        ]:
            if candidate.exists():
                folders = []
                for line in open(candidate):
                    rel = line.strip().lstrip("/")
                    if rel:
                        folders.append(rel)
                splits[name] = folders
                break
        else:
            splits[name] = []
    return splits


def load_splits_from_csv(repo_root: Path, data_root: Path):
    """Load folder lists from cubicasa5k.csv."""
    # CSV can be in repo root or data root
    for candidate in [
        repo_root / "cubicasa5k.csv",
        data_root / "cubicasa5k.csv",
        data_root.parent / "cubicasa5k.csv",
    ]:
        if candidate.exists():
            import csv as _csv
            with open(candidate) as f:
                rows = list(_csv.DictReader(f))

            print(f"  CSV found: {candidate}  columns={list(rows[0].keys()) if rows else 'empty'}", flush=True)

            if not rows:
                continue

            # Figure out which column has the image path
            path_col = None
            for col in ("image", "path", "filename", "file", rows[0].keys().__iter__().__next__()):
                if col in rows[0]:
                    path_col = col
                    break

            if path_col is None:
                print(f"  Cannot find path column in CSV", flush=True)
                continue

            # Extract folder from image path
            # e.g. "high_quality/00000001/F1_original.png" → "high_quality/00000001"
            def to_folder(row):
                p = row[path_col].strip().lstrip("/")
                parts = Path(p).parts
                # Return everything except the filename
                if len(parts) >= 2:
                    return str(Path(*parts[:-1]))
                return p

            # Check if CSV has a split column
            split_col = None
            for col in ("split", "set", "subset", "partition"):
                if col in rows[0]:
                    split_col = col
                    break

            if split_col:
                train = [to_folder(r) for r in rows if r[split_col].lower() in ("train", "training")]
                val   = [to_folder(r) for r in rows if r[split_col].lower() in ("val", "valid", "validation")]
                test  = [to_folder(r) for r in rows if r[split_col].lower() in ("test", "testing")]
            else:
                # No split column — make our own 80/10/10 split
                random.shuffle(rows)
                n     = len(rows)
                all_f = [to_folder(r) for r in rows]
                train = all_f[:int(n * 0.8)]
                val   = all_f[int(n * 0.8):int(n * 0.9)]
                test  = all_f[int(n * 0.9):]

            return {"train": train, "val": val, "test": test}

    return None


def make_mask(folder: Path) -> tuple:
    """
    Load SVG via CubiCasa House class and produce 4-class mask.
    Returns (mask, img) or (None, None).
    """
    svg_files = list(folder.glob("*.svg"))
    if not svg_files:
        return None, None

    img_files = list(folder.glob("F1_original.png"))
    if not img_files:
        img_files = list(folder.glob("*.png")) + list(folder.glob("*.jpg"))
    if not img_files:
        return None, None

    img = cv2.imread(str(img_files[0]))
    if img is None or min(img.shape[:2]) < MIN_SIZE:
        return None, None

    ih, iw = img.shape[:2]

    try:
        from floortrans.loaders.house import House
        house = House(str(svg_files[0]), ih, iw)

        mask = np.zeros((ih, iw), dtype=np.uint8)

        # Walls
        if hasattr(house, "walls") and house.walls is not None:
            wm = np.array(house.walls)
            if wm.ndim == 3:
                wall_channels = [c for c in [1, 7] if c < wm.shape[2]]
                if wall_channels:
                    mask[wm[:, :, wall_channels].sum(axis=2) > 0] = 1
            elif wm.ndim == 2:
                mask[np.isin(wm, [2, 8])] = 1

        if 1 not in np.unique(mask):
            return None, None

        # Icons
        if hasattr(house, "icons") and house.icons is not None:
            ic = np.array(house.icons)
            if ic.ndim == 3:
                if ic.shape[2] > 2:
                    mask[ic[:, :, 2] > 0] = 2   # door
                if ic.shape[2] > 1:
                    mask[ic[:, :, 1] > 0] = 3   # window
            elif ic.ndim == 2:
                mask[ic == 2] = 2
                mask[ic == 1] = 3

        return mask, img

    except Exception as e:
        return None, None


def process_folders(folders, data_root, out_images, out_masks, split_name):
    records = []
    skipped = 0

    for i, folder_rel in enumerate(folders):
        # folder_rel can be absolute or relative
        folder = Path(folder_rel)
        if not folder.is_absolute():
            folder = data_root / folder_rel
        if not folder.exists():
            skipped += 1
            continue

        mask, img = make_mask(folder)
        if mask is None:
            skipped += 1
            continue

        img_rs  = cv2.resize(img,  (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_LANCZOS4)
        mask_rs = cv2.resize(mask, (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_NEAREST)
        mask_rs = np.clip(mask_rs, 0, 3).astype(np.uint8)

        uid       = str(folder_rel).replace("/", "_").replace("\\", "_")
        img_path  = out_images / f"{uid}.png"
        mask_path = out_masks  / f"{uid}_mask.png"
        cv2.imwrite(str(img_path),  img_rs)
        cv2.imwrite(str(mask_path), mask_rs)

        records.append({"image": str(img_path), "mask": str(mask_path), "source": "cubicasa"})

        if (i + 1) % 200 == 0:
            print(f"  {split_name}: {i+1}/{len(folders)} done, {skipped} skipped", flush=True)

    print(f"  {split_name}: {len(records)} valid, {skipped} skipped", flush=True)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cubicasa-root",   required=True)
    parser.add_argument("--output-root",     required=True)
    parser.add_argument("--floortrans-repo", required=True)
    args = parser.parse_args()

    data_root       = Path(args.cubicasa_root)
    out_root        = Path(args.output_root)
    floortrans_root = Path(args.floortrans_repo)

    if str(floortrans_root) not in sys.path:
        sys.path.insert(0, str(floortrans_root))

    out_images = out_root / "images"
    out_masks  = out_root / "masks"
    out_splits = out_root / "splits"
    for d in [out_images, out_masks, out_splits]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Load splits ───────────────────────────────────────────────────────────
    print("Loading splits...", flush=True)

    # Try txt files first
    splits = load_splits_from_txt(floortrans_root)
    if any(splits.values()):
        print(f"  Loaded from txt: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}", flush=True)
    else:
        # Try CSV
        splits = load_splits_from_csv(floortrans_root, data_root)
        if splits and any(splits.values()):
            print(f"  Loaded from CSV: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}", flush=True)
        else:
            # Last resort: walk the entire data directory
            print("  No txt/CSV splits found — scanning directory for plan folders...", flush=True)
            all_folders = find_all_plan_folders(data_root)
            print(f"  Found {len(all_folders)} plan folders by scanning", flush=True)
            random.shuffle(all_folders)
            n = len(all_folders)
            # Use absolute paths since we found them by walking
            all_abs = [str(f) for f in all_folders]
            splits = {
                "train": all_abs[:int(n * 0.8)],
                "val":   all_abs[int(n * 0.8):int(n * 0.9)],
                "test":  all_abs[int(n * 0.9):],
            }

    if not any(splits.values()):
        print("ERROR: Could not find any floor plan folders.", flush=True)
        sys.exit(1)

    # ── Process ───────────────────────────────────────────────────────────────
    all_records = {}
    for split_name, folders in splits.items():
        if not folders:
            all_records[split_name] = []
            continue
        print(f"\nProcessing {split_name} ({len(folders)} folders)...", flush=True)
        records = process_folders(folders, data_root, out_images, out_masks, split_name)
        all_records[split_name] = records
        with open(out_splits / f"{split_name}.json", "w") as f:
            json.dump(records, f, indent=2)

    # ── Sanity check ──────────────────────────────────────────────────────────
    train_records = all_records.get("train", [])
    print(f"\nTotal: train={len(train_records)} val={len(all_records.get('val',[]))} test={len(all_records.get('test',[]))}", flush=True)

    if not train_records:
        print("ERROR: no records produced.", flush=True)
        sys.exit(1)

    ok = 0
    for s in random.sample(train_records, min(10, len(train_records))):
        m = cv2.imread(s["mask"], cv2.IMREAD_GRAYSCALE)
        if m is not None and set(np.unique(m).tolist()).issubset({0,1,2,3}) and 1 in np.unique(m):
            ok += 1
    print(f"Sanity check: {ok}/{min(10, len(train_records))} passed {'✓' if ok > 0 else '— check House loader'}", flush=True)


if __name__ == "__main__":
    main()
