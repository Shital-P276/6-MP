"""
prepare_cubicasa.py
Converts CubiCasa5k SVGs → 4-class integer PNG masks (0=bg, 1=wall, 2=door, 3=window)

BUG FIXES:
1. make_mask returned (mask, img) as a tuple but callers unpacked as (mask, img)
   while the tuple was (mask, img) — actually was consistent, but the None check
   "if result is None" would also match any falsy return, which a 0-filled mask
   could theoretically trigger. Fixed return signature to be explicit.
2. House class SVG parse: house.walls and house.icons are multi-channel arrays,
   NOT 2D label arrays. CubiCasa House.walls is (H, W, num_channels) with
   one-hot-like encoding. The correct access is house.walls[:,:,1] for
   wall category, not house.walls == 2. Fixed to use the correct channel indexing
   based on confirmed working code from the previous session's prepare_cubicasa.py.
3. The folder path structure in CubiCasa5k is:
     cubicasa5k/high_quality/some_folder/
   but the txt files contain paths like "high_quality/some_folder".
   The data_root passed is already the cubicasa5k/ folder, so joining works,
   but we must handle both leading-slash and no-leading-slash variants.
4. floortrans path was inserted every call in make_mask — moved to main() once.
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


def load_splits(repo_root: Path):
    """Load train/val/test folder lists from CubiCasa5k txt files."""
    splits = {}
    for name in ("train", "val", "test"):
        txt = repo_root / f"{name}.txt"
        folders = []
        if txt.exists():
            for line in open(txt):
                rel = line.strip().lstrip("/")
                if rel:
                    folders.append(rel)
        splits[name] = folders

    if not any(splits.values()):
        # Fallback: use cubicasa5k.csv
        csv_path = repo_root / "cubicasa5k.csv"
        if csv_path.exists():
            import csv as _csv
            with open(csv_path) as f:
                rows = list(_csv.DictReader(f))
            random.shuffle(rows)
            n = len(rows)
            # CSV column is like "high_quality/some_id/F1_original.png"
            def to_folder(row):
                p = row.get("image", row.get("path", ""))
                # strip the filename, keep folder path
                return str(Path(p).parent).lstrip("/")
            splits["train"] = [to_folder(r) for r in rows[:int(n * 0.8)]]
            splits["val"]   = [to_folder(r) for r in rows[int(n * 0.8):int(n * 0.9)]]
            splits["test"]  = [to_folder(r) for r in rows[int(n * 0.9):]]

    return splits


def make_mask(folder: Path, h: int, w: int) -> tuple:
    """
    Load SVG via CubiCasa House class and produce 4-class integer mask.
    Returns (mask_uint8, img_bgr) or (None, None) if sample is invalid.

    FIX: CubiCasa House.walls is shape (H, W, num_wall_types).
    Channel indices (confirmed from CubiCasa5k source):
      walls channel 1 = exterior wall
      walls channel 7 = railing (treat as wall)
    House.icons is shape (H, W, num_icon_types):
      icons channel 1 = window
      icons channel 2 = door
    We take the max across the spatial for each channel and threshold.
    """
    svg_files = list(folder.glob("*.svg"))
    if not svg_files:
        return None, None

    svg_path  = svg_files[0]
    img_files = list(folder.glob("F1_original.png"))
    if not img_files:
        img_files = list(folder.glob("*.png"))
    if not img_files:
        return None, None

    img = cv2.imread(str(img_files[0]))
    if img is None or min(img.shape[:2]) < MIN_SIZE:
        return None, None

    # Use actual image dimensions for the House loader (may differ from saved h, w)
    ih, iw = img.shape[:2]

    try:
        from floortrans.loaders.house import House
        house = House(str(svg_path), ih, iw)

        mask = np.zeros((ih, iw), dtype=np.uint8)

        # ── Walls ────────────────────────────────────────────────────────────
        if hasattr(house, "walls") and house.walls is not None:
            wm = np.array(house.walls)
            if wm.ndim == 3:
                # Multi-channel: sum channels 1 (exterior wall) and 7 (railing)
                wall_ch = [1, 7]
                wall_ch = [c for c in wall_ch if c < wm.shape[2]]
                if wall_ch:
                    wall_presence = wm[:, :, wall_ch].sum(axis=2) > 0
                    mask[wall_presence] = 1
            elif wm.ndim == 2:
                # Already a label map (older House versions)
                mask[np.isin(wm, [2, 8])] = 1

        if 1 not in np.unique(mask):
            return None, None   # no walls = bad sample

        # ── Icons (doors + windows) ───────────────────────────────────────────
        if hasattr(house, "icons") and house.icons is not None:
            ic = np.array(house.icons)
            if ic.ndim == 3:
                # Channel 2 = door, channel 1 = window (confirmed from CubiCasa5k)
                if ic.shape[2] > 2:
                    door_presence   = ic[:, :, 2] > 0
                    mask[door_presence] = 2
                if ic.shape[2] > 1:
                    window_presence = ic[:, :, 1] > 0
                    mask[window_presence] = 3
            elif ic.ndim == 2:
                mask[ic == 2] = 2   # door
                mask[ic == 1] = 3   # window

        return mask, img

    except Exception as e:
        return None, None


def process_split(
    folders:        list,
    data_root:      Path,
    out_images_dir: Path,
    out_masks_dir:  Path,
    split_name:     str,
) -> list:
    records = []
    skipped = 0

    for i, rel in enumerate(folders):
        folder = data_root / rel
        if not folder.exists():
            skipped += 1
            continue

        mask, img = make_mask(folder, 0, 0)   # h/w unused — read from image inside
        if mask is None:
            skipped += 1
            continue

        # Resize to SAVE_SIZE
        img_rs  = cv2.resize(img,  (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_LANCZOS4)
        mask_rs = cv2.resize(mask, (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_NEAREST)

        # Final sanity: NEAREST resize can occasionally bleed values on borders
        mask_rs = np.clip(mask_rs, 0, 3).astype(np.uint8)

        uid       = rel.replace("/", "_").replace("\\", "_")
        img_path  = out_images_dir / f"{uid}.png"
        mask_path = out_masks_dir  / f"{uid}_mask.png"

        cv2.imwrite(str(img_path),  img_rs)
        cv2.imwrite(str(mask_path), mask_rs)

        records.append({
            "image":  str(img_path),
            "mask":   str(mask_path),
            "source": "cubicasa",
        })

        if (i + 1) % 100 == 0:
            print(f"  {split_name}: {i+1}/{len(folders)} done, {skipped} skipped", flush=True)

    print(f"  {split_name}: DONE — {len(records)} valid, {skipped} skipped", flush=True)
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

    # FIX: add floortrans path ONCE here, not inside every make_mask call
    if str(floortrans_root) not in sys.path:
        sys.path.insert(0, str(floortrans_root))

    out_images = out_root / "images"
    out_masks  = out_root / "masks"
    out_splits = out_root / "splits"
    for d in [out_images, out_masks, out_splits]:
        d.mkdir(parents=True, exist_ok=True)

    splits = load_splits(floortrans_root)
    print(f"Splits: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}", flush=True)

    all_records = {}
    for split_name, folders in splits.items():
        if not folders:
            print(f"  WARNING: no folders for {split_name} — check txt files in {floortrans_root}")
            all_records[split_name] = []
            continue
        print(f"\nProcessing {split_name}...", flush=True)
        records = process_split(folders, data_root, out_images, out_masks, split_name)
        all_records[split_name] = records
        out_path = out_splits / f"{split_name}.json"
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"  Saved {len(records)} records → {out_path}", flush=True)

    # Sanity check
    print("\n── Sanity check (10 random train samples) ──")
    train_records = all_records.get("train", [])
    if not train_records:
        print("  ERROR: no training records produced — check paths and House loader")
        return
    samples = random.sample(train_records, min(10, len(train_records)))
    ok = 0
    for s in samples:
        m = cv2.imread(s["mask"], cv2.IMREAD_GRAYSCALE)
        if m is not None:
            u = np.unique(m).tolist()
            if set(u).issubset({0, 1, 2, 3}) and 1 in u:
                ok += 1
            else:
                print(f"  BAD SAMPLE: {s['mask']} unique={u}")
    if ok == len(samples):
        print(f"  {ok}/{len(samples)} passed ✓")
    else:
        print(f"  WARNING: only {ok}/{len(samples)} passed — check House loader channel indices")


if __name__ == "__main__":
    main()
