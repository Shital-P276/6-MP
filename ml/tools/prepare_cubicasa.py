# tools/prepare_cubicasa.py
#
# Converts CubiCasa5k SVG annotations → PNG masks and builds train/val/test
# split JSON files for FloorPlanDataset.
#
# Works with the Zenodo download (https://zenodo.org/record/2613548) which
# contains train.txt / val.txt / test.txt instead of cubicasa5k.csv.
#
# Each line in those txt files is a folder path like:
#   /high_quality_architectural/6044/
#   /high_quality/1234/
#
# Usage (Kaggle):
#   python tools/prepare_cubicasa.py \
#       --cubicasa-root /kaggle/working/data/raw/cubicasa5k \
#       --output-root   /kaggle/working/data/processed \
#       --floortrans-repo /kaggle/working/cubicasa_repo
#
# What it does:
#   1. Reads train.txt, val.txt, test.txt from cubicasa-root
#   2. For each folder, renders SVG annotations → 4-class pixel mask
#      using CubiCasa5k's floortrans loader (the RIGHT way, not manual SVG parsing)
#   3. Saves resized images + masks to processed/images/ and processed/masks/
#   4. Writes processed/splits/train.json, val.json, test.json
#
# Mask class encoding:
#   0 = background  (rooms, furniture, stairs, empty space)
#   1 = wall        (Wall + Railing merged)
#   2 = door        (opening gap only)
#   3 = window      (opening gap only)

import os
import sys
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Defaults (all overridable via CLI args)
# ─────────────────────────────────────────────────────────────────────────────

CUBICASA_ROOT   = Path('./data/raw/cubicasa5k')
OUTPUT_ROOT     = Path('./data/processed')
FLOORTRANS_REPO = Path('./data/raw/cubicasa_repo')   # CubiCasa5k GitHub clone
SAVE_SIZE       = 512
MIN_SIZE        = 200

# CubiCasa5k room_channels / icon_channels → our 4 classes
# Based on floortrans/data_conversions.py in the CubiCasa5k repo:
#   rooms[0]  = Background → 0
#   rooms[1]  = Outdoor    → 0
#   rooms[2]  = Wall       → 1
#   rooms[3]  = Kitchen    → 0
#   rooms[4]  = Living Room→ 0
#   rooms[5]  = Bedroom    → 0
#   rooms[6]  = Bath       → 0
#   rooms[7]  = Hallway    → 0
#   rooms[8]  = Railing    → 1  (merge with wall)
#   rooms[9]  = Storage    → 0
#   rooms[10] = Garage     → 0
#   rooms[11] = Undefined  → 0
#   icons[0]  = No Icon    → 0
#   icons[1]  = Window     → 3
#   icons[2]  = Door       → 2
#   icons[3]  = Closet     → 0
#   icons[4]  = Elect.     → 0
#   icons[5]  = Toilet     → 0
#   icons[6]  = Sink       → 0
#   icons[7]  = Sauna      → 0
#   icons[8]  = Bath       → 0
#   icons[9]  = Fireplace  → 0
#   icons[10] = Stairs     → 0
#   icons[11] = Stairs mirrored → 0

ROOM_CLASS_MAP = {
    2: 1,   # Wall  → wall
    8: 1,   # Railing → wall
}
ICON_CLASS_MAP = {
    1: 3,   # Window → window
    2: 2,   # Door   → door
}


# ─────────────────────────────────────────────────────────────────────────────
# Split loading — handles both txt files (Zenodo) and CSV (GitHub)
# ─────────────────────────────────────────────────────────────────────────────

def load_splits_from_txt(cubicasa_root: Path) -> tuple:
    """
    Read train.txt / val.txt / test.txt from the Zenodo download.

    Each line is a path like:  /high_quality_architectural/6044/
    Returns three lists of Path objects pointing to existing folders.
    """
    splits = {}
    for split_name in ('train', 'val', 'test'):
        txt_path = cubicasa_root / f'{split_name}.txt'
        if not txt_path.exists():
            print(f"  ⚠ {txt_path} not found — {split_name} will be empty")
            splits[split_name] = []
            continue

        folders = []
        with open(txt_path) as f:
            for line in f:
                rel = line.strip().strip('/')
                if not rel:
                    continue
                folder = cubicasa_root / rel
                if folder.exists():
                    folders.append(folder)
                # else: silently skip — some Zenodo entries may be missing

        splits[split_name] = folders
        print(f"  {split_name}.txt → {len(folders)} folders found")

    return splits['train'], splits['val'], splits['test']


def load_splits_from_csv(cubicasa_root: Path) -> tuple:
    """
    Fallback: read cubicasa5k.csv (GitHub format).
    CSV columns: path, split
    """
    import csv
    csv_path = cubicasa_root / 'cubicasa5k.csv'
    train_folders, val_folders, test_folders = [], [], []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            folder = cubicasa_root / row['path'].strip('/')
            if not folder.exists():
                continue
            split = row.get('split', row.get('fold', '')).strip().lower()
            if split == 'train':
                train_folders.append(folder)
            elif split == 'val':
                val_folders.append(folder)
            elif split == 'test':
                test_folders.append(folder)

    return train_folders, val_folders, test_folders


# ─────────────────────────────────────────────────────────────────────────────
# Sample loading — floortrans (preferred) or fallback
# ─────────────────────────────────────────────────────────────────────────────

def load_with_floortrans(folder_path: Path, floortrans_root: Path):
    """
    Load image + masks using CubiCasa5k's official floortrans loader.

    floortrans produces:
      image:  (H, W, 3) uint8 RGB
      heatmaps: (H, W, 21) float — first 12 are room channels, next 11 are icon channels
                We use argmax per-pixel within each group.
    """
    try:
        from floortrans.loaders.house import HouseExpoLoader

        loader = HouseExpoLoader(
            str(folder_path),
            set_name='',
            is_transform=False,
        )
        sample  = loader[0]
        image   = sample['image']       # (3, H, W) or (H, W, 3)
        heatmap = sample['heatmaps']    # (21, H, W) or (H, W, 21) float

        # Normalise axis order to (H, W, ...)
        if isinstance(image, np.ndarray):
            if image.shape[0] == 3:
                image = image.transpose(1, 2, 0)
        if isinstance(heatmap, np.ndarray):
            if heatmap.shape[0] == 21:
                heatmap = heatmap.transpose(1, 2, 0)

        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

        # Build 4-class mask from argmax of room + icon channels
        room_hm = heatmap[:, :, :12]   # channels 0-11
        icon_hm = heatmap[:, :, 12:]   # channels 12-20 (9 icon classes)

        room_pred = room_hm.argmax(axis=2)  # (H, W) room class index
        icon_pred = icon_hm.argmax(axis=2)  # (H, W) icon class index

        h, w = room_pred.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        # Apply room map
        for room_id, our_id in ROOM_CLASS_MAP.items():
            mask[room_pred == room_id] = our_id

        # Apply icon map (icons override rooms — doors/windows take priority)
        for icon_id, our_id in ICON_CLASS_MAP.items():
            mask[icon_pred == icon_id] = our_id

        return image, mask

    except Exception as e:
        return None, str(e)


def load_fallback_image_only(folder_path: Path):
    """
    Emergency fallback: load image only, return all-background mask.
    Mask is NOT usable for training — but lets the script complete.
    """
    for fname in ['F1_original.png', 'F1_floorplan.png', 'floorplan.png', 'F1.png']:
        p = folder_path / fname
        if p.exists():
            img = np.array(Image.open(p).convert('RGB'))
            return img, np.zeros(img.shape[:2], dtype=np.uint8)
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Process one split
# ─────────────────────────────────────────────────────────────────────────────

def process_split(
    folders: list,
    split_name: str,
    image_out_dir: Path,
    mask_out_dir: Path,
    floortrans_root: Path,
    use_floortrans: bool,
) -> list:
    samples   = []
    skipped   = 0
    processed = 0
    errors    = 0
    total     = len(folders)

    for i, folder in enumerate(folders):
        if (i + 1) % 200 == 0 or (i + 1) == total:
            print(f"  [{split_name}] {i+1}/{total} "
                  f"ok={processed} skip={skipped} err={errors}")

        plan_id = folder.name

        # ── Load ──────────────────────────────────────────────────────
        if use_floortrans:
            image, mask_or_err = load_with_floortrans(folder, floortrans_root)
            if image is None:
                # floortrans failed — try image-only fallback
                image, mask_or_err = load_fallback_image_only(folder)
                if image is None:
                    skipped += 1
                    continue
                # mask_or_err is None here from fallback
                mask = mask_or_err if mask_or_err is not None else np.zeros(image.shape[:2], dtype=np.uint8)
            else:
                mask = mask_or_err
        else:
            image, mask = load_fallback_image_only(folder)
            if image is None:
                skipped += 1
                continue

        # ── Size check ────────────────────────────────────────────────
        h, w = image.shape[:2]
        if h < MIN_SIZE or w < MIN_SIZE:
            skipped += 1
            continue

        # ── Resize ────────────────────────────────────────────────────
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_resized  = cv2.resize(img_bgr, (SAVE_SIZE, SAVE_SIZE),
                                   interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (SAVE_SIZE, SAVE_SIZE),
                                   interpolation=cv2.INTER_NEAREST)

        # Safety: clamp mask values
        mask_resized = np.clip(mask_resized, 0, 3).astype(np.uint8)

        # ── Save ──────────────────────────────────────────────────────
        img_path  = image_out_dir / f"{plan_id}.png"
        mask_path = mask_out_dir  / f"{plan_id}.png"
        cv2.imwrite(str(img_path),  img_resized)
        cv2.imwrite(str(mask_path), mask_resized)

        samples.append({
            'image_path': str(img_path),
            'mask_path':  str(mask_path),
        })
        processed += 1

    print(f"  [{split_name}] DONE  ok={processed}  skipped={skipped}  errors={errors}  total={total}")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Quick mask sanity check
# ─────────────────────────────────────────────────────────────────────────────

def verify_masks(split_json: list, n_check: int = 10):
    """Check that a sample of masks have wall pixels (value=1)."""
    import random
    has_walls = 0
    sample = random.sample(split_json, min(n_check, len(split_json)))
    for s in sample:
        m = np.array(Image.open(s['mask_path']).convert('L'))
        if (m == 1).any():
            has_walls += 1

    print(f"  Mask check: {has_walls}/{len(sample)} samples have wall pixels")
    if has_walls == 0:
        print("  ⚠ WARNING: No wall pixels found! floortrans may not be working correctly.")
        print("  ⚠ Check that floortrans_repo path is correct and the loader ran without errors.")
    return has_walls > 0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    cubicasa_root   = Path(args.cubicasa_root)
    output_root     = Path(args.output_root)
    floortrans_repo = Path(args.floortrans_repo)

    # ── Setup floortrans on path ───────────────────────────────────────
    use_floortrans = False
    if floortrans_repo.exists():
        sys.path.insert(0, str(floortrans_repo))
        try:
            from floortrans.loaders.house import HouseExpoLoader
            use_floortrans = True
            print(f"✓ floortrans loaded from {floortrans_repo}")
        except ImportError as e:
            print(f"⚠ floortrans import failed: {e}")
            print(f"  Will use image-only fallback (masks = all background, NOT usable for training)")
    else:
        print(f"⚠ floortrans_repo not found: {floortrans_repo}")
        print(f"  Pass --floortrans-repo to specify the CubiCasa5k GitHub clone path")

    # ── Load splits ───────────────────────────────────────────────────
    print(f"\nLoading splits from: {cubicasa_root}")
    if (cubicasa_root / 'train.txt').exists():
        print("  Format: txt files (Zenodo download)")
        train_folders, val_folders, test_folders = load_splits_from_txt(cubicasa_root)
    elif (cubicasa_root / 'cubicasa5k.csv').exists():
        print("  Format: CSV (GitHub)")
        train_folders, val_folders, test_folders = load_splits_from_csv(cubicasa_root)
    else:
        print(f"ERROR: Neither train.txt nor cubicasa5k.csv found in {cubicasa_root}")
        sys.exit(1)

    print(f"\nSplit sizes:")
    print(f"  train: {len(train_folders)}")
    print(f"  val:   {len(val_folders)}")
    print(f"  test:  {len(test_folders)}")

    # ── Create output dirs ────────────────────────────────────────────
    image_dir = output_root / 'images'
    mask_dir  = output_root / 'masks'
    split_dir = output_root / 'splits'
    for d in [image_dir, mask_dir, split_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Process ───────────────────────────────────────────────────────
    all_splits = {}
    for name, folders in [
        ('train', train_folders),
        ('val',   val_folders),
        ('test',  test_folders),
    ]:
        print(f"\nProcessing {name}...")
        samples = process_split(
            folders, name, image_dir, mask_dir, floortrans_repo, use_floortrans
        )
        all_splits[name] = samples

        out_path = split_dir / f'{name}.json'
        with open(out_path, 'w') as f:
            json.dump(samples, f, indent=2)
        print(f"  Wrote {out_path} ({len(samples)} samples)")

    # ── Verify masks have actual wall pixels ─────────────────────────
    print(f"\nVerifying masks...")
    verify_masks(all_splits['train'])

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Done!")
    print(f"  Images: {image_dir}")
    print(f"  Masks:  {mask_dir}")
    print(f"  Splits: {split_dir}")
    print(f"\nNext step: python src/train_segformer.py --data-root {output_root.parent}")

    if not use_floortrans:
        print(f"\n⚠ IMPORTANT: Ran without floortrans — masks are all-background!")
        print(f"  Training on these masks will produce a useless model.")
        print(f"  Fix: re-run with --floortrans-repo /path/to/CubiCasa5k-clone")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cubicasa-root',   default=str(CUBICASA_ROOT),
                   help='Path containing high_quality/, train.txt, etc.')
    p.add_argument('--output-root',     default=str(OUTPUT_ROOT),
                   help='Where to write images/, masks/, splits/')
    p.add_argument('--floortrans-repo', default=str(FLOORTRANS_REPO),
                   help='Path to CubiCasa5k GitHub clone (contains floortrans/)')
    p.add_argument('--save-size',       type=int, default=SAVE_SIZE)
    p.add_argument('--min-size',        type=int, default=MIN_SIZE)
    args = p.parse_args()
    SAVE_SIZE = args.save_size
    MIN_SIZE  = args.min_size
    main(args)
