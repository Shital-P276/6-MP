# tools/prepare_cubicasa.py
#
# Converts CubiCasa5k SVG annotations → PNG masks and builds train/val/test
# split JSON files for FloorPlanDataset.
#
# Run ONCE after cloning CubiCasa5k:
#   python tools/prepare_cubicasa.py
#
# What it does:
#   1. Reads cubicasa5k/cubicasa5k.csv — official train/val/test splits
#   2. For each floor plan, loads the SVG annotation via CubiCasa5k's own
#      house.py dataloader (do NOT re-parse SVGs manually)
#   3. Extracts 4-class pixel masks: bg=0, wall=1, door=2, window=3
#   4. Saves resized images + masks to data/processed/images/ and masks/
#   5. Writes data/processed/splits/train.json, val.json, test.json
#
# Expected input structure (after git clone):
#   data/raw/cubicasa5k/
#     cubicasa5k.csv
#     high_quality/
#       <id>/
#         F1_original.png   ← floor plan image
#         F1_floorplan.svg  ← annotation SVG
#     high_quality_architectural/
#       (same structure)
#
# Output structure:
#   data/processed/
#     images/<id>.png       resized to SAVE_SIZE
#     masks/<id>.png        uint8 single-channel, values 0-3
#     splits/train.json     [{"image_path": ..., "mask_path": ...}, ...]
#     splits/val.json
#     splits/test.json
#
# Requires CubiCasa5k repo on PYTHONPATH:
#   export PYTHONPATH=data/raw/cubicasa5k:$PYTHONPATH
#
# CubiCasa5k class → our 4 classes mapping:
#   CubiCasa category 1 (Background)   → 0 background
#   CubiCasa category 2 (Wall)         → 1 wall
#   CubiCasa category 8 (Railing)      → 1 wall  (merge)
#   CubiCasa category 3 (Door)         → 2 door
#   CubiCasa category 4 (Window)       → 3 window
#   All other categories               → 0 background
#     (furniture, stairs, rooms, etc.)

import os
import sys
import json
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

CUBICASA_ROOT  = Path('./data/raw/cubicasa5k')
OUTPUT_ROOT    = Path('./data/processed')
SAVE_SIZE      = 512        # All images/masks saved at this resolution
MIN_SIZE       = 200        # Skip images smaller than this (CubiCasa5k known issue)

# CubiCasa5k category IDs → our class IDs
# Full category list: https://github.com/CubiCasa/CubiCasa5k/blob/master/README.md
CATEGORY_MAP = {
    1:  0,   # Background   → background
    2:  1,   # Wall         → wall
    3:  2,   # Door         → door
    4:  3,   # Window       → window
    5:  0,   # Stairs       → background
    6:  0,   # Elevator     → background
    7:  0,   # Escalator    → background
    8:  1,   # Railing      → wall (merge — railings treated as thin walls)
    9:  0,   # Garage       → background
    10: 0,   # Column       → background  (NOTE: our augmentation adds these separately)
    # Room categories (11-99+) → background
}
DEFAULT_CLASS = 0  # Any unmapped CubiCasa category → background


# ─────────────────────────────────────────────────────────────────────────────
# SVG → mask via CubiCasa5k's own dataloader
# ─────────────────────────────────────────────────────────────────────────────

def load_cubicasa_sample_with_house_py(folder_path: Path) -> tuple:
    """
    Use CubiCasa5k's official house.py to load image + category mask.

    This is the RIGHT way — do not reparse SVGs manually.
    house.py handles all the polygon rendering, scale, and coordinate transforms.

    Returns:
        image:    (H, W, 3) uint8 RGB
        cat_mask: (H, W) uint8 — CubiCasa category IDs
        None, None if loading fails
    """
    try:
        # CubiCasa5k's house.py must be on PYTHONPATH
        from floortrans.loaders.house import FloorPlanGraph

        graph = FloorPlanGraph(str(folder_path))
        image, rooms, icons = graph.get_segmentation_maps()
        # rooms contains room-level masks, icons contains door/window icons
        # We want the combined segmentation: 'cat' channel
        cat_mask = graph.get_segmentation_maps()[1]
        return np.array(image), cat_mask
    except ImportError:
        raise ImportError(
            "Cannot import floortrans.loaders.house.\n"
            "Make sure CubiCasa5k repo is on PYTHONPATH:\n"
            "  export PYTHONPATH=data/raw/cubicasa5k:$PYTHONPATH"
        )
    except Exception as e:
        return None, None


def load_cubicasa_sample_fallback(folder_path: Path) -> tuple:
    """
    Fallback: Load image only (no mask) if house.py is not available.
    
    This lets you at least verify the data structure before setting up
    the full CubiCasa5k repo on PYTHONPATH.
    
    Returns dummy all-background mask — NOT usable for training.
    """
    # Try common image filenames used by CubiCasa5k
    for fname in ['F1_original.png', 'F1_floorplan.png', 'floorplan.png']:
        img_path = folder_path / fname
        if img_path.exists():
            img = np.array(Image.open(img_path).convert('RGB'))
            dummy_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            return img, dummy_mask
    return None, None


def cubicasa_mask_to_our_classes(cat_mask: np.ndarray) -> np.ndarray:
    """
    Convert CubiCasa5k category mask (80+ classes) → our 4-class mask.

    Args:
        cat_mask: (H, W) uint8 — CubiCasa category IDs

    Returns:
        (H, W) uint8 — values 0-3
    """
    our_mask = np.full_like(cat_mask, DEFAULT_CLASS)
    for cubicasa_id, our_id in CATEGORY_MAP.items():
        our_mask[cat_mask == cubicasa_id] = our_id
    return our_mask


# ─────────────────────────────────────────────────────────────────────────────
# Main processing
# ─────────────────────────────────────────────────────────────────────────────

def process_split(
    folders: list,
    split_name: str,
    image_out_dir: Path,
    mask_out_dir:  Path,
    use_house_py:  bool = True,
) -> list:
    """
    Process a list of floor plan folders, save images + masks, return split JSON.

    Args:
        folders:      List of Path objects pointing to plan directories.
        split_name:   'train', 'val', or 'test' (for logging only).
        image_out_dir: Directory to save processed images.
        mask_out_dir:  Directory to save processed masks.
        use_house_py:  If True, use house.py; if False, use fallback.

    Returns:
        List of dicts: [{"image_path": str, "mask_path": str}, ...]
    """
    samples   = []
    skipped   = 0
    processed = 0
    total     = len(folders)

    for i, folder in enumerate(folders):
        if (i + 1) % 100 == 0:
            print(f"  [{split_name}] {i+1}/{total} — processed={processed} skipped={skipped}")

        plan_id = folder.name

        # Load image and mask
        if use_house_py:
            image, cat_mask = load_cubicasa_sample_with_house_py(folder)
        else:
            image, cat_mask = load_cubicasa_sample_fallback(folder)

        if image is None:
            skipped += 1
            continue

        # Skip tiny images (CubiCasa5k known issue: some < 200px)
        h, w = image.shape[:2]
        if h < MIN_SIZE or w < MIN_SIZE:
            skipped += 1
            continue

        # Convert mask classes
        if use_house_py:
            our_mask = cubicasa_mask_to_our_classes(cat_mask)
        else:
            our_mask = cat_mask  # fallback: already 0-only dummy

        # Resize to SAVE_SIZE (preserve original aspect ratio would be better,
        # but fixed size simplifies storage; transforms handle augmentation)
        image_resized = cv2.resize(
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            (SAVE_SIZE, SAVE_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )
        mask_resized = cv2.resize(
            our_mask,
            (SAVE_SIZE, SAVE_SIZE),
            interpolation=cv2.INTER_NEAREST,  # IMPORTANT: no interpolation for class labels
        )

        # Save
        img_path  = image_out_dir / f"{plan_id}.png"
        mask_path = mask_out_dir  / f"{plan_id}.png"

        cv2.imwrite(str(img_path), image_resized)
        cv2.imwrite(str(mask_path), mask_resized)

        samples.append({
            'image_path': str(img_path),
            'mask_path':  str(mask_path),
        })
        processed += 1

    print(f"  [{split_name}] Done. Processed={processed} Skipped={skipped}/{total}")
    return samples


def main(args):
    # ── Verify CubiCasa5k is present ──────────────────────────────────────
    csv_path = CUBICASA_ROOT / 'cubicasa5k.csv'
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        print("Run:  git clone https://github.com/CubiCasa/CubiCasa5k.git data/raw/cubicasa5k")
        sys.exit(1)

    # ── Check if house.py is importable ───────────────────────────────────
    use_house_py = True
    try:
        sys.path.insert(0, str(CUBICASA_ROOT))
        from floortrans.loaders.house import FloorPlanGraph
        print("✓ floortrans.loaders.house found — will use official SVG parser")
    except ImportError:
        print("⚠ floortrans.loaders.house NOT found — using fallback (no masks)")
        print("  Set PYTHONPATH=data/raw/cubicasa5k to enable SVG parsing")
        use_house_py = False

    # ── Read official splits from cubicasa5k.csv ──────────────────────────
    #  CSV format: path,split
    #  path = relative path to folder (e.g. high_quality/1234)
    #  split = train | val | test
    train_folders, val_folders, test_folders = [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            folder = CUBICASA_ROOT / row['path'].strip('/')
            if not folder.exists():
                continue
            split = row.get('split', row.get('fold', '')).strip().lower()
            if split == 'train':
                train_folders.append(folder)
            elif split == 'val':
                val_folders.append(folder)
            elif split == 'test':
                test_folders.append(folder)

    print(f"\nCubiCasa5k split sizes:")
    print(f"  Train: {len(train_folders)}")
    print(f"  Val:   {len(val_folders)}")
    print(f"  Test:  {len(test_folders)}")

    # ── Set up output directories ──────────────────────────────────────────
    image_dir = OUTPUT_ROOT / 'images'
    mask_dir  = OUTPUT_ROOT / 'masks'
    split_dir = OUTPUT_ROOT / 'splits'
    for d in [image_dir, mask_dir, split_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Process each split ─────────────────────────────────────────────────
    print("\nProcessing train split...")
    train_samples = process_split(train_folders, 'train', image_dir, mask_dir, use_house_py)

    print("\nProcessing val split...")
    val_samples = process_split(val_folders, 'val', image_dir, mask_dir, use_house_py)

    print("\nProcessing test split...")
    test_samples = process_split(test_folders, 'test', image_dir, mask_dir, use_house_py)

    # ── Write split JSON files ─────────────────────────────────────────────
    for name, samples in [('train', train_samples), ('val', val_samples), ('test', test_samples)]:
        out = split_dir / f'{name}.json'
        with open(out, 'w') as f:
            json.dump(samples, f, indent=2)
        print(f"Wrote {out} ({len(samples)} samples)")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Data preparation complete!")
    print(f"  Images:  {image_dir}")
    print(f"  Masks:   {mask_dir}")
    print(f"  Splits:  {split_dir}")
    print(f"\nNext step:")
    print(f"  python src/train_segformer.py")

    if not use_house_py:
        print(f"\n⚠ WARNING: Ran with fallback (no SVG parsing).")
        print(f"  Masks are all background — NOT usable for training.")
        print(f"  Fix: export PYTHONPATH=data/raw/cubicasa5k:$PYTHONPATH")
        print(f"  Then re-run this script.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cubicasa-root', default=str(CUBICASA_ROOT))
    parser.add_argument('--output-root',   default=str(OUTPUT_ROOT))
    parser.add_argument('--save-size',     type=int, default=SAVE_SIZE)
    args = parser.parse_args()

    CUBICASA_ROOT = Path(args.cubicasa_root)
    OUTPUT_ROOT   = Path(args.output_root)
    SAVE_SIZE     = args.save_size

    main(args)
