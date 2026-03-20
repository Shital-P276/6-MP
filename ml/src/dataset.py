"""
dataset.py — FloorPlanDataset with fault-tolerant sample loading

Bad samples (corrupt image, wrong mask values, transform crash) are skipped
silently. The loader tries up to MAX_RETRIES neighbours before giving up,
so a DataLoader batch is never short even if individual files are broken.
"""

import json
import numpy as np
import cv2
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMAGE_SIZE    = 512
MAX_RETRIES   = 10   # how many neighbours to try before raising


def get_train_transforms() -> A.Compose:
    return A.Compose([
        # FIX: albumentations 1.4+ requires size=(H, W) instead of height=, width=
        A.RandomResizedCrop(
            size=(IMAGE_SIZE, IMAGE_SIZE),
            scale=(0.7, 1.0), ratio=(0.75, 1.33),
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.0, p=0.7),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        # A.ToGray keeps 3 channels (RGB dims preserved) — safe for MiT-B2
        A.ToGray(p=0.15),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms() -> A.Compose:
    return A.Compose([
        # FIX: albumentations 1.4+ — positional (H, W) works; height=/width= removed
        A.Resize(IMAGE_SIZE, IMAGE_SIZE, interpolation=cv2.INTER_LANCZOS4),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def _load_sample(sample: dict):
    """
    Load and validate a single {image, mask} pair.
    Returns (img_tensor, mask_tensor) or raises ValueError on any problem.
    """
    img = cv2.imread(sample["image"])
    if img is None:
        raise ValueError(f"Cannot read image: {sample['image']}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(sample["mask"], cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask: {sample['mask']}")

    # Validate mask values
    unique  = np.unique(mask).tolist()
    invalid = set(unique) - {0, 1, 2, 3}
    if invalid:
        raise ValueError(f"Invalid mask values {invalid} in {sample['mask']}")
    if 1 not in unique:
        raise ValueError(f"No wall pixels in {sample['mask']}")

    return img, mask.astype(np.uint8)


class FloorPlanDataset(Dataset):
    """
    Loads floor plan image + 4-class integer mask pairs.
    Skips broken samples gracefully — never crashes the DataLoader.

    JSON format:
    [{"image": "/abs/path.png", "mask": "/abs/path_mask.png", "source": "cubicasa"}, ...]
    """

    def __init__(self, split_json: str, transforms=None):
        with open(split_json) as f:
            self.samples = json.load(f)
        self.transforms = transforms
        if not self.samples:
            raise ValueError(f"Empty split: {split_json}")
        self._bad_indices = set()   # track known-bad indices to avoid repeated warnings

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fault-tolerant load: if the requested sample fails, try MAX_RETRIES
        neighbours before giving up. This ensures DataLoader never receives
        None or raises mid-batch.
        """
        for attempt in range(MAX_RETRIES):
            try_idx = (idx + attempt) % len(self.samples)
            sample  = self.samples[try_idx]

            try:
                img, mask = _load_sample(sample)
            except ValueError as e:
                if try_idx not in self._bad_indices:
                    self._bad_indices.add(try_idx)
                    print(f"[WARN] Skipping sample {try_idx}: {e}", flush=True)
                continue

            # Apply transforms — wrap in try/except in case augmentation fails
            # (e.g. albumentations version mismatch, degenerate crop)
            try:
                if self.transforms:
                    aug    = self.transforms(image=img, mask=mask)
                    img_t  = aug["image"]           # (3, H, W) float tensor
                    mask_t = aug["mask"].long()      # (H, W) int64 tensor
                else:
                    img_t  = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
                    mask_t = torch.from_numpy(mask.astype(np.int64))
            except Exception as e:
                if try_idx not in self._bad_indices:
                    self._bad_indices.add(try_idx)
                    print(f"[WARN] Transform failed on sample {try_idx}: {e}", flush=True)
                continue

            # Final check: correct shapes and no NaN
            if img_t.shape[0] != 3:
                print(f"[WARN] Sample {try_idx}: image has {img_t.shape[0]} channels (expected 3)", flush=True)
                continue
            if torch.isnan(img_t).any():
                print(f"[WARN] Sample {try_idx}: NaN in image tensor after normalisation", flush=True)
                continue

            return img_t, mask_t

        # If we get here all MAX_RETRIES neighbours also failed — return a zero tensor
        # so DataLoader can still form a batch (train_one_epoch will detect NaN loss and skip)
        print(f"[ERROR] All {MAX_RETRIES} attempts failed around idx={idx} — returning zeros", flush=True)
        return (
            torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE),
            torch.zeros(IMAGE_SIZE, IMAGE_SIZE, dtype=torch.long),
        )


def build_dataloaders(
    train_json:  str,
    val_json:    str,
    batch_size:  int = 8,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = FloorPlanDataset(train_json, transforms=get_train_transforms())
    val_ds   = FloorPlanDataset(val_json,   transforms=get_val_transforms())

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader


if __name__ == "__main__":
    import sys
    split = sys.argv[1] if len(sys.argv) > 1 else "train.json"
    ds = FloorPlanDataset(split, transforms=get_val_transforms())
    print(f"Dataset size: {len(ds)}")
    img, mask = ds[0]
    print(f"Image: {img.shape} {img.dtype}  min={img.min():.2f} max={img.max():.2f}")
    print(f"Mask:  {mask.shape} {mask.dtype}  unique={mask.unique().tolist()}")
    assert img.shape[0] == 3, f"Expected 3-channel image, got {img.shape}"
    print("Dataset OK ✓")
