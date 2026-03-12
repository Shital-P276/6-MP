# src/datasets.py
#
# PyTorch Dataset for floor plan segmentation.
#
# Expects data prepared by tools/prepare_cubicasa.py which outputs:
#
#   data/processed/
#     images/          <id>.png  (RGB, any size — transforms handle resize)
#     masks/           <id>.png  (uint8 single-channel, values 0-3)
#     splits/
#       train.json     [{"image_path": "...", "mask_path": "..."}, ...]
#       val.json
#       test.json
#
# Mask class encoding:
#   0 = background  (floor, furniture, room labels, empty space)
#   1 = wall        (all wall types + railings + columns)
#   2 = door        (opening gap only — not the swing arc or leaf)
#   3 = window      (opening gap only — not the sill projection)

import os
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class FloorPlanDataset(Dataset):
    """
    Loads (image, mask) pairs from pre-processed floor plan data.

    Args:
        data_root:  Root directory containing processed/ subdirectory.
        split:      One of 'train', 'val', 'test'.
        transform:  Albumentations Compose pipeline (from augmentations.py).
                    Must accept keyword args: image=, mask=
                    Must return dict with keys: 'image', 'mask'

    Returns per __getitem__:
        dict with keys:
          'pixel_values'  FloatTensor (3, H, W)  — normalized image
          'labels'        LongTensor  (H, W)     — class indices 0-3
          'image_id'      str                    — filename stem for debugging

    Example:
        from datasets import FloorPlanDataset
        from augmentations import get_train_transforms

        ds = FloorPlanDataset('./data', 'train', get_train_transforms(512))
        sample = ds[0]
        print(sample['pixel_values'].shape)   # torch.Size([3, 512, 512])
        print(sample['labels'].shape)         # torch.Size([512, 512])
        print(sample['labels'].unique())      # tensor([0, 1, 2, 3])
    """

    NUM_CLASSES = 4
    CLASS_NAMES = ['background', 'wall', 'door', 'window']

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        transform=None,
    ):
        assert split in ('train', 'val', 'test'), \
            f"split must be one of train/val/test, got '{split}'"

        self.transform = transform
        self.data_root = Path(data_root)

        split_file = self.data_root / 'processed' / 'splits' / f'{split}.json'
        if not split_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_file}\n"
                f"Run tools/prepare_cubicasa.py first to generate splits."
            )

        with open(split_file) as f:
            self.samples = json.load(f)

        print(f"[FloorPlanDataset] {split}: {len(self.samples)} samples loaded")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        image_path = s['image_path']
        mask_path  = s['mask_path']

        # Load image as RGB numpy array (H, W, 3) uint8
        image = np.array(Image.open(image_path).convert('RGB'))

        # Load mask as single-channel numpy array (H, W) uint8 values 0-3
        mask = np.array(Image.open(mask_path).convert('L'))

        # Clamp mask to valid range — safety check for bad annotations
        mask = np.clip(mask, 0, self.NUM_CLASSES - 1).astype(np.uint8)

        # Apply augmentations (albumentations handles image+mask jointly)
        if self.transform:
            aug   = self.transform(image=image, mask=mask)
            image = aug['image']   # After Normalize: float32 (H, W, 3)
            mask  = aug['mask']    # Still uint8 (H, W)

        # Convert to tensors
        # image: (H, W, 3) float32 → (3, H, W) float32
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image)
        ).permute(2, 0, 1).float()

        # mask: (H, W) uint8 → (H, W) int64
        mask_tensor = torch.from_numpy(
            np.ascontiguousarray(mask)
        ).long()

        return {
            'pixel_values': image_tensor,
            'labels':        mask_tensor,
            'image_id':      Path(image_path).stem,
        }

    def get_class_weights(self) -> torch.Tensor:
        """
        Return class weights for weighted cross-entropy loss.
        Pre-computed from CubiCasa5k pixel statistics:
          background ~85%, wall ~12%, door ~2%, window ~1%

        Inverse-frequency weights, clipped to avoid extreme values.
        Recommended: pass these to F.cross_entropy(weight=...) in training.
        """
        # From analysis of CubiCasa5k training set pixel counts:
        # bg=0.85, wall=0.12, door=0.02, window=0.01
        # weight = 1 / freq, normalized so wall=3.0
        return torch.tensor([0.5, 3.0, 5.0, 5.0], dtype=torch.float32)


class CombinedFloorPlanDataset(Dataset):
    """
    Merges multiple FloorPlanDataset instances for mixed training.

    Useful for combining CubiCasa5k + ResPlan (Phase 1/2) or adding
    Indian plan data with oversampling (Phase 3 fine-tuning).

    Args:
        datasets:  List of FloorPlanDataset instances.
        weights:   Optional sampling weights per dataset.
                   e.g. [1.0, 3.0] oversamples second dataset 3x.
                   If None, all datasets are weighted equally.

    Example (Phase 3 Indian fine-tuning):
        base_ds   = FloorPlanDataset('./data', 'train', transform)
        indian_ds = FloorPlanDataset('./data/indian', 'train', transform)
        combined  = CombinedFloorPlanDataset(
            [base_ds, indian_ds],
            weights=[1.0, 3.0]   # oversample Indian data 3x
        )
    """

    def __init__(self, datasets: list, weights: Optional[list] = None):
        self.datasets = datasets
        if weights is None:
            weights = [1.0] * len(datasets)
        assert len(weights) == len(datasets)

        # Build a flat index: each entry is (dataset_idx, sample_idx)
        self.index = []
        for ds_idx, (ds, w) in enumerate(zip(datasets, weights)):
            reps = max(1, round(w))
            for _ in range(reps):
                for sample_idx in range(len(ds)):
                    self.index.append((ds_idx, sample_idx))

        print(f"[CombinedFloorPlanDataset] Total: {len(self.index)} samples "
              f"from {len(datasets)} datasets (weights={weights})")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        ds_idx, sample_idx = self.index[idx]
        return self.datasets[ds_idx][sample_idx]
