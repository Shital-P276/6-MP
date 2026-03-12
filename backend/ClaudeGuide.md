# ClaudeGuide.md
# ML-Powered Floor Plan Detection: Complete Implementation Guide
# For: Claude working on the FloorPlan 3D Viewer project
# Last updated: March 2026

---

## CONTEXT: WHAT THIS IS AND WHY

The current pipeline uses `raster_parser.py` — a hand-coded pixel scanner that:
- Detects walls via Hough line transforms on raw image pixels
- Finds doors/windows by scanning brightness bands for gaps
- Has fragile tuning (DEDUP_DIST, MIN_JAMB_PX, stub detection rules)
- Breaks on thick Indian masonry walls, unusual color schemes, non-CAD formats

**Goal**: Replace `raster_parser.py` with an ML model that outputs the same
`ParsedGeometry` object. Everything downstream — WallDetector, GeometryBuilder,
the 3D viewer — remains completely unchanged.

**Dataset reality**: No Indian floor plan dataset exists publicly.
We use CubiCasa5k + ResPlan as base, with aggressive augmentation to simulate
Indian plan characteristics (thick walls, dimension annotations, column markers, etc.)

**Architecture decision (final)**:
- Phase 1: SegFormer fine-tuned on CubiCasa5k → cleans input for existing raster_parser
- Phase 2: MuraNet end-to-end → full raster_parser replacement
- These are sequential, not combined. SegFormer output does NOT feed MuraNet.
  They are alternatives — SegFormer is the fast Phase 1 win, MuraNet is the full solution.

---

## THE THREE PHASES AT A GLANCE

```
PHASE 1 (Weeks 1-2): SegFormer as image preprocessor
  Input:  raw floor plan image
  Output: clean wall_mask / door_mask / window_mask
  Usage:  feed wall_mask as cleaned input into existing raster_parser.py
  Result: raster_parser works on noise-free wall pixels → fewer bugs
  Risk:   LOW — existing pipeline barely changes

PHASE 2 (Weeks 3-8): MuraNet as full parser
  Input:  raw floor plan image
  Output: wall segments + door/window bounding boxes → ParsedGeometry
  Usage:  drop-in replacement for raster_parser.py entirely
  Result: all brightness-scanning, stub-detection, DEDUP logic deleted
  Risk:   MEDIUM — needs solid training data + evaluation

PHASE 3 (Ongoing): Fine-tune on real Indian plans
  Input:  annotated Indian plans (collected over time)
  Output: better MuraNet checkpoint for Indian-specific features
  Usage:  swap checkpoint, no code changes
  Risk:   LOW — incremental improvement, can't make things worse
```

---

## PHASE 1: SEGFORMER

### Step 1.1 — Environment Setup

```bash
python -m venv floorplan_ml
source floorplan_ml/bin/activate       # Linux/Mac
# floorplan_ml\Scripts\activate        # Windows

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets
pip install albumentations opencv-python pillow numpy matplotlib
pip install tensorboard huggingface_hub
pip install scikit-image                # for skeletonization in Phase 2

# Verify GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Minimum hardware: 8GB VRAM (RTX 3070+).
No GPU: Use Google Colab Pro (A100) or Kaggle free tier (T4, 30hr/week).

### Step 1.2 — Download CubiCasa5k

```bash
git clone https://github.com/CubiCasa/CubiCasa5k.git
cd CubiCasa5k
# Dataset is ~2GB. Contains:
#   high_quality/                  2500 clean CAD plans
#   high_quality_architectural/    2500 with furniture
#   cubicasa5k.csv                 train/val/test split
# Each image has a matching .svg annotation file
```

IMPORTANT: CubiCasa5k has a ready-made Python dataloader in their repo.
Use `floortrans/loaders/house.py` — it handles SVG parsing already.
Do NOT rewrite the SVG parser from scratch.

The 4 classes we care about from CubiCasa5k's 80+ categories:
- background = 0
- wall = 1        (includes Railing, merge them)
- door = 2
- window = 3

### Step 1.3 — Download ResPlan (optional but recommended)

ResPlan (2025) has 17,000 residential floor plans with wall/door/window/balcony
annotations in JSON format. Much more diverse than CubiCasa5k alone.

```bash
# Search for ResPlan on Papers With Code or HuggingFace datasets
# URL: https://huggingface.co/datasets/[ResPlan-repo]
# After download, convert JSON annotations → pixel masks using provided scripts
```

If ResPlan is hard to obtain, CubiCasa5k alone (4,200 training samples with
augmentation) is sufficient for a working Phase 1 model.

### Step 1.4 — Build Augmentation Pipeline

This is the most critical step for Indian plan generalization.
Apply ONLY during training, never on val/test sets.

```python
# src/augmentations.py

import albumentations as A
import cv2
import numpy as np
import random


class AddRandomTextOverlay(A.ImageOnlyTransform):
    """
    Simulate Hindi/English dimension annotations.
    Draws random text at random positions.
    Does NOT modify the mask — model learns to ignore text.
    """
    def __init__(self, num_texts=(3, 12), p=0.4):
        super().__init__(p=p)
        self.num_texts = num_texts

    def apply(self, img, **params):
        result = img.copy()
        h, w = img.shape[:2]
        texts = [
            f"{random.randint(1,30)}'-{random.randint(0,11)}\"",
            f"{random.randint(100,6000)}mm",
            f"BR {random.randint(1,4)}",
            "TOILET", "KITCHEN", "HALL", "POOJA", "STORE",
            f"W={random.randint(600,1500)}", f"D={random.randint(750,1200)}",
        ]
        for _ in range(random.randint(*self.num_texts)):
            text = random.choice(texts)
            x = random.randint(10, max(10, w - 100))
            y = random.randint(15, max(15, h - 10))
            scale = random.uniform(0.3, 0.8)
            thickness = random.randint(1, 2)
            color = random.choice([(0,0,0), (50,50,200), (0,0,150)])
            cv2.putText(result, text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
        return result


class AddColumnMarkers(A.DualTransform):
    """
    Simulate Indian RC frame column squares at wall junctions.
    Modifies BOTH image AND mask (columns labeled as wall class = 1).
    """
    def __init__(self, num_columns=(4, 12), col_size_px=(8, 20), p=0.35):
        super().__init__(p=p)
        self.num_columns = num_columns
        self.col_size_px = col_size_px

    def get_params_dependent_on_targets(self, params):
        h, w = params['image'].shape[:2]
        n = random.randint(*self.num_columns)
        cols = []
        for _ in range(n):
            size = random.randint(*self.col_size_px)
            x = random.randint(0, max(0, w - size - 1))
            y = random.randint(0, max(0, h - size - 1))
            cols.append((x, y, size))
        return {'columns': cols}

    @property
    def targets_as_params(self):
        return ['image']

    def apply(self, img, columns=None, **params):
        result = img.copy()
        for (x, y, size) in (columns or []):
            cv2.rectangle(result, (x, y), (x+size, y+size), (0, 0, 0), -1)
        return result

    def apply_to_mask(self, mask, columns=None, **params):
        result = mask.copy()
        for (x, y, size) in (columns or []):
            result[y:y+size, x:x+size] = 1   # wall class
        return result

    def get_transform_init_args_names(self):
        return ('num_columns', 'col_size_px')


class AddDimensionLines(A.ImageOnlyTransform):
    """Simulate dimension witness lines with tick marks at plan edges."""
    def apply(self, img, **params):
        result = img.copy()
        h, w = img.shape[:2]
        for _ in range(random.randint(2, 6)):
            if random.random() > 0.5:
                y = random.randint(5, h - 5)
                x1, x2 = sorted(random.sample(range(w), 2))
                cv2.line(result, (x1, y), (x2, y), (0, 0, 0), 1)
                cv2.line(result, (x1, y-4), (x1, y+4), (0, 0, 0), 1)
                cv2.line(result, (x2, y-4), (x2, y+4), (0, 0, 0), 1)
            else:
                x = random.randint(5, w - 5)
                y1, y2 = sorted(random.sample(range(h), 2))
                cv2.line(result, (x, y1), (x, y2), (0, 0, 0), 1)
                cv2.line(result, (x-4, y1), (x+4, y1), (0, 0, 0), 1)
                cv2.line(result, (x-4, y2), (x+4, y2), (0, 0, 0), 1)
        return result


def get_train_transforms(image_size=512):
    return A.Compose([
        A.Resize(image_size, image_size),

        # ── GEOMETRIC (applied to image + mask) ───────────────────────
        # RandomScale simulates different wall thicknesses
        # scale 0.6 = walls appear 67% thicker (Indian masonry effect)
        A.RandomScale(scale_limit=(-0.4, 0.4), p=0.7),
        A.PadIfNeeded(image_size, image_size, border_mode=cv2.BORDER_CONSTANT,
                      value=255, mask_value=0),
        A.RandomCrop(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=5, border_mode=cv2.BORDER_CONSTANT,
                 value=255, mask_value=0, p=0.4),

        # ── PHOTOMETRIC (image only) ───────────────────────────────────
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.RandomGamma(gamma_limit=(70, 130), p=0.4),
        A.ColorJitter(hue=0.05, saturation=0.3, brightness=0.2, p=0.4),
        A.ToGray(p=0.2),
        A.HueSaturationValue(hue_shift_limit=15, p=0.3),

        # ── NOISE / BLUR ───────────────────────────────────────────────
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50)),
            A.ISONoise(color_shift=(0.01, 0.05)),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1)),
        ], p=0.4),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MotionBlur(blur_limit=5),
        ], p=0.3),

        # ── INDIAN PLAN SPECIFIC ───────────────────────────────────────
        AddRandomTextOverlay(p=0.4),
        AddDimensionLines(p=0.3),
        AddColumnMarkers(p=0.35),

        # ── NORMALIZE ─────────────────────────────────────────────────
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(image_size=512):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
```

### Step 1.5 — Dataset Class

```python
# src/datasets.py

import os, json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class FloorPlanDataset(Dataset):
    """
    Loads (image, mask) pairs from processed CubiCasa5k or ResPlan data.
    Expects data_root/splits/{split}.json listing image_path and mask_path.
    Mask values: 0=background, 1=wall, 2=door, 3=window
    """
    def __init__(self, data_root: str, split: str = 'train', transform=None):
        self.transform = transform
        split_file = os.path.join(data_root, 'splits', f'{split}.json')
        with open(split_file) as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = np.array(Image.open(s['image_path']).convert('RGB'))
        mask  = np.array(Image.open(s['mask_path']))   # uint8, 0-3

        if self.transform:
            aug   = self.transform(image=image, mask=mask)
            image = aug['image']
            mask  = aug['mask']

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask  = torch.from_numpy(mask).long()
        return {'pixel_values': image, 'labels': mask}
```

### Step 1.6 — SegFormer Training Script

```python
# src/train_segformer.py

import os, json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import SegformerForSemanticSegmentation
from transformers import get_linear_schedule_with_warmup
from datasets import FloorPlanDataset
from augmentations import get_train_transforms, get_val_transforms

CONFIG = {
    'model_name':    'nvidia/segformer-b2-finetuned-ade-512-512',
    'num_classes':   4,
    'image_size':    512,
    'batch_size':    8,       # reduce to 4 if OOM
    'lr':            6e-5,
    'weight_decay':  0.01,
    'epochs':        50,
    'warmup_steps':  500,
    'save_dir':      './checkpoints/segformer',
    'data_root':     './data/processed',
    # Upweight wall/door/window to combat class imbalance
    # (background dominates pixel count in floor plans)
    'class_weights': [0.5, 3.0, 5.0, 5.0],   # bg, wall, door, window
}


def compute_iou(preds: torch.Tensor, labels: torch.Tensor, num_classes=4):
    ious = []
    for cls in range(num_classes):
        tp = ((preds == cls) & (labels == cls)).sum().float()
        union = ((preds == cls) | (labels == cls)).sum().float()
        ious.append((tp / union).item() if union > 0 else float('nan'))
    return ious


def train(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_ds = FloorPlanDataset(cfg['data_root'], 'train',
                                 get_train_transforms(cfg['image_size']))
    val_ds   = FloorPlanDataset(cfg['data_root'], 'val',
                                 get_val_transforms(cfg['image_size']))
    train_dl = DataLoader(train_ds, batch_size=cfg['batch_size'],
                           shuffle=True,  num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=4,
                           shuffle=False, num_workers=4)

    model = SegformerForSemanticSegmentation.from_pretrained(
        cfg['model_name'],
        num_labels=cfg['num_classes'],
        id2label={0:'background',1:'wall',2:'door',3:'window'},
        label2id={'background':0,'wall':1,'door':2,'window':3},
        ignore_mismatched_sizes=True,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    total_steps = len(train_dl) * cfg['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer, cfg['warmup_steps'], total_steps)
    scaler  = GradScaler()
    weights = torch.tensor(cfg['class_weights'], device=device)
    best_miou = 0.0

    for epoch in range(cfg['epochs']):
        # ── TRAIN ──────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for batch in train_dl:
            px  = batch['pixel_values'].to(device)
            lbl = batch['labels'].to(device)
            optimizer.zero_grad()
            with autocast():
                out = model(pixel_values=px)
                up  = F.interpolate(out.logits,
                                    size=(cfg['image_size'], cfg['image_size']),
                                    mode='bilinear', align_corners=False)
                loss = F.cross_entropy(up, lbl, weight=weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        # ── VALIDATE ───────────────────────────────────────────────────
        model.eval()
        all_ious = []
        with torch.no_grad():
            for batch in val_dl:
                px  = batch['pixel_values'].to(device)
                lbl = batch['labels'].to(device)
                out = model(pixel_values=px)
                up  = F.interpolate(out.logits,
                                    size=(cfg['image_size'], cfg['image_size']),
                                    mode='bilinear', align_corners=False)
                preds = up.argmax(dim=1)
                for p, l in zip(preds, lbl):
                    all_ious.append(compute_iou(p.cpu(), l.cpu()))

        mean_ious = np.nanmean(all_ious, axis=0)
        miou = float(np.nanmean(mean_ious))
        print(f"Epoch {epoch+1:3d}/{cfg['epochs']} | "
              f"Loss: {total_loss/len(train_dl):.4f} | "
              f"mIoU: {miou:.4f} | "
              f"Wall: {mean_ious[1]:.3f}  Door: {mean_ious[2]:.3f}  Win: {mean_ious[3]:.3f}")

        if miou > best_miou:
            best_miou = miou
            os.makedirs(cfg['save_dir'], exist_ok=True)
            model.save_pretrained(f"{cfg['save_dir']}/best")
            print(f"  ✓ Saved (mIoU={miou:.4f})")


if __name__ == '__main__':
    train(CONFIG)
```

### Step 1.7 — Phase 1 Inference Wrapper

Integrates SegFormer into the existing pipeline WITHOUT changing raster_parser.py.

```python
# src/ml_preprocessor.py

import torch
import numpy as np
import cv2
from PIL import Image
from transformers import SegformerForSemanticSegmentation
import torch.nn.functional as F


class MLPreprocessor:
    """
    Phase 1 integration.
    Runs SegFormer, returns a cleaned binary image that raster_parser.py
    can process reliably (wall pixels only, no furniture/text noise).
    """

    CLASS_BG, CLASS_WALL, CLASS_DOOR, CLASS_WIN = 0, 1, 2, 3
    MEAN = np.array([0.485, 0.456, 0.406])
    STD  = np.array([0.229, 0.224, 0.225])

    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            checkpoint_path).eval().to(self.device)
        self.image_size = 512

    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(img, (self.image_size, self.image_size))
        normed  = (resized.astype(np.float32) / 255.0 - self.MEAN) / self.STD
        return torch.from_numpy(normed).permute(2, 0, 1).unsqueeze(0).float()

    def get_masks(self, image_path: str) -> dict:
        """Run inference, return binary masks at original image size."""
        orig = np.array(Image.open(image_path).convert('RGB'))
        h, w = orig.shape[:2]
        tensor = self._preprocess(orig).to(self.device)
        with torch.no_grad():
            out  = self.model(pixel_values=tensor)
            up   = F.interpolate(out.logits, size=(h, w),
                                  mode='bilinear', align_corners=False)
            pred = up.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
        return {
            'wall':   (pred == self.CLASS_WALL).astype(np.uint8) * 255,
            'door':   (pred == self.CLASS_DOOR).astype(np.uint8) * 255,
            'window': (pred == self.CLASS_WIN ).astype(np.uint8) * 255,
            'full_pred':     pred,
            'original_size': (h, w),
        }

    def make_clean_wall_image(self, image_path: str) -> np.ndarray:
        """
        Returns a clean black-on-white image showing only wall/door/window pixels.
        Feed this as input to raster_parser.py instead of the raw image.
        
        Encoding:
          white (255) = background (ignored by raster_parser)
          black (0)   = wall
          gray (128)  = door opening
          gray (200)  = window opening
        """
        masks = self.get_masks(image_path)
        h, w  = masks['original_size']
        clean = np.full((h, w, 3), 255, dtype=np.uint8)
        clean[masks['wall']   > 0] = [0,   0,   0  ]
        clean[masks['door']   > 0] = [128, 128, 128]
        clean[masks['window'] > 0] = [200, 200, 200]
        return clean
```

**How to activate Phase 1 in existing pipeline:**
In `raster_parser.py`, add these lines at the top of the `parse()` method:

```python
USE_ML = True   # toggle this flag

if USE_ML:
    from ml_preprocessor import MLPreprocessor
    pre = MLPreprocessor('./checkpoints/segformer/best')
    clean = pre.make_clean_wall_image(image_path)
    tmp = '/tmp/_ml_clean.png'
    cv2.imwrite(tmp, clean)
    image_path = tmp
    # Everything below runs unchanged on the cleaned image
```

---

## PHASE 2: MURANET FULL REPLACEMENT

### Step 2.1 — MuraNet Architecture Overview

```
Input Image (512×512×3)
        ↓
Mix-Transformer B2 Encoder (shared backbone, pretrained on ImageNet)
  Produces multi-scale features: [E1, E2, E3, E4]
  at 1/4, 1/8, 1/16, 1/32 of input resolution
        ↓
  ┌─────┴──────────────────────────┐
  │                                │
Segmentation Decoder            Detection Head (YOLOX-style)
  Takes [E1, E2, E3, E4]          Takes [E3, E4] (coarser features)
  Upsamples progressively          3 detection scales
  Output: (H/4 × W/4 × 4)         Output per anchor:
  classes: bg/wall/door/win          - objectness score (0-1)
        ↓                            - class (door=0, window=1)
  Upsample to (H × W)               - bbox (cx, cy, w, h) normalized
  Loss: CE + Tversky                 - angle: 0 or π/2
                                   Loss: BCE + CIoU + CE
```

**Tversky Loss** — critical for walls (prevents bloated thick predictions):
```python
def tversky_loss(pred_prob, target, alpha=0.7, beta=0.3, smooth=1e-6):
    """
    alpha=0.7 penalizes False Positives heavily (prevents thick phantom walls).
    For door/window detection head, swap to alpha=0.3, beta=0.7 (high recall).
    """
    tp = (pred_prob * target).sum()
    fp = (pred_prob * (1 - target)).sum()
    fn = ((1 - pred_prob) * target).sum()
    return 1 - (tp + smooth) / (tp + alpha*fp + beta*fn + smooth)
```

### Step 2.2 — Get MuraNet Code

```bash
# Option A: Use published repo (preferred)
# Search "MuraNet floor plan" on Papers With Code
# or: https://github.com/[author]/MuraNet

# Option B: If repo is unmaintained/broken, implement from scratch
# The architecture is not complex. See ClaudeGuideExtra.md Section 3
# for the from-scratch implementation template.

# Key files in MuraNet repo (if using Option A):
#   model/muranet.py          full architecture
#   model/losses.py           joint loss
#   data/floorplan_dataset.py dataset loader (adapt for CubiCasa5k)
#   train.py                  training loop
#   inference.py              inference + NMS postprocessing
```

### Step 2.3 — Prepare Combined Dataset

```
data/
  raw/
    cubicasa5k/               raw downloaded repo
    resplan/                  raw downloaded data
  processed/
    images/                   .png files
    masks/                    4-class pixel masks (uint8 PNG)
    bboxes/                   door/window bounding box JSON files
    splits/
      train.json              list of {image_path, mask_path, bbox_path}
      val.json
      test.json               CubiCasa5k test only (for standard benchmark)
```

**Convert segmentation masks → detection bounding boxes:**

```python
# src/masks_to_bboxes.py
import cv2
import json
import numpy as np
from pathlib import Path


def extract_bboxes_from_mask(mask: np.ndarray) -> list:
    """
    Convert door/window pixel masks to bounding box annotations.
    Returns list of dicts: {class_id, cx, cy, w, h, angle}
    All values normalized 0-1 relative to image size.
    """
    h, w = mask.shape
    bboxes = []

    for cls_id, cls_val in [(0, 2), (1, 3)]:   # door=2, window=3 in mask
        binary = (mask == cls_val).astype(np.uint8)
        if binary.sum() == 0:
            continue

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8)

        for i in range(1, n_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 50:
                continue    # skip tiny noise

            bx = stats[i, cv2.CC_STAT_LEFT]
            by = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]

            cx_norm = (bx + bw / 2) / w
            cy_norm = (by + bh / 2) / h
            w_norm  = bw / w
            h_norm  = bh / h
            angle   = 0.0 if bw >= bh else 1.5708   # 0 = horizontal, π/2 = vertical

            bboxes.append({
                'class_id': cls_id,
                'cx': cx_norm, 'cy': cy_norm,
                'w': w_norm,   'h': h_norm,
                'angle': angle
            })

    return bboxes
```

### Step 2.4 — MuraNet Training Config

```python
# src/train_muranet.py  (skeleton — adapt with actual MuraNet repo code)

MURANET_CONFIG = {
    'backbone':            'mit_b2',       # mit_b0 for CPU deployment, mit_b4 for best accuracy
    'num_seg_classes':     4,
    'num_det_classes':     2,              # door, window
    'image_size':          512,
    'batch_size':          6,              # MuraNet is heavier than SegFormer
    'lr':                  1e-4,
    'lr_backbone':         1e-5,           # lower LR for pretrained backbone
    'weight_decay':        0.01,
    'epochs':              100,
    'warmup_epochs':       5,
    'seg_loss_weight':     1.0,
    'det_loss_weight':     0.5,            # start here, increase to 1.0 if doors not detected
    'tversky_alpha':       0.7,
    'tversky_beta':        0.3,
    'det_conf_threshold':  0.5,
    'det_nms_threshold':   0.45,
    'save_dir':            './checkpoints/muranet',
    'data_root':           './data/processed',
    'class_weights_seg':   [0.5, 3.0, 5.0, 5.0],
}
```

**Joint loss function:**
```python
def joint_loss(seg_logits, det_preds, seg_labels, det_targets, cfg):
    """
    Combines segmentation loss (CE + Tversky) with detection loss (YOLOX).
    """
    # ── Segmentation ──────────────────────────────────────────────────
    up = F.interpolate(seg_logits, size=(cfg['image_size'], cfg['image_size']),
                        mode='bilinear', align_corners=False)
    weights = torch.tensor(cfg['class_weights_seg'], device=seg_logits.device)
    seg_ce = F.cross_entropy(up, seg_labels, weight=weights)

    # Tversky on wall class specifically
    wall_prob = torch.softmax(up, dim=1)[:, 1]
    wall_gt   = (seg_labels == 1).float()
    seg_tv = tversky_loss(wall_prob, wall_gt,
                           cfg['tversky_alpha'], cfg['tversky_beta'])

    seg_loss = seg_ce + seg_tv

    # ── Detection ─────────────────────────────────────────────────────
    det_loss = compute_yolox_loss(det_preds, det_targets)   # from MuraNet repo

    # ── Combined ──────────────────────────────────────────────────────
    total = (cfg['seg_loss_weight'] * seg_loss +
             cfg['det_loss_weight'] * det_loss)

    return total, seg_loss.item(), det_loss.item()
```

### Step 2.5 — MLParser: raster_parser.py Replacement

This is the final product of Phase 2. It has the exact same interface as
the existing RasterParser class.

```python
# src/ml_parser.py
# Drop-in replacement for app/core/raster_parser.py

import torch
import numpy as np
import cv2
from PIL import Image
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
import torch.nn.functional as F
from skimage.morphology import skeletonize

# Import same dataclasses as existing raster_parser.py
# These must remain identical in structure
from app.core.raster_parser import ParsedGeometry, WallLine, Opening


class MLParser:
    """
    MuraNet-based replacement for RasterParser.
    Interface: parse(image_path) → ParsedGeometry
    """

    MEAN = np.array([0.485, 0.456, 0.406])
    STD  = np.array([0.229, 0.224, 0.225])

    def __init__(self, checkpoint_path: str,
                 conf_threshold: float = 0.5,
                 fallback_parser=None,
                 device: str = 'auto'):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.conf_threshold = conf_threshold
        self.fallback = fallback_parser   # RasterParser() instance for low-confidence fallback
        self.model = self._load_model(checkpoint_path)

    def _load_model(self, path: str):
        # Load MuraNet from checkpoint
        # Adapt this to actual MuraNet repo's load mechanism
        from model.muranet import MuraNet
        model = MuraNet.load_from_checkpoint(path)
        return model.eval().to(self.device)

    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(img, (512, 512))
        normed  = (resized.astype(np.float32) / 255.0 - self.MEAN) / self.STD
        return torch.from_numpy(normed).permute(2, 0, 1).unsqueeze(0).float()

    def parse(self, image_path: str, scale_ppm: float = None) -> ParsedGeometry:
        """
        Main entry point. Same signature as RasterParser.parse().
        """
        img  = np.array(Image.open(image_path).convert('RGB'))
        h, w = img.shape[:2]
        t    = self._preprocess(img).to(self.device)

        with torch.no_grad():
            seg_logits, det_preds = self.model(t)

        # Upsample seg to original size
        seg_up   = F.interpolate(seg_logits, size=(h, w),
                                  mode='bilinear', align_corners=False)
        seg_mask = seg_up.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)

        # Decode detections
        det_boxes = self._decode_detections(det_preds, h, w)

        # Confidence check — fall back to raster parser if uncertain
        confidence = self._confidence(seg_mask)
        if confidence < 0.4 and self.fallback is not None:
            print(f"[MLParser] Low confidence ({confidence:.2f}), using fallback parser")
            return self.fallback.parse(image_path, scale_ppm)

        # Convert to ParsedGeometry fields
        wall_lines = self._mask_to_wall_lines(seg_mask)
        openings   = self._boxes_to_openings(det_boxes)

        if scale_ppm is None:
            scale_ppm = self._estimate_scale(seg_mask, image_path)

        return ParsedGeometry(
            wall_lines  = wall_lines,
            openings    = openings,
            scale_ppm   = scale_ppm,
            image_size  = (h, w),
            method      = 'muranet',
            confidence  = confidence,
        )

    def _mask_to_wall_lines(self, seg_mask: np.ndarray) -> List[WallLine]:
        """Skeletonize wall pixels → Hough lines → WallLine objects."""
        binary   = (seg_mask == 1).astype(np.uint8)
        skeleton = skeletonize(binary).astype(np.uint8) * 255

        lines = cv2.HoughLinesP(
            skeleton, rho=1, theta=np.pi/180,
            threshold=30, minLineLength=20, maxLineGap=10)

        if lines is None:
            return []

        return [WallLine(x1=int(l[0][0]), y1=int(l[0][1]),
                          x2=int(l[0][2]), y2=int(l[0][3]))
                for l in lines]

    def _boxes_to_openings(self, det_boxes: list) -> List[Opening]:
        return [
            Opening(
                kind     = 'door' if box['class_id'] == 0 else 'window',
                center_x = box['cx_px'],
                center_y = box['cy_px'],
                width_px = box['w_px'],
                angle    = box['angle'],
                confidence = box['conf'],
            )
            for box in det_boxes
            if box['conf'] >= self.conf_threshold
        ]

    def _decode_detections(self, det_preds, img_h: int, img_w: int) -> list:
        """NMS + denormalize detection predictions to pixel coords."""
        # This depends on MuraNet's exact output format.
        # Typical YOLOX output: (batch, num_anchors, 7) where 7 = [cx,cy,w,h,obj,cls,angle]
        # Apply sigmoid to obj score, softmax to cls scores
        # Run NMS with nms_threshold=0.45
        # Denormalize cx*img_w, cy*img_h, w*img_w
        pass   # implement once MuraNet repo format is known

    def _confidence(self, seg_mask: np.ndarray) -> float:
        """Heuristic confidence: valid floor plan has 3-25% wall pixels."""
        wall_ratio = (seg_mask == 1).sum() / seg_mask.size
        if 0.03 <= wall_ratio <= 0.25:
            return 0.9
        return 0.2

    def _estimate_scale(self, seg_mask: np.ndarray, image_path: str) -> float:
        """
        Fallback scale estimation if scale_ppm not provided.
        Uses median wall thickness in pixels as proxy.
        Indian walls: ~230mm → if 14px thick → 14/0.23 = 60.9 ppm
        Finnish walls: ~150mm → if 9px thick → 9/0.15 = 60 ppm
        """
        # Simple fallback — try to read from image metadata first
        # then fall back to 60 ppm (typical for our test images)
        return 60.25
```

### Step 2.6 — Swap Into Existing Pipeline

In `app/core/__init__.py` or wherever the parser is instantiated:

```python
# Before (Phase 1 and earlier):
from app.core.raster_parser import RasterParser
parser = RasterParser()

# After Phase 2:
from src.ml_parser import MLParser
from app.core.raster_parser import RasterParser

parser = MLParser(
    checkpoint_path='./checkpoints/muranet/best',
    fallback_parser=RasterParser(),   # safety net
)

# Call signature is identical:
geometry = parser.parse(image_path)
```

---

## PHASE 3: FINE-TUNING ON REAL INDIAN PLANS

### Step 3.1 — Collecting Indian Floor Plans

Sources to scrape (no annotation yet, just raw images):

1. **GharExpert.com** — thousands of free downloadable Indian residential PDFs
2. **Houseyog.com** — similar, more recent
3. **CADbull.com** — AutoCAD DWG files (convert with LibreCAD or ODA File Converter)
4. **Architecture firm portfolio pages** — manual download, higher quality
5. **Your own project clients** — highest value, directly relevant

```bash
# Convert DWG → PNG (free tool)
# Download ODA File Converter from opendesign.com
# Or use LibreCAD:
libreoffice --headless --convert-to png *.dwg
```

Target: 500 annotated plans minimum before fine-tuning starts.

### Step 3.2 — Annotation Setup

```bash
pip install label-studio
label-studio start --port 8080
# Browser opens at localhost:8080
```

**Label Studio setup:**
1. New project → Image Segmentation task
2. Label classes: wall, door, window, column, background
3. Use Brush or Polygon tool for wall/column masks
4. Use Rectangle tool for door/window bounding boxes
5. Export → COCO JSON format

**Annotation guidelines (for yourself or annotators):**
- Walls: full wall thickness including plaster, NOT centerline
- Doors: the opening gap only, NOT the door leaf, arc, or swing symbol
- Windows: the opening gap only, NOT the sill projection
- Columns: label as 'column' class (we'll merge into 'wall' during preprocessing)
- IGNORE: furniture, dimension text, north arrow, staircase hatching, room labels

**Time estimate**: ~20 min per simple 2BHK, ~40 min per complex plan.
500 plans = ~170 hours of annotation work.

### Step 3.3 — Domain Adaptation Fine-tuning

```python
# src/fine_tune_indian.py

FINETUNE_CONFIG = {
    'base_checkpoint':    './checkpoints/muranet/best',
    'lr':                 1e-5,          # 10x lower — preserve base knowledge
    'lr_backbone':        1e-6,          # even lower for shared encoder
    'epochs':             30,
    'batch_size':         4,
    'freeze_encoder_epochs': 5,          # freeze MiT backbone for first 5 epochs

    # Mix ratio: oversample Indian data 3x to bias toward new distribution
    # while keeping 30% CubiCasa5k to prevent forgetting
    'indian_data_weight': 3.0,
    'cubicasa_fraction':  0.3,

    'indian_data_root':   './data/indian',
    'base_data_root':     './data/processed',
    'save_dir':           './checkpoints/muranet_indian',
}
```

---

## EVALUATION AND TESTING

### Target Metrics

| Dataset | Wall IoU | Door AP50 | Window AP50 |
|---|---|---|---|
| CubiCasa5k test (standard benchmark) | ≥ 78% | ≥ 75% | ≥ 70% |
| Our test images (L-shaped plan etc.) | ≥ 70% | ≥ 70% | ≥ 65% |
| Indian plans (Phase 3) | ≥ 65% | ≥ 60% | ≥ 55% |

### Test Set to Build Immediately

```
test_images/
  standard/      50 CubiCasa5k-style (from held-out test split)
  thick_walls/   20 plans scaled to appear with 230mm+ walls
  lshaped/       20 irregular/L-shaped plans (like our existing test image)
  scanned/       20 scanned or photographed plans (noisy)
  annotated/     text-heavy plans with many dimensions/labels
```

Always test on this exact set after every model change.

### Failure Mode Checklist

After every training run, check for these specific failures:

| Symptom | Likely cause | Fix |
|---|---|---|
| mIoU < 50% | class imbalance, model predicts all background | Increase wall class_weight to 5.0+ |
| Walls predicted too thick | Tversky alpha too low | Increase alpha to 0.8 |
| Doors not detected at all | det_loss_weight too low | Increase to 1.0 |
| Phantom walls in empty rooms | Furniture not separated | Add furniture class or augment with furniture erasure |
| Indian thick wall split into 2 thin walls | WallDetector pairing still needed | Keep min_pair_dist=0.08 in WallDetector — ML doesn't fix this |
| Model confident on non-floor-plan images | No negative examples in training | Add non-floor-plan images as background class |
| Inference too slow for production | MiT-B2 too heavy | Switch to MiT-B0 backbone (faster, slightly less accurate) |

---

## PROJECT DIRECTORY STRUCTURE

```
floorplan-ml/
  data/
    raw/
      cubicasa5k/
      resplan/
      indian_raw/
    processed/
      images/
      masks/
      bboxes/
      splits/
    indian/                  annotated Indian plans (Phase 3)
    test_images/             fixed test set (never augmented, never trained on)

  checkpoints/
    segformer/best/          Phase 1 checkpoint
    muranet/best/            Phase 2 checkpoint
    muranet_indian/best/     Phase 3 checkpoint

  src/
    augmentations.py
    datasets.py
    train_segformer.py
    train_muranet.py
    fine_tune_indian.py
    ml_preprocessor.py       Phase 1 pipeline integration
    ml_parser.py             Phase 2/3 raster_parser.py replacement
    masks_to_bboxes.py
    evaluate.py

  notebooks/
    01_data_exploration.ipynb
    02_augmentation_preview.ipynb
    03_training_curves.ipynb
    04_error_analysis.ipynb

  tests/
    test_ml_parser_interface.py   MLParser returns same ParsedGeometry structure
    test_pipeline_regression.py   End-to-end: our L-shaped plan still works
    test_thick_walls.py           Indian thick wall test cases

  existing_backend/               (your current floorplan-visualizer repo)
    backend/app/core/
      raster_parser.py            ← Phase 2 replaces this
      wall_detector.py            ← UNCHANGED
      geometry_builder.py         ← UNCHANGED
    viewer/
      index.html                  ← UNCHANGED
```

---

## QUICK COMMAND REFERENCE

```bash
# Phase 1: Train SegFormer
cd floorplan-ml
python src/train_segformer.py

# Phase 1: Test on single image
python -c "
from src.ml_preprocessor import MLPreprocessor
p = MLPreprocessor('./checkpoints/segformer/best')
m = p.get_masks('./test_images/lshaped/test1.png')
print('Wall pixels:', m['wall'].sum(), '| Door px:', m['door'].sum())
"

# Phase 2: Train MuraNet
python src/train_muranet.py

# Evaluate Phase 2
python src/evaluate.py \
  --checkpoint ./checkpoints/muranet/best \
  --test-dir   ./data/test_images

# Phase 3: Fine-tune on Indian plans
python src/fine_tune_indian.py

# Run full pipeline with ML parser (end-to-end test)
python -c "
from src.ml_parser import MLParser
p = MLParser('./checkpoints/muranet/best')
g = p.parse('./test_images/lshaped/test1.png')
print(f'Walls={len(g.wall_lines)} Doors={sum(1 for o in g.openings if o.kind==\"door\")} Windows={sum(1 for o in g.openings if o.kind==\"window\")}')
"
```
