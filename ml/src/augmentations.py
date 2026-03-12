# src/augmentations.py
#
# Augmentation pipeline for floor plan segmentation training.
#
# Key design principle:
#   - Geometric transforms (flip, rotate, scale, crop) are applied to BOTH
#     image and mask so they stay aligned.
#   - Photometric transforms (brightness, noise, blur) are applied to image ONLY.
#   - Indian-specific transforms simulate annotation noise that appears on real
#     Indian floor plans but NOT in CubiCasa5k training data.
#
# Usage:
#   from augmentations import get_train_transforms, get_val_transforms
#   transform = get_train_transforms(image_size=512)
#   aug = transform(image=img_np, mask=mask_np)
#   image, mask = aug['image'], aug['mask']

import random
import cv2
import numpy as np
import albumentations as A


# ─────────────────────────────────────────────────────────────────────────────
# Custom transforms — Indian floor plan specifics
# ─────────────────────────────────────────────────────────────────────────────

class AddRandomTextOverlay(A.ImageOnlyTransform):
    """
    Simulate Hindi/English dimension annotations and room labels.
    Draws random text strings at random positions on the image.

    Mask is NOT modified — the model must learn to ignore annotation text
    and still correctly classify wall/door/window pixels beneath it.

    Typical text on Indian plans:
      - Dimension strings: "12'-6\"", "3650mm", "W=900"
      - Room labels: "BR 1", "TOILET", "POOJA ROOM", "KITCHEN"
      - Structural notes: "RCC COLUMN", "LOAD BEARING"
    """

    def __init__(self, num_texts=(3, 12), p=0.4):
        super().__init__(p=p)
        self.num_texts = num_texts

    def apply(self, img, **params):
        result = img.copy()
        h, w = img.shape[:2]

        text_pool = [
            # Dimension strings
            f"{random.randint(1,30)}'-{random.randint(0,11)}\"",
            f"{random.randint(100, 6000)}mm",
            f"W={random.randint(600, 1500)}",
            f"D={random.randint(750, 1200)}",
            f"{random.randint(1,9)}.{random.randint(0,9)}m",
            # Room labels
            "BR 1", "BR 2", "BR 3",
            "TOILET", "BATHROOM", "WC",
            "KITCHEN", "HALL", "LIVING",
            "POOJA", "STORE", "BALCONY",
            "DINING", "PASSAGE", "LOBBY",
            # Structural notes
            "RCC COL", "TYP.", "ALL DIMS IN MM",
            f"AREA={random.randint(8,40)} SQ.M",
        ]

        n = random.randint(*self.num_texts)
        for _ in range(n):
            text = random.choice(text_pool)
            x = random.randint(10, max(10, w - 120))
            y = random.randint(15, max(15, h - 10))
            scale = random.uniform(0.25, 0.75)
            thickness = random.randint(1, 2)
            # Colors: black (most common), dark blue (CAD), dark red (markup)
            color = random.choice([
                (0, 0, 0),
                (50, 50, 180),
                (0, 0, 140),
                (120, 0, 0),
            ])
            font = random.choice([
                cv2.FONT_HERSHEY_SIMPLEX,
                cv2.FONT_HERSHEY_PLAIN,
                cv2.FONT_HERSHEY_DUPLEX,
            ])
            cv2.putText(result, text, (x, y), font, scale, color, thickness,
                        cv2.LINE_AA)
        return result

    def get_transform_init_args_names(self):
        return ('num_texts',)


class AddColumnMarkers(A.DualTransform):
    """
    Simulate Indian RC frame column squares at wall junctions.

    In Indian masonry/RC frame construction, columns appear as solid black
    squares where walls intersect. CubiCasa5k (Finnish plans) has almost none.

    Both image AND mask are modified:
      - Image: black filled rectangle drawn
      - Mask:  same rectangle filled with class 1 (wall)

    This trains the model to treat column squares as part of the wall class,
    matching our annotation guideline: "columns → merge into wall class".
    """

    def __init__(self, num_columns=(4, 12), col_size_px=(8, 22), p=0.35):
        super().__init__(p=p)
        self.num_columns = num_columns
        self.col_size_px = col_size_px

    @property
    def targets_as_params(self):
        return ['image']

    def get_params_dependent_on_targets(self, params):
        h, w = params['image'].shape[:2]
        n = random.randint(*self.num_columns)
        columns = []
        for _ in range(n):
            size = random.randint(*self.col_size_px)
            x = random.randint(0, max(0, w - size - 1))
            y = random.randint(0, max(0, h - size - 1))
            columns.append((x, y, size))
        return {'columns': columns}

    def apply(self, img, columns=None, **params):
        result = img.copy()
        for (x, y, size) in (columns or []):
            # Solid black square (column cross-section)
            cv2.rectangle(result, (x, y), (x + size, y + size), (0, 0, 0), -1)
            # Thin outline to match typical CAD rendering
            cv2.rectangle(result, (x, y), (x + size, y + size), (40, 40, 40), 1)
        return result

    def apply_to_mask(self, mask, columns=None, **params):
        result = mask.copy()
        for (x, y, size) in (columns or []):
            result[y:y + size, x:x + size] = 1  # wall class
        return result

    def get_transform_init_args_names(self):
        return ('num_columns', 'col_size_px')


class AddDimensionLines(A.ImageOnlyTransform):
    """
    Simulate dimension witness lines with tick marks at plan edges.

    Indian plans typically have dimension chains around the perimeter:
    horizontal and vertical lines with tick marks at each measurement point.

    Mask is NOT modified — these are annotation artifacts, not building elements.
    """

    def __init__(self, num_lines=(2, 7), p=0.35):
        super().__init__(p=p)
        self.num_lines = num_lines

    def apply(self, img, **params):
        result = img.copy()
        h, w = img.shape[:2]
        color = (0, 0, 0)
        thickness = 1

        for _ in range(random.randint(*self.num_lines)):
            if random.random() > 0.5:
                # Horizontal dimension line
                y = random.randint(5, h - 5)
                x1, x2 = sorted(random.sample(range(w), 2))
                cv2.line(result, (x1, y), (x2, y), color, thickness)
                # Tick marks at each end
                cv2.line(result, (x1, y - 5), (x1, y + 5), color, thickness)
                cv2.line(result, (x2, y - 5), (x2, y + 5), color, thickness)
                # Occasional intermediate ticks
                if random.random() > 0.5:
                    xm = (x1 + x2) // 2
                    cv2.line(result, (xm, y - 4), (xm, y + 4), color, thickness)
            else:
                # Vertical dimension line
                x = random.randint(5, w - 5)
                y1, y2 = sorted(random.sample(range(h), 2))
                cv2.line(result, (x, y1), (x, y2), color, thickness)
                cv2.line(result, (x - 5, y1), (x + 5, y1), color, thickness)
                cv2.line(result, (x - 5, y2), (x + 5, y2), color, thickness)
        return result

    def get_transform_init_args_names(self):
        return ('num_lines',)


class SimulateThickWalls(A.DualTransform):
    """
    Simulate Indian masonry wall thickness by scaling the image DOWN then padding.

    Indian load-bearing walls are ~230mm thick vs Finnish ~150mm.
    At the same pixels-per-meter, Indian walls appear ~53% thicker in pixels.

    Effect: scaling to 0.65x makes walls appear thicker relative to the image,
    simulating a higher-density plan (more px/meter used to render Indian walls).

    Both image and mask are transformed identically.
    """

    def __init__(self, scale_range=(0.60, 0.80), p=0.3):
        super().__init__(p=p)
        self.scale_range = scale_range

    @property
    def targets_as_params(self):
        return ['image']

    def get_params_dependent_on_targets(self, params):
        return {'scale': random.uniform(*self.scale_range)}

    def apply(self, img, scale=0.7, **params):
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # Pad back to original size (white background = 255)
        padded = np.full((h, w, 3), 255, dtype=np.uint8)
        padded[:new_h, :new_w] = small
        return padded

    def apply_to_mask(self, mask, scale=0.7, **params):
        h, w = mask.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        small = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        padded = np.zeros((h, w), dtype=np.uint8)
        padded[:new_h, :new_w] = small
        return padded

    def get_transform_init_args_names(self):
        return ('scale_range',)


# ─────────────────────────────────────────────────────────────────────────────
# Composed pipelines
# ─────────────────────────────────────────────────────────────────────────────

def get_train_transforms(image_size: int = 512) -> A.Compose:
    """
    Full training augmentation pipeline.
    Applied to BOTH image and mask unless noted as image-only.

    Order matters:
      1. Resize first (fixed size for batching)
      2. Geometric transforms (scale, flip, rotate, crop)
      3. Photometric transforms (brightness, color, noise)
      4. Indian-specific noise (text, columns, dimension lines)
      5. Normalize last (ImageNet mean/std)
    """
    return A.Compose([
        # ── 1. RESIZE ─────────────────────────────────────────────────────
        A.Resize(image_size, image_size),

        # ── 2. GEOMETRIC (image + mask) ───────────────────────────────────
        # RandomScale: simulates different wall thicknesses across plans
        #   scale 0.6x → walls appear ~67% thicker (Indian masonry effect)
        #   scale 1.4x → walls appear thinner (large building / small detail)
        A.RandomScale(scale_limit=(-0.4, 0.4), p=0.7),
        A.PadIfNeeded(
            image_size, image_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=255,       # white background for image
            mask_value=0,    # background class for mask
            p=1.0,
        ),
        A.RandomCrop(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        # Small rotations only — floor plans are mostly axis-aligned
        A.Rotate(
            limit=5,
            border_mode=cv2.BORDER_CONSTANT,
            value=255,
            mask_value=0,
            p=0.4,
        ),

        # ── 3. PHOTOMETRIC (image only) ───────────────────────────────────
        A.RandomBrightnessContrast(
            brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.RandomGamma(gamma_limit=(70, 130), p=0.4),
        A.ColorJitter(
            hue=0.05, saturation=0.3, brightness=0.2, contrast=0.2, p=0.4),
        # Occasional grayscale (simulates B&W scans / photocopies)
        A.ToGray(p=0.2),
        A.HueSaturationValue(
            hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=0.3),

        # ── 4. NOISE / BLUR ───────────────────────────────────────────────
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50)),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1)),
        ], p=0.4),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=3),
        ], p=0.3),

        # ── 5. INDIAN PLAN SPECIFICS ──────────────────────────────────────
        # These simulate visual noise present in Indian plans but NOT in
        # CubiCasa5k (Finnish). Crucial for generalization.
        AddRandomTextOverlay(num_texts=(3, 12), p=0.4),
        AddDimensionLines(num_lines=(2, 7), p=0.35),
        AddColumnMarkers(num_columns=(4, 12), col_size_px=(8, 22), p=0.35),
        SimulateThickWalls(scale_range=(0.60, 0.80), p=0.25),

        # ── 6. NORMALIZE ──────────────────────────────────────────────────
        # ImageNet mean/std — required because SegFormer encoder is pretrained
        # on ImageNet. Do NOT change these values.
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_val_transforms(image_size: int = 512) -> A.Compose:
    """
    Validation/test transform — resize and normalize ONLY.
    No augmentation on val/test sets (ever).
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Quick visual check — run this file directly to preview augmentations
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    import matplotlib.pyplot as plt

    # Usage: python augmentations.py path/to/image.png path/to/mask.png
    if len(sys.argv) < 3:
        print("Usage: python augmentations.py <image_path> <mask_path>")
        print("Generates a grid showing 8 augmented versions of the input.")
        sys.exit(1)

    from PIL import Image

    img  = np.array(Image.open(sys.argv[1]).convert('RGB'))
    mask = np.array(Image.open(sys.argv[2]))
    tf   = get_train_transforms(512)

    fig, axes = plt.subplots(2, 8, figsize=(24, 6))
    for i in range(8):
        aug = tf(image=img, mask=mask)
        # Denormalize for display
        disp = aug['image'] * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        disp = np.clip(disp, 0, 1)
        axes[0, i].imshow(disp)
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Aug {i+1}')
        # Mask: 0=bg(white), 1=wall(black), 2=door(blue), 3=window(green)
        mask_rgb = np.zeros((*aug['mask'].shape, 3), dtype=np.uint8)
        mask_rgb[aug['mask'] == 0] = [240, 240, 240]
        mask_rgb[aug['mask'] == 1] = [30,  30,  30 ]
        mask_rgb[aug['mask'] == 2] = [50,  100, 200]
        mask_rgb[aug['mask'] == 3] = [50,  180, 80 ]
        axes[1, i].imshow(mask_rgb)
        axes[1, i].axis('off')

    axes[0, 0].set_title('Images', loc='left', fontsize=10)
    axes[1, 0].set_title('Masks', loc='left', fontsize=10)
    plt.tight_layout()
    plt.savefig('augmentation_preview.png', dpi=120)
    print("Saved augmentation_preview.png")
    plt.show()
