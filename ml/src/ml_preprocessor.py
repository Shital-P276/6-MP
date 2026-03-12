# src/ml_preprocessor.py
#
# Phase 1 integration: SegFormer inference + raster_parser.py compatibility.
#
# This module does two things:
#
#   1. MLPreprocessor.get_masks(image_path)
#      → Runs SegFormer, returns separate binary masks for wall/door/window.
#      → Use this for evaluation or visualisation.
#
#   2. MLPreprocessor.make_clean_wall_image(image_path)
#      → Returns a clean synthetic floor plan image (black walls on white bg)
#         that can be passed directly to the existing raster_parser.py.
#      → Only 4 lines of code needed in raster_parser.py to activate this.
#
# ─────────────────────────────────────────────────────────────────────────────
# HOW TO ACTIVATE IN raster_parser.py
# ─────────────────────────────────────────────────────────────────────────────
#
#   In app/core/raster_parser.py, at the TOP of the parse() method, add:
#
#       USE_ML = True   # set False to revert to original behaviour
#
#       if USE_ML:
#           from src.ml_preprocessor import MLPreprocessor
#           _pre = MLPreprocessor('./checkpoints/segformer/best')
#           clean = _pre.make_clean_wall_image(image_path)
#           import tempfile, cv2, os
#           tmp = tempfile.mktemp(suffix='.png')
#           cv2.imwrite(tmp, clean)
#           image_path = tmp   # raster_parser processes the clean image
#           # ↑ Everything below this line runs completely unchanged.
#
#   That's it. The rest of raster_parser.py, WallDetector, GeometryBuilder
#   are completely untouched.
#
# ─────────────────────────────────────────────────────────────────────────────

import sys
import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation


class MLPreprocessor:
    """
    Runs SegFormer inference on a floor plan image and produces:
      - Binary masks for wall / door / window at original image resolution
      - A clean wall image compatible with the existing raster_parser.py

    Args:
        checkpoint_path:  Path to HuggingFace save_pretrained directory.
                          (from train_segformer.py → checkpoints/segformer/best)
        device:           'auto', 'cuda', 'cpu', or 'mps' (Apple Silicon)
        image_size:       Inference resolution. Must match training (512).
        confidence_threshold:
                          Minimum softmax confidence to accept a prediction.
                          Pixels below threshold → treated as background.
                          Set 0.0 to disable (use argmax only).

    Class constants:
        CLASS_BG   = 0  background
        CLASS_WALL = 1  wall (+ columns)
        CLASS_DOOR = 2  door opening
        CLASS_WIN  = 3  window opening
    """

    CLASS_BG   = 0
    CLASS_WALL = 1
    CLASS_DOOR = 2
    CLASS_WIN  = 3

    # ImageNet normalization (must match training)
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'auto',
        image_size: int = 512,
        confidence_threshold: float = 0.0,
    ):
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        self.device = torch.device(device)
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold

        print(f"[MLPreprocessor] Loading checkpoint: {checkpoint_path}")
        print(f"[MLPreprocessor] Device: {self.device}")

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            checkpoint_path
        ).eval().to(self.device)

        # Disable gradient computation globally for this model
        for param in self.model.parameters():
            param.requires_grad_(False)

    # ─────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────

    def _preprocess(self, image_rgb: np.ndarray) -> torch.Tensor:
        """
        Resize + normalize image for model input.

        Args:
            image_rgb: (H, W, 3) uint8 RGB array

        Returns:
            (1, 3, image_size, image_size) float32 tensor
        """
        resized = cv2.resize(
            image_rgb, (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        normed = (resized.astype(np.float32) / 255.0 - self.MEAN) / self.STD
        tensor = torch.from_numpy(normed).permute(2, 0, 1).unsqueeze(0).float()
        return tensor

    def _run_inference(
        self,
        image_rgb: np.ndarray,
    ) -> tuple:
        """
        Run model forward pass.

        Returns:
            pred_mask:  (H, W) uint8 — argmax class indices at original resolution
            confidence: (H, W) float32 — max softmax probability at each pixel
            orig_size:  (H, W) tuple
        """
        h, w = image_rgb.shape[:2]
        tensor = self._preprocess(image_rgb).to(self.device)

        with torch.no_grad():
            outputs = self.model(pixel_values=tensor)
            # SegFormer outputs logits at 1/4 resolution → upsample to original
            logits = F.interpolate(
                outputs.logits,
                size=(h, w),
                mode='bilinear',
                align_corners=False,
            )
            probs = torch.softmax(logits, dim=1).squeeze(0)   # (4, H, W)

        probs_np      = probs.cpu().numpy()
        confidence    = probs_np.max(axis=0).astype(np.float32)
        pred_mask     = probs_np.argmax(axis=0).astype(np.uint8)

        # Apply confidence threshold: low-confidence pixels → background
        if self.confidence_threshold > 0:
            pred_mask[confidence < self.confidence_threshold] = self.CLASS_BG

        return pred_mask, confidence, (h, w)

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def get_masks(self, image_path: str) -> dict:
        """
        Run inference on a floor plan image.

        Returns dict with keys:
            'wall'        (H, W) uint8 0/255 binary mask — wall pixels
            'door'        (H, W) uint8 0/255 binary mask — door opening pixels
            'window'      (H, W) uint8 0/255 binary mask — window opening pixels
            'full_pred'   (H, W) uint8 0-3  full class prediction
            'confidence'  (H, W) float32    per-pixel max softmax probability
            'original_size' (H, W) tuple
        """
        image_rgb = np.array(Image.open(image_path).convert('RGB'))
        pred, confidence, orig_size = self._run_inference(image_rgb)

        return {
            'wall':          (pred == self.CLASS_WALL).astype(np.uint8) * 255,
            'door':          (pred == self.CLASS_DOOR).astype(np.uint8) * 255,
            'window':        (pred == self.CLASS_WIN ).astype(np.uint8) * 255,
            'full_pred':     pred,
            'confidence':    confidence,
            'original_size': orig_size,
        }

    def make_clean_wall_image(self, image_path: str) -> np.ndarray:
        """
        Returns a clean BGR image suitable for feeding into raster_parser.py.

        Pixel encoding (chosen to match raster_parser.py's color expectations):
            white (255,255,255)  = background  → raster_parser ignores this
            black (  0,  0,  0)  = wall        → Hough detection finds these
            gray  (128,128,128)  = door opening → brightness scan finds gaps
            gray  (200,200,200)  = window       → brightness scan finds gaps

        The clean image looks like a simplified CAD floor plan with:
        - No furniture
        - No room labels
        - No dimension text
        - No hatching
        - Clean sharp wall boundaries

        This is what raster_parser was designed for — we're just generating it
        from the raw messy image automatically instead of requiring a clean scan.
        """
        masks = self.get_masks(image_path)
        h, w  = masks['original_size']

        # Start with white background (raster_parser background)
        clean = np.full((h, w, 3), 255, dtype=np.uint8)

        # Draw walls as black (raster_parser looks for dark pixels)
        clean[masks['wall']   > 0] = [0,   0,   0  ]

        # Draw door openings as mid-gray (raster_parser brightness scan gap)
        clean[masks['door']   > 0] = [128, 128, 128]

        # Draw window openings as light gray (slightly different from doors)
        clean[masks['window'] > 0] = [200, 200, 200]

        return clean   # BGR format (OpenCV convention)

    def get_confidence_stats(self, image_path: str) -> dict:
        """
        Returns confidence statistics for the prediction.
        Used to decide whether to fall back to raster_parser (no ML).

        A wall_ratio < 3% or > 25% usually means something went wrong:
        - < 3%: model probably predicted all background (no walls found)
        - > 25%: model probably over-predicted walls (image not a floor plan)

        In both cases, fall back to raster_parser for safety.
        """
        masks = self.get_masks(image_path)
        h, w  = masks['original_size']
        total = h * w

        wall_px   = (masks['wall']   > 0).sum()
        door_px   = (masks['door']   > 0).sum()
        window_px = (masks['window'] > 0).sum()

        return {
            'wall_ratio':       wall_px   / total,
            'door_ratio':       door_px   / total,
            'window_ratio':     window_px / total,
            'mean_confidence':  float(masks['confidence'].mean()),
            'wall_px':          int(wall_px),
            'door_px':          int(door_px),
            'window_px':        int(window_px),
            'looks_valid':      0.03 <= (wall_px / total) <= 0.25,
        }


class MLPreprocessorWithFallback:
    """
    Drop-in wrapper that uses MLPreprocessor if the prediction looks valid,
    and falls back to the original raster_parser if not.

    Use this in production where robustness > accuracy.

    Args:
        checkpoint_path:  Path to SegFormer checkpoint.
        fallback_fn:      Callable that takes image_path → raster_parser result.
                          Typically: lambda p: raster_parser.parse(p)
        wall_ratio_min:   Minimum wall pixel ratio to trust ML prediction.
        wall_ratio_max:   Maximum wall pixel ratio to trust ML prediction.

    Example:
        from app.core.raster_parser import RasterParser
        rp = RasterParser()

        ml = MLPreprocessorWithFallback(
            checkpoint_path='./checkpoints/segformer/best',
            fallback_fn=lambda p: rp.parse(p),
        )

        # Returns clean image path if ML confident, original path if not
        path_to_parse = ml.get_clean_image_path(original_path)
        geometry = rp.parse(path_to_parse)
    """

    def __init__(
        self,
        checkpoint_path: str,
        fallback_fn=None,
        wall_ratio_min: float = 0.03,
        wall_ratio_max: float = 0.25,
    ):
        self.preprocessor    = MLPreprocessor(checkpoint_path)
        self.fallback_fn     = fallback_fn
        self.wall_ratio_min  = wall_ratio_min
        self.wall_ratio_max  = wall_ratio_max
        self._tmp_files      = []

    def get_clean_image_path(self, image_path: str) -> tuple:
        """
        Returns (path_to_use, used_ml: bool).

        If ML prediction looks valid, writes clean image to a temp file
        and returns that path. Otherwise returns the original path.
        """
        try:
            stats = self.preprocessor.get_confidence_stats(image_path)
            if stats['looks_valid']:
                clean = self.preprocessor.make_clean_wall_image(image_path)
                tmp = tempfile.mktemp(suffix='_ml_clean.png')
                cv2.imwrite(tmp, clean)
                self._tmp_files.append(tmp)
                return tmp, True
            else:
                print(f"[MLPreprocessor] Falling back: wall_ratio={stats['wall_ratio']:.3f}")
                return image_path, False
        except Exception as e:
            print(f"[MLPreprocessor] Error, falling back: {e}")
            return image_path, False

    def cleanup(self):
        """Delete temporary clean image files."""
        import os
        for f in self._tmp_files:
            try:
                os.unlink(f)
            except Exception:
                pass
        self._tmp_files.clear()


# ─────────────────────────────────────────────────────────────────────────────
# CLI — test on a single image
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run SegFormer Phase 1 inference on a floor plan image'
    )
    parser.add_argument('--image',      required=True,
                        help='Path to input floor plan image')
    parser.add_argument('--checkpoint', default='./checkpoints/segformer/best',
                        help='Path to SegFormer checkpoint')
    parser.add_argument('--out-dir',    default='./outputs',
                        help='Directory to save output images')
    parser.add_argument('--show',       action='store_true',
                        help='Display results with matplotlib')
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    pre = MLPreprocessor(args.checkpoint)

    print(f"\nRunning inference on: {args.image}")
    masks = pre.get_masks(args.image)
    stats = pre.get_confidence_stats(args.image)

    print(f"\nResults:")
    print(f"  Wall pixels:    {stats['wall_px']:,}  ({stats['wall_ratio']*100:.1f}%)")
    print(f"  Door pixels:    {stats['door_px']:,}  ({stats['door_ratio']*100:.1f}%)")
    print(f"  Window pixels:  {stats['window_px']:,}  ({stats['window_ratio']*100:.1f}%)")
    print(f"  Mean confidence: {stats['mean_confidence']:.3f}")
    print(f"  Prediction valid: {'✓ YES' if stats['looks_valid'] else '✗ NO (will fallback)'}")

    # Save outputs
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    orig     = np.array(Image.open(args.image).convert('RGB'))
    clean    = pre.make_clean_wall_image(args.image)
    out_path = out_dir / (Path(args.image).stem + '_clean.png')
    cv2.imwrite(str(out_path), clean)
    print(f"\nClean image saved → {out_path}")

    if args.show:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(orig)
        axes[0].set_title('Original')
        axes[0].axis('off')

        # Colorised prediction
        pred_rgb = np.zeros((*masks['full_pred'].shape, 3), dtype=np.uint8)
        pred_rgb[masks['full_pred'] == 0] = [240, 240, 240]  # bg = light gray
        pred_rgb[masks['full_pred'] == 1] = [30,  30,  30 ]  # wall = dark
        pred_rgb[masks['full_pred'] == 2] = [50,  100, 200]  # door = blue
        pred_rgb[masks['full_pred'] == 3] = [50,  180, 80 ]  # window = green
        axes[1].imshow(pred_rgb)
        axes[1].set_title('Prediction (wall=dark, door=blue, window=green)')
        axes[1].axis('off')

        axes[2].imshow(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB))
        axes[2].set_title('Clean image (raster_parser input)')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()
