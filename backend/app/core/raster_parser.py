"""
Raster Parser — extracts wall geometry from PNG/JPG/PDF floor plan images.

Pipeline:
  1. Load image (or convert PDF page to image)
  2. Preprocess: grayscale → denoise → binarize → morphological cleanup
  3. Edge detection (Canny)
  4. Probabilistic Hough Transform → line segments
  5. Convert pixel lines → metric segments

Requirements: opencv-python, numpy
Optional: pdf2image + poppler (for PDF support)
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import math

from .dxf_parser import Segment, Point2D, ParsedGeometry

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import pdf2image
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Defaults (all tunable via constructor)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PIXELS_PER_METER = 100   # rough default; user should calibrate
CANNY_LOW    = 50
CANNY_HIGH   = 150
HOUGH_RHO    = 1
HOUGH_THETA  = math.pi / 180
MIN_PX_LEN   = 20               # ignore segments shorter than this (pixels)
MIN_WALL_LINE_THICKNESS_PX = 1.6  # reject very thin linework (dimensions/annotations)


# ─────────────────────────────────────────────────────────────────────────────
# Image loading
# ─────────────────────────────────────────────────────────────────────────────

def _require_cv2():
    if not CV2_AVAILABLE:
        raise ImportError("opencv-python is required for image parsing. Run: pip install opencv-python")

def _load_cv2(filepath: str) -> "np.ndarray":
    _require_cv2()
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError(f"Could not read image file: {filepath}")
    return img

def _pdf_to_cv2(filepath: str, dpi: int = 200) -> "np.ndarray":
    if not PDF2IMAGE_AVAILABLE:
        raise ImportError(
            "pdf2image is required for PDF support.\n"
            "Install: pip install pdf2image\n"
            "Also install Poppler: https://github.com/oschwartz10612/poppler-windows/releases (Windows)"
        )
    _require_cv2()
    pages = pdf2image.convert_from_path(filepath, dpi=dpi, first_page=1, last_page=1)
    if not pages:
        raise ValueError("PDF is empty or unreadable")
    arr = np.array(pages[0])
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess(img: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
    """
    Build two masks:
      1) binary_lines: high-contrast linework for Hough
      2) wall_mask: thicker dark regions likely to be structural walls
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # General linework extraction
    _, binary_lines = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary_lines = cv2.morphologyEx(
        binary_lines, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )

    # Dark-region extraction: tends to keep walls and reject light/colored dimensions
    dark_thresh = int(min(140, max(80, np.mean(enhanced) - 25)))
    _, wall_mask = cv2.threshold(enhanced, dark_thresh, 255, cv2.THRESH_BINARY_INV)
    wall_mask = cv2.morphologyEx(
        wall_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )
    wall_mask = cv2.morphologyEx(
        wall_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    )

    # Remove tiny connected components (dimension text / ticks / noise)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(wall_mask, connectivity=8)
    cleaned = np.zeros_like(wall_mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 80:
            cleaned[labels == i] = 255

    return binary_lines, cleaned


# ─────────────────────────────────────────────────────────────────────────────
# Line detection
# ─────────────────────────────────────────────────────────────────────────────

def _hough_lines(
    binary: "np.ndarray",
    threshold: int,
    min_length: int,
    max_gap: int,
) -> list[tuple[int, int, int, int]]:
    edges = cv2.Canny(binary, CANNY_LOW, CANNY_HIGH)
    raw = cv2.HoughLinesP(
        edges,
        rho=HOUGH_RHO,
        theta=HOUGH_THETA,
        threshold=threshold,
        minLineLength=min_length,
        maxLineGap=max_gap,
    )
    if raw is None:
        return []
    return [(int(l[0][0]), int(l[0][1]), int(l[0][2]), int(l[0][3])) for l in raw]


def _filter_lines_with_wall_mask(
    lines: list[tuple[int, int, int, int]],
    wall_mask: "np.ndarray",
) -> tuple[list[tuple[int, int, int, int]], int]:
    """
    Keep lines that overlap dark/thick wall regions and are not extremely thin.
    Returns (kept_lines, filtered_out_count).
    """
    kept = []
    filtered_out = 0

    # Distance transform approximates local half-thickness in wall regions.
    dist = cv2.distanceTransform((wall_mask > 0).astype(np.uint8), cv2.DIST_L2, 3)

    for x1, y1, x2, y2 in lines:
        line_img = np.zeros_like(wall_mask)
        cv2.line(line_img, (x1, y1), (x2, y2), 255, 2)
        line_px = cv2.countNonZero(line_img)
        if line_px == 0:
            filtered_out += 1
            continue

        overlap_px = cv2.countNonZero(cv2.bitwise_and(line_img, wall_mask))
        overlap_ratio = overlap_px / line_px

        ys, xs = np.where(line_img > 0)
        thickness_est = float(np.median(dist[ys, xs]) * 2.0) if len(xs) else 0.0

        if overlap_ratio >= 0.35 and thickness_est >= MIN_WALL_LINE_THICKNESS_PX:
            kept.append((x1, y1, x2, y2))
        else:
            filtered_out += 1

    return kept, filtered_out


def _px_to_segments(
    lines: list[tuple[int, int, int, int]],
    ppm: float,
    img_h: int,
) -> list[Segment]:
    segs = []
    for x1, y1, x2, y2 in lines:
        if math.hypot(x2 - x1, y2 - y1) < MIN_PX_LEN:
            continue
        # Flip Y: image Y=0 is top, we want Y=0 at bottom
        segs.append(Segment(
            start=Point2D(x1 / ppm, (img_h - y1) / ppm),
            end=Point2D(x2 / ppm,   (img_h - y2) / ppm),
            layer="WALL",
            source_type="HOUGH",
        ))
    return segs


# ─────────────────────────────────────────────────────────────────────────────
# Main parser class
# ─────────────────────────────────────────────────────────────────────────────

class RasterParser:
    def __init__(
        self,
        pixels_per_meter: float = DEFAULT_PIXELS_PER_METER,
        pdf_dpi: int = 200,
        hough_threshold: int = 50,
        hough_min_length: int = 30,
        hough_max_gap: int = 15,
    ):
        self.pixels_per_meter = pixels_per_meter
        self.pdf_dpi = pdf_dpi
        self.hough_threshold = hough_threshold
        self.hough_min_length = hough_min_length
        self.hough_max_gap = hough_max_gap

    def parse(self, filepath: str) -> ParsedGeometry:
        suffix = Path(filepath).suffix.lower()

        if suffix == ".pdf":
            img = _pdf_to_cv2(filepath, dpi=self.pdf_dpi)
        elif suffix in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            img = _load_cv2(filepath)
        else:
            raise ValueError(f"Unsupported raster format: {suffix}")

        h, w = img.shape[:2]
        binary_lines, wall_mask = _preprocess(img)
        raw_lines = _hough_lines(binary_lines, self.hough_threshold, self.hough_min_length, self.hough_max_gap)
        lines, filtered_out = _filter_lines_with_wall_mask(raw_lines, wall_mask)
        segments = _px_to_segments(lines, self.pixels_per_meter, h)

        result = ParsedGeometry()
        result.wall_segments = segments
        result.units = "meters (pixel-estimated)"

        all_x = [s.start.x for s in segments] + [s.end.x for s in segments]
        all_y = [s.start.y for s in segments] + [s.end.y for s in segments]
        if all_x:
            result.bounds = {
                "minx": min(all_x), "miny": min(all_y),
                "maxx": max(all_x), "maxy": max(all_y),
            }

        result.metadata_extra = {
            "source": "raster",
            "image_size": f"{w}x{h}px",
            "lines_detected": len(lines),
            "raw_lines_detected": len(raw_lines),
            "filtered_non_wall_lines": filtered_out,
            "pixels_per_meter": self.pixels_per_meter,
        }

        return result

    def save_debug_image(self, filepath: str, output_path: str = "debug_lines.png") -> str:
        """Render detected lines onto original image and save. Useful for tuning."""
        suffix = Path(filepath).suffix.lower()
        img = _pdf_to_cv2(filepath, self.pdf_dpi) if suffix == ".pdf" else _load_cv2(filepath)
        binary_lines, wall_mask = _preprocess(img)
        raw_lines = _hough_lines(binary_lines, self.hough_threshold, self.hough_min_length, self.hough_max_gap)
        lines, _ = _filter_lines_with_wall_mask(raw_lines, wall_mask)
        vis = img.copy()
        for x1, y1, x2, y2 in lines:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 200, 50), 2)
        cv2.imwrite(output_path, vis)
        return output_path
