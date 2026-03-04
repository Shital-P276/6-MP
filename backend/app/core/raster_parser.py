"""
Raster Parser — extracts wall geometry from PNG/JPG/PDF floor plan images.

Pipeline:
  1. Load image (or convert PDF page to image)
  2. Grayscale conversion
  3. Gaussian blur denoising
  4. Adaptive thresholding
  5. Canny edge detection
  6. Edge-map cleanup
  7. Probabilistic Hough Transform → line segments
  8. Convert pixel lines → metric segments

Requirements: opencv-python, numpy
Optional: pdf2image + poppler (for PDF support)
"""

from __future__ import annotations
from pathlib import Path
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

DEFAULT_PIXELS_PER_METER = 100
CANNY_LOW = 50
CANNY_HIGH = 150
HOUGH_RHO = 1
HOUGH_THETA = math.pi / 180
MIN_PX_LEN = 20

# Preprocess constants for clean structural edges
GAUSSIAN_KERNEL = (5, 5)
ADAPTIVE_BLOCK_SIZE = 31
ADAPTIVE_C = 5
EDGE_CLEAN_KERNEL = (3, 3)


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

def _preprocess_to_edges(img: "np.ndarray") -> "np.ndarray":
    """
    Build a clean edge map suitable for structural line detection.

    Steps:
      grayscale -> gaussian blur -> adaptive threshold -> canny -> edge cleanup
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    blurred = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL, 0)

    # Adaptive threshold handles uneven lighting/background in scans.
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        ADAPTIVE_BLOCK_SIZE,
        ADAPTIVE_C,
    )

    edges = cv2.Canny(binary, CANNY_LOW, CANNY_HIGH)

    # Light morphological close to connect short broken edge fragments.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, EDGE_CLEAN_KERNEL)
    clean_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    return clean_edges


# ─────────────────────────────────────────────────────────────────────────────
# Line detection
# ─────────────────────────────────────────────────────────────────────────────

def _hough_lines(
    edge_map: "np.ndarray",
    threshold: int,
    min_length: int,
    max_gap: int,
) -> list[tuple[int, int, int, int]]:
    raw = cv2.HoughLinesP(
        edge_map,
        rho=HOUGH_RHO,
        theta=HOUGH_THETA,
        threshold=threshold,
        minLineLength=min_length,
        maxLineGap=max_gap,
    )
    if raw is None:
        return []
    return [(int(l[0][0]), int(l[0][1]), int(l[0][2]), int(l[0][3])) for l in raw]


def _px_to_segments(
    lines: list[tuple[int, int, int, int]],
    ppm: float,
    img_h: int,
) -> list[Segment]:
    segs = []
    for x1, y1, x2, y2 in lines:
        if math.hypot(x2 - x1, y2 - y1) < MIN_PX_LEN:
            continue
        segs.append(Segment(
            start=Point2D(x1 / ppm, (img_h - y1) / ppm),
            end=Point2D(x2 / ppm, (img_h - y2) / ppm),
            layer="WALL",
            source_type="HOUGH",
        ))
    return segs


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

    def _load_image(self, filepath: str) -> "np.ndarray":
        suffix = Path(filepath).suffix.lower()
        if suffix == ".pdf":
            return _pdf_to_cv2(filepath, dpi=self.pdf_dpi)
        if suffix in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            return _load_cv2(filepath)
        raise ValueError(f"Unsupported raster format: {suffix}")

    def parse(self, filepath: str) -> ParsedGeometry:
        img = self._load_image(filepath)
        h, w = img.shape[:2]

        edge_map = _preprocess_to_edges(img)
        lines = _hough_lines(edge_map, self.hough_threshold, self.hough_min_length, self.hough_max_gap)
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
            "pixels_per_meter": self.pixels_per_meter,
            "edge_pixels": int((edge_map > 0).sum()),
        }

        return result

    def save_debug_image(self, filepath: str, output_path: str = "debug_lines.png") -> str:
        """Render detected lines on top of an edge-map backdrop for tuning."""
        img = self._load_image(filepath)
        edge_map = _preprocess_to_edges(img)
        lines = _hough_lines(edge_map, self.hough_threshold, self.hough_min_length, self.hough_max_gap)

        vis = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR)
        for x1, y1, x2, y2 in lines:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 200, 50), 2)

        cv2.imwrite(output_path, vis)
        return output_path
