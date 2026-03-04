"""Dimension detection using OCR + dimension-line extraction.

Steps:
1) OCR text in floorplan
2) parse candidate numeric dimension labels
3) detect nearby dimension lines
4) compute scale = real_length / pixel_length
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math
import re

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


NUM_RE = re.compile(r"(?<!\d)(\d+(?:[\.,]\d+)?)(?:\s*(m|mm|cm))?(?!\d)", re.IGNORECASE)


@dataclass
class DimensionEvidence:
    text: str
    real_length_m: float
    pixel_length: float
    scale: float
    bbox: tuple[int, int, int, int]

    def to_dict(self) -> dict:
        x, y, w, h = self.bbox
        return {
            "text": self.text,
            "real_length_m": round(self.real_length_m, 6),
            "pixel_length": round(self.pixel_length, 3),
            "scale": round(self.scale, 8),
            "bbox": {"x": x, "y": y, "w": w, "h": h},
        }


class DimensionDetector:
    def __init__(self):
        self.last_evidence: list[DimensionEvidence] = []

    def detect_scale(self, filepath: str) -> tuple[float | None, dict]:
        self.last_evidence = []
        if not CV2_AVAILABLE:
            return None, {"method": "ocr", "status": "opencv-missing", "evidence": []}

        path = Path(filepath)
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            return None, {"method": "ocr", "status": "unsupported-format", "evidence": []}

        img = cv2.imread(str(path))
        if img is None:
            return None, {"method": "ocr", "status": "read-failed", "evidence": []}

        if not TESSERACT_AVAILABLE:
            return None, {"method": "ocr", "status": "tesseract-missing", "evidence": []}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        bin_img = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7
        )

        data = pytesseract.image_to_data(bin_img, output_type=pytesseract.Output.DICT)
        evidences: list[DimensionEvidence] = []

        for i, txt in enumerate(data.get("text", [])):
            txt = (txt or "").strip()
            if not txt:
                continue
            conf = float(data.get("conf", ["-1"])[i])
            if conf < 30:
                continue

            m = NUM_RE.search(txt.replace(",", "."))
            if not m:
                continue

            value = float(m.group(1).replace(",", "."))
            unit = (m.group(2) or "").lower()
            real_m = self._to_meters(value, unit)
            if real_m <= 0:
                continue

            x = int(data["left"][i]); y = int(data["top"][i]); w = int(data["width"][i]); h = int(data["height"][i])
            px_len = self._nearest_dimension_line_length(bin_img, x, y, w, h)
            if px_len is None or px_len < 5:
                continue

            scale = real_m / px_len
            if 1e-6 <= scale <= 1.0:
                evidences.append(DimensionEvidence(txt, real_m, px_len, scale, (x, y, w, h)))

        if not evidences:
            return None, {"method": "ocr", "status": "no-dimensions", "evidence": []}

        # Robust median scale
        scales = sorted(e.scale for e in evidences)
        mid = len(scales) // 2
        scale = scales[mid] if len(scales) % 2 == 1 else (scales[mid - 1] + scales[mid]) / 2

        self.last_evidence = evidences
        return scale, {
            "method": "ocr",
            "status": "ok",
            "samples": len(evidences),
            "evidence": [e.to_dict() for e in evidences[:10]],
        }

    def _nearest_dimension_line_length(self, bin_img, x: int, y: int, w: int, h: int) -> float | None:
        pad = 80
        h_img, w_img = bin_img.shape[:2]
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(w_img, x + w + pad); y1 = min(h_img, y + h + pad)
        roi = bin_img[y0:y1, x0:x1]
        if roi.size == 0:
            return None

        edges = cv2.Canny(roi, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, math.pi / 180, threshold=25, minLineLength=20, maxLineGap=8)
        if lines is None:
            return None

        cx = x + w / 2
        cy = y + h / 2
        best = None
        best_dist = float("inf")

        for l in lines:
            x1l, y1l, x2l, y2l = [int(v) for v in l[0]]
            gx1 = x1l + x0; gy1 = y1l + y0
            gx2 = x2l + x0; gy2 = y2l + y0
            mx = (gx1 + gx2) / 2; my = (gy1 + gy2) / 2
            dist = math.hypot(mx - cx, my - cy)
            L = math.hypot(gx2 - gx1, gy2 - gy1)
            if dist < best_dist and L >= 20:
                best_dist = dist
                best = L

        return best

    @staticmethod
    def _to_meters(value: float, unit: str) -> float:
        if unit == "mm":
            return value / 1000.0
        if unit == "cm":
            return value / 100.0
        # default meters
        return value
