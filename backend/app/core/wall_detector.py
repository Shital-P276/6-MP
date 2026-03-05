"""
Wall Detector

Primary path (raster/image):
  1) Detect line segments with OpenCV HoughLinesP
  2) Estimate/filter by wall thickness
  3) Merge collinear + overlapping segments
  4) Output structured Wall objects

Fallback path (vector/DXF):
  - Build walls from parsed wall segments with parallel-line pairing + single-line fallback.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math
import collections

from .dxf_parser import Segment, Point2D, ParsedGeometry

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

MIN_WALL_LENGTH = 0.1
MAX_PAIR_DIST = 0.8
MIN_PAIR_DIST = 0.01
PARALLEL_TOL_DEG = 5.0
AXIAL_OVERLAP_TOL = 0.3
DEFAULT_HEIGHT = 3.0
DEFAULT_THICKNESS = 0.2

# OpenCV wall extraction defaults
CANNY_LOW = 50
CANNY_HIGH = 150
HOUGH_RHO = 1
HOUGH_THETA = math.pi / 180
HOUGH_THRESHOLD = 60
HOUGH_MIN_LENGTH = 30
HOUGH_MAX_GAP = 10
DOMINANT_AXIS_TOL_DEG = 12.0

COLLINEAR_PERP_TOL_M = 0.10
COLLINEAR_GAP_TOL_M = 0.20


def _len(s):
    return math.hypot(s.end.x - s.start.x, s.end.y - s.start.y)


def _angle(s):
    return math.degrees(math.atan2(s.end.y - s.start.y, s.end.x - s.start.x)) % 180


def _mid(s):
    return Point2D((s.start.x + s.end.x) / 2, (s.start.y + s.end.y) / 2)


def _unit(s):
    dx = s.end.x - s.start.x
    dy = s.end.y - s.start.y
    L = math.hypot(dx, dy)
    return (dx / L, dy / L) if L > 1e-9 else (1.0, 0.0)


def _perp_dist(a, b):
    ux, uy = _unit(a)
    nx, ny = -uy, ux
    ma, mb = _mid(a), _mid(b)
    return abs((mb.x - ma.x) * nx + (mb.y - ma.y) * ny)


def _axial_proj(a, pt):
    ux, uy = _unit(a)
    return (pt.x - a.start.x) * ux + (pt.y - a.start.y) * uy


def _axial_overlap(a, b, tol=AXIAL_OVERLAP_TOL):
    a0, a1 = sorted([_axial_proj(a, a.start), _axial_proj(a, a.end)])
    b0, b1 = sorted([_axial_proj(a, b.start), _axial_proj(a, b.end)])
    return a1 + tol >= b0 and b1 + tol >= a0


def _angle_diff(a, b):
    d = abs(_angle(a) - _angle(b))
    return min(d, 180 - d)


def _centerline(a, b):
    if _axial_proj(a, b.start) > _axial_proj(a, b.end):
        b = Segment(b.end, b.start, b.layer, b.source_type)
    return (
        Point2D((a.start.x + b.start.x) / 2, (a.start.y + b.start.y) / 2),
        Point2D((a.end.x + b.end.x) / 2, (a.end.y + b.end.y) / 2),
    )


def infer_scale(bounds):
    if not bounds:
        return 1.0
    span = max(bounds.get("maxx", 0) - bounds.get("minx", 0),
               bounds.get("maxy", 0) - bounds.get("miny", 0))
    if span <= 0:
        return 1.0
    if span <= 100:
        return 1.0
    if span <= 2000:
        return 0.0254
    return 0.001


@dataclass
class Wall:
    start: Point2D
    end: Point2D
    thickness: float = DEFAULT_THICKNESS
    height: float = DEFAULT_HEIGHT
    layer: str = "WALL"
    paired: bool = False
    confidence: float = 1.0

    @property
    def length(self):
        return math.hypot(self.end.x - self.start.x, self.end.y - self.start.y)

    @property
    def angle_deg(self):
        return math.degrees(math.atan2(self.end.y - self.start.y, self.end.x - self.start.x))

    def to_dict(self):
        return {
            "start": {"x": round(self.start.x, 4), "y": round(self.start.y, 4)},
            "end": {"x": round(self.end.x, 4), "y": round(self.end.y, 4)},
            "thickness": round(self.thickness, 4),
            "height": round(self.height, 4),
            "length": round(self.length, 4),
            "paired": self.paired,
            "confidence": round(self.confidence, 3),
        }


def pair_double_lines(segs, height, default_thickness):
    used = set()
    walls = []
    buckets = collections.defaultdict(list)
    for i, s in enumerate(segs):
        b = int(_angle(s) / 5)
        buckets[b].append(i)
        buckets[(b + 1) % 36].append(i)
    for i, sa in enumerate(segs):
        if i in used:
            continue
        best_j = None
        best_d = float("inf")
        for j in buckets.get(int(_angle(sa) / 5), []):
            if j <= i or j in used:
                continue
            sb = segs[j]
            if _angle_diff(sa, sb) > PARALLEL_TOL_DEG:
                continue
            d = _perp_dist(sa, sb)
            if d < MIN_PAIR_DIST or d > MAX_PAIR_DIST:
                continue
            if not _axial_overlap(sa, sb):
                continue
            if d < best_d:
                best_d = d
                best_j = j
        if best_j is not None:
            used.add(i)
            used.add(best_j)
            cs, ce = _centerline(sa, segs[best_j])
            ra = min(_len(sa), _len(segs[best_j])) / (max(_len(sa), _len(segs[best_j])) + 1e-9)
            walls.append(Wall(start=cs, end=ce, thickness=round(best_d, 4), height=height,
                              layer=sa.layer, paired=True, confidence=round(min(1.0, 0.7 + 0.3 * ra), 3)))
    unpaired = [segs[i] for i in range(len(segs)) if i not in used]
    return walls, unpaired


def _line_points(x1, y1, x2, y2):
    n = int(max(abs(x2 - x1), abs(y2 - y1))) + 1
    if n <= 1:
        return [(int(round(x1)), int(round(y1)))]
    return [
        (int(round(x1 + (x2 - x1) * t / (n - 1))), int(round(y1 + (y2 - y1) * t / (n - 1))))
        for t in range(n)
    ]


def _estimate_line_thickness_px(line, dist_map):
    x1, y1, x2, y2 = line
    pts = _line_points(x1, y1, x2, y2)
    vals = []
    h, w = dist_map.shape[:2]
    for x, y in pts:
        if 0 <= x < w and 0 <= y < h:
            vals.append(dist_map[y, x])
    if not vals:
        return 0.0
    # distance transform gives half-thickness to nearest edge in binary wall mask
    return float(2.0 * np.median(vals))


def _merge_collinear_segments(segments: list[Wall], perp_tol_m: float, gap_tol_m: float) -> list[Wall]:
    changed = True
    merged = segments[:]
    while changed:
        changed = False
        out = []
        used = set()
        for i, a in enumerate(merged):
            if i in used:
                continue
            current = a
            for j in range(i + 1, len(merged)):
                if j in used:
                    continue
                b = merged[j]
                sa = Segment(current.start, current.end, "WALL", "MERGED")
                sb = Segment(b.start, b.end, "WALL", "MERGED")
                if _angle_diff(sa, sb) > PARALLEL_TOL_DEG:
                    continue
                if _perp_dist(sa, sb) > perp_tol_m:
                    continue
                if not _axial_overlap(sa, sb, tol=gap_tol_m):
                    continue

                ux, uy = _unit(sa)
                pts = [current.start, current.end, b.start, b.end]
                projs = [p.x * ux + p.y * uy for p in pts]
                i_min = projs.index(min(projs))
                i_max = projs.index(max(projs))
                p0 = pts[i_min]
                p1 = pts[i_max]
                current = Wall(
                    start=Point2D(p0.x, p0.y),
                    end=Point2D(p1.x, p1.y),
                    thickness=max(current.thickness, b.thickness),
                    height=current.height,
                    layer="WALL",
                    paired=False,
                    confidence=min(1.0, max(current.confidence, b.confidence) + 0.05),
                )
                used.add(j)
                changed = True
            out.append(current)
            used.add(i)
        merged = out
    return merged


def _line_angle_deg(line: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = line
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180


def _line_len_px(line: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = line
    return math.hypot(x2 - x1, y2 - y1)


def _angular_distance_deg(a: float, b: float) -> float:
    d = abs(a - b) % 180
    return min(d, 180 - d)


def _find_dominant_axes(lines: list[tuple[int, int, int, int]]) -> tuple[float, float] | None:
    """Estimate main orthogonal wall axes from Hough lines.

    Many floor plans contain diagonal annotations (door arcs, dimensions, text)
    that can dominate raw Hough output. We select the strongest angle by
    length-weighted voting and derive its orthogonal partner.
    """
    if not lines:
        return None

    hist = [0.0] * 180
    for line in lines:
        angle = int(round(_line_angle_deg(line))) % 180
        hist[angle] += _line_len_px(line)

    dominant = float(max(range(180), key=lambda i: hist[i]))
    return dominant, (dominant + 90.0) % 180.0


def _filter_to_dominant_axes(
    lines: list[tuple[int, int, int, int]],
    tolerance_deg: float = DOMINANT_AXIS_TOL_DEG,
) -> list[tuple[int, int, int, int]]:
    axes = _find_dominant_axes(lines)
    if axes is None:
        return lines

    a0, a1 = axes
    filtered = [
        line for line in lines
        if min(_angular_distance_deg(_line_angle_deg(line), a0),
               _angular_distance_deg(_line_angle_deg(line), a1)) <= tolerance_deg
    ]

    # Fallback safety: if the filter is too aggressive, keep original lines.
    return filtered if len(filtered) >= max(4, int(0.3 * len(lines))) else lines


class WallDetector:
    def __init__(
        self,
        scale=1.0,
        auto_scale=True,
        default_thickness=DEFAULT_THICKNESS,
        default_height=DEFAULT_HEIGHT,
        is_raster=False,
        pixels_per_meter: float = 100.0,
        min_wall_thickness_m: float = 0.08,
        max_wall_thickness_m: float = 0.8,
    ):
        self.scale = scale
        self.auto_scale = auto_scale
        self.default_thickness = default_thickness
        self.default_height = default_height
        self.is_raster = is_raster
        self.pixels_per_meter = pixels_per_meter
        self.min_wall_thickness_m = min_wall_thickness_m
        self.max_wall_thickness_m = max_wall_thickness_m
        self._applied_scale = scale

    @property
    def applied_scale(self):
        return self._applied_scale

    def _detect_from_image(self, image_path: str) -> list[Wall]:
        if not CV2_AVAILABLE:
            return []
        img = cv2.imread(image_path)
        if img is None:
            return []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 5
        )
        edges = cv2.Canny(binary, CANNY_LOW, CANNY_HIGH)

        # Use wall mask for thickness estimation.
        wall_mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        dist_map = cv2.distanceTransform(wall_mask, cv2.DIST_L2, 5)

        raw = cv2.HoughLinesP(
            edges,
            rho=HOUGH_RHO,
            theta=HOUGH_THETA,
            threshold=HOUGH_THRESHOLD,
            minLineLength=HOUGH_MIN_LENGTH,
            maxLineGap=HOUGH_MAX_GAP,
        )
        if raw is None:
            return []

        raw_lines = [tuple(map(int, l[0])) for l in raw]
        candidate_lines = _filter_to_dominant_axes(raw_lines)

        h = img.shape[0]
        walls = []
        for x1, y1, x2, y2 in candidate_lines:
            thickness_px = _estimate_line_thickness_px((x1, y1, x2, y2), dist_map)
            thickness_m = max(thickness_px / max(self.pixels_per_meter, 1e-6), 0.0)

            if not (self.min_wall_thickness_m <= thickness_m <= self.max_wall_thickness_m):
                continue

            sx, sy = x1 / self.pixels_per_meter, (h - y1) / self.pixels_per_meter
            ex, ey = x2 / self.pixels_per_meter, (h - y2) / self.pixels_per_meter
            w = Wall(
                start=Point2D(sx, sy),
                end=Point2D(ex, ey),
                thickness=max(thickness_m, self.default_thickness),
                height=self.default_height,
                layer="WALL",
                paired=False,
                confidence=0.85,
            )
            if w.length >= MIN_WALL_LENGTH:
                walls.append(w)

        if not walls:
            return []

        return _merge_collinear_segments(
            walls,
            perp_tol_m=COLLINEAR_PERP_TOL_M,
            gap_tol_m=COLLINEAR_GAP_TOL_M,
        )

    def _detect_from_segments(self, geometry: ParsedGeometry) -> list[Wall]:
        segs = list(geometry.wall_segments)

        if self.auto_scale and self.scale == 1.0:
            self._applied_scale = infer_scale(geometry.bounds)
        else:
            self._applied_scale = self.scale

        if self._applied_scale != 1.0:
            s = self._applied_scale
            segs = [
                Segment(Point2D(sg.start.x * s, sg.start.y * s),
                        Point2D(sg.end.x * s, sg.end.y * s),
                        sg.layer, sg.source_type)
                for sg in segs
            ]
            if geometry.bounds:
                geometry.bounds = {k: v * s for k, v in geometry.bounds.items()}

        segs = [s for s in segs if _len(s) >= MIN_WALL_LENGTH]
        walls, unpaired = pair_double_lines(segs, self.default_height, self.default_thickness)

        for seg in unpaired:
            if _len(seg) >= MIN_WALL_LENGTH:
                walls.append(Wall(start=seg.start, end=seg.end,
                                  thickness=self.default_thickness,
                                  height=self.default_height,
                                  layer=seg.layer, paired=False, confidence=0.8))
        return walls

    def detect(self, geometry: ParsedGeometry, image_path: str | None = None) -> list[Wall]:
        # Raster path: detect directly from image with OpenCV Hough + thickness filtering.
        if self.is_raster and image_path:
            suffix = Path(image_path).suffix.lower()
            if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                walls = self._detect_from_image(image_path)
                if walls:
                    self._applied_scale = 1.0
                    return walls

        # Vector/fallback path.
        return self._detect_from_segments(geometry)
    def detect_into(self, floorplan, geometry: ParsedGeometry, image_path: str | None = None):
        floorplan.walls = self.detect(geometry, image_path=image_path)
        return floorplan
