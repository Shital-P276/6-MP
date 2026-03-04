"""Window detection from small parallel segments lying within wall boundaries."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math

from .dxf_parser import Point2D, Segment, ParsedGeometry
from .opening_detector import Opening
from .wall_detector import Wall

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class Window:
    start: Point2D
    end: Point2D
    wall_reference: int
    confidence: float = 0.5

    @property
    def length(self) -> float:
        return math.hypot(self.end.x - self.start.x, self.end.y - self.start.y)

    def to_dict(self) -> dict:
        return {
            "start": {"x": round(self.start.x, 4), "y": round(self.start.y, 4)},
            "end": {"x": round(self.end.x, 4), "y": round(self.end.y, 4)},
            "wall_reference": self.wall_reference,
            "confidence": round(self.confidence, 3),
        }

    def to_opening(self, walls: list[Wall]) -> Opening | None:
        if self.wall_reference < 0 or self.wall_reference >= len(walls):
            return None
        wall = walls[self.wall_reference]
        L = wall.length
        if L < 1e-6:
            return None
        mx = (self.start.x + self.end.x) / 2
        my = (self.start.y + self.end.y) / 2
        dx = wall.end.x - wall.start.x
        dy = wall.end.y - wall.start.y
        t = ((mx - wall.start.x) * dx + (my - wall.start.y) * dy) / (L * L)
        t = max(0.0, min(1.0, t))
        return Opening(
            wall_idx=self.wall_reference,
            t_center=t,
            width=min(self.length, L * 0.9),
            kind="window",
            x=mx,
            y=my,
            angle=math.atan2(dy, dx),
        )

    @classmethod
    def from_opening(cls, opening: Opening, walls: list[Wall]) -> "Window" | None:
        if opening.wall_idx < 0 or opening.wall_idx >= len(walls):
            return None
        wall = walls[opening.wall_idx]
        L = wall.length
        if L < 1e-6:
            return None
        half = min(opening.width / 2, L / 2)
        t0 = max(0.0, opening.t_center - half / L)
        t1 = min(1.0, opening.t_center + half / L)
        sx = wall.start.x + (wall.end.x - wall.start.x) * t0
        sy = wall.start.y + (wall.end.y - wall.start.y) * t0
        ex = wall.start.x + (wall.end.x - wall.start.x) * t1
        ey = wall.start.y + (wall.end.y - wall.start.y) * t1
        return cls(start=Point2D(sx, sy), end=Point2D(ex, ey), wall_reference=opening.wall_idx, confidence=0.6)


class WindowDetector:
    def __init__(
        self,
        pixels_per_meter: float = 100.0,
        min_seg_len_m: float = 0.3,
        max_seg_len_m: float = 2.5,
    ):
        self.pixels_per_meter = pixels_per_meter
        self.min_seg_len_m = min_seg_len_m
        self.max_seg_len_m = max_seg_len_m

    def detect(
        self,
        geometry: ParsedGeometry,
        walls: list[Wall],
        image_path: str | None = None,
    ) -> list[Window]:
        candidates = self._candidate_segments(geometry, image_path)
        if not candidates or not walls:
            return []

        # 1) detect small parallel segments + 2) ensure inside wall boundaries
        tagged = []  # (segment, wall_idx)
        for seg in candidates:
            wall_idx = self._wall_reference(seg, walls)
            if wall_idx is None:
                continue
            tagged.append((seg, wall_idx))

        # 3) group parallel in-wall segments into windows
        return self._group_to_windows(tagged, walls)

    def _candidate_segments(self, geometry: ParsedGeometry, image_path: str | None) -> list[Segment]:
        segs = []

        # Existing classified window segments (DXF path)
        for s in geometry.window_segments:
            if self.min_seg_len_m <= self._len(s) <= self.max_seg_len_m:
                segs.append(s)

        # Raster fallback: detect short line segments from edge map
        segs.extend(self._segments_from_image(image_path))
        return segs

    def _segments_from_image(self, image_path: str | None) -> list[Segment]:
        if not image_path or not CV2_AVAILABLE:
            return []
        suffix = Path(image_path).suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            return []

        img = cv2.imread(image_path)
        if img is None:
            return []
        h = img.shape[0]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            5,
        )
        edges = cv2.Canny(binary, 50, 150)

        min_len_px = max(8, int(self.min_seg_len_m * self.pixels_per_meter))
        max_len_px = max(min_len_px + 1, int(self.max_seg_len_m * self.pixels_per_meter))

        raw = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=math.pi / 180,
            threshold=35,
            minLineLength=min_len_px,
            maxLineGap=6,
        )
        if raw is None:
            return []

        out = []
        for l in raw:
            x1, y1, x2, y2 = map(int, l[0])
            Lpx = math.hypot(x2 - x1, y2 - y1)
            if Lpx < min_len_px or Lpx > max_len_px:
                continue
            out.append(Segment(
                start=Point2D(x1 / self.pixels_per_meter, (h - y1) / self.pixels_per_meter),
                end=Point2D(x2 / self.pixels_per_meter, (h - y2) / self.pixels_per_meter),
                layer="WINDOW",
                source_type="HOUGH",
            ))
        return out

    def _wall_reference(self, seg: Segment, walls: list[Wall]) -> int | None:
        mx = (seg.start.x + seg.end.x) / 2
        my = (seg.start.y + seg.end.y) / 2

        best = None
        best_dist = float("inf")
        for wi, w in enumerate(walls):
            t, qx, qy, dist = _project_point(mx, my, w)
            if t < -0.05 or t > 1.05:
                continue
            tol = max(0.15, w.thickness * 0.9)
            if dist <= tol and dist < best_dist:
                # Segment should be roughly parallel to wall direction
                if _angle_diff(_seg_angle(seg), w.angle_deg) <= 15.0:
                    best = wi
                    best_dist = dist
        return best

    def _group_to_windows(self, tagged: list[tuple[Segment, int]], walls: list[Wall]) -> list[Window]:
        by_wall: dict[int, list[Segment]] = {}
        for seg, wi in tagged:
            by_wall.setdefault(wi, []).append(seg)

        windows: list[Window] = []
        for wi, segs in by_wall.items():
            used = set()
            wall = walls[wi]
            for i, a in enumerate(segs):
                if i in used:
                    continue
                best_j = None
                best_d = float("inf")
                for j in range(i + 1, len(segs)):
                    if j in used:
                        continue
                    b = segs[j]
                    if _angle_diff(_seg_angle(a), _seg_angle(b)) > 10.0:
                        continue
                    # lines should be close/parallel (thin double-line window symbol)
                    d = _seg_perp_dist(a, b)
                    if d < 0.02 or d > 0.35:
                        continue
                    if not _seg_axial_overlap(a, b, tol=0.10):
                        continue
                    if d < best_d:
                        best_d = d
                        best_j = j

                if best_j is None:
                    continue

                b = segs[best_j]
                used.add(i)
                used.add(best_j)

                # Build one window span from overlap projected on wall axis.
                t0, t1 = _window_interval_on_wall(a, b, wall)
                if t1 <= t0:
                    continue
                sx = wall.start.x + (wall.end.x - wall.start.x) * t0
                sy = wall.start.y + (wall.end.y - wall.start.y) * t0
                ex = wall.start.x + (wall.end.x - wall.start.x) * t1
                ey = wall.start.y + (wall.end.y - wall.start.y) * t1
                w = Window(
                    start=Point2D(sx, sy),
                    end=Point2D(ex, ey),
                    wall_reference=wi,
                    confidence=0.8,
                )
                if self.min_seg_len_m <= w.length <= self.max_seg_len_m:
                    windows.append(w)

        return windows


    def detect_into(self, floorplan, geometry: ParsedGeometry, image_path: str | None = None):
        floorplan.windows = self.detect(
            geometry=geometry,
            walls=floorplan.walls,
            image_path=image_path,
        )
        return floorplan
    @staticmethod
    def _len(seg: Segment) -> float:
        return math.hypot(seg.end.x - seg.start.x, seg.end.y - seg.start.y)


def _seg_angle(seg: Segment) -> float:
    return math.degrees(math.atan2(seg.end.y - seg.start.y, seg.end.x - seg.start.x))


def _angle_diff(a: float, b: float) -> float:
    d = abs((a - b) % 180)
    return min(d, 180 - d)


def _seg_perp_dist(a: Segment, b: Segment) -> float:
    ux, uy = _seg_unit(a)
    nx, ny = -uy, ux
    ma = Point2D((a.start.x + a.end.x) / 2, (a.start.y + a.end.y) / 2)
    mb = Point2D((b.start.x + b.end.x) / 2, (b.start.y + b.end.y) / 2)
    return abs((mb.x - ma.x) * nx + (mb.y - ma.y) * ny)


def _seg_unit(seg: Segment) -> tuple[float, float]:
    dx = seg.end.x - seg.start.x
    dy = seg.end.y - seg.start.y
    L = math.hypot(dx, dy)
    return (dx / L, dy / L) if L > 1e-9 else (1.0, 0.0)


def _seg_axial_overlap(a: Segment, b: Segment, tol: float = 0.1) -> bool:
    ua = _seg_unit(a)
    def proj(p: Point2D):
        return (p.x - a.start.x) * ua[0] + (p.y - a.start.y) * ua[1]

    a0, a1 = sorted([proj(a.start), proj(a.end)])
    b0, b1 = sorted([proj(b.start), proj(b.end)])
    return a1 + tol >= b0 and b1 + tol >= a0


def _project_point(px: float, py: float, wall: Wall) -> tuple[float, float, float, float]:
    ax, ay = wall.start.x, wall.start.y
    bx, by = wall.end.x, wall.end.y
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-9:
        return 0.0, ax, ay, math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / L2
    qx, qy = ax + t * dx, ay + t * dy
    return t, qx, qy, math.hypot(px - qx, py - qy)


def _window_interval_on_wall(a: Segment, b: Segment, wall: Wall) -> tuple[float, float]:
    dx = wall.end.x - wall.start.x
    dy = wall.end.y - wall.start.y
    L2 = dx * dx + dy * dy
    if L2 < 1e-9:
        return 0.0, 0.0

    def t_of(p: Point2D):
        return ((p.x - wall.start.x) * dx + (p.y - wall.start.y) * dy) / L2

    ta0, ta1 = sorted([t_of(a.start), t_of(a.end)])
    tb0, tb1 = sorted([t_of(b.start), t_of(b.end)])
    t0 = max(0.0, max(ta0, tb0))
    t1 = min(1.0, min(ta1, tb1))
    if t1 < t0:
        # if no strict overlap, use average span midpoint with minimum safe length
        ca = (ta0 + ta1) / 2
        cb = (tb0 + tb1) / 2
        c = max(0.0, min(1.0, (ca + cb) / 2))
        half = max(0.02, abs(ta1 - ta0) * 0.5)
        t0 = max(0.0, c - half)
        t1 = min(1.0, c + half)
    return t0, t1
