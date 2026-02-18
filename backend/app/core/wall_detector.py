"""
Wall Detection — converts raw segments into structured wall objects.

Strategy:
1. Group segments into parallel pairs (detect wall thickness)
2. Merge collinear segments
3. Produce Wall objects with centerline + thickness
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import math

from .dxf_parser import Segment, Point2D, ParsedGeometry


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Wall:
    start: Point2D
    end: Point2D
    thickness: float = 0.2   # meters
    height: float = 3.0       # default wall height in meters
    layer: str = "WALL"

    @property
    def length(self) -> float:
        return math.hypot(self.end.x - self.start.x, self.end.y - self.start.y)

    @property
    def angle_deg(self) -> float:
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        return math.degrees(math.atan2(dy, dx))

    def to_dict(self) -> dict:
        return {
            "start": {"x": round(self.start.x, 4), "y": round(self.start.y, 4)},
            "end": {"x": round(self.end.x, 4), "y": round(self.end.y, 4)},
            "thickness": round(self.thickness, 4),
            "height": round(self.height, 4),
            "length": round(self.length, 4),
            "angle_deg": round(self.angle_deg, 2),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _segment_angle(seg: Segment) -> float:
    """Normalized angle in [0, 180) degrees."""
    dx = seg.end.x - seg.start.x
    dy = seg.end.y - seg.start.y
    angle = math.degrees(math.atan2(dy, dx)) % 180
    return angle


def _segment_length(seg: Segment) -> float:
    return math.hypot(seg.end.x - seg.start.x, seg.end.y - seg.start.y)


def _midpoint(seg: Segment) -> Point2D:
    return Point2D(
        (seg.start.x + seg.end.x) / 2,
        (seg.start.y + seg.end.y) / 2,
    )


def _perpendicular_distance(seg_a: Segment, seg_b: Segment) -> float:
    """Approximate perpendicular distance between two parallel segments via midpoints."""
    ma = _midpoint(seg_a)
    mb = _midpoint(seg_b)
    return math.hypot(ma.x - mb.x, ma.y - mb.y)


def _are_parallel(seg_a: Segment, seg_b: Segment, angle_tol: float = 5.0) -> bool:
    """Check if two segments are approximately parallel."""
    diff = abs(_segment_angle(seg_a) - _segment_angle(seg_b))
    return diff < angle_tol or abs(diff - 180) < angle_tol


def _segments_overlap_axially(seg_a: Segment, seg_b: Segment, tol: float = 0.5) -> bool:
    """
    Check if two parallel segments overlap along their shared axis.
    Projects both onto the primary axis of seg_a.
    """
    dx = seg_a.end.x - seg_a.start.x
    dy = seg_a.end.y - seg_a.start.y
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return False

    ux, uy = dx / length, dy / length  # unit vector along seg_a

    def project(pt: Point2D) -> float:
        return (pt.x - seg_a.start.x) * ux + (pt.y - seg_a.start.y) * uy

    a_min, a_max = sorted([project(seg_a.start), project(seg_a.end)])
    b_min, b_max = sorted([project(seg_b.start), project(seg_b.end)])

    # Overlap if ranges intersect (with tolerance)
    return a_max + tol >= b_min and b_max + tol >= a_min


def _centerline(seg_a: Segment, seg_b: Segment) -> tuple[Point2D, Point2D]:
    """Compute the centerline between two parallel segments."""
    sx = (seg_a.start.x + seg_b.start.x) / 2
    sy = (seg_a.start.y + seg_b.start.y) / 2
    ex = (seg_a.end.x + seg_b.end.x) / 2
    ey = (seg_a.end.y + seg_b.end.y) / 2
    return Point2D(sx, sy), Point2D(ex, ey)


# ─────────────────────────────────────────────────────────────────────────────
# Main detector
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_WALL_THICKNESS = 0.2   # meters — fallback when pairing fails
DEFAULT_WALL_HEIGHT = 3.0       # meters
MIN_WALL_LENGTH = 0.3           # ignore tiny segments
MAX_WALL_PAIR_DISTANCE = 1.0    # max distance between parallel lines to form a wall pair


class WallDetector:
    def __init__(
        self,
        scale: float = 1.0,
        default_thickness: float = DEFAULT_WALL_THICKNESS,
        default_height: float = DEFAULT_WALL_HEIGHT,
    ):
        """
        Args:
            scale: multiply all coordinates by this factor (for unit conversion)
            default_thickness: fallback wall thickness in meters
            default_height: wall height in meters
        """
        self.scale = scale
        self.default_thickness = default_thickness
        self.default_height = default_height

    def detect(self, geometry: ParsedGeometry) -> list[Wall]:
        """Detect walls from parsed DXF geometry."""
        segments = geometry.wall_segments

        # Scale if needed
        if self.scale != 1.0:
            segments = self._scale_segments(segments)

        # Remove tiny segments
        segments = [s for s in segments if _segment_length(s) >= MIN_WALL_LENGTH]

        # Attempt to pair parallel segments into wall pairs
        walls, unpaired = self._pair_segments(segments)

        # Remaining unpaired segments → single-line walls with default thickness
        for seg in unpaired:
            walls.append(Wall(
                start=seg.start,
                end=seg.end,
                thickness=self.default_thickness,
                height=self.default_height,
                layer=seg.layer,
            ))

        return walls

    def _scale_segments(self, segments: list[Segment]) -> list[Segment]:
        scaled = []
        for s in segments:
            scaled.append(Segment(
                start=Point2D(s.start.x * self.scale, s.start.y * self.scale),
                end=Point2D(s.end.x * self.scale, s.end.y * self.scale),
                layer=s.layer,
                source_type=s.source_type,
            ))
        return scaled

    def _pair_segments(self, segments: list[Segment]) -> tuple[list[Wall], list[Segment]]:
        """Pair parallel overlapping segments into walls."""
        used = set()
        walls = []

        for i, seg_a in enumerate(segments):
            if i in used:
                continue

            best_j = None
            best_dist = float("inf")

            for j, seg_b in enumerate(segments):
                if j <= i or j in used:
                    continue
                if not _are_parallel(seg_a, seg_b):
                    continue
                dist = _perpendicular_distance(seg_a, seg_b)
                if dist > MAX_WALL_PAIR_DISTANCE:
                    continue
                if not _segments_overlap_axially(seg_a, seg_b):
                    continue
                if dist < best_dist:
                    best_dist = dist
                    best_j = j

            if best_j is not None:
                used.add(i)
                used.add(best_j)

                seg_b = segments[best_j]
                center_start, center_end = _centerline(seg_a, seg_b)
                walls.append(Wall(
                    start=center_start,
                    end=center_end,
                    thickness=round(best_dist, 3),
                    height=self.default_height,
                    layer=seg_a.layer,
                ))

        unpaired = [s for i, s in enumerate(segments) if i not in used]
        return walls, unpaired
