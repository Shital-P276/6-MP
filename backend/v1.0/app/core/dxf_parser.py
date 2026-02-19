"""
DXF Parser — extracts raw geometry from a DXF file.

Handles:
- Lines and polylines from WALL layer
- Arc approximation
- Layer-based filtering
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math

try:
    import ezdxf
    from ezdxf.document import Drawing
    from ezdxf import select
except ImportError:
    raise ImportError("ezdxf is required: pip install ezdxf")


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Point2D:
    x: float
    y: float

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass
class Segment:
    """A 2D line segment."""
    start: Point2D
    end: Point2D
    layer: str = "0"
    source_type: str = "LINE"  # LINE, POLYLINE, ARC


@dataclass
class ParsedGeometry:
    """Raw geometry extracted from a DXF file."""
    wall_segments: list[Segment] = field(default_factory=list)
    door_segments: list[Segment] = field(default_factory=list)
    window_segments: list[Segment] = field(default_factory=list)
    other_segments: list[Segment] = field(default_factory=list)
    units: str = "unknown"
    bounds: Optional[dict] = None  # {minx, miny, maxx, maxy}


# ─────────────────────────────────────────────────────────────────────────────
# Layer classification
# ─────────────────────────────────────────────────────────────────────────────

WALL_KEYWORDS = ["wall", "mur", "wand", "cloison"]
DOOR_KEYWORDS = ["door", "porte", "tür", "tur"]
WINDOW_KEYWORDS = ["window", "fenetre", "fenêtre", "fenster"]


def classify_layer(layer_name: str) -> str:
    """Map a layer name to a semantic category."""
    name = layer_name.lower().strip()
    for kw in WALL_KEYWORDS:
        if kw in name:
            return "WALL"
    for kw in DOOR_KEYWORDS:
        if kw in name:
            return "DOOR"
    for kw in WINDOW_KEYWORDS:
        if kw in name:
            return "WINDOW"
    return "OTHER"


# ─────────────────────────────────────────────────────────────────────────────
# Arc approximation
# ─────────────────────────────────────────────────────────────────────────────

def arc_to_segments(
    center: tuple[float, float],
    radius: float,
    start_angle_deg: float,
    end_angle_deg: float,
    layer: str,
    steps: int = 12,
) -> list[Segment]:
    """Approximate an arc as a series of line segments."""
    start_rad = math.radians(start_angle_deg)
    end_rad = math.radians(end_angle_deg)

    if end_rad <= start_rad:
        end_rad += 2 * math.pi

    angle_step = (end_rad - start_rad) / steps
    points = []
    for i in range(steps + 1):
        angle = start_rad + i * angle_step
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        points.append(Point2D(x, y))

    return [
        Segment(start=points[i], end=points[i + 1], layer=layer, source_type="ARC")
        for i in range(len(points) - 1)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Main parser
# ─────────────────────────────────────────────────────────────────────────────

class DXFParser:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._doc: Optional[Drawing] = None

    def load(self) -> "DXFParser":
        self._doc = ezdxf.readfile(self.filepath)
        return self

    def parse(self) -> ParsedGeometry:
        if self._doc is None:
            self.load()

        result = ParsedGeometry()
        result.units = self._get_units()

        msp = self._doc.modelspace()

        all_segments: list[Segment] = []

        for entity in msp:
            segments = self._entity_to_segments(entity)
            all_segments.extend(segments)

        # Classify into buckets
        for seg in all_segments:
            cat = classify_layer(seg.layer)
            if cat == "WALL":
                result.wall_segments.append(seg)
            elif cat == "DOOR":
                result.door_segments.append(seg)
            elif cat == "WINDOW":
                result.window_segments.append(seg)
            else:
                result.other_segments.append(seg)

        result.bounds = self._compute_bounds(all_segments)
        return result

    # ── Entity handlers ──────────────────────────────────────────────────────

    def _entity_to_segments(self, entity) -> list[Segment]:
        t = entity.dxftype()
        layer = entity.dxf.layer if hasattr(entity.dxf, "layer") else "0"

        if t == "LINE":
            return self._handle_line(entity, layer)
        elif t in ("LWPOLYLINE", "POLYLINE"):
            return self._handle_polyline(entity, layer)
        elif t == "ARC":
            return self._handle_arc(entity, layer)
        return []

    def _handle_line(self, entity, layer: str) -> list[Segment]:
        s = entity.dxf.start
        e = entity.dxf.end
        return [Segment(
            start=Point2D(s.x, s.y),
            end=Point2D(e.x, e.y),
            layer=layer,
            source_type="LINE",
        )]

    def _handle_polyline(self, entity, layer: str) -> list[Segment]:
        segments = []
        try:
            points = list(entity.get_points())
        except AttributeError:
            try:
                points = [v.dxf.location for v in entity.vertices]
            except Exception:
                return []

        for i in range(len(points) - 1):
            ax, ay = points[i][0], points[i][1]
            bx, by = points[i + 1][0], points[i + 1][1]
            segments.append(Segment(
                start=Point2D(ax, ay),
                end=Point2D(bx, by),
                layer=layer,
                source_type="POLYLINE",
            ))

        # Close if flagged as closed
        try:
            if entity.is_closed and len(points) >= 2:
                ax, ay = points[-1][0], points[-1][1]
                bx, by = points[0][0], points[0][1]
                segments.append(Segment(
                    start=Point2D(ax, ay),
                    end=Point2D(bx, by),
                    layer=layer,
                    source_type="POLYLINE",
                ))
        except AttributeError:
            pass

        return segments

    def _handle_arc(self, entity, layer: str) -> list[Segment]:
        center = entity.dxf.center
        radius = entity.dxf.radius
        start_angle = entity.dxf.start_angle
        end_angle = entity.dxf.end_angle
        return arc_to_segments(
            center=(center.x, center.y),
            radius=radius,
            start_angle_deg=start_angle,
            end_angle_deg=end_angle,
            layer=layer,
        )

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _get_units(self) -> str:
        try:
            unit_code = self._doc.header.get("$INSUNITS", 0)
            unit_map = {0: "unitless", 1: "inches", 2: "feet", 4: "mm", 5: "cm", 6: "meters"}
            return unit_map.get(unit_code, f"code:{unit_code}")
        except Exception:
            return "unknown"

    def _compute_bounds(self, segments: list[Segment]) -> dict:
        if not segments:
            return {"minx": 0, "miny": 0, "maxx": 0, "maxy": 0}

        all_x = [s.start.x for s in segments] + [s.end.x for s in segments]
        all_y = [s.start.y for s in segments] + [s.end.y for s in segments]

        return {
            "minx": min(all_x),
            "miny": min(all_y),
            "maxx": max(all_x),
            "maxy": max(all_y),
        }
