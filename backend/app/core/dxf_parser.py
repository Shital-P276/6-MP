"""
DXF Parser — extracts raw 2D geometry from a DXF file.
Handles: LINE, LWPOLYLINE, POLYLINE, ARC
Classifies segments by layer name into WALL / DOOR / WINDOW / OTHER.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math

try:
    import ezdxf
except ImportError:
    raise ImportError("ezdxf required: pip install ezdxf")


@dataclass
class Point2D:
    x: float
    y: float


@dataclass
class Segment:
    start: Point2D
    end: Point2D
    layer: str = "0"
    source_type: str = "LINE"


@dataclass
class ParsedGeometry:
    wall_segments:   list[Segment] = field(default_factory=list)
    door_segments:   list[Segment] = field(default_factory=list)
    window_segments: list[Segment] = field(default_factory=list)
    other_segments:  list[Segment] = field(default_factory=list)
    ignored_segments:list[Segment] = field(default_factory=list)
    text_labels:     list          = field(default_factory=list)  # [(x, y, text), ...]
    units: str = "unknown"
    bounds: Optional[dict] = None


# ── Layer classification ──────────────────────────────────────────────────────

def classify_layer(name: str) -> str:
    n = name.lower()
    for kw in ("dim", "dimension", "measure", "cote", "cota", "annot", "axis", "grid"):
        if kw in n: return "IGNORE"
    for kw in ("wall", "mur", "wand", "cloison"):
        if kw in n: return "WALL"
    for kw in ("door", "doors", "porte", "puerta", "tür", "tur", "d-"):
        if kw in n: return "DOOR"
    for kw in ("window", "windows", "fenetre", "fenêtre", "fenster", "win", "glazing"):
        if kw in n: return "WINDOW"
    return "OTHER"


# ── Arc approximation ─────────────────────────────────────────────────────────

def _arc_to_segments(center, radius, start_deg, end_deg, layer, steps=12):
    s = math.radians(start_deg)
    e = math.radians(end_deg)
    if e <= s:
        e += 2 * math.pi
    step = (e - s) / steps
    pts = [Point2D(center[0] + radius * math.cos(s + i * step),
                   center[1] + radius * math.sin(s + i * step))
           for i in range(steps + 1)]
    return [Segment(pts[i], pts[i+1], layer, "ARC") for i in range(steps)]


# ── Main parser ───────────────────────────────────────────────────────────────

class DXFParser:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def parse(self) -> ParsedGeometry:
        doc = ezdxf.readfile(self.filepath)
        result = ParsedGeometry()
        result.units = self._units(doc)
        all_segs: list[Segment] = []

        for entity in doc.modelspace():
            all_segs.extend(self._to_segments(entity))
            # Collect TEXT / MTEXT entities for room labeling
            if entity.dxftype() in ("TEXT", "MTEXT"):
                try:
                    if entity.dxftype() == "TEXT":
                        ins = entity.dxf.insert
                        txt = entity.dxf.text.strip()
                    else:  # MTEXT
                        ins = entity.dxf.insert
                        txt = entity.plain_mtext().strip()
                    if txt:
                        result.text_labels.append((ins.x, ins.y, txt))
                except Exception:
                    pass

        for seg in all_segs:
            cat = classify_layer(seg.layer)
            if cat == "WALL":   result.wall_segments.append(seg)
            elif cat == "DOOR": result.door_segments.append(seg)
            elif cat == "WINDOW": result.window_segments.append(seg)
            elif cat == "IGNORE": result.ignored_segments.append(seg)
            else: result.other_segments.append(seg)

        result.bounds = self._bounds(all_segs)
        return result

    def _to_segments(self, entity) -> list[Segment]:
        t = entity.dxftype()
        layer = getattr(entity.dxf, "layer", "0")
        try:
            if t == "LINE":
                s, e = entity.dxf.start, entity.dxf.end
                return [Segment(Point2D(s.x, s.y), Point2D(e.x, e.y), layer, "LINE")]

            if t in ("LWPOLYLINE", "POLYLINE"):
                try:
                    pts = list(entity.get_points())
                except AttributeError:
                    pts = [v.dxf.location for v in entity.vertices]
                segs = []
                for i in range(len(pts) - 1):
                    segs.append(Segment(
                        Point2D(pts[i][0], pts[i][1]),
                        Point2D(pts[i+1][0], pts[i+1][1]),
                        layer, "POLYLINE"
                    ))
                try:
                    if entity.is_closed and len(pts) >= 2:
                        segs.append(Segment(
                            Point2D(pts[-1][0], pts[-1][1]),
                            Point2D(pts[0][0], pts[0][1]),
                            layer, "POLYLINE"
                        ))
                except AttributeError:
                    pass
                return segs

            if t == "ARC":
                c = entity.dxf.center
                return _arc_to_segments(
                    (c.x, c.y), entity.dxf.radius,
                    entity.dxf.start_angle, entity.dxf.end_angle, layer
                )
        except Exception:
            pass
        return []

    def _units(self, doc) -> str:
        try:
            code = doc.header.get("$INSUNITS", 0)
            return {0:"unitless",1:"inches",2:"feet",4:"mm",5:"cm",6:"meters"}.get(code, f"code:{code}")
        except Exception:
            return "unknown"

    def _bounds(self, segs: list[Segment]) -> Optional[dict]:
        if not segs:
            return None
        xs = [s.start.x for s in segs] + [s.end.x for s in segs]
        ys = [s.start.y for s in segs] + [s.end.y for s in segs]
        return {"minx": min(xs), "miny": min(ys), "maxx": max(xs), "maxy": max(ys)}
