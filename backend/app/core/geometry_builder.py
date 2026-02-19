"""
Geometry Builder — Wall + Opening + Room objects → Three.js JSON.

Coordinate mapping:
  DXF X  →  Three.js X
  DXF Y  →  Three.js -Z
  height →  Three.js Y (up)
  rotation_y = -atan2(dy, dx)

Wall with openings:
  Split into solid pieces + opening descriptors.
  Doors  → dark void (no geometry) + door-leaf thin box + swing arc line
  Windows→ two short wall stubs (sill + header) + glass panel insert
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math, json
from .wall_detector import Wall
from .opening_detector import Opening, split_wall_at_openings

SILL_HEIGHT   = 0.9   # window sill height (m)
WIN_HEIGHT    = 1.2   # window opening height (m)
DOOR_LEAF_T   = 0.05  # door leaf thickness (m)


@dataclass
class BuildingModel:
    walls:    list[dict] = field(default_factory=list)
    floors:   list[dict] = field(default_factory=list)
    rooms:    list[dict] = field(default_factory=list)
    doors:    list[dict] = field(default_factory=list)
    windows:  list[dict] = field(default_factory=list)
    metadata: dict       = field(default_factory=dict)

    def to_dict(self):
        return {"metadata": self.metadata, "walls": self.walls,
                "floors": self.floors, "rooms": self.rooms,
                "doors": self.doors, "windows": self.windows}

    def to_json(self, indent=2):
        return json.dumps(self.to_dict(), indent=indent)


# ── Coordinate helpers ────────────────────────────────────────────────────────

def _box(cx, cy, length, height, thickness, rot_y):
    """Build a box dict in Three.js coords (cy already negated before call)."""
    return {
        "type":       "box",
        "position":   {"x": round(cx,4), "y": round(height/2,4), "z": round(cy,4)},
        "dimensions": {"width": round(length,4), "height": round(height,4),
                       "depth": round(thickness,4)},
        "rotation_y": round(rot_y, 6),
    }


def _midpoint_3js(sx, sy, ex, ey, wall_height):
    """Mid-position of a segment in Three.js coords."""
    return (
        (sx + ex) / 2,
        wall_height / 2,
        -((sy + ey) / 2),
    )


# ── Wall → boxes (with opening support) ──────────────────────────────────────

def wall_to_boxes(wall: Wall, openings: list[Opening]) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Returns (wall_boxes, door_dicts, window_dicts).
    Openings split the wall into solid pieces; doors/windows get their own dicts.
    """
    L = wall.length
    if L < 1e-6:
        return [], [], []

    dx = wall.end.x - wall.start.x
    dy = wall.end.y - wall.start.y
    rot_y  = -math.atan2(dy, dx)
    thick  = max(wall.thickness, 0.05)
    h      = wall.height

    if not openings:
        # Simple whole wall
        cx = (wall.start.x + wall.end.x) / 2
        cy = -((wall.start.y + wall.end.y) / 2)
        box = _box(cx, cy, L if L > thick else thick, h, thick if L > thick else L, rot_y)
        box.update({"layer": wall.layer, "paired": wall.paired, "confidence": wall.confidence})
        return [box], [], []

    pieces = split_wall_at_openings(wall, openings)
    wall_boxes = []
    door_dicts = []
    win_dicts  = []

    for p in pieces:
        seg_len = p["length"]
        if seg_len < 1e-4:
            continue
        cx_dxf = (p["sx"] + p["ex"]) / 2
        cy_dxf = (p["sy"] + p["ey"]) / 2

        if not p["is_opening"]:
            box = _box(cx_dxf, -cy_dxf, seg_len, h, thick, rot_y)
            box.update({"layer": wall.layer, "paired": wall.paired, "confidence": wall.confidence,
                        "length": round(seg_len,4), "thickness": round(thick,4)})
            wall_boxes.append(box)

        elif p.get("kind") == "door":
            # Door: void (no geometry) + door leaf + swing indicator
            door_dicts.append({
                "position": {
                    "x": round(cx_dxf, 4),
                    "y": round(h / 2, 4),
                    "z": round(-cy_dxf, 4),
                },
                "width":    round(seg_len, 4),
                "height":   round(h, 4),
                "depth":    round(thick, 4),
                "rotation_y": round(rot_y, 6),
                # Door leaf starts at one side of opening
                "leaf": _door_leaf(p["sx"], p["sy"], p["ex"], p["ey"],
                                   thick, h, rot_y),
            })

        elif p.get("kind") == "window":
            # Window: sill box + header box + glass panel
            win_dicts.append({
                "position": {
                    "x": round(cx_dxf, 4),
                    "y": round(h / 2, 4),
                    "z": round(-cy_dxf, 4),
                },
                "width":    round(seg_len, 4),
                "height":   round(h, 4),
                "depth":    round(thick, 4),
                "rotation_y": round(rot_y, 6),
                "sill_h":   SILL_HEIGHT,
                "win_h":    WIN_HEIGHT,
                "pieces":   _window_pieces(p["sx"], p["sy"], p["ex"], p["ey"],
                                           thick, h, rot_y),
            })

    return wall_boxes, door_dicts, win_dicts


def _door_leaf(sx, sy, ex, ey, thick, h, rot_y):
    """Thin door leaf box at the hinge side of the opening."""
    L = math.hypot(ex-sx, ey-sy)
    # Leaf sits at the start side, rotated open 45° visually
    cx = sx + (ex-sx) * 0.5
    cy = sy + (ey-sy) * 0.5
    return {
        "position":   {"x": round(cx,4), "y": round(h*0.5,4), "z": round(-cy,4)},
        "dimensions": {"width": round(L,4), "height": round(h,4),
                       "depth": round(DOOR_LEAF_T,4)},
        "rotation_y": round(rot_y + math.pi/4, 6),  # 45° open
    }


def _window_pieces(sx, sy, ex, ey, thick, h, rot_y):
    """Sill, header, and glass panel for a window opening."""
    L  = math.hypot(ex-sx, ey-sy)
    cx = (sx + ex) / 2
    cy = (sy + ey) / 2
    win_top = SILL_HEIGHT + WIN_HEIGHT

    pieces = []
    # Sill (below window)
    if SILL_HEIGHT > 0.05:
        pieces.append({
            "kind": "sill",
            "position":   {"x": round(cx,4), "y": round(SILL_HEIGHT/2,4), "z": round(-cy,4)},
            "dimensions": {"width": round(L,4), "height": round(SILL_HEIGHT,4),
                           "depth": round(thick,4)},
            "rotation_y": round(rot_y, 6),
        })
    # Header (above window)
    header_h = h - win_top
    if header_h > 0.05:
        pieces.append({
            "kind": "header",
            "position":   {"x": round(cx,4), "y": round(win_top + header_h/2,4), "z": round(-cy,4)},
            "dimensions": {"width": round(L,4), "height": round(header_h,4),
                           "depth": round(thick,4)},
            "rotation_y": round(rot_y, 6),
        })
    # Glass panel
    pieces.append({
        "kind": "glass",
        "position":   {"x": round(cx,4), "y": round(SILL_HEIGHT + WIN_HEIGHT/2,4), "z": round(-cy,4)},
        "dimensions": {"width": round(L,4), "height": round(WIN_HEIGHT,4),
                       "depth": round(thick * 0.15, 4)},  # thin glass
        "rotation_y": round(rot_y, 6),
    })

    return pieces


# ── Room label ────────────────────────────────────────────────────────────────

def room_to_label(room, wall_height: float = 3.0):
    return {
        "id":        room.id,
        "label":     room.label,
        "room_type": room.room_type,
        "color":     room.color,
        "area":      round(room.area, 1),
        "position": {
            "x": round(room.centroid_x, 3),
            "y": round(wall_height * 0.55, 3),
            "z": round(-room.centroid_y, 3),
        },
        "confidence": room.confidence,
    }


# ── Main builder ──────────────────────────────────────────────────────────────

class GeometryBuilder:
    def build(self, walls, bounds=None, rooms=None,
              openings=None, wall_height: float = 3.0):

        model = BuildingModel()

        # Group openings by wall index
        openings_by_wall: dict[int, list[Opening]] = {}
        if openings:
            for op in openings:
                openings_by_wall.setdefault(op.wall_idx, []).append(op)

        for wi, wall in enumerate(walls):
            wall_openings = openings_by_wall.get(wi, [])
            wall_boxes, door_dicts, win_dicts = wall_to_boxes(wall, wall_openings)
            model.walls.extend(wall_boxes)
            model.doors.extend(door_dicts)
            model.windows.extend(win_dicts)

        if bounds:
            model.floors.append(self._floor(bounds))

        if rooms:
            for r in rooms:
                model.rooms.append(room_to_label(r, wall_height))

        total_wall_len = sum(w.get("length", w["dimensions"]["width"])
                             for w in model.walls)

        model.metadata = {
            "wall_count":        len(model.walls),
            "floor_count":       len(model.floors),
            "room_count":        len(model.rooms),
            "door_count":        len(model.doors),
            "window_count":      len(model.windows),
            "total_wall_length": round(total_wall_len, 2),
            "paired_walls":      sum(1 for w in model.walls if w.get("paired")),
            "bounds":            bounds,
            "format":            "floorplan-json-v1",
        }
        return model

    def _floor(self, b):
        return {
            "type":       "floor",
            "position":   {"x": round((b["minx"]+b["maxx"])/2,4), "y": 0.0,
                           "z": round(-(b["miny"]+b["maxy"])/2,4)},
            "dimensions": {"width": round(b["maxx"]-b["minx"],4),
                           "depth": round(b["maxy"]-b["miny"],4)},
        }
