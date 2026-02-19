"""
3D Geometry Builder — converts detected walls into 3D mesh data.

Output is a JSON structure that can be consumed by Three.js directly,
or later converted to GLTF.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math
import json

from .wall_detector import Wall


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BuildingModel:
    """Complete 3D model ready for the viewer."""
    walls: list[dict] = field(default_factory=list)
    floors: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "metadata": self.metadata,
            "walls": self.walls,
            "floors": self.floors,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ─────────────────────────────────────────────────────────────────────────────
# Wall → 3D box mesh
# ─────────────────────────────────────────────────────────────────────────────

def wall_to_box(wall: Wall) -> dict:
    """
    Convert a Wall to a box mesh description.

    The box is centered on the wall centerline, with:
    - length along the wall direction
    - width = wall thickness
    - height = wall height

    Returns a dict with position, dimensions, and rotation — easy for
    Three.js to instantiate as a BoxGeometry.
    """
    cx = (wall.start.x + wall.end.x) / 2
    cy = (wall.start.y + wall.end.y) / 2

    length = wall.length
    if length < 1e-6:
        return None

    dx = wall.end.x - wall.start.x
    dy = wall.end.y - wall.start.y
    rotation_y = -math.atan2(dy, dx)   # Three.js uses right-hand Y-up

    return {
        "type": "box",
        "position": {
            "x": round(cx, 4),
            "y": round(wall.height / 2, 4),   # center vertically
            "z": round(cy, 4),                  # DXF Y → Three.js Z
        },
        "dimensions": {
            "width": round(length, 4),
            "height": round(wall.height, 4),
            "depth": round(max(wall.thickness, 0.05), 4),
        },
        "rotation_y": round(rotation_y, 6),
        "layer": wall.layer,
        "length": round(length, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main builder
# ─────────────────────────────────────────────────────────────────────────────

class GeometryBuilder:
    def __init__(self, floor_height: float = 0.0):
        self.floor_height = floor_height

    def build(self, walls: list[Wall], bounds: dict | None = None) -> BuildingModel:
        model = BuildingModel()

        # Build wall meshes
        for wall in walls:
            box = wall_to_box(wall)
            if box:
                model.walls.append(box)

        # Build a simple ground floor if we have bounds
        if bounds:
            model.floors.append(self._build_floor(bounds))

        # Metadata
        model.metadata = {
            "wall_count": len(model.walls),
            "floor_count": len(model.floors),
            "total_wall_length": round(sum(w["length"] for w in model.walls), 2),
            "bounds": bounds,
            "format": "floorplan-json-v1",
        }

        return model

    def _build_floor(self, bounds: dict) -> dict:
        minx = bounds.get("minx", 0)
        miny = bounds.get("miny", 0)
        maxx = bounds.get("maxx", 10)
        maxy = bounds.get("maxy", 10)

        cx = (minx + maxx) / 2
        cz = (miny + maxy) / 2
        width = maxx - minx
        depth = maxy - miny

        return {
            "type": "floor",
            "position": {
                "x": round(cx, 4),
                "y": round(self.floor_height, 4),
                "z": round(cz, 4),
            },
            "dimensions": {
                "width": round(width, 4),
                "depth": round(depth, 4),
            },
        }
