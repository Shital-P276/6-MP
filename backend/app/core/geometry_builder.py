"""
Geometry Builder — Wall objects → Three.js JSON.

Coordinate mapping:
  DXF X  →  Three.js X
  DXF Y  →  Three.js Z
  height →  Three.js Y (up)

rotation_y = +atan2(dy, dx)  [NOT negated — negation caused mirror image]
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math, json
from .wall_detector import Wall

@dataclass
class BuildingModel:
    walls:    list[dict] = field(default_factory=list)
    floors:   list[dict] = field(default_factory=list)
    metadata: dict       = field(default_factory=dict)
    def to_dict(self): return {"metadata":self.metadata,"walls":self.walls,"floors":self.floors}
    def to_json(self, indent=2): return json.dumps(self.to_dict(), indent=indent)

def wall_to_box(wall: Wall):
    length = wall.length
    if length < 1e-6: return None
    thickness = max(wall.thickness, 0.05)
    if thickness > length: length, thickness = thickness, length

    cx = (wall.start.x + wall.end.x) / 2
    cy = (wall.start.y + wall.end.y) / 2
    dx = wall.end.x - wall.start.x
    dy = wall.end.y - wall.start.y

    # Positive atan2 — corrects the mirror image that negative caused
    rotation_y = math.atan2(dy, dx)

    return {
        "type": "box",
        "position": {"x":round(cx,4), "y":round(wall.height/2,4), "z":round(cy,4)},
        "dimensions": {"width":round(length,4), "height":round(wall.height,4), "depth":round(thickness,4)},
        "rotation_y": round(rotation_y, 6),
        "length":    round(length, 4),
        "thickness": round(thickness, 4),
        "layer":     wall.layer,
        "paired":    wall.paired,
        "confidence":wall.confidence,
    }

class GeometryBuilder:
    def build(self, walls, bounds=None):
        model = BuildingModel()
        for w in walls:
            box = wall_to_box(w)
            if box: model.walls.append(box)
        if bounds: model.floors.append(self._floor(bounds))
        model.metadata = {
            "wall_count":        len(model.walls),
            "floor_count":       len(model.floors),
            "total_wall_length": round(sum(w["length"] for w in model.walls), 2),
            "paired_walls":      sum(1 for w in model.walls if w.get("paired")),
            "bounds":            bounds,
            "format":            "floorplan-json-v1",
        }
        return model

    def _floor(self, b):
        return {
            "type":"floor",
            "position":{"x":round((b["minx"]+b["maxx"])/2,4),"y":0.0,"z":round((b["miny"]+b["maxy"])/2,4)},
            "dimensions":{"width":round(b["maxx"]-b["minx"],4),"depth":round(b["maxy"]-b["miny"],4)},
        }
