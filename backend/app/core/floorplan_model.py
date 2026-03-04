"""Unified FloorPlan structure shared across detection/render stages."""

from __future__ import annotations
from dataclasses import dataclass, field

from .door_detector import Door
from .opening_detector import Opening
from .wall_detector import Wall
from .wall_graph import WallGraph
from .window_detector import Window


@dataclass
class FloorPlan:
    walls: list[Wall] = field(default_factory=list)
    doors: list[Door] = field(default_factory=list)
    windows: list[Window] = field(default_factory=list)
    rooms: list = field(default_factory=list)
    scale: float = 1.0
    source_type: str = "unknown"
    wall_graph: WallGraph | None = None
    bounds: dict | None = None
    scale_source: str = "heuristic"
    scale_evidence: dict | None = None

    @property
    def door_openings(self) -> list[Opening]:
        openings: list[Opening] = []
        for d in self.doors:
            op = d.to_opening(self.walls)
            if op is not None:
                openings.append(op)
        return openings

    @property
    def window_openings(self) -> list[Opening]:
        openings: list[Opening] = []
        for w in self.windows:
            op = w.to_opening(self.walls)
            if op is not None:
                openings.append(op)
        return openings

    @property
    def openings(self) -> list[Opening]:
        return [*self.door_openings, *self.window_openings]

    def to_dict(self) -> dict:
        return {
            "source_type": self.source_type,
            "scale": self.scale,
            "scale_source": self.scale_source,
            "scale_evidence": self.scale_evidence,
            "walls": [w.to_dict() for w in self.walls],
            "wall_graph": self.wall_graph.to_dict() if self.wall_graph else None,
            "doors": [d.to_dict() for d in self.doors],
            "windows": [w.to_dict() for w in self.windows],
            "rooms": [r.to_dict() if hasattr(r, "to_dict") else r for r in self.rooms],
            "bounds": self.bounds,
        }


# Backward compatibility for earlier refactors.
FloorplanDataModel = FloorPlan
