"""Room detection from wall graph cycles.

Algorithm:
1. find closed cycles in wall graph
2. each cycle is a room candidate
3. compute polygon + area
4. attach doors/windows that belong to the room boundary/interior
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math

from .door_detector import Door
from .window_detector import Window
from .wall_graph import WallGraph


@dataclass
class Room:
    id: int
    polygon: list[dict]
    area: float
    doors: list[int] = field(default_factory=list)
    windows: list[int] = field(default_factory=list)

    @property
    def centroid_x(self) -> float:
        if not self.polygon:
            return 0.0
        return sum(p["x"] for p in self.polygon) / len(self.polygon)

    @property
    def centroid_y(self) -> float:
        if not self.polygon:
            return 0.0
        return sum(p["y"] for p in self.polygon) / len(self.polygon)

    @property
    def label(self) -> str:
        return f"Room {self.id + 1}"

    @property
    def room_type(self) -> str:
        return "room"

    @property
    def color(self) -> str:
        return "#333344"

    @property
    def confidence(self) -> float:
        return 0.8

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "polygon": self.polygon,
            "area": round(self.area, 3),
            "doors": self.doors,
            "windows": self.windows,
            "centroid": {"x": round(self.centroid_x, 4), "y": round(self.centroid_y, 4)},
        }


class GraphRoomDetector:
    def __init__(self, max_cycle_len: int = 12, min_area: float = 1.0):
        self.max_cycle_len = max_cycle_len
        self.min_area = min_area

    def detect(self, wall_graph: WallGraph, doors: list[Door] | None = None, windows: list[Window] | None = None) -> list[Room]:
        if not wall_graph or not wall_graph.nodes or not wall_graph.edges:
            return []

        doors = doors or []
        windows = windows or []

        adjacency = self._build_node_adjacency(wall_graph)
        cycles = self._find_cycles(adjacency)

        rooms: list[Room] = []
        for i, cyc in enumerate(cycles):
            poly = self._cycle_to_polygon(cyc, wall_graph)
            if len(poly) < 3:
                continue
            area = abs(self._polygon_area(poly))
            if area < self.min_area:
                continue

            room = Room(id=len(rooms), polygon=poly, area=area)
            room.doors = self._doors_in_room(room, doors)
            room.windows = self._windows_in_room(room, windows)
            rooms.append(room)

        return rooms


    def detect_into(self, floorplan):
        floorplan.rooms = self.detect(
            wall_graph=floorplan.wall_graph,
            doors=floorplan.doors,
            windows=floorplan.windows,
        )
        return floorplan
    def _build_node_adjacency(self, wall_graph: WallGraph) -> dict[int, set[int]]:
        g: dict[int, set[int]] = {n.id: set() for n in wall_graph.nodes}
        for e in wall_graph.edges:
            g[e.start_node].add(e.end_node)
            g[e.end_node].add(e.start_node)
        return g

    def _find_cycles(self, g: dict[int, set[int]]) -> list[list[int]]:
        cycles: set[tuple[int, ...]] = set()

        def canonical(cycle: list[int]) -> tuple[int, ...]:
            # cycle includes repeated start at end; drop duplicate for canonicalization
            c = cycle[:-1]
            mins = min(c)
            idxs = [i for i, v in enumerate(c) if v == mins]
            variants = []
            for i in idxs:
                r = c[i:] + c[:i]
                variants.append(tuple(r))
                variants.append(tuple(reversed(r)))
            return min(variants)

        for start in g:
            stack = [(start, [start], {start})]
            while stack:
                node, path, visited = stack.pop()
                if len(path) > self.max_cycle_len:
                    continue
                for nb in g[node]:
                    if nb == start and len(path) >= 3:
                        cyc = path + [start]
                        cycles.add(canonical(cyc))
                    elif nb not in visited and nb >= start:
                        stack.append((nb, path + [nb], visited | {nb}))

        out = []
        for c in cycles:
            out.append(list(c) + [c[0]])
        return out

    def _cycle_to_polygon(self, cycle: list[int], wall_graph: WallGraph) -> list[dict]:
        id_to_node = {n.id: n for n in wall_graph.nodes}
        poly = []
        for nid in cycle[:-1]:
            n = id_to_node.get(nid)
            if n is None:
                continue
            poly.append({"x": round(n.x, 4), "y": round(n.y, 4)})
        return poly

    @staticmethod
    def _polygon_area(poly: list[dict]) -> float:
        a = 0.0
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]["x"], poly[i]["y"]
            x2, y2 = poly[(i + 1) % n]["x"], poly[(i + 1) % n]["y"]
            a += x1 * y2 - x2 * y1
        return 0.5 * a

    def _doors_in_room(self, room: Room, doors: list[Door]) -> list[int]:
        out = []
        for i, d in enumerate(doors):
            if self._point_in_polygon(d.hinge_point.x, d.hinge_point.y, room.polygon):
                out.append(i)
        return out

    def _windows_in_room(self, room: Room, windows: list[Window]) -> list[int]:
        out = []
        for i, w in enumerate(windows):
            mx = (w.start.x + w.end.x) / 2
            my = (w.start.y + w.end.y) / 2
            if self._point_in_polygon(mx, my, room.polygon):
                out.append(i)
        return out

    @staticmethod
    def _point_in_polygon(x: float, y: float, poly: list[dict]) -> bool:
        inside = False
        n = len(poly)
        if n < 3:
            return False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]["x"], poly[i]["y"]
            xj, yj = poly[j]["x"], poly[j]["y"]
            intersect = ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
            )
            if intersect:
                inside = not inside
            j = i
        return inside
