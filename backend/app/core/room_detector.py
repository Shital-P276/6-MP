"""
Room Detector — finds enclosed rooms from wall segments using flood-fill.

Algorithm:
  1. Rasterize all wall segments onto a 2D grid (Bresenham line drawing)
  2. Flood-fill each unvisited empty region → one region per room
  3. Discard the exterior region (any region touching the grid border)
  4. Compute centroid, area, bounding box for each room
  5. Infer room type from geometry (aspect ratio + area heuristics)
  6. If DXF TEXT entities are available, use those labels instead of heuristics

Output: list of Room dicts ready to embed in the model JSON and render in Three.js.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
import math


# ── Resolution ────────────────────────────────────────────────────────────────
# 0.05m = 5cm per grid cell. Fine enough to detect 1m hallways (20 cells wide),
# coarse enough to keep the grid small for typical floor plans (< 500×500 cells).
GRID_RES   = 0.05   # meters per cell
GRID_PAD   = 0.6    # padding around bounds (meters)
MIN_CELLS  = 15     # ignore regions smaller than this (noise / wall gaps)


# ── Room type heuristics ──────────────────────────────────────────────────────
# Ordered: most specific first.
ROOM_TYPE_RULES = [
    # (min_area, max_area, min_aspect, max_aspect, type_name)
    (0,    4,    1.0, 99,  "closet"),
    (0,    10,   1.0, 99,  "bathroom"),
    (0,    99,   3.5, 99,  "hallway"),    # very elongated = corridor
    (10,   18,   1.0, 3.5, "bedroom"),
    (18,   30,   1.0, 3.5, "bedroom"),   # could also be kitchen, but bedroom is safer guess
    (30,   99,   1.0, 3.5, "living_room"),
]

TYPE_LABELS = {
    "living_room": "Living Room",
    "bedroom":     "Bedroom",
    "bathroom":    "Bathroom",
    "kitchen":     "Kitchen",
    "hallway":     "Hallway",
    "closet":      "Closet",
    "dining":      "Dining Room",
    "office":      "Office",
    "unknown":     "Room",
}

TYPE_COLORS = {
    "living_room": "#2d6a9f",
    "bedroom":     "#4a3d7a",
    "bathroom":    "#2d7a5a",
    "kitchen":     "#7a5a2d",
    "hallway":     "#3a3a3a",
    "closet":      "#2a4a3a",
    "dining":      "#7a3a2d",
    "office":      "#2d4a7a",
    "unknown":     "#333344",
}


@dataclass
class Room:
    id:         int
    centroid_x: float   # DXF coordinates
    centroid_y: float
    area:       float   # m²
    width:      float
    depth:      float
    room_type:  str
    label:      str
    color:      str
    confidence: float = 0.5

    def to_dict(self) -> dict:
        return {
            "id":         self.id,
            "centroid_x": round(self.centroid_x, 3),
            "centroid_y": round(self.centroid_y, 3),
            "area":       round(self.area, 2),
            "width":      round(self.width, 2),
            "depth":      round(self.depth, 2),
            "room_type":  self.room_type,
            "label":      self.label,
            "color":      self.color,
            "confidence": round(self.confidence, 2),
        }


# ── Main detector ─────────────────────────────────────────────────────────────

class RoomDetector:
    def __init__(self, resolution: float = GRID_RES):
        self.resolution = resolution

    def detect(
        self,
        wall_segments: list,          # list of Segment objects (start/end Point2D)
        text_labels: list | None = None,  # optional list of (x, y, text) from DXF TEXT layer
    ) -> list[Room]:
        if not wall_segments:
            return []

        # ── Build grid ───────────────────────────────────────────────────────
        xs = [p for s in wall_segments for p in (s.start.x, s.end.x)]
        ys = [p for s in wall_segments for p in (s.start.y, s.end.y)]
        self.minx = min(xs) - GRID_PAD
        self.miny = min(ys) - GRID_PAD
        self.maxx = max(xs) + GRID_PAD
        self.maxy = max(ys) + GRID_PAD

        self.W = int((self.maxx - self.minx) / self.resolution) + 2
        self.H = int((self.maxy - self.miny) / self.resolution) + 2

        grid = [[0] * self.W for _ in range(self.H)]

        # ── Rasterize walls ──────────────────────────────────────────────────
        for seg in wall_segments:
            self._draw_line(grid, seg.start.x, seg.start.y, seg.end.x, seg.end.y)

        # ── Flood fill ───────────────────────────────────────────────────────
        visited = [[-1] * self.W for _ in range(self.H)]
        regions = []   # list of cell lists

        for gy in range(self.H):
            for gx in range(self.W):
                if grid[gy][gx] == 0 and visited[gy][gx] == -1:
                    cells = self._flood_fill(grid, visited, gx, gy, len(regions))
                    if len(cells) >= MIN_CELLS:
                        regions.append(cells)

        # ── Discard exterior (touches grid border) ───────────────────────────
        interior = []
        for cells in regions:
            touches_border = any(
                x == 0 or x == self.W - 1 or y == 0 or y == self.H - 1
                for x, y in cells
            )
            if not touches_border:
                interior.append(cells)

        # ── Build Room objects ───────────────────────────────────────────────
        rooms = []
        for i, cells in enumerate(interior):
            room = self._cells_to_room(i, cells)
            rooms.append(room)

        # ── Override labels from DXF TEXT if available ──────────────────────
        if text_labels:
            rooms = self._apply_text_labels(rooms, text_labels)

        # ── Number duplicate type labels (Bedroom 1, Bedroom 2 …) ───────────
        rooms = self._number_duplicates(rooms)

        return rooms

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _w2g(self, x, y):
        """World → grid coordinates."""
        return (
            int((x - self.minx) / self.resolution),
            int((y - self.miny) / self.resolution),
        )

    def _draw_line(self, grid, x1, y1, x2, y2):
        """Bresenham line rasterization."""
        gx1, gy1 = self._w2g(x1, y1)
        gx2, gy2 = self._w2g(x2, y2)
        dx = abs(gx2 - gx1); dy = abs(gy2 - gy1)
        sx = 1 if gx1 < gx2 else -1
        sy = 1 if gy1 < gy2 else -1
        err = dx - dy
        x, y = gx1, gy1
        while True:
            if 0 <= x < self.W and 0 <= y < self.H:
                grid[y][x] = 1
                # Thicken walls by 1 cell in each direction so gaps don't leak
                for ddx, ddy in [(1,0),(-1,0),(0,1),(0,-1)]:
                    nx, ny = x+ddx, y+ddy
                    if 0 <= nx < self.W and 0 <= ny < self.H:
                        grid[ny][nx] = 1
            if x == gx2 and y == gy2:
                break
            e2 = 2 * err
            if e2 > -dy: err -= dy; x += sx
            if e2 <  dx: err += dx; y += sy

    def _flood_fill(self, grid, visited, start_x, start_y, region_id):
        cells = []
        q = deque([(start_x, start_y)])
        while q:
            x, y = q.popleft()
            if x < 0 or x >= self.W or y < 0 or y >= self.H:
                continue
            if grid[y][x] == 1 or visited[y][x] != -1:
                continue
            visited[y][x] = region_id
            cells.append((x, y))
            q.append((x+1, y)); q.append((x-1, y))
            q.append((x, y+1)); q.append((x, y-1))
        return cells

    def _cells_to_room(self, room_id: int, cells: list) -> Room:
        r = self.resolution
        world_xs = [gx * r + self.minx for gx, gy in cells]
        world_ys = [gy * r + self.miny for gx, gy in cells]

        cx = sum(world_xs) / len(world_xs)
        cy = sum(world_ys) / len(world_ys)
        area = len(cells) * r * r
        width = max(world_xs) - min(world_xs)
        depth = max(world_ys) - min(world_ys)

        room_type, confidence = self._infer_type(area, width, depth)
        label = TYPE_LABELS.get(room_type, "Room")
        color = TYPE_COLORS.get(room_type, "#333344")

        return Room(
            id=room_id, centroid_x=cx, centroid_y=cy,
            area=area, width=width, depth=depth,
            room_type=room_type, label=label,
            color=color, confidence=confidence,
        )

    def _infer_type(self, area: float, width: float, depth: float):
        aspect = max(width, depth) / max(min(width, depth), 0.1)
        for min_a, max_a, min_asp, max_asp, rtype in ROOM_TYPE_RULES:
            if min_a <= area < max_a and min_asp <= aspect < max_asp:
                return rtype, 0.6
        return "unknown", 0.3

    def _apply_text_labels(self, rooms: list[Room], text_labels: list) -> list[Room]:
        """
        Match DXF text entities to rooms by proximity of text position to room centroid.
        Ignores area/dimension annotations (contains 'm²', digits only, etc.)
        """
        # Filter to likely room name texts only
        def is_room_name(text: str) -> bool:
            t = text.strip()
            if not t or len(t) < 2: return False
            if 'm²' in t or 'm2' in t.lower(): return False
            if t.replace('.','').replace(',','').isdigit(): return False
            return True

        name_labels = [(x, y, txt) for x, y, txt in text_labels if is_room_name(txt)]
        if not name_labels:
            return rooms

        used_labels = set()
        for room in rooms:
            best_dist = float('inf')
            best_txt = None
            for lx, ly, txt in name_labels:
                if txt in used_labels:
                    continue
                dist = math.hypot(lx - room.centroid_x, ly - room.centroid_y)
                if dist < best_dist:
                    best_dist = dist
                    best_txt = txt

            # Only apply if the label is within the room's bounding box (roughly)
            max_dim = max(room.width, room.depth)
            if best_txt and best_dist < max_dim * 0.8:
                room.label = best_txt
                room.confidence = min(1.0, room.confidence + 0.3)
                used_labels.add(best_txt)
                # Also update room_type from label if recognisable
                lt = best_txt.lower()
                for kw, rtype in [
                    ("living","living_room"),("lounge","living_room"),
                    ("kitchen","kitchen"),("dining","dining"),
                    ("bedroom","bedroom"),("bed ","bedroom"),
                    ("bathroom","bathroom"),("bath","bathroom"),("wc","bathroom"),
                    ("hallway","hallway"),("hall","hallway"),("corridor","hallway"),
                    ("office","office"),("study","office"),
                    ("closet","closet"),("storage","closet"),("store","closet"),
                ]:
                    if kw in lt:
                        room.room_type = rtype
                        room.color = TYPE_COLORS.get(rtype, room.color)
                        break

        return rooms

    def _number_duplicates(self, rooms: list[Room]) -> list[Room]:
        """Add numbers to duplicate labels: Bedroom → Bedroom 1, Bedroom 2 …"""
        from collections import Counter
        counts = Counter(r.label for r in rooms)
        seen = Counter()
        for room in rooms:
            if counts[room.label] > 1:
                seen[room.label] += 1
                room.label = f"{room.label} {seen[room.label]}"
        return rooms
