"""
Opening Detector — finds doors and windows in wall segments.

Architectural conventions (from real DXF floor plans):
  DOOR:   LINE (door leaf) + ARC (swing path) on DOOR layer.
          Arc center = hinge point, arc radius = door width (~0.7-1.0m).
          The arc center lies on the wall face.

  WINDOW: LINE or short segment on WINDOW layer.
          Midpoint lies within the wall thickness.
          Length = window width (~0.6-1.5m).

Algorithm:
  1. Parse door arcs → (hinge_x, hinge_y, radius) from ARC entities
     (fall back to door LINE midpoints if no arcs found)
  2. Parse window lines → midpoint + width
  3. For each opening, find the nearest wall within MAX_DIST
  4. Project opening onto wall centerline → get t position [0,1]
  5. Return Opening objects: wall_idx, t_center, width, type
  6. Wall splitter: given a wall + its openings, return the sub-segments to render
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math

MAX_WALL_DIST   = 1.5   # max distance from opening to wall centerline (m)
MIN_DOOR_WIDTH  = 0.5   # smaller arcs are probably noise
MAX_DOOR_WIDTH  = 1.5
MIN_WIN_WIDTH   = 0.3
MAX_WIN_WIDTH   = 3.0
MIN_PIECE_LEN   = 0.15  # don't emit wall pieces shorter than this


@dataclass
class Opening:
    wall_idx:  int
    t_center:  float       # position along wall [0,1]
    width:     float       # opening width in meters
    kind:      str         # "door" | "window"
    x:         float = 0.0 # world position (DXF coords)
    y:         float = 0.0
    angle:     float = 0.0 # wall angle (radians)


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _project(px, py, ax, ay, bx, by):
    """Project P onto segment AB. Returns (t, proj_x, proj_y, dist)."""
    dx, dy = bx - ax, by - ay
    L2 = dx*dx + dy*dy
    if L2 < 1e-9:
        return 0.0, ax, ay, math.hypot(px-ax, py-ay)
    t = max(0.0, min(1.0, ((px-ax)*dx + (py-ay)*dy) / L2))
    qx, qy = ax + t*dx, ay + t*dy
    return t, qx, qy, math.hypot(px-qx, py-qy)


# ── Main detector ─────────────────────────────────────────────────────────────

class OpeningDetector:
    def detect(self, geometry, walls) -> list[Opening]:
        """
        geometry: ParsedGeometry (has door_segments, window_segments)
        walls:    list of Wall objects (from WallDetector)
        Returns:  list of Opening
        """
        openings = []

        # ── Doors ────────────────────────────────────────────────────────────
        door_points = self._extract_door_points(geometry.door_segments)
        for hx, hy, radius in door_points:
            best = self._find_wall(hx, hy, walls)
            if best is None:
                continue
            wi, t, dist = best
            wall = walls[wi]
            wall_len = wall.length
            if wall_len < 1e-6:
                continue
            openings.append(Opening(
                wall_idx=wi, t_center=t,
                width=min(radius, wall_len * 0.9),
                kind="door", x=hx, y=hy,
                angle=math.atan2(wall.end.y - wall.start.y,
                                  wall.end.x - wall.start.x),
            ))

        # ── Windows ──────────────────────────────────────────────────────────
        win_points = self._extract_window_points(geometry.window_segments)
        for mx, my, width in win_points:
            best = self._find_wall(mx, my, walls)
            if best is None:
                continue
            wi, t, dist = best
            wall = walls[wi]
            wall_len = wall.length
            if wall_len < 1e-6:
                continue
            openings.append(Opening(
                wall_idx=wi, t_center=t,
                width=min(width, wall_len * 0.9),
                kind="window", x=mx, y=my,
                angle=math.atan2(wall.end.y - wall.start.y,
                                  wall.end.x - wall.start.x),
            ))

        return openings

    def _extract_door_points(self, door_segs) -> list[tuple]:
        """
        Extract (hinge_x, hinge_y, radius) from door segments.
        Arc segments are approximated polylines — we recover the true center
        using the circumcenter of 3 spread points on the arc.
        """
        arc_segs  = [s for s in door_segs if s.source_type == "ARC"]
        line_segs = [s for s in door_segs if s.source_type != "ARC"]
        results = []

        if arc_segs:
            used = set()
            for i, sa in enumerate(arc_segs):
                if i in used:
                    continue
                cluster = [sa]; cluster_ids = {i}
                for j, sb in enumerate(arc_segs):
                    if j in used or j == i:
                        continue
                    dist = min(
                        math.hypot(sa.start.x-sb.start.x, sa.start.y-sb.start.y),
                        math.hypot(sa.start.x-sb.end.x,   sa.start.y-sb.end.y),
                        math.hypot(sa.end.x-sb.start.x,   sa.end.y-sb.start.y),
                        math.hypot(sa.end.x-sb.end.x,     sa.end.y-sb.end.y),
                    )
                    if dist < 0.3:
                        cluster.append(sb); cluster_ids.add(j)
                used.update(cluster_ids)

                # Collect all points on the arc
                pts = []
                for seg in cluster:
                    pts.append((seg.start.x, seg.start.y))
                pts.append((cluster[-1].end.x, cluster[-1].end.y))

                if len(pts) < 3:
                    continue

                # Recover arc center using circumcenter of 3 spread points
                p1 = pts[0]
                p2 = pts[len(pts)//2]
                p3 = pts[-1]
                center = self._circumcenter(p1, p2, p3)
                if center is None:
                    continue
                hx, hy = center
                radius = math.hypot(pts[0][0]-hx, pts[0][1]-hy)

                if MIN_DOOR_WIDTH <= radius <= MAX_DOOR_WIDTH:
                    results.append((hx, hy, radius))

        if not results and line_segs:
            for seg in line_segs:
                length = math.hypot(seg.end.x-seg.start.x, seg.end.y-seg.start.y)
                if MIN_DOOR_WIDTH <= length <= MAX_DOOR_WIDTH:
                    mx = (seg.start.x + seg.end.x) / 2
                    my = (seg.start.y + seg.end.y) / 2
                    results.append((mx, my, length))

        return results

    @staticmethod
    def _circumcenter(p1, p2, p3):
        """Circumcenter of 3 points — the center of the circle passing through all three."""
        ax, ay = p1; bx, by = p2; cx, cy = p3
        d = 2 * (ax*(by-cy) + bx*(cy-ay) + cx*(ay-by))
        if abs(d) < 1e-9:
            return None
        ux = ((ax**2+ay**2)*(by-cy) + (bx**2+by**2)*(cy-ay) + (cx**2+cy**2)*(ay-by)) / d
        uy = ((ax**2+ay**2)*(cx-bx) + (bx**2+by**2)*(ax-cx) + (cx**2+cy**2)*(bx-ax)) / d
        return ux, uy

    def _extract_window_points(self, win_segs) -> list[tuple]:
        """Extract (midpoint_x, midpoint_y, width) from window segments."""
        results = []
        line_segs = [s for s in win_segs if s.source_type not in ("ARC",)]
        for seg in line_segs:
            length = math.hypot(seg.end.x - seg.start.x,
                                 seg.end.y - seg.start.y)
            if MIN_WIN_WIDTH <= length <= MAX_WIN_WIDTH:
                mx = (seg.start.x + seg.end.x) / 2
                my = (seg.start.y + seg.end.y) / 2
                results.append((mx, my, length))
        return results

    def _find_wall(self, px, py, walls) -> tuple | None:
        """Find nearest wall to point (px, py). Returns (wall_idx, t, dist)."""
        best_wi = None; best_t = 0.0; best_dist = MAX_WALL_DIST

        for wi, wall in enumerate(walls):
            t, _, _, dist = _project(
                px, py,
                wall.start.x, wall.start.y,
                wall.end.x,   wall.end.y,
            )
            if dist < best_dist:
                best_dist = dist; best_t = t; best_wi = wi

        if best_wi is None:
            return None
        return best_wi, best_t, best_dist


# ── Wall splitting ────────────────────────────────────────────────────────────

def split_wall_at_openings(wall, openings_on_wall: list[Opening]) -> list[dict]:
    """
    Given a wall and its openings, return list of wall-piece dicts and opening dicts.
    Each piece has: start_t, end_t, start (Point2D), end (Point2D), is_opening, kind
    """
    L = wall.length
    if L < 1e-6:
        return []

    # Convert openings to (t_start, t_end, kind) intervals, sorted by position
    intervals = []
    for op in openings_on_wall:
        half = (op.width / 2) / L
        ts = max(0.0, op.t_center - half)
        te = min(1.0, op.t_center + half)
        intervals.append((ts, te, op.kind))
    intervals.sort()

    # Merge overlapping intervals
    merged = []
    for ts, te, kind in intervals:
        if merged and ts <= merged[-1][1] + 0.01:
            merged[-1] = (merged[-1][0], max(merged[-1][1], te), merged[-1][2])
        else:
            merged.append([ts, te, kind])

    # Build piece list: alternating solid / opening
    pieces = []
    cursor = 0.0
    for ts, te, kind in merged:
        if ts - cursor > MIN_PIECE_LEN / L:
            pieces.append({"t0": cursor, "t1": ts, "is_opening": False})
        pieces.append({"t0": ts, "t1": te, "is_opening": True, "kind": kind})
        cursor = te
    if 1.0 - cursor > MIN_PIECE_LEN / L:
        pieces.append({"t0": cursor, "t1": 1.0, "is_opening": False})

    # Attach world positions
    for p in pieces:
        sx = wall.start.x + p["t0"] * (wall.end.x - wall.start.x)
        sy = wall.start.y + p["t0"] * (wall.end.y - wall.start.y)
        ex = wall.start.x + p["t1"] * (wall.end.x - wall.start.x)
        ey = wall.start.y + p["t1"] * (wall.end.y - wall.start.y)
        p.update({"sx": sx, "sy": sy, "ex": ex, "ey": ey,
                  "length": math.hypot(ex-sx, ey-sy)})

    return pieces
