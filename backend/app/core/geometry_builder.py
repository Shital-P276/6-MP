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
from .dxf_parser import Point2D

SILL_HEIGHT   = 0.9   # window sill height (m)
WIN_HEIGHT    = 1.2   # window opening height (m)
DOOR_LEAF_T   = 0.05  # door leaf thickness (m)

# Corner tolerance: endpoint must be within this perp distance to be considered a junction.
# Uses sum-of-half-thicknesses + raster drift allowance (set in _compute_wall_extensions).
CORNER_RASTER_DRIFT = 0.10   # metres of endpoint drift to tolerate for raster walls

# Wall splitting / room assignment
SNAP_TOL = 0.35   # max perp distance for a T-endpoint to be considered "on" a wall

# Room colour palette — dark base colours, one per room (cycles for >7 rooms)
_ROOM_COLORS = [
    "#1a6b8a",  # teal
    "#8a4f1a",  # amber
    "#2a7a3a",  # green
    "#7a2a7a",  # purple
    "#7a7a1a",  # olive
    "#1a3a8a",  # blue
    "#8a1a2a",  # red
]


# ── Corner extension ──────────────────────────────────────────────────────────

def _compute_wall_extensions(walls):
    """
    For each pair of perpendicular walls meeting at a junction, extend the wall
    endpoints outward so their FACES meet exactly (no protruding half-cut edges).

    L-corner  (endpoint near another wall's endpoint):
        Both walls extend by the other wall's thickness / 2.

    T-junction (endpoint near the MIDDLE of another wall):
        Only the incoming (stub) wall extends by the through-wall's thickness / 2.
        The through-wall is NOT modified — it passes through unbroken.

    Returns a list of (ext_start_m, ext_end_m) per wall, where each value is
    the extra metres to add at that end in the wall's own travel direction.

    Key correctness notes vs old _compute_corner_trims():
    - Extends OUTWARD (longer walls) rather than trimming INWARD (shorter walls)
      → filling the corner void, not creating a gap
    - Tolerance = wa.thickness/2 + wb.thickness/2 + CORNER_RASTER_DRIFT
      → adapts to actual measured thicknesses, handles raster endpoint drift
    - T-junction logic: only stub extends, through-wall is untouched
      → old approach modified both which cut through-walls short
    """
    n = len(walls)
    ext_start = [0.0] * n
    ext_end   = [0.0] * n

    for i, wa in enumerate(walls):
        La = wa.length
        if La < 1e-6:
            continue
        dxa = (wa.end.x - wa.start.x) / La
        dya = (wa.end.y - wa.start.y) / La

        for j, wb in enumerate(walls):
            if i == j:
                continue

            # Only process walls that are roughly perpendicular (75°–105°)
            ang = abs(wa.angle_deg - wb.angle_deg) % 180
            if not (75.0 <= ang <= 105.0):
                continue

            # Tolerance: sum of half-thicknesses + raster drift
            tol = wa.thickness / 2.0 + wb.thickness / 2.0 + CORNER_RASTER_DRIFT

            # Check each endpoint of wb against wa's centerline
            for wb_ep, ext_list_j in (('start', ext_start), ('end', ext_end)):
                pt = wb.start if wb_ep == 'start' else wb.end

                # Project pt onto wa's infinite centerline
                t_raw = ((pt.x - wa.start.x) * dxa + (pt.y - wa.start.y) * dya) / La
                proj_x = wa.start.x + t_raw * La * dxa
                proj_y = wa.start.y + t_raw * La * dya
                perp = math.hypot(pt.x - proj_x, pt.y - proj_y)

                if perp > tol:
                    continue   # too far off-axis — not a real junction

                t_c = max(0.0, min(1.0, t_raw))

                if t_c < 0.08:
                    # Junction near wa's START → L-corner at wa-start / wb-end
                    ext_list_j[j] = max(ext_list_j[j], wa.thickness / 2.0)
                    ext_start[i]  = max(ext_start[i],  wb.thickness / 2.0)

                elif t_c > 0.92:
                    # Junction near wa's END → L-corner at wa-end / wb-end
                    ext_list_j[j] = max(ext_list_j[j], wa.thickness / 2.0)
                    ext_end[i]    = max(ext_end[i],    wb.thickness / 2.0)

                else:
                    # Junction in the middle of wa → T-junction
                    # Only the incoming stub (wb) extends; wa passes through unbroken
                    ext_list_j[j] = max(ext_list_j[j], wa.thickness / 2.0)

    return list(zip(ext_start, ext_end))


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
                                   thick, h, rot_y,
                                   p.get("swing_side", "right")),
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


def _door_leaf(sx, sy, ex, ey, thick, h, rot_y, swing_side='right'):
    """
    Thin door leaf hinged at one end of the opening, swung ~80° open into the room.

    Coordinate system:
      (sx,sy)→(ex,ey) are DXF world coords (Y up).
      Three.js output: x=world_x, y=height, z=-world_y.
      rot_y is -atan2(dy,dx) — the Three.js Y-rotation for a box along this wall.

    We use the wall's own unit vector (ux,uy) to compute hinge and leaf positions
    directly in world space, avoiding trig confusion between wall-angle and rot_y.
    """
    L = math.hypot(ex - sx, ey - sy)
    if L < 1e-6:
        return None

    leaf_h = min(h - 0.05, 2.1)   # standard door height cap

    # Unit vector ALONG wall, and its perpendicular (normal)
    ux = (ex - sx) / L
    uy = (ey - sy) / L
    # Normal: rotate 90° CCW in world XY.
    # 'right' side (walking start→end) = positive normal (+90°) = (-uy, ux)
    # 'left'  side                     = negative normal (-90°) = ( uy,-ux)
    sign = 1.0 if swing_side == 'right' else -1.0
    nx = -uy * sign
    ny =  ux * sign

    # Swing angle ~80° — leaf center is L/2 away from hinge in the swept direction
    # Direction of swung leaf = wall_dir rotated ~80° toward room side
    sweep = math.pi * 0.44   # ~80°
    cos_s = math.cos(sweep)
    sin_s = math.sin(sweep)
    # Rotate (ux,uy) by +sweep toward normal side
    swung_x = ux * cos_s - uy * sin_s * sign
    swung_y = ux * sin_s * sign + uy * cos_s

    # Hinge at the start end of the opening
    hinge_x = sx
    hinge_y = sy

    # Leaf center: hinge + (L/2) in swung direction
    cx_world = hinge_x + (L / 2) * swung_x
    cy_world = hinge_y + (L / 2) * swung_y

    # Leaf rotation in Three.js: -atan2 of swung direction
    leaf_rot_y = -math.atan2(swung_y, swung_x)

    return {
        "position":   {"x": round(cx_world, 4),
                       "y": round(leaf_h * 0.5, 4),
                       "z": round(-cy_world, 4)},   # Three.js z = -world_y
        "dimensions": {"width":  round(L, 4),
                       "height": round(leaf_h, 4),
                       "depth":  round(DOOR_LEAF_T, 4)},
        "rotation_y": round(leaf_rot_y, 6),
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
        "dimensions": {
            "width": round(room.width, 3),
            "depth": round(room.depth, 3),
        },
        "confidence": room.confidence,
    }


# ── Wall splitting at room junctions ─────────────────────────────────────────

def _centerline_intersection_t(wa: Wall, wb: Wall):
    """
    Returns (t_a, t_b): parameters along wa and wb where their INFINITE
    centerlines cross.  Returns None if parallel.

    Does NOT check whether the segments actually reach each other — the caller
    decides that based on t_a / t_b values and wall geometry.
    """
    dx1 = wa.end.x - wa.start.x;  dy1 = wa.end.y - wa.start.y
    dx2 = wb.end.x - wb.start.x;  dy2 = wb.end.y - wb.start.y
    denom = dx1 * dy2 - dy1 * dx2
    if abs(denom) < 1e-9:
        return None   # parallel / collinear
    dx3 = wb.start.x - wa.start.x
    dy3 = wb.start.y - wa.start.y
    t_a = (dx3 * dy2 - dy3 * dx2) / denom
    t_b = (dx3 * dy1 - dy3 * dx1) / denom
    return t_a, t_b


def _split_walls_at_junctions(walls: list) -> list:
    """
    Cut wall A at every point where wall B physically intersects it.

    "Physically intersects" means wall B's body reaches wall A's centerline —
    whether B ends right at A (T-junction), passes through A (+ junction), or
    slightly overlaps A (raster overshoot).

    Criterion: the crossing point on wa (t_a) must be inside wa's extent, AND
    the crossing point must be within wb's physical extent, defined as:
        -wb.thickness/2 - RASTER_DRIFT  ≤  t_b * Lb  ≤  Lb + wb.thickness/2 + RASTER_DRIFT

    This handles:
      • T-junction: inner wall ends exactly at outer wall face (t_b ≈ 0 or 1) ✓
      • + junction: both walls cross each other's midspan ✓
      • Raster overshoot: endpoint is a few px past the centerline ✓
      • Short stubs that don't reach wa: t_b*Lb < 0 → not cut ✓
      • Collinear near-neighbors: angle gate filters these out ✓

    Does NOT hardcode wall thickness thresholds — uses each wall's own measured
    thickness so the logic scales correctly across different floor plans and PPMs.
    """
    RASTER_DRIFT = 0.15   # metres of endpoint slop to tolerate (≈5px @ 33ppm)

    result = []

    for i, wa in enumerate(walls):
        La = wa.length
        if La < 1e-6:
            result.append(wa)
            continue

        dxa = (wa.end.x - wa.start.x) / La
        dya = (wa.end.y - wa.start.y) / La

        cut_ts: list = []

        for j, wb in enumerate(walls):
            if i == j:
                continue

            # Only perpendicular walls create room boundaries
            ang = abs(wa.angle_deg - wb.angle_deg) % 180
            if not (70.0 <= ang <= 110.0):
                continue

            ret = _centerline_intersection_t(wa, wb)
            if ret is None:
                continue
            t_a, t_b = ret
            Lb = wb.length

            # t_a: crossing must be strictly inside wa (not at its own endpoints)
            if t_a <= 0.02 or t_a >= 0.98:
                continue

            # wb must physically REACH the crossing point.
            # t_b * Lb is the distance along wb from its start to the crossing.
            # We allow wb to end anywhere from -(thickness/2 + drift) to Lb+(thickness/2+drift)
            # so that T-junctions (endpoint right at wa's face) are included.
            reach = wb.thickness / 2.0 + RASTER_DRIFT
            if t_b * Lb < -reach or t_b * Lb > Lb + reach:
                continue

            cut_ts.append(round(t_a, 6))

        if not cut_ts:
            result.append(wa)
            continue

        # Sort and deduplicate cuts within 1% of wall length
        cut_ts.sort()
        merged: list = []
        for t in cut_ts:
            if not merged or t - merged[-1] > 0.01:
                merged.append(t)

        boundaries = [0.0] + merged + [1.0]
        for k in range(len(boundaries) - 1):
            t0, t1 = boundaries[k], boundaries[k + 1]
            if (t1 - t0) * La < 0.05:   # skip slivers < 5cm
                continue
            result.append(Wall(
                start=Point2D(wa.start.x + t0 * La * dxa, wa.start.y + t0 * La * dya),
                end=Point2D(wa.start.x + t1 * La * dxa,   wa.start.y + t1 * La * dya),
                thickness=wa.thickness, height=wa.height,
                layer=wa.layer, paired=wa.paired, confidence=wa.confidence,
            ))

    return result


# ── Room ID assignment ────────────────────────────────────────────────────────

def _assign_room_ids(wall_boxes: list[dict], rooms: list) -> list[dict]:
    """
    Tag every wall box dict with room_id for future per-room texture assignment.
    No colour is set here — the viewer uses a single uniform wall colour until
    per-room texturing is explicitly enabled.

    Matching: nearest room centroid in world XZ.
    If no rooms exist, room_id is None on all boxes.
    """
    if not rooms:
        for box in wall_boxes:
            box["room_id"] = None
        return wall_boxes

    room_data = [(r.centroid_x, r.centroid_y, r.id) for r in rooms]

    for box in wall_boxes:
        bx = box["position"]["x"]
        by = -box["position"]["z"]   # Three.js z = -world_y
        best_id   = room_data[0][2]
        best_dist = float("inf")
        for (rx, ry, rid) in room_data:
            d = math.hypot(bx - rx, by - ry)
            if d < best_dist:
                best_dist = d
                best_id   = rid
        box["room_id"] = best_id

    return wall_boxes


# ── Opening re-projection after wall splitting ────────────────────────────────

def _reproj_openings(openings, walls):
    """
    After _split_walls_at_junctions re-indexes walls, two things must be fixed:
      1. wall_idx  — points to the wrong (pre-split) wall index
      2. t_center  — is a fraction of the OLD full-length wall; on the new
                     shorter sub-wall it places the opening at the wrong position

    Fix: for each opening, find the sub-wall whose axis contains op.x/op.y,
    update wall_idx, then recompute t_center as the projection of the opening's
    world position onto that sub-wall's [0..1] range.

    Opening dataclass fields: wall_idx, t_center, width, kind, x, y, angle, swing_side
    NOTE: there is NO t_start/t_end on Opening — split_wall_at_openings derives
    those from t_center ± (width/2)/wall_length internally.

    Freestanding openings (wall_idx == -1) are left untouched.
    """
    if not openings or not walls:
        return openings

    updated = []
    for op in openings:
        if op.wall_idx == -1:
            updated.append(op)
            continue

        best_idx  = op.wall_idx
        best_t    = op.t_center   # fallback: keep original
        best_dist = float("inf")

        for wi, wall in enumerate(walls):
            L = wall.length
            if L < 1e-6:
                continue
            dxa = (wall.end.x - wall.start.x) / L
            dya = (wall.end.y - wall.start.y) / L
            t = ((op.x - wall.start.x) * dxa + (op.y - wall.start.y) * dya) / L
            if not (-0.05 <= t <= 1.05):
                continue
            proj_x = wall.start.x + t * L * dxa
            proj_y = wall.start.y + t * L * dya
            d = math.hypot(op.x - proj_x, op.y - proj_y)
            if d < best_dist:
                best_dist = d
                best_idx  = wi
                best_t    = max(0.0, min(1.0, t))

        from dataclasses import replace as dc_replace
        updated.append(dc_replace(
            op,
            wall_idx = best_idx,
            t_center = round(best_t, 4),
        ))

    return updated


# ── Main builder ──────────────────────────────────────────────────────────────

class GeometryBuilder:
    def build(self, walls, bounds=None, rooms=None,
              openings=None, wall_height: float = 3.0):

        model = BuildingModel()

        # ── Step 1: split walls at room junctions ─────────────────────────────
        # Must happen BEFORE opening grouping so wall indices stay consistent.
        if walls:
            walls = _split_walls_at_junctions(walls)

        # ── Step 2: re-project openings onto the new (split) wall list ────────
        if openings:
            openings = _reproj_openings(openings, walls)

        # ── Step 3: group openings by wall index ───────────────────────────────
        openings_by_wall: dict[int, list[Opening]] = {}
        if openings:
            for op in openings:
                openings_by_wall.setdefault(op.wall_idx, []).append(op)

        # ── Step 4: corner extensions so wall faces meet cleanly ───────────────
        extensions = _compute_wall_extensions(walls) if walls else []

        for wi, wall in enumerate(walls):
            # Apply endpoint extensions for this wall
            ext_s, ext_e = extensions[wi] if extensions else (0.0, 0.0)
            w = wall
            if (ext_s > 0.0 or ext_e > 0.0) and wall.length > ext_s + ext_e + 0.05:
                L  = wall.length
                dx = (wall.end.x - wall.start.x) / L
                dy = (wall.end.y - wall.start.y) / L
                new_start = Point2D(wall.start.x - dx * ext_s,
                                    wall.start.y - dy * ext_s)
                new_end   = Point2D(wall.end.x   + dx * ext_e,
                                    wall.end.y   + dy * ext_e)
                w = Wall(start=new_start, end=new_end,
                         thickness=wall.thickness, height=wall.height,
                         layer=wall.layer, paired=wall.paired,
                         confidence=wall.confidence)

            wall_openings = openings_by_wall.get(wi, [])
            wall_boxes, door_dicts, win_dicts = wall_to_boxes(w, wall_openings)
            model.walls.extend(wall_boxes)
            model.doors.extend(door_dicts)
            model.windows.extend(win_dicts)

        # ── Step 5: tag every wall box with room_id + room_color ──────────────
        _assign_room_ids(model.walls, rooms or [])

        # ── Freestanding inter-segment openings (wall_idx == -1) ─────────────
        # These are openings that live in the gap between two collinear wall stubs.
        # The stubs themselves are already rendered correctly (no overlap).
        # We only need to emit the door leaf / window glass geometry.
        freestanding = [op for op in (openings or []) if op.wall_idx == -1]
        for op in freestanding:
            # Use a representative wall thickness from the nearest wall
            thick = 0.2
            if walls:
                # Find nearest wall to get its thickness
                min_d = float('inf')
                for wall in walls:
                    mid_x = (wall.start.x + wall.end.x) / 2
                    mid_y = (wall.start.y + wall.end.y) / 2
                    d = math.hypot(op.x - mid_x, op.y - mid_y)
                    if d < min_d:
                        min_d = d
                        thick = max(wall.thickness, 0.05)

            rot_y = -op.angle
            cx_dxf = op.x
            cy_dxf = op.y
            h = wall_height

            if op.kind == "door":
                # Hinge at one end of the opening, leaf swung open
                half = op.width / 2
                sx = cx_dxf - half * math.cos(op.angle)
                sy = cy_dxf - half * math.sin(op.angle)
                ex = cx_dxf + half * math.cos(op.angle)
                ey = cy_dxf + half * math.sin(op.angle)
                model.doors.append({
                    "position": {
                        "x": round(cx_dxf, 4),
                        "y": round(h / 2, 4),
                        "z": round(-cy_dxf, 4),
                    },
                    "width":      round(op.width, 4),
                    "height":     round(h, 4),
                    "depth":      round(thick, 4),
                    "rotation_y": round(rot_y, 6),
                    "leaf": _door_leaf(sx, sy, ex, ey, thick, h, rot_y,
                                       getattr(op, 'swing_side', 'right')),
                })
            elif op.kind == "window":
                half = op.width / 2
                sx = cx_dxf - half * math.cos(op.angle)
                sy = cy_dxf - half * math.sin(op.angle)
                ex = cx_dxf + half * math.cos(op.angle)
                ey = cy_dxf + half * math.sin(op.angle)
                model.windows.append({
                    "position": {
                        "x": round(cx_dxf, 4),
                        "y": round(h / 2, 4),
                        "z": round(-cy_dxf, 4),
                    },
                    "width":      round(op.width, 4),
                    "height":     round(h, 4),
                    "depth":      round(thick, 4),
                    "rotation_y": round(rot_y, 6),
                    "sill_h":     SILL_HEIGHT,
                    "win_h":      WIN_HEIGHT,
                    "pieces": _window_pieces(sx, sy, ex, ey, thick, h, rot_y),
                })

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
