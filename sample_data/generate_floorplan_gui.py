"""
Floor Plan Generator GUI

A tkinter application that lets you specify room counts, dimensions,
and style, then generates a realistic randomized DXF floor plan.

Run: python generate_floorplan_gui.py
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import math
import random
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

try:
    import ezdxf
    EZDXF_OK = True
except ImportError:
    EZDXF_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# Floor plan data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Room:
    name: str
    x: float
    y: float
    w: float
    h: float
    room_type: str = "room"   # room, bathroom, kitchen, living, bedroom, hallway

    @property
    def cx(self): return self.x + self.w / 2
    @property
    def cy(self): return self.y + self.h / 2
    @property
    def area(self): return self.w * self.h


@dataclass
class FloorPlan:
    rooms: list[Room] = field(default_factory=list)
    width: float = 0
    height: float = 0
    wall_thickness: float = 0.2


# ─────────────────────────────────────────────────────────────────────────────
# Floor plan generation algorithms
# ─────────────────────────────────────────────────────────────────────────────

# Room size ranges in meters
ROOM_SIZES = {
    "living":   (4.0, 7.0, 3.5, 6.0),   # min_w, max_w, min_h, max_h
    "bedroom":  (3.0, 4.5, 3.0, 4.5),
    "kitchen":  (2.5, 4.5, 2.5, 4.0),
    "bathroom": (1.5, 2.5, 2.0, 3.0),
    "hallway":  (1.0, 2.0, 2.0, 5.0),
    "dining":   (3.0, 4.5, 3.0, 4.5),
    "office":   (2.5, 4.0, 2.5, 4.0),
    "storage":  (1.0, 2.0, 1.5, 2.5),
}


def _rand_size(room_type: str) -> tuple[float, float]:
    sizes = ROOM_SIZES.get(room_type, (2.5, 4.0, 2.5, 4.0))
    w = round(random.uniform(sizes[0], sizes[1]), 1)
    h = round(random.uniform(sizes[2], sizes[3]), 1)
    return w, h


def generate_floor_plan(
    bedrooms: int = 2,
    bathrooms: int = 1,
    has_kitchen: bool = True,
    has_living: bool = True,
    has_dining: bool = False,
    has_office: bool = False,
    has_hallway: bool = True,
    style: str = "rectangular",   # "rectangular", "L-shape", "open-plan"
    seed: Optional[int] = None,
) -> FloorPlan:
    if seed is not None:
        random.seed(seed)

    # Build room list
    room_types = []
    if has_living:   room_types.append("living")
    if has_kitchen:  room_types.append("kitchen")
    if has_dining:   room_types.append("dining")
    if has_office:   room_types.append("office")
    for _ in range(bedrooms):  room_types.append("bedroom")
    for _ in range(bathrooms): room_types.append("bathroom")
    if has_hallway:  room_types.append("hallway")

    if not room_types:
        room_types = ["living", "bedroom", "bathroom"]

    # Layout strategy
    if style == "open-plan":
        return _layout_open_plan(room_types)
    elif style == "L-shape":
        return _layout_l_shape(room_types)
    else:
        return _layout_rectangular(room_types)


def _layout_rectangular(room_types: list[str]) -> FloorPlan:
    """Pack rooms in a roughly rectangular grid layout."""
    rooms = []
    wt = 0.2   # wall thickness

    # Split into columns: public (living/kitchen) and private (bedrooms/bathrooms)
    public   = [r for r in room_types if r in ("living", "kitchen", "dining", "office")]
    private  = [r for r in room_types if r in ("bedroom", "bathroom", "storage")]
    hallways = [r for r in room_types if r == "hallway"]

    # Place public rooms on the left column
    public_rooms, col_w, col_h = _stack_column(public, wt, start_x=0, start_y=0)
    rooms.extend(public_rooms)

    # Hallway in the middle
    hallway_w = 1.2 if hallways else 0
    hx = col_w

    # Place private rooms on the right column
    private_rooms, priv_w, priv_h = _stack_column(private, wt, start_x=col_w + hallway_w, start_y=0)
    rooms.extend(private_rooms)

    total_h = max(col_h, priv_h)

    # Add hallway
    if hallways:
        rooms.append(Room(
            name="Hallway", x=hx, y=0,
            w=hallway_w, h=total_h,
            room_type="hallway"
        ))

    total_w = col_w + hallway_w + priv_w

    plan = FloorPlan(rooms=rooms, width=round(total_w, 2), height=round(total_h, 2))
    return plan


def _stack_column(
    room_types: list[str],
    wt: float,
    start_x: float,
    start_y: float,
) -> tuple[list[Room], float, float]:
    """Stack rooms vertically in a column."""
    rooms = []
    if not room_types:
        return rooms, 0, 0

    max_w = 0
    cy = start_y

    for i, rt in enumerate(room_types):
        w, h = _rand_size(rt)
        name = _room_name(rt, room_types[:i].count(rt) + 1)
        rooms.append(Room(name=name, x=start_x, y=cy, w=w, h=h, room_type=rt))
        max_w = max(max_w, w)
        cy += h

    # Normalize widths in column to max_w
    for r in rooms:
        r.w = max_w

    return rooms, max_w, cy - start_y


def _layout_l_shape(room_types: list[str]) -> FloorPlan:
    """Generate an L-shaped floor plan."""
    plan = _layout_rectangular(room_types)

    # Remove bottom-right corner rooms to create L shape
    if len(plan.rooms) > 3:
        # Find rooms in bottom-right quadrant and remove ~1/4 of them
        mid_x = plan.width / 2
        mid_y = plan.height / 2
        bottom_right = [r for r in plan.rooms if r.cx > mid_x and r.cy < mid_y]
        for r in bottom_right[:len(bottom_right)//2]:
            plan.rooms.remove(r)

    plan.width = max((r.x + r.w) for r in plan.rooms) if plan.rooms else plan.width
    plan.height = max((r.y + r.h) for r in plan.rooms) if plan.rooms else plan.height
    return plan


def _layout_open_plan(room_types: list[str]) -> FloorPlan:
    """Open-plan: living/kitchen/dining as one large space, bedrooms separate."""
    open_types = [r for r in room_types if r in ("living", "kitchen", "dining")]
    private = [r for r in room_types if r not in ("living", "kitchen", "dining", "hallway")]
    hallways = [r for r in room_types if r == "hallway"]

    rooms = []

    # Big open space
    open_w = round(random.uniform(6, 10), 1)
    open_h = round(random.uniform(4, 7), 1)
    label = " / ".join(t.capitalize() for t in open_types) if open_types else "Open Plan"
    rooms.append(Room(name=label, x=0, y=0, w=open_w, h=open_h, room_type="living"))

    # Stack private rooms to the right
    priv_rooms, priv_w, priv_h = _stack_column(private, 0.2, start_x=open_w + (1.2 if hallways else 0), start_y=0)
    rooms.extend(priv_rooms)

    if hallways:
        rooms.append(Room(name="Hallway", x=open_w, y=0, w=1.2, h=max(open_h, priv_h), room_type="hallway"))

    total_w = open_w + (1.2 if hallways else 0) + priv_w
    total_h = max(open_h, priv_h)
    return FloorPlan(rooms=rooms, width=round(total_w, 2), height=round(total_h, 2))


def _room_name(room_type: str, count: int) -> str:
    names = {
        "living": "Living Room", "kitchen": "Kitchen", "dining": "Dining Room",
        "bedroom": f"Bedroom {count}", "bathroom": f"Bathroom {count}" if count > 1 else "Bathroom",
        "hallway": "Hallway", "office": "Office", "storage": "Storage",
    }
    return names.get(room_type, room_type.capitalize())


# ─────────────────────────────────────────────────────────────────────────────
# DXF export — with wall deduplication
#
# Rooms are placed edge-to-edge, so adjacent rooms share a wall.
# We collect every wall edge from every room, then deduplicate before drawing.
# Each unique wall is drawn exactly once as a double-line pair.
# ─────────────────────────────────────────────────────────────────────────────

def _wall_key(x1, y1, x2, y2, tol=0.01):
    """
    Canonical key for a wall segment — normalised so the lower-coordinate
    endpoint always comes first, rounded to avoid float noise.
    Two walls that are the same edge (even if drawn in opposite directions)
    produce the same key and are deduplicated.
    """
    r = lambda v: round(v / tol) * tol
    p1 = (r(x1), r(y1))
    p2 = (r(x2), r(y2))
    return (min(p1, p2), max(p1, p2))


def export_to_dxf(plan: FloorPlan, filepath: str, wall_thickness: float = 0.2):
    if not EZDXF_OK:
        raise ImportError("ezdxf required: pip install ezdxf")

    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 6   # meters

    msp = doc.modelspace()
    doc.layers.add("WALL",   color=7)
    doc.layers.add("ROOM",   color=3)
    doc.layers.add("TEXT",   color=2)
    doc.layers.add("DOOR",   color=1)
    doc.layers.add("WINDOW", color=5)

    wt  = wall_thickness
    wh  = {"layer": "WALL"}
    dh  = {"layer": "DOOR"}
    winh = {"layer": "WINDOW"}

    # ── Step 1: collect all wall edges from all rooms ──────────────────────
    # Each room contributes 4 edges. Adjacent rooms share edges exactly.
    # We store: key → (x1,y1,x2,y2) to deduplicate.
    wall_edges: dict = {}   # key → (x1,y1,x2,y2)

    for room in plan.rooms:
        rx, ry, rw, rh = room.x, room.y, room.w, room.h
        edges = [
            (rx,        ry,        rx + rw, ry),          # bottom
            (rx + rw,   ry,        rx + rw, ry + rh),     # right
            (rx,        ry + rh,   rx + rw, ry + rh),     # top
            (rx,        ry,        rx,      ry + rh),      # left
        ]
        for x1, y1, x2, y2 in edges:
            k = _wall_key(x1, y1, x2, y2)
            wall_edges[k] = (x1, y1, x2, y2)

    # ── Step 2: draw each unique wall as a single centerline ──────────────
    # Single-line export is simpler and works correctly with the detector's
    # single-line fallback mode. Double-line pairs break when rooms share walls
    # because deduplication produces single lines that can't be paired.
    for x1, y1, x2, y2 in wall_edges.values():
        length = math.hypot(x2 - x1, y2 - y1)
        if length > 1e-6:
            msp.add_line((x1, y1), (x2, y2), dxfattribs=wh)

    # ── Step 3: build edge ownership map for door/window placement ──────────
    # We need to know which walls are exterior (count=1) vs interior shared (count=2)
    # so we can correctly place doors on shared walls and windows on exterior walls.
    def _edge_key(x1, y1, x2, y2, tol=0.01):
        r = lambda v: round(v / tol) * tol
        p1 = (r(x1), r(y1)); p2 = (r(x2), r(y2))
        return (min(p1, p2), max(p1, p2))

    def _room_edges(room):
        rx, ry, rw, rh = room.x, room.y, room.w, room.h
        return {
            'bottom': (rx, ry, rx + rw, ry),
            'top':    (rx, ry + rh, rx + rw, ry + rh),
            'left':   (rx, ry, rx, ry + rh),
            'right':  (rx + rw, ry, rx + rw, ry + rh),
        }

    edge_count: dict = {}
    for room in plan.rooms:
        for _, (x1, y1, x2, y2) in _room_edges(room).items():
            k = _edge_key(x1, y1, x2, y2)
            edge_count[k] = edge_count.get(k, 0) + 1

    exterior_edges = {k for k, v in edge_count.items() if v == 1}
    interior_edges = {k for k, v in edge_count.items() if v == 2}

    # ── Step 4: annotations (labels, doors, windows) per room ────────────────
    door_w = 0.9   # standard door width (m)
    win_w  = 1.2   # standard window width (m)

    for room in plan.rooms:
        rx, ry, rw, rh = room.x, room.y, room.w, room.h
        edges = _room_edges(room)

        # Room name label
        msp.add_text(
            room.name,
            dxfattribs={
                "layer": "TEXT", "height": 0.25,
                "insert": (room.cx - len(room.name) * 0.07, room.cy),
            }
        )

        # Area label
        msp.add_text(
            f"{room.area:.1f}m²",
            dxfattribs={
                "layer": "TEXT", "height": 0.18,
                "insert": (room.cx - 0.3, room.cy - 0.35),
            }
        )

        # ── Door: one per room on a shared INTERIOR wall ─────────────────────
        # Priority: shared wall with hallway > any shared wall
        # Hallway itself gets no door (it IS the circulation space)
        if room.room_type != "hallway":
            door_placed = False
            # Try interior walls first (correct — opens between rooms/hallway)
            for side, (x1, y1, x2, y2) in edges.items():
                k = _edge_key(x1, y1, x2, y2)
                if k not in interior_edges:
                    continue
                L = math.hypot(x2 - x1, y2 - y1)
                if L < door_w + 0.3:
                    continue

                # Place door at 30% along the shared wall from the lower end
                t = 0.3
                hinge_x = x1 + t * (x2 - x1)
                hinge_y = y1 + t * (y2 - y1)

                # Door leaf: from hinge to hinge + door_w along wall direction
                dx_n = (x2 - x1) / L
                dy_n = (y2 - y1) / L
                leaf_ex = hinge_x + door_w * dx_n
                leaf_ey = hinge_y + door_w * dy_n

                # Door swing arc: quarter circle sweeping INTO the room
                # Determine sweep direction: perpendicular pointing inward
                # Inward normal: rotate wall direction 90° toward room center
                perp_x = -dy_n
                perp_y =  dx_n
                room_mid_x = rx + rw / 2
                room_mid_y = ry + rh / 2
                to_room_x = room_mid_x - hinge_x
                to_room_y = room_mid_y - hinge_y
                # If perp doesn't point toward room, flip it
                if perp_x * to_room_x + perp_y * to_room_y < 0:
                    perp_x, perp_y = -perp_x, -perp_y

                # Arc start angle: direction of door leaf (along wall)
                start_angle = math.degrees(math.atan2(dy_n, dx_n))
                # Arc end angle: 90° swept into room
                end_angle = math.degrees(math.atan2(perp_y, perp_x))
                # Ensure positive sweep (always draw arc going counter-clockwise)
                if end_angle < start_angle:
                    end_angle += 360

                # Draw door leaf line
                msp.add_line((hinge_x, hinge_y), (leaf_ex, leaf_ey), dxfattribs=dh)
                # Draw swing arc
                msp.add_arc(
                    center=(hinge_x, hinge_y),
                    radius=door_w,
                    start_angle=start_angle,
                    end_angle=end_angle,
                    dxfattribs=dh,
                )
                door_placed = True
                break

            # Fallback: place door on any wall wide enough (for rooms with no shared walls)
            if not door_placed:
                for side in ('bottom', 'left', 'right', 'top'):
                    x1, y1, x2, y2 = edges[side]
                    L = math.hypot(x2 - x1, y2 - y1)
                    if L < door_w + 0.3:
                        continue
                    t = 0.3
                    hinge_x = x1 + t * (x2 - x1)
                    hinge_y = y1 + t * (y2 - y1)
                    dx_n = (x2 - x1) / L; dy_n = (y2 - y1) / L
                    leaf_ex = hinge_x + door_w * dx_n
                    leaf_ey = hinge_y + door_w * dy_n
                    start_angle = math.degrees(math.atan2(dy_n, dx_n))
                    end_angle   = start_angle + 90
                    msp.add_line((hinge_x, hinge_y), (leaf_ex, leaf_ey), dxfattribs=dh)
                    msp.add_arc(center=(hinge_x, hinge_y), radius=door_w,
                                start_angle=start_angle, end_angle=end_angle, dxfattribs=dh)
                    break

        # ── Window: one per room on an EXTERIOR wall ─────────────────────────
        # Skip rooms that don't typically have windows
        if room.room_type not in ("hallway", "storage"):
            for side in ('bottom', 'top', 'left', 'right'):
                x1, y1, x2, y2 = edges[side]
                k = _edge_key(x1, y1, x2, y2)
                if k not in exterior_edges:
                    continue
                L = math.hypot(x2 - x1, y2 - y1)
                if L < win_w + 0.4:
                    continue
                # Centre window on the wall
                ww = min(win_w, L * 0.45)
                t_mid = 0.5
                t0 = t_mid - ww / (2 * L)
                t1 = t_mid + ww / (2 * L)
                wx0 = x1 + t0 * (x2 - x1); wy0 = y1 + t0 * (y2 - y1)
                wx1 = x1 + t1 * (x2 - x1); wy1 = y1 + t1 * (y2 - y1)
                msp.add_line((wx0, wy0), (wx1, wy1), dxfattribs=winh)
                break

    # Title block
    msp.add_text(
        f"Floor Plan — {len(plan.rooms)} Rooms  |  {plan.width:.1f}m × {plan.height:.1f}m",
        dxfattribs={"layer": "TEXT", "height": 0.35, "insert": (0, -1.0)}
    )

    doc.saveas(filepath)


# ─────────────────────────────────────────────────────────────────────────────
# Preview canvas (simple 2D top-down drawing)
# ─────────────────────────────────────────────────────────────────────────────

ROOM_COLORS = {
    "living":   "#2a4a6b",
    "kitchen":  "#4a6b2a",
    "dining":   "#3a5a4a",
    "bedroom":  "#4a2a6b",
    "bathroom": "#2a5a6b",
    "hallway":  "#3a3a4a",
    "office":   "#5a4a2a",
    "storage":  "#3a3a3a",
}


def draw_plan_on_canvas(canvas: tk.Canvas, plan: FloorPlan, padding: int = 30):
    canvas.delete("all")
    if not plan.rooms:
        return

    cw = int(canvas["width"])
    ch = int(canvas["height"])

    available_w = cw - padding * 2
    available_h = ch - padding * 2

    scale = min(available_w / max(plan.width, 0.1), available_h / max(plan.height, 0.1))

    def tx(x): return padding + x * scale
    def ty(y): return ch - padding - y * scale  # flip Y

    # Background grid
    grid_step = 1.0   # 1 meter
    gx = 0
    while gx <= plan.width + 1:
        x = tx(gx)
        canvas.create_line(x, padding, x, ch - padding, fill="#1a1a2a", width=1)
        gx += grid_step
    gy = 0
    while gy <= plan.height + 1:
        y = ty(gy)
        canvas.create_line(padding, y, cw - padding, y, fill="#1a1a2a", width=1)
        gy += grid_step

    # Rooms
    for room in plan.rooms:
        x1, y1 = tx(room.x), ty(room.y + room.h)
        x2, y2 = tx(room.x + room.w), ty(room.y)
        color = ROOM_COLORS.get(room.room_type, "#2a3a4a")

        canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="#00e5ff", width=2)

        # Room name
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        canvas.create_text(cx, cy - 8, text=room.name, fill="#e8e8f0",
                           font=("Consolas", 9, "bold"), anchor="center")
        canvas.create_text(cx, cy + 8, text=f"{room.area:.1f}m²", fill="#00e5ff",
                           font=("Consolas", 8), anchor="center")

    # Dimensions
    canvas.create_text(
        cw // 2, ch - 8,
        text=f"↔ {plan.width:.1f}m",
        fill="#666680", font=("Consolas", 9)
    )
    canvas.create_text(
        8, ch // 2,
        text=f"↕ {plan.height:.1f}m",
        fill="#666680", font=("Consolas", 9), angle=90
    )


# ─────────────────────────────────────────────────────────────────────────────
# GUI Application
# ─────────────────────────────────────────────────────────────────────────────

class FloorPlanGeneratorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Floor Plan Generator")
        self.root.configure(bg="#0a0a0f")
        self.root.resizable(True, True)

        self.current_plan: Optional[FloorPlan] = None
        self._setup_styles()
        self._build_ui()

        # Generate on startup
        self.root.after(100, self.generate)

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(".", background="#0a0a0f", foreground="#e8e8f0",
                        fieldbackground="#1a1a24", selectbackground="#00e5ff",
                        selectforeground="#000000", font=("Consolas", 10))
        style.configure("TFrame",      background="#0a0a0f")
        style.configure("TLabelframe", background="#111118", foreground="#00e5ff",
                        bordercolor="#2a2a38", font=("Consolas", 10, "bold"))
        style.configure("TLabelframe.Label", background="#111118", foreground="#00e5ff")
        style.configure("TLabel",  background="#111118", foreground="#e8e8f0")
        style.configure("TButton", background="#00e5ff", foreground="#000000",
                        font=("Consolas", 10, "bold"), borderwidth=0, focuscolor="none")
        style.map("TButton",
                  background=[("active", "#33eeff"), ("pressed", "#0099cc")])
        style.configure("Secondary.TButton", background="#2a2a38", foreground="#e8e8f0")
        style.map("Secondary.TButton", background=[("active", "#3a3a4a")])
        style.configure("TCombobox", fieldbackground="#1a1a24", foreground="#e8e8f0",
                        selectbackground="#2a2a38", selectforeground="#00e5ff")
        style.configure("TCheckbutton", background="#111118", foreground="#e8e8f0")
        style.map("TCheckbutton", background=[("active", "#111118")])
        style.configure("TScale", background="#111118", troughcolor="#2a2a38",
                        slidercolor="#00e5ff")
        style.configure("TSpinbox", fieldbackground="#1a1a24", foreground="#e8e8f0")

    def _build_ui(self):
        self.root.geometry("1100x720")
        main = ttk.Frame(self.root, padding=0)
        main.pack(fill="both", expand=True)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        # ── Left panel ───────────────────────────────────────────────────────
        left = tk.Frame(main, bg="#111118", width=320)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        # Header
        hdr = tk.Frame(left, bg="#0a0a0f", height=56)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="FLOOR", bg="#0a0a0f", fg="#e8e8f0",
                 font=("Consolas", 18, "bold")).pack(side="left", padx=(16, 0), pady=12)
        tk.Label(hdr, text="PLAN", bg="#0a0a0f", fg="#00e5ff",
                 font=("Consolas", 18, "bold")).pack(side="left")
        tk.Label(hdr, text="GEN", bg="#0a0a0f", fg="#666680",
                 font=("Consolas", 10)).pack(side="left", padx=(4, 0), pady=16)

        # Scrollable content
        canvas_scroll = tk.Canvas(left, bg="#111118", highlightthickness=0)
        scrollbar = ttk.Scrollbar(left, orient="vertical", command=canvas_scroll.yview)
        canvas_scroll.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas_scroll.pack(side="left", fill="both", expand=True)

        content = tk.Frame(canvas_scroll, bg="#111118")
        content_window = canvas_scroll.create_window((0, 0), window=content, anchor="nw")

        def on_frame_configure(e):
            canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all"))
        def on_canvas_configure(e):
            canvas_scroll.itemconfig(content_window, width=e.width)

        content.bind("<Configure>", on_frame_configure)
        canvas_scroll.bind("<Configure>", on_canvas_configure)

        def _on_mousewheel(e):
            canvas_scroll.yview_scroll(int(-1 * (e.delta / 120)), "units")
        canvas_scroll.bind_all("<MouseWheel>", _on_mousewheel)

        pad = {"padx": 12, "pady": 4}

        # ── Room counts ──────────────────────────────────────────────────────
        self._section(content, "ROOMS")

        self.bedrooms_var = tk.IntVar(value=2)
        self._spinner(content, "Bedrooms", self.bedrooms_var, 0, 6, **pad)

        self.bathrooms_var = tk.IntVar(value=1)
        self._spinner(content, "Bathrooms", self.bathrooms_var, 0, 4, **pad)

        # ── Optional rooms ───────────────────────────────────────────────────
        self._section(content, "OPTIONAL ROOMS")

        self.living_var    = tk.BooleanVar(value=True)
        self.kitchen_var   = tk.BooleanVar(value=True)
        self.dining_var    = tk.BooleanVar(value=False)
        self.office_var    = tk.BooleanVar(value=False)
        self.hallway_var   = tk.BooleanVar(value=True)

        for label, var in [
            ("Living Room",  self.living_var),
            ("Kitchen",      self.kitchen_var),
            ("Dining Room",  self.dining_var),
            ("Home Office",  self.office_var),
            ("Hallway",      self.hallway_var),
        ]:
            cb = tk.Checkbutton(content, text=label, variable=var,
                                bg="#111118", fg="#e8e8f0", activebackground="#111118",
                                activeforeground="#00e5ff", selectcolor="#0a0a0f",
                                font=("Consolas", 10), cursor="hand2")
            cb.pack(anchor="w", **pad)

        # ── Style ────────────────────────────────────────────────────────────
        self._section(content, "LAYOUT STYLE")

        self.style_var = tk.StringVar(value="rectangular")
        style_frame = tk.Frame(content, bg="#111118")
        style_frame.pack(fill="x", **pad)
        for val, label in [("rectangular", "Rectangular"), ("L-shape", "L-Shape"), ("open-plan", "Open Plan")]:
            rb = tk.Radiobutton(style_frame, text=label, variable=self.style_var, value=val,
                                bg="#111118", fg="#e8e8f0", activebackground="#111118",
                                activeforeground="#00e5ff", selectcolor="#0a0a0f",
                                font=("Consolas", 10), cursor="hand2")
            rb.pack(anchor="w")

        # ── Wall settings ────────────────────────────────────────────────────
        self._section(content, "WALL SETTINGS")

        self.wall_thickness_var = tk.DoubleVar(value=0.2)
        self._slider(content, "Wall Thickness (m)", self.wall_thickness_var, 0.1, 0.5, **pad)

        # ── Seed ─────────────────────────────────────────────────────────────
        self._section(content, "RANDOMIZATION")

        seed_frame = tk.Frame(content, bg="#111118")
        seed_frame.pack(fill="x", **pad)
        tk.Label(seed_frame, text="Seed (blank = random)", bg="#111118",
                 fg="#666680", font=("Consolas", 9)).pack(anchor="w")

        self.seed_var = tk.StringVar(value="")
        seed_entry = tk.Entry(seed_frame, textvariable=self.seed_var,
                              bg="#1a1a24", fg="#e8e8f0", insertbackground="#00e5ff",
                              relief="flat", font=("Consolas", 11), bd=6)
        seed_entry.pack(fill="x", pady=(4, 0))

        # ── Buttons ──────────────────────────────────────────────────────────
        btn_frame = tk.Frame(content, bg="#111118")
        btn_frame.pack(fill="x", padx=12, pady=16)

        tk.Button(btn_frame, text="⟳  Generate", command=self.generate,
                  bg="#00e5ff", fg="#000000", font=("Consolas", 11, "bold"),
                  relief="flat", cursor="hand2", pady=10).pack(fill="x", pady=(0, 6))

        tk.Button(btn_frame, text="💾  Export DXF", command=self.export_dxf,
                  bg="#2a2a38", fg="#e8e8f0", font=("Consolas", 10),
                  relief="flat", cursor="hand2", pady=8).pack(fill="x", pady=(0, 4))

        tk.Button(btn_frame, text="📋  Copy Stats", command=self.copy_stats,
                  bg="#2a2a38", fg="#666680", font=("Consolas", 9),
                  relief="flat", cursor="hand2", pady=6).pack(fill="x")

        # ── Stats display ────────────────────────────────────────────────────
        self._section(content, "PLAN STATS")
        self.stats_label = tk.Label(content, text="Generate a plan to see stats",
                                    bg="#111118", fg="#666680", font=("Consolas", 9),
                                    justify="left", wraplength=260)
        self.stats_label.pack(anchor="w", padx=12, pady=4)

        # ── Right panel (preview) ────────────────────────────────────────────
        right = tk.Frame(main, bg="#0a0a0f")
        right.pack(side="left", fill="both", expand=True)

        preview_hdr = tk.Frame(right, bg="#111118", height=40)
        preview_hdr.pack(fill="x")
        preview_hdr.pack_propagate(False)
        tk.Label(preview_hdr, text="2D PREVIEW", bg="#111118", fg="#666680",
                 font=("Consolas", 9, "bold")).pack(side="left", padx=16, pady=10)
        self.plan_info = tk.Label(preview_hdr, text="", bg="#111118", fg="#00e5ff",
                                  font=("Consolas", 9))
        self.plan_info.pack(side="right", padx=16)

        self.preview_canvas = tk.Canvas(right, bg="#0a0a0f", highlightthickness=0)
        self.preview_canvas.pack(fill="both", expand=True, padx=2, pady=2)
        self.preview_canvas.bind("<Configure>", lambda e: self._redraw_preview())

    def _section(self, parent, title):
        tk.Label(parent, text=title, bg="#0a0a0f", fg="#00e5ff",
                 font=("Consolas", 8, "bold")).pack(
            fill="x", padx=0, pady=(12, 2), ipady=4)

    def _spinner(self, parent, label, var, min_val, max_val, **kwargs):
        f = tk.Frame(parent, bg="#111118")
        f.pack(fill="x", **kwargs)
        tk.Label(f, text=label, bg="#111118", fg="#e8e8f0",
                 font=("Consolas", 10), width=14, anchor="w").pack(side="left")
        spin = tk.Spinbox(f, from_=min_val, to=max_val, textvariable=var, width=4,
                          bg="#1a1a24", fg="#00e5ff", insertbackground="#00e5ff",
                          relief="flat", font=("Consolas", 11, "bold"),
                          buttonbackground="#2a2a38", bd=4)
        spin.pack(side="left")

    def _slider(self, parent, label, var, min_val, max_val, **kwargs):
        f = tk.Frame(parent, bg="#111118")
        f.pack(fill="x", **kwargs)
        tk.Label(f, text=label, bg="#111118", fg="#e8e8f0",
                 font=("Consolas", 9), anchor="w").pack(anchor="w")
        row = tk.Frame(f, bg="#111118")
        row.pack(fill="x")
        sl = ttk.Scale(row, from_=min_val, to=max_val, variable=var, orient="horizontal")
        sl.pack(side="left", fill="x", expand=True)
        val_label = tk.Label(row, textvariable=var, bg="#111118", fg="#00e5ff",
                             font=("Consolas", 9), width=5)
        val_label.pack(side="left")
        # Format to 2dp
        var.trace_add("write", lambda *_: val_label.config(
            text=f"{var.get():.2f}"))

    # ── Actions ───────────────────────────────────────────────────────────────

    def generate(self):
        seed_str = self.seed_var.get().strip()
        seed = int(seed_str) if seed_str.isdigit() else None

        plan = generate_floor_plan(
            bedrooms=self.bedrooms_var.get(),
            bathrooms=self.bathrooms_var.get(),
            has_kitchen=self.kitchen_var.get(),
            has_living=self.living_var.get(),
            has_dining=self.dining_var.get(),
            has_office=self.office_var.get(),
            has_hallway=self.hallway_var.get(),
            style=self.style_var.get(),
            seed=seed,
        )

        self.current_plan = plan
        self._update_stats(plan)
        self._redraw_preview()

    def _redraw_preview(self):
        if self.current_plan:
            draw_plan_on_canvas(self.preview_canvas, self.current_plan)

    def _update_stats(self, plan: FloorPlan):
        total_area = sum(r.area for r in plan.rooms)
        room_lines = "\n".join(
            f"  {r.name:<20} {r.w:.1f}×{r.h:.1f}m  ({r.area:.1f}m²)"
            for r in plan.rooms
        )
        stats = (
            f"Total: {plan.width:.1f}m × {plan.height:.1f}m\n"
            f"Floor area: {total_area:.1f}m²\n"
            f"Rooms: {len(plan.rooms)}\n\n"
            f"{room_lines}"
        )
        self.stats_label.config(text=stats, fg="#aaaacc")
        self.plan_info.config(
            text=f"{plan.width:.1f}m × {plan.height:.1f}m  |  {len(plan.rooms)} rooms  |  {total_area:.0f}m²"
        )

    def export_dxf(self):
        if not self.current_plan:
            messagebox.showwarning("No Plan", "Generate a floor plan first.")
            return
        if not EZDXF_OK:
            messagebox.showerror("Missing Dependency", "ezdxf is required.\nRun: pip install ezdxf")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".dxf",
            filetypes=[("DXF files", "*.dxf"), ("All files", "*.*")],
            title="Save Floor Plan DXF",
            initialfile="floor_plan.dxf",
        )
        if not filepath:
            return

        try:
            export_to_dxf(self.current_plan, filepath, self.wall_thickness_var.get())
            messagebox.showinfo("Exported", f"Saved to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Export Failed", str(e))

    def copy_stats(self):
        if not self.current_plan:
            return
        plan = self.current_plan
        lines = [f"Floor Plan — {plan.width:.1f}m × {plan.height:.1f}m"]
        for r in plan.rooms:
            lines.append(f"{r.name}: {r.w:.1f}×{r.h:.1f}m ({r.area:.1f}m²)")
        self.root.clipboard_clear()
        self.root.clipboard_append("\n".join(lines))
        messagebox.showinfo("Copied", "Stats copied to clipboard.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    app = FloorPlanGeneratorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
