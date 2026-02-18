"""
Generates a simple sample floor plan DXF file for testing.

Run: python generate_sample.py
Output: sample_floorplan.dxf
"""

import ezdxf


def create_sample_floorplan():
    doc = ezdxf.new("R2010")

    # Create layers
    msp = doc.modelspace()
    doc.layers.add("WALL", color=7)
    doc.layers.add("DOOR", color=2)
    doc.layers.add("WINDOW", color=5)

    wall_attribs = {"layer": "WALL"}
    door_attribs = {"layer": "DOOR"}
    window_attribs = {"layer": "WINDOW"}

    # ─────────────────────────────────────────────
    # Outer walls (10m x 8m building footprint)
    # Wall thickness represented by two parallel lines 0.2m apart
    # ─────────────────────────────────────────────

    def add_wall_pair(start, end, thickness=0.2):
        """Draw two parallel lines representing a wall."""
        sx, sy = start
        ex, ey = end

        import math
        dx = ex - sx
        dy = ey - sy
        length = math.hypot(dx, dy)
        nx = -dy / length * thickness
        ny = dx / length * thickness

        msp.add_line((sx, sy), (ex, ey), dxfattribs=wall_attribs)
        msp.add_line(
            (sx + nx, sy + ny), (ex + nx, ey + ny), dxfattribs=wall_attribs
        )

    # Outer walls
    add_wall_pair((0, 0), (10, 0))   # bottom
    add_wall_pair((10, 0), (10, 8))  # right
    add_wall_pair((10, 8), (0, 8))   # top
    add_wall_pair((0, 8), (0, 0))    # left

    # Interior wall: vertical divider at x=6 from y=0 to y=8 (splits living/bedroom)
    add_wall_pair((6, 0), (6, 5))

    # Interior wall: horizontal at y=5 from x=6 to x=10 (bathroom/bedroom split)
    add_wall_pair((6, 5), (10, 5))

    # ─────────────────────────────────────────────
    # Doors (represented as arcs on DOOR layer)
    # ─────────────────────────────────────────────

    # Front door at bottom wall around x=2
    msp.add_line((2, 0), (2.9, 0), dxfattribs=door_attribs)  # door gap marker

    # Interior door between living and bedroom at x=6, y=2.5
    msp.add_line((6, 2), (6, 2.9), dxfattribs=door_attribs)

    # Bathroom door at y=5, x=8
    msp.add_line((8, 5), (8.9, 5), dxfattribs=door_attribs)

    # ─────────────────────────────────────────────
    # Windows (short lines on WINDOW layer)
    # ─────────────────────────────────────────────

    # Living room window on bottom wall
    msp.add_line((4, 0), (5.5, 0), dxfattribs=window_attribs)

    # Bedroom window on right wall
    msp.add_line((10, 6), (10, 7.5), dxfattribs=window_attribs)

    # Bathroom window on right wall
    msp.add_line((10, 5.5), (10, 4.5), dxfattribs=window_attribs)

    # ─────────────────────────────────────────────
    # Save
    # ─────────────────────────────────────────────
    output_path = "sample_floorplan.dxf"
    doc.saveas(output_path)
    print(f"✅ Sample floor plan saved to: {output_path}")
    print("   Rooms: Living room (6x8m), Bedroom (4x5m), Bathroom (4x3m)")
    print("   Layers: WALL, DOOR, WINDOW")


if __name__ == "__main__":
    create_sample_floorplan()
