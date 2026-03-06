# PROJECTMAP — Floor Plan 3D Visualizer

> Complete map of the project: what each file does, what each function does,
> what the current status is, and what still needs work.
> Read alongside LEARNMAP.md and NEXT_STEPS_PLAN.md.

---

## Repository

GitHub: `https://github.com/Shital-P276/6-MP/tree/v2.2`
Stack: FastAPI (Python backend) + Three.js (browser frontend)

---

## File Map

```
project/
├── backend/
│   └── app/
│       ├── main.py                  ← FastAPI routes + API params
│       └── core/
│           ├── dxf_parser.py        ← DXF entity extraction (UNCHANGED)
│           ├── pipeline.py          ← Orchestrates all steps
│           ├── raster_parser.py     ← Image/PDF → ParsedGeometry  ★ MOST CHANGES
│           ├── wall_detector.py     ← Segments → Wall objects      ★ NEEDS FIXES
│           ├── opening_detector.py  ← DXF door/window detection   (UNCHANGED)
│           ├── room_detector.py     ← Room polygon detection       (UNCHANGED)
│           └── geometry_builder.py  ← Wall+Opening → Three.js JSON ★ NEEDS FIX
└── viewer/
    └── index_v2.html                ← Three.js viewer              (STABLE)
```

---

## main.py

**Purpose:** FastAPI entry point. Defines API endpoints.

**Key params (all in the process endpoint):**
```python
scale: float = 0.0           # 0 = auto-detect
wall_height: float = 3.0     # metres
wall_thickness: float = 0.2  # default fallback (overridden by measured)
pixels_per_meter: float = 0.0  # ← CRITICAL: must be 0.0 (auto-detect)
                                #   Was 100.0 — caused all PPM bugs. FIXED.
```

**Status:** ✅ Correct. Do not change.

---

## pipeline.py

**Purpose:** Orchestrates parse → detect → build. Routes by file type.

**Key functions:**
- `run(filepath)` → `PipelineResult`
- `_parse_dxf()` → DXF path
- `_parse_raster()` → raster/PDF path (calls `RasterParser`)
- `_build_raster_openings()` → converts pixel openings to world Opening objects
  - Intra-segment: match to nearest wall by perp distance
  - Inter-segment (t_center==-1.0): freestanding, match by orientation

**Status:** ✅ Correct. Do not change.

**Note:** `simpledraw_parser.py` imports are GONE (reverted). Do not re-add.

---

## raster_parser.py ★

**Purpose:** Converts floor plan image → ParsedGeometry with wall segments,
door/window openings, room polygons.

**Pipeline steps:**
```
1. _autocrop()          Find floor plan panel in UI screenshot
2. _detect_ppm()        Read green tick marks → pixels per metre
   _detect_border_crop() Dynamic green frame inner edge detection  ← NEW
3. _segment()           Classify pixels: WALL(82-148), ROOM(153-244), green masked
4. _skeleton()          Morphological thinning → 1px centerlines
5. _detect_walls()      Hough → snap H/V → merge → dedup → refine centerlines
6. _complete_outer_walls() Add missing outer wall sides from bounding box
   [TODO: _filter_edge_stubs() AFTER this — see NEXT_STEPS_PLAN Problem 1]
7. _detect_ppm() again for PDF case
8. _detect_openings_from_image()
   Pass 1: intra-segment brightness scan + window symbol scan
   Pass 2: inter-segment collinear stub gaps
   Pass 3: T-junction gaps (endpoint near perpendicular wall body)
9. _detect_rooms()      Seal doorways → connected components
```

**Key constants:**
```
BORDER_CROP_PX    = 4       # minimum — overridden by _detect_border_crop()
MERGE_GAP         = 60      # max collinear gap to bridge
MIN_WALL_PX       = 40      # minimum segment length (pixels)
SCAN_HALF_PX      = 14      # brightness sample band (TODO: make dynamic)
OPENING_BRIGHT    = 148     # brightness threshold for gap detection
MIN_OPENING_M     = 0.35    # minimum opening width (metres)
MAX_DOOR_M        = 1.40    # door/window cutoff
MAX_WINDOW_M      = 3.50    # max opening before noise
WALL_HALF_PX      = 20      # T-junction proximity tolerance
EDGE_GUARD        = 20      # image-edge exclusion for T-junction
FALLBACK_PPM      = 65.0    # if tick detection fails
GREEN_DIFF        = 22      # green channel dominance threshold
```

**Known issues to fix (see NEXT_STEPS_PLAN):**
- [ ] Problem 1: Edge stubs becoming phantom doors (add `_filter_edge_stubs`)
- [ ] Problem 2: Thin wall door detection (dynamic SCAN_HALF)
- [ ] Problem 4: Window symbols only on V-walls (add to H-walls)

**Output attached to ParsedGeometry:**
```python
result._raster_openings   # list of opening dicts with pixel coords
result._rooms_px          # list of room dicts with pixel coords
result._ppm               # detected pixels per metre
result._fp_h              # cropped floor plan height in pixels
result._wall_mask         # binary mask of wall pixels (for thickness measurement)
result.metadata_extra     # dict with source, sizes, ppm_source, counts
```

---

## wall_detector.py ★

**Purpose:** Converts Segment list → Wall list with thickness, height, layer.

**Key functions:**
- `WallDetector.detect(geometry)` → `list[Wall]`
  1. Scale inference (DXF only)
  2. Filter short segments (< MIN_WALL_LENGTH=0.1m)
  3. Raster-only: `merge_collinear_fragments()` (bridges Hough gaps)
  4. `pair_double_lines()` for external CAD DXFs (double-edge → one centerline)
  5. Single-line walls from remaining unpaired segments
  6. Raster-only: `_measure_wall_thicknesses()` (reads pixel mask)

- `_measure_wall_thicknesses(walls, wall_mask, ppm, bounds)` → updated walls
  - Samples perpendicular to each wall at 7 points along its length
  - Measures wall pixel width via scan
  - **BUGS (see NEXT_STEPS_PLAN Problem 3):**
    - SCAN_HALF=30 too wide (fix to 15)
    - Span measurement overestimates (fix to consecutive run)
    - Y-flip missing in image-space perpendicular direction
    - No blend for robustness

**Known issues to fix:**
- [ ] Problem 3: All 4 thickness measurement bugs in `_measure_wall_thicknesses`

---

## geometry_builder.py ★

**Purpose:** Converts Wall + Opening + Room objects → Three.js-compatible JSON dicts.

**Key functions:**
- `GeometryBuilder.build(walls, bounds, rooms, openings, wall_height)` → `BuildingModel`
- `wall_to_boxes(wall, openings)` → `(wall_boxes, door_dicts, win_dicts)`
  - Splits wall at opening positions
  - Door: void gap + door leaf box + swing indicator
  - Window: sill stub + header stub + glass panel
- `_floor(bounds)` → floor plane box
- `room_to_label(room)` → room label dict

**What was removed:**
- `_compute_corner_trims()` — REVERTED (trimmed inward, created gaps)

**What needs to be added:**
- [ ] Problem 5: `_compute_wall_extensions()` — extends outward to fill corner gaps

**Constants:**
```
SILL_HEIGHT  = 0.9    # window sill height
WIN_HEIGHT   = 1.2    # window opening height
DOOR_LEAF_T  = 0.05   # door leaf thickness
CORNER_TOL   = 0.35   # max endpoint distance for junction detection
```

---

## opening_detector.py

**Purpose:** DXF-specific door/window detection from arc/line geometry.
**Status:** ✅ Unchanged. Only used for DXF input. Raster uses raster_parser's detection.

---

## room_detector.py

**Purpose:** Finds room polygons from wall segment network.
**Status:** ✅ Unchanged. Works correctly.

---

## dxf_parser.py

**Purpose:** Reads DXF files, extracts LINE/ARC/POLYLINE entities by layer.
**Status:** ✅ Unchanged.

**Key data structures defined here (used everywhere):**
```python
@dataclass
class Point2D: x: float; y: float

@dataclass
class Segment:
    start: Point2D; end: Point2D
    layer: str = "WALL"; source_type: str = "dxf"

@dataclass
class ParsedGeometry:
    wall_segments: list[Segment]
    door_segments: list[Segment]
    window_segments: list[Segment]
    other_segments: list[Segment]
    bounds: dict       # {minx, miny, maxx, maxy}
    units: str
    metadata_extra: dict
    text_labels: list
    # Raster-only attributes (attached after parsing):
    _raster_openings, _rooms_px, _ppm, _fp_h, _wall_mask
```

---

## index_v2.html (viewer)

**Purpose:** Three.js browser viewer. Uploads files, displays 3D model.

**Key behaviors:**
- Sends `&pixels_per_meter=0` in process request ← CRITICAL (was missing, caused PPM=100 bug)
- Accepts: PNG, JPG, JPEG, PDF, DXF
- Renders: wall boxes, door leaves + swing arcs, window sill/glass, room labels
- Controls: orbit camera, wall height slider, scale slider

**Status:** ✅ Stable. Do not change unless viewer feature needed.

---

## Data Flow Detail

```
Image file
    │
    ▼
RasterParser.parse()
    │  wall_mask (binary, wall pixels)
    │  h_walls [(x1,y,x2,y), ...]   ← pixel coordinates
    │  v_walls [(x,y1,x,y2), ...]
    │  img_openings [{"orient","fixed","x_px","y_px","width_m","kind",...}]
    │  _ppm (float)
    │  _fp_h (int, pixel height of cropped floor plan)
    ▼
ParsedGeometry (wall_segments in metres, door/window segments as markers)
    + _raster_openings, _ppm, _fp_h, _wall_mask attached
    │
    ▼
WallDetector.detect()
    → [Wall(start, end, thickness, height, layer, paired, confidence)]
    thickness from _measure_wall_thicknesses() reading wall_mask
    │
    ▼
pipeline._build_raster_openings(img_openings, walls, fp_h, ppm)
    → [Opening(wall_idx, t_center, width, kind, x, y, angle, swing_side)]
    │
    ▼
RoomDetector.detect(wall_segments)
    → [Room(id, label, centroid_x, centroid_y, area, width, depth, ...)]
    │
    ▼
GeometryBuilder.build(walls, bounds, rooms, openings, wall_height)
    → BuildingModel
        .walls   [box dicts]        → Three.js BoxGeometry
        .doors   [door dicts]       → door leaf + swing arc
        .windows [window dicts]     → sill + glass + header
        .rooms   [label dicts]      → text sprites
        .floors  [floor dict]       → PlaneGeometry
        .metadata {counts, bounds}
    │
    ▼
JSON response → viewer/index_v2.html → Three.js scene
```

---

## Current Open Problems (Summary)

| # | Problem | Severity | Files | Status |
|---|---------|----------|-------|--------|
| 1 | Phantom doors from tick marks | Medium | raster_parser.py | Plan written |
| 2 | Thin wall door detection | Medium | raster_parser.py | Plan written |
| 3 | Wall thickness rendering | High | wall_detector.py | Plan written |
| 4 | Window symbols on H-walls only | Low | raster_parser.py | Plan written |
| 5 | Corner half-cut edges | High | geometry_builder.py | Plan written |

---

## Testing Approach

1. Upload `fp4.png` — should show 4 rooms, outer walls thicker than inner
2. Check warning message for PPM (should show ~33 px/m, source="auto")
3. Verify no phantom doors at corners
4. Verify doors at all T-junctions (4-5 doors in fp4)
5. Upload the larger 672x670 screenshot — should still work with different PPM

**Known test images:**
- `fp4.png` (640×355): 4-room L-shaped plan, PPM=33.2, outer walls 9px, inner 6px
- `Door.png` (672×670): annotated floor plan showing door/window identification
- Large screenshot from sessions 1-3: multi-room CAD export, ~65 px/m
