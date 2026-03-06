# PROJECTMAP — Floor Plan 3D Visualizer

> Complete map of the project: what each file does, what each function does,
> what the current status is, and what still needs work.
> Read alongside LEARNMAP.md.

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
│           ├── pipeline.py          ← Orchestrates all steps (UNCHANGED)
│           ├── raster_parser.py     ← Image/PDF → ParsedGeometry  ★ MOST CHANGES
│           ├── wall_detector.py     ← Segments → Wall objects      (STABLE)
│           ├── opening_detector.py  ← DXF door/window detection   (UNCHANGED)
│           ├── room_detector.py     ← Room polygon detection       (UNCHANGED)
│           └── geometry_builder.py  ← Wall+Opening → Three.js JSON ★ RECENTLY CHANGED
└── viewer/
    └── index_v2.html                ← Three.js viewer              ★ RECENTLY CHANGED
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
   _detect_border_crop() Dynamic green frame inner edge detection
3. _segment()           Classify pixels: WALL(82-148), ROOM(153-244), green masked
4. _skeleton()          Morphological thinning → 1px centerlines
5. _detect_walls()      Hough → snap H/V → merge → dedup → refine centerlines
6. _complete_outer_walls() Add missing outer wall sides from bounding box
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

**Known issues to fix:**
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

## wall_detector.py

**Purpose:** Converts Segment list → Wall list with thickness, height, layer.

**Status:** ✅ Stable. Thickness blend removed — raw measured value used directly.
Outer walls measure ~0.271m, inner partitions ~0.180m on fp4.png.

**Key functions:**
- `WallDetector.detect(geometry)` → `list[Wall]`
  1. Scale inference (DXF only)
  2. Filter short segments (< MIN_WALL_LENGTH=0.1m)
  3. Raster-only: `merge_collinear_fragments()` (bridges Hough gaps)
  4. `pair_double_lines()` for external CAD DXFs
  5. Single-line walls from remaining unpaired segments
  6. Raster-only: `_measure_wall_thicknesses()` (reads pixel mask)

**Known issues (not yet fixed):**
- [ ] Problem 3: SCAN_HALF=30 too wide; span measurement overestimates; Y-flip missing in perp direction

---

## opening_detector.py

**Purpose:** DXF-specific door/window detection from arc/line geometry.

**Opening dataclass fields (critical — do NOT add fields that don't exist here):**
```python
@dataclass
class Opening:
    wall_idx:   int         # wall index (-1 = freestanding)
    t_center:   float       # position along wall [0,1], or -1.0 for freestanding
    width:      float       # opening width in metres
    kind:       str         # "door" | "window"
    x:          float = 0.0 # world position
    y:          float = 0.0
    angle:      float = 0.0 # wall angle (radians)
    swing_side: str   = 'right'
```

**⚠️ NO t_start or t_end fields.** `split_wall_at_openings()` derives those internally
from `t_center ± (width/2)/wall_length`. Never pass t_start/t_end to dc_replace().

**Status:** ✅ Unchanged. Only used for DXF input.

---

## geometry_builder.py ★

**Purpose:** Converts Wall + Opening + Room objects → Three.js-compatible JSON dicts.

**Build order (must be this sequence):**
```
1. _split_walls_at_junctions(walls)    ← FIRST, before opening grouping
2. _reproj_openings(openings, walls)   ← fix wall_idx + t_center after split
3. Group openings by wall_idx
4. _compute_wall_extensions(walls)     ← corner gap fill (exists, untested)
5. wall_to_boxes() per wall            ← render each wall
6. _assign_room_ids(wall_boxes, rooms) ← tag after all boxes built
7. Freestanding openings (wall_idx == -1)
```

**Key functions:**

### `_split_walls_at_junctions(walls)` ✅ NEW
Cuts wall A wherever wall B physically reaches it (T-junctions, + junctions).
- Gate: `t_a ∈ (0.02, 0.98)` — crossing inside wall A, not at its own endpoints
- Gate: `t_b * Lb ∈ [-reach, Lb+reach]` where `reach = wb.thickness/2 + 0.15m`
- Deduplicates cuts within 1% of wall length; skips slivers < 5cm
- Each sub-wall inherits all parent attributes (thickness, height, layer, etc.)

### `_reproj_openings(openings, walls)` ✅ FIXED (two iterations)
After splitting, re-projects each opening onto the correct sub-wall.
- **v1 bug:** only updated wall_idx, left t_center as fraction of old full wall → window pushed to side
- **v2 bug:** tried to dc_replace t_start/t_end which don't exist on Opening dataclass → TypeError crash
- **v3 (current):** updates only `wall_idx` and `t_center` (the only two fields that need fixing).
  t_center is recomputed as projection of op.x/op.y onto the matched sub-wall's [0..1] range.
  t_start/t_end are NOT set here — split_wall_at_openings derives them from t_center internally.

```python
# Correct fields to dc_replace — ONLY these two:
updated.append(dc_replace(op, wall_idx=best_idx, t_center=round(best_t, 4)))
```

### `_assign_room_ids(wall_boxes, rooms)` ✅ NEW
Tags every wall box with `room_id` (nearest room centroid in world XZ).
Stored in `mesh.userData.room_id` in the viewer for future texture filtering.

### `_door_leaf()` ✅ FIXED
Old bug: used `rot_y` (Three.js rotation angle) in `math.cos/sin` for world-space
positioning — wrong convention.
Fix: uses wall unit vector `(ux, uy)` and perpendicular `(nx, ny)` directly.
Rotates travel direction by sweep angle using wall's own coordinate frame.
`leaf_rot_y = -atan2(swung_y, swung_x)`

**Constants:**
```
SILL_HEIGHT         = 0.9    # window sill height (m)
WIN_HEIGHT          = 1.2    # window opening height (m)
DOOR_LEAF_T         = 0.05   # door leaf thickness (m)
CORNER_RASTER_DRIFT = 0.10   # endpoint drift tolerance for corner extensions
SNAP_TOL            = 0.35   # (kept but unused in current split logic)
RASTER_DRIFT        = 0.15   # metres of endpoint slop in split function
```

**Known issues:**
- [ ] Problem 5: `_compute_wall_extensions()` exists but untested on real floor plans
- [ ] Door pushed into wall (cosmetic, deferred)

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

## index_v2.html (viewer) ★

**Purpose:** Three.js browser viewer. Uploads files, displays 3D model.

**Key behaviors:**
- Sends `&pixels_per_meter=0` in process request ← CRITICAL (was missing, caused PPM=100 bug)
- Accepts: PNG, JPG, JPEG, PDF, DXF
- Renders: wall boxes, door leaves + swing arcs, window sill/glass, room labels
- Controls: orbit camera, wall height slider, scale slider
- Wall thickness visual: relative (median-based), no hardcoded thresholds
  - Outer walls (`thick >= median * 1.15`): dark navy `0x0d2233`, bright cyan wireframe 50%
  - Inner partitions: `0x1a3d55`, opacity 0.85, dim wireframe 25%

**View modes (NEW):**
- **◈ BLUEPRINT** (default): dark navy walls, cyan wireframes, dark scene background
- **◉ REALISTIC**: daylight scene, textured walls and floor, wireframes hidden

**Material editor (NEW — only active in Realistic mode):**
- Scope: `ALL WALLS` or `SELECT` (click walls in viewport to pick them, orange outline)
- Wall finishes (6 procedural textures, no external files): Plaster, Brick, Concrete, Wood, Marble, White Tile
- Custom solid colour: colour picker
- Tile size slider: 0.3m–4m controls texture repeat
- ▶ APPLY: commits finish to target walls (all or selected)
- ↺ RESET ALL: clears all overrides, back to default realistic materials
- Floor finishes (6 swatches): Tile, Parquet, Marble, Concrete, Stone, Carpet

**Material system internals:**
- `_blueprintMats` WeakMap: stores original blueprint material per mesh for mode restore
- `_meshOverrides` WeakMap: stores `{ texId, color, tileSize }` override per mesh
- `makeTex(size, drawFn)` → `THREE.CanvasTexture` (all textures generated via 2D canvas)
- `setMode('blueprint'|'realistic')`: restores blueprint mats OR applies overrides/defaults
- `applyPendingMaterial()`: reads scope + pending tex + tile size → writes to target meshes
- `handleWallClick(event)`: raycasts into `wallMeshes`, toggles selection, adds/removes orange EdgesGeometry outline

**mesh.userData fields:**
```javascript
{
  room_id:             string | null,  // nearest room centroid tag
  thickness:           float,          // wall depth in metres
  is_outer:            bool,           // true if thickness >= median * 1.15
  selected:            bool,           // selection state
  _selectionOutline:   THREE.LineSegments | null,  // orange outline mesh
}
```

---

## Data Flow Detail

```
Image file
    │
    ▼
RasterParser.parse()
    │  wall_mask, h_walls, v_walls, img_openings, _ppm, _fp_h
    ▼
ParsedGeometry (wall_segments in metres)
    + _raster_openings, _ppm, _fp_h, _wall_mask attached
    │
    ▼
WallDetector.detect()
    → [Wall(start, end, thickness, height, layer, paired, confidence)]
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
    1. _split_walls_at_junctions(walls)
    2. _reproj_openings(openings, walls)   ← updates wall_idx + t_center only
    3. _assign_room_ids(wall_boxes, rooms)
    → BuildingModel
        .walls   [box dicts with room_id]  → Three.js BoxGeometry
        .doors   [door dicts]              → door leaf + swing arc
        .windows [window dicts]            → sill + glass + header
        .rooms   [label dicts]             → text sprites
        .floors  [floor dict]              → PlaneGeometry
        .metadata {counts, bounds}
    │
    ▼
JSON response → viewer/index_v2.html → Three.js scene
  Blueprint mode: navy boxes + cyan wireframes
  Realistic mode: procedural textures + floor material + daylight
```

---

## Current Open Problems

| # | Problem | Severity | File | Status |
|---|---------|----------|------|--------|
| 1 | Phantom doors from tick marks | Medium | raster_parser.py | Not started |
| 2 | Thin wall door detection (dynamic SCAN_HALF) | Medium | raster_parser.py | Not started |
| 3 | Wall thickness measurement bugs | High | wall_detector.py | Not started |
| 4 | Window symbols only detected on V-walls | Low | raster_parser.py | Not started |
| 5 | Corner gap edges (_compute_wall_extensions) | High | geometry_builder.py | Exists, untested |

---

## Testing Approach

1. Upload `fp4.png` — should show 4 rooms, outer walls thicker than inner
2. Check warning message for PPM (should show ~33 px/m, source="auto")
3. Verify no phantom doors at corners
4. Verify doors at all T-junctions (4-5 doors in fp4)
5. Switch to Realistic mode — walls should turn warm plaster colour, floor tiled
6. Select individual walls, apply brick texture, verify only selected walls change
7. Upload the larger 672x670 screenshot — should still work with different PPM

**Known test images:**
- `fp4.png` (640×355): 4-room L-shaped plan, PPM=33.2, outer walls 9px, inner 6px
- `Door.png` (672×670): annotated floor plan showing door/window identification
- `New_project__3_.png`: simple 2-room plan, 1 door bottom wall, 1 window right side
- Large screenshot from sessions 1-3: multi-room CAD export, ~65 px/m
