# FloorPlan 3D — Change Log & Debugging Context

> This document exists so future Claude sessions can quickly understand what was tried,
> what worked, what broke, and what the current state is. Read this before touching any file.

---

## Project Structure

```
backend/app/
├── main.py                  ← FastAPI routes — THE ROOT CAUSE FILE (see below)
└── core/
    ├── pipeline.py          ← Orchestrates parse→detect→build
    ├── raster_parser.py     ← Image/PDF CV pipeline (primary parser for raster)
    ├── simpledraw_parser.py ← NEW: Black-on-white drawing parser (separate path)
    ├── wall_detector.py     ← Segments → Wall objects
    ├── opening_detector.py  ← Door/window detection (DXF path)
    ├── room_detector.py     ← Room polygon detection
    ├── geometry_builder.py  ← Wall+Opening+Room → Three.js JSON
    └── dxf_parser.py        ← DXF entity extraction

viewer/index_v2.html         ← Three.js viewer
```

---

## THE ROOT CAUSE (most important thing to know)

**`main.py` had `pixels_per_meter: float = Query(default=100.0)`**

This was silently overriding all auto-detection in `raster_parser.py`. The viewer never
sent `pixels_per_meter` in its request, so the API always used 100 px/m as an override,
bypassing `_detect_ppm()` completely. Every single image was processed as if 100px = 1m
regardless of actual content.

**Fix applied:**
- `main.py`: changed default to `0.0` (0 = auto-detect)
- `index_v2.html`: viewer now explicitly sends `&pixels_per_meter=0` in process request

**Verification:** After fix, warning message changes from `"100 px/m (override)"` to
`"65 px/m (ticks)"` or similar auto-detected value.

---

## Files Changed This Session

### `main.py` ✅ FIXED — Deploy this
- Line 97: `default=100.0` → `default=0.0` for `pixels_per_meter`
- No other changes

### `index_v2.html` ✅ FIXED — Deploy this
- Line 1147: process URL now includes `&pixels_per_meter=0`
- Prevents viewer from ever accidentally passing 100 as default

### `simpledraw_parser.py` ✅ NEW FILE — Drop into `backend/app/core/`
- Handles black-on-white architectural drawings (SimpleDraw, hand-drawn style)
- Detects format via `is_simpledraw_format()`: >65% white background, truly black lines, no green
- Morphological separation: outer walls (15px erode), inner walls (5px erode), thin (window/arc)
- PPM estimated from outer wall thickness (assumed 0.30m exterior wall)
- Window detection: paired parallel thin lines within 18px
- Door detection: arc contours after removing straight lines, `minEnclosingCircle`
- Does NOT affect raster_parser.py path at all — completely separate

### `pipeline.py` ✅ FIXED — Deploy this
- Added simpledraw format auto-detection before raster parsing
- `is_simpledraw_format()` check routes to SimpleDrawParser or RasterParser
- Fixed `_build_raster_openings`: was accidentally lost in an edit, now restored
- Fixed inter-segment door angle: uses nearest wall by orientation, not coordinate matching
- **CRITICAL**: `_build_raster_openings` method MUST exist — previous version had it dropped

### `raster_parser.py` ⚠️ PARTIALLY FIXED — Has improvements but also regressions
**What improved:**
- `MERGE_GAP` is now dynamic: `int(MIN_OPENING_M * ppm * 0.35)` — scales with resolution
- `MIN_WALL_PX` is now dynamic: `int(0.25 * ppm)` — scales with resolution  
- Both `_detect_walls()` and `_complete_outer_walls()` now accept `ppm` parameter
- `_wall_mask` is now attached to ParsedGeometry for wall_detector thickness measurement
- Endpoint guard relaxed: only drops gaps where BOTH sides < 2px (not the aggressive version)
- H-wall window symbol scan added (was only running on V-walls before)

**What may have regressed:**
- `ppm_source` metadata key changed from `"ticks"/"fallback"` to `"override"/"auto"`
  - This broke the warning string format in `_parse_raster` (looks for `ppm_source`)
  - TODO: align ppm_source values between raster_parser and pipeline warning strings

### `geometry_builder.py` ⚠️ CHANGED BUT CAUSING ISSUES
**What was added:**
- `_compute_corner_trims()` function: trims wall endpoints at L-corners and T-junctions
- Applied in `build()` before rendering each wall

**Problem observed (Image 2):**
- Corner trim is being too aggressive or miscalculating junction types
- Walls appear cut off at junctions rather than meeting cleanly
- The trim logic uses `perp < thickness/2 + 0.12` tolerance which may be too loose,
  causing non-junction walls to get incorrectly trimmed

**Recommendation:**
- The corner trim concept is correct but the tolerance is wrong
- For raster input: wall endpoints from Hough detection are often ±5-10px off the true
  junction. At 65px/m this is ±0.08-0.15m. Current tolerance 0.12m may not be enough.
- Consider making tolerance a function of PPM: `tol = max(0.12, 3/ppm_estimate)`

### `wall_detector.py` ⚠️ CHANGED — May cause issues for simpledraw
**What was added:**
- `_measure_wall_thicknesses()`: samples wall_mask perpendicular to each wall at 7 points
- Called when `geometry._wall_mask` is available (raster input only)
- Uses median pixel run width, converts to metres, clamps to [0.08, 0.55]
- Outer walls expected ~0.30m, inner walls ~0.10-0.15m

**Known issue:**
- The Y-flip between pixel coords (image top=0) and world coords (y increases upward)
  means the perpendicular sampling may hit wrong pixels
- `py = int(FH - wy * ppm)` converts world y to image y — this looks correct
- But `ny` (perpendicular in image space) may be flipped: world +Y = image -Y

---

## Architecture — How Data Flows

### Raster path (CAD format: green ticks, grey walls)
```
PNG/JPG → RasterParser.parse()
    → _autocrop()         (find floor plan panel inside UI screenshot)
    → _detect_ppm()       (read green tick marks, fallback to FALLBACK_PPM=65)
    → _segment()          (pixel classify: WALL=grey 82-148, ROOM=grey 153-244)
    → _detect_walls(ppm)  (skeleton → Hough → merge(dynamic) → dedup → complete)
    → _detect_openings_from_image()  (scan wall bands for bright gaps and window symbols)
    → _detect_rooms()     (seal doorways → connected components on ROOM pixels)
    → ParsedGeometry with _raster_openings, _rooms_px, _ppm, _fp_h, _wall_mask

pipeline._build_raster_openings()
    → converts pixel openings to Opening objects
    → intra-segment: matched to nearest wall by perpendicular distance
    → inter-segment (t_center==-1.0): freestanding, matched by orientation

GeometryBuilder.build()
    → _compute_corner_trims() [NEW — may be causing issues]
    → wall_to_boxes() with openings → wall pieces + door/window dicts
```

### SimpleDraw path (black-on-white: thick=exterior, thin=interior, arc=door, double-line=window)
```
PNG/JPG → is_simpledraw_format() check → SimpleDrawParser.parse()
    → _binarise()           (threshold at 50 → binary mask)
    → _separate_layers()    (erode at 15px → outer, 5px → inner, subtract → thin)
    → _estimate_ppm()       (measure outer wall body width, divide by 0.30m)
    → _hough_walls()        (separate Hough on outer_mask and inner_mask)
    → _measure_wall_thickness_px()  (per-wall perpendicular sampling)
    → _detect_windows()     (pair thin parallel lines within 18px)
    → _detect_doors()       (arc contours after removing straight lines)
    → _detect_rooms_simpledraw()  (white space between walls)
    → ParsedGeometry with same _raster_openings, _ppm, _fp_h fields as raster path

Then same pipeline._build_raster_openings() → GeometryBuilder.build()
```

### DXF path (unchanged)
```
DXF → DXFParser → ParsedGeometry → WallDetector → OpeningDetector → GeometryBuilder
```

---

## Known Remaining Issues

### Issue 1: Phantom walls (raster format)
**Symptom:** Extra wall appears in middle of a real wall, sometimes with phantom window
**Root cause:** Window symbol (two dark lines) surviving as a separate wall segment
**Status:** MERGE_GAP fix helps but not fully resolved
**Where to look:** `_dedup()` in raster_parser.py — the window symbol lines may not be
  collapsed because they're on a slightly different y coordinate than the wall centerline

### Issue 2: Missing corner walls (raster format)  
**Symptom:** Corner where two walls meet has one wall missing or truncated
**Root cause:** Short wall stubs between a door and a corner junction fall below MIN_WALL_PX
**Status:** MIN_WALL_PX lowered to `int(0.25*ppm)` — should be better
**Where to look:** `_merge()` in raster_parser.py, `_complete_outer_walls()`

### Issue 3: Door sizes incorrect (raster format)
**Symptom:** Door leaf rendered at wrong size (too small or too large)
**Root cause:** `gpx` (bright pixel run) is shorter than true door because:
  - The scan band `±14px` averages over door frame pixels (dark) which shrink the run
  - At higher PPM the door in pixels is larger, but if PPM is wrong the width_m is wrong
**Status:** Partially fixed by PPM auto-detection. Still imprecise.
**Where to look:** `_find_gap_openings()` in raster_parser.py, SCAN_HALF_PX constant

### Issue 4: Corner overlap/cut in 3D (geometry_builder)
**Symptom:** Walls at junctions have visible overlaps or gaps rather than clean joins
**Root cause:** `_compute_corner_trims()` tolerance is wrong — perp threshold 0.12m
  may be too small for raster junctions (±0.15m error) or too large (false trims)
**Status:** NEEDS FIXING
**Approach:** 
  1. Increase perp tolerance to `wall.thickness * 0.8` rather than a fixed 0.12m
  2. OR: replace trim approach with "extend to intersection point" — compute exact
     intersection of the two wall centerlines and set endpoint to that point

### Issue 5: SimpleDraw producing too many walls (25 inner walls)
**Symptom:** Image 1 shows 25 inner walls for a simple 3-room plan
**Root cause:** `_hough_walls()` on `inner_mask` picks up arc curves as wall segments
  (arcs are ~10px thick = same as inner walls, survive the 5px erosion)
**Fix needed:** In `simpledraw_parser._hough_walls()` for inner walls, add a check:
  only accept lines where the aspect ratio of the containing bbox is > 3:1 (wall-like)
  Arcs produce roughly square bboxes — this would filter them out

### Issue 6: Door placement (both formats)
**Symptom:** Door leaves rendered in wrong position or multiple doors at same location
**Root cause:** Both `_build_raster_openings` (intra) and `freestanding` paths can emit
  the same door if it's detected in both passes
**Fix needed:** Deduplicate openings in `_build_raster_openings` by position proximity

---

## Constants That Matter

### raster_parser.py
```python
FALLBACK_PPM      = 65.0    # used when tick detection fails
MIN_OPENING_M     = 0.35    # minimum gap to be an opening
MAX_DOOR_M        = 1.40    # door vs window cutoff
SCAN_HALF_PX      = 14      # wall band scan width for opening detection
OPENING_BRIGHT    = 148     # brightness threshold for "open space"
DEDUP_DIST        = 20      # max px between parallel double-edges to collapse

# These are now DYNAMIC (computed from ppm):
merge_gap   = max(4, int(MIN_OPENING_M * ppm * 0.35))   # ~35% of smallest opening
min_wall_px = max(12, int(0.25 * ppm))                   # ~0.25m minimum
```

### simpledraw_parser.py
```python
OUTER_KERNEL_PX = 15    # erosion that kills inner walls (~10px) but keeps outer (~30px)
INNER_KERNEL_PX = 4     # erosion that kills window lines but keeps inner walls
ASSUMED_OUTER_M = 0.30  # assumed exterior wall thickness for PPM estimation
WIN_MAX_PAIR_GAP = 18   # max px gap between two lines to be a window pair
ARC_MIN_R = 25          # minimum arc radius in pixels
ARC_MAX_R = 250         # maximum arc radius
```

---

## What To Do Next Session

1. **Fix Issue 5 first** (SimpleDraw 25 inner walls): filter arc-shaped contours from
   inner wall Hough by bbox aspect ratio. Quick fix, high impact.

2. **Fix Issue 4** (corner overlaps): replace fixed 0.12m perp tolerance with
   `wb.thickness * 0.8`. Or better: compute actual intersection point of wall centerlines
   and snap endpoints there.

3. **Test raster format** with the PPM fix in place (main.py default=0.0). The base
   detection should be much better now that 100px/m override is removed.

4. **Fix Issue 6** (door deduplication): in `_build_raster_openings`, after building
   all openings, remove any where two openings are within `min(op.width for op in openings)`
   of each other (same door detected twice).

5. **Verify wall_detector thickness measurement**: check Y-flip in `_measure_wall_thicknesses`
   — `py = int(FH - wy * ppm)` should be correct but verify with debug output.

---

## Files That Were NOT Changed (safe to use originals)

- `opening_detector.py` — unchanged, works correctly for DXF path
- `room_detector.py` — unchanged
- `dxf_parser.py` — unchanged

---

## Deploy Order

Replace files in this order, restart uvicorn after all are in place:

1. `backend/app/main.py`                     ← fix ppm default
2. `viewer/index_v2.html`                    ← send ppm=0
3. `backend/app/core/pipeline.py`            ← simpledraw routing + method fix
4. `backend/app/core/raster_parser.py`       ← dynamic thresholds
5. `backend/app/core/simpledraw_parser.py`   ← NEW, drop into core/
6. `backend/app/core/geometry_builder.py`    ← corner trim (may need further tuning)
7. `backend/app/core/wall_detector.py`       ← per-segment thickness

Restart: `cd backend && uvicorn app.main:app --reload --port 8000`
