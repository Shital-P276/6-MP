# LEARNMAP — Floor Plan 3D Visualizer

> What we've learned through trial and error. Read before making any change.
> This is the accumulated knowledge of 4+ sessions of debugging.

---

## Architecture Overview

```
PNG/JPG/PDF → raster_parser.py → ParsedGeometry
                                        ↓
                              pipeline.py orchestrates:
                                        ↓
                              wall_detector.py → [Wall]
                                        ↓
                              opening_detector.py (DXF) OR
                              pipeline._build_raster_openings (raster)
                                        ↓
                              room_detector.py → [Room]
                                        ↓
                              geometry_builder.py → BuildingModel (Three.js JSON)
                                        ↓
                              viewer/index_v2.html (Three.js)
```

---

## What Worked

### PPM Detection
- Green tick marks along image borders reliably encode scale
- `_detect_ppm()`: scans top 5% of image for green spans, uses inter-span gaps > 50px
- fp4.png: 133px gaps = 4m bays → 33.2 px/m ✓
- Larger screenshot: ~130px gaps similarly detected
- **Fallback PPM=65 is only used when tick detection fails**

### Dynamic Border Crop (last fix)
- `_detect_border_crop()`: scans inward from each edge until green coverage < 10%
- For fp4.png (3px green bar): returns ~17px → outer walls at y=24 are preserved ✓
- For large screenshots (18px green bar): returns ~26px → same behavior ✓
- Safety cap: never crops > 8% of smaller image dimension

### Green Mask Suppression
- `GREEN_DIFF=22`: (G-R > 22) AND (G-B > 22) correctly identifies annotation pixels
- Dilate by 4px catches fringe anti-aliased green pixels
- This removes dimension lines, tick marks, and annotation text from wall detection

### Wall Segmentation Range
- WALL_LO=82, WALL_HI=148 (gray range for wall pixels)
- ROOM_LO=153, ROOM_HI=244 (gray range for room floor pixels)
- These thresholds work for both dark brown walls (fp4, gray≈107-120) and gray CAD walls
- The gap between WALL_HI=148 and ROOM_LO=153 is intentional — catches neither

### Hough + Merge Pipeline
- MERGE_GAP=60: bridges doorway gaps (~50px at 33ppm or ~25px at 65ppm)
- MIN_WALL_PX=40: filters noise/arc artifacts without losing short real wall stubs
- COORD_GROUP_TOL=11: groups parallel double-edges into one centerline
- DEDUP_DIST=20: collapses both faces of a thick wall into single centerline

### Inter-Segment Opening Detection (Pass 2)
- Groups walls by fixed coord (H by y, V by x)
- Finds gaps between collinear stubs on same line
- Correctly detects doors that are too wide for MERGE_GAP to bridge

### T-Junction Opening Detection (Pass 3)
- Finds doors between perpendicular wall endpoints within WALL_HALF_PX=20
- EDGE_GUARD=20 prevents tick marks at image corners from becoming phantom doors
- Correctly handles all 4 T-junction orientations

### Pipeline _build_raster_openings
- Converts pixel-space openings to world-space Opening objects
- Intra-segment: matched to nearest wall by perpendicular distance
- Inter-segment (t_center==-1.0): freestanding, matched by orientation to nearest wall

### Autocrop
- Finds white floor plan panel inside UI screenshots
- Uses fill ratio (fraction of bright pixels) to select the floor plan bounding box
- Handles dark 3D viewer background with cyan walls

---

## What Failed (and Why)

### Fixed BORDER_CROP_PX=52 ← DON'T USE
- Was calibrated for 670px screenshots
- For fp4.png (355px), it removed y=0-52, cutting off outer walls at y=24-32
- **The dynamic `_detect_border_crop()` replaced this entirely**

### Fixed BORDER_CROP_PX=36 ← DON'T USE
- Better than 52 but still too aggressive for small/tight floor plans
- Tick marks at y=36-48 would leak through

### _compute_corner_trims (geometry_builder) ← REVERTED
- Concept: trim wall endpoints at junctions to prevent overlap
- Problem: perp tolerance of 0.12m was too tight for raster endpoints (drift ±0.15m)
- It was trimming walls that didn't actually overlap AND missing ones that did
- Result: walls shorter than they should be, creating gaps at junctions
- **Replacement: `_compute_wall_extensions()` extending outward instead**
- Key insight: better to overshoot (extend) than undershoot (trim)

### SimpleDraw Parser ← REMOVED
- Attempted to handle black-on-white floor plans (thick=exterior, thin=interior, arc=door)
- Added `is_simpledraw_format()` detection into pipeline.py routing
- Caused regressions: 25 inner walls detected for simple plans (arcs survived inner erosion)
- The main raster pipeline handles both formats adequately when PPM is correct
- Removed entirely; reverted to original uploads + only PPM fix

### Hardcoded pixels_per_meter=100 ← FIXED
- `main.py` had `default=100.0` for pixels_per_meter API parameter
- The viewer never sent this parameter → API always used 100 as override
- All dynamic threshold calculations (merge_gap, min_wall) used wrong PPM=100
- Fix: `default=0.0` (0 = auto-detect), viewer sends `&pixels_per_meter=0`

### Wall Thickness measurement (SCAN_HALF=30) ← PARTIALLY BROKEN
- The `_measure_wall_thicknesses()` function exists and is called
- Bug 1: SCAN_HALF=30 too wide — picks up floor pixels beyond wall body
- Bug 2: span measurement (`hits[-1]-hits[0]+1`) includes gaps, overestimates
- Bug 3: Y-flip missing — `ny` in image space should be negated relative to world `ny`
- **These bugs are documented in NEXT_STEPS_PLAN.md — not yet fixed**

---

## Coordinate System (Critical)

```
World space (metres):
  X increases right
  Y increases UP (standard math)
  Origin: bottom-left of floor plan

Image space (pixels):
  x increases right (same)
  y increases DOWN (image convention)
  Origin: top-left

Conversion:
  px = world_x * ppm
  py = FH - world_y * ppm     ← Y IS FLIPPED

Three.js space:
  X = world_x
  Y = height (vertical)
  Z = -world_y                ← Y becomes negative Z
  rotation_y = -atan2(dy, dx) ← negated because Z is flipped
```

**This Y-flip is the source of the wall thickness measurement bug** — the perpendicular
direction `ny` must be negated when converting from world to image space.

---

## Floor Plan Image Anatomy

For the CAD format with green annotations:

```
┌─────────────────────────────────────────┐
│  Green measurement frame (3-20px thick) │  ← excluded by green mask + border crop
│  ┌───────────────────────────────────┐  │
│  │  Outer wall (gray 82-148)        │  │  ← 9-20px thick at 33ppm
│  │  ┌─────────────────────────────┐ │  │
│  │  │  Room floor (gray 153-244) │ │  │  ← tan/beige color
│  │  │  Door gap (bright >148)    │ │  │  ← detected by brightness scan
│  │  │  Window: ▓░░░▓ (dark-bright-dark) │  ← detected by symbol scan
│  │  │  Inner wall (gray 82-148)  │ │  │  ← thinner than outer
│  │  └─────────────────────────────┘ │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

---

## Wall Thickness Values (fp4.png)

| Wall type | Pixel width | At PPM=33.2 | Architectural reality |
|-----------|-------------|-------------|----------------------|
| Outer wall | 9px | 0.271m (27cm) | ✓ realistic exterior |
| Inner partition | 6px | 0.181m (18cm) | ✓ realistic interior |
| Bottom outer (H) | 9px | 0.271m (27cm) | ✓ same as top/sides |

The ratio 9:6 = 1.5:1 is consistent across all images of this type.

---

## Opening Detection Logic

```
Pass 1: Intra-segment
  For each wall [a..b]: scan band ±SCAN_HALF_PX around centerline
  If brightness > OPENING_BRIGHT: it's a gap
  Gap width ≤ MAX_DOOR_M → door
  Gap width MAX_DOOR_M..MAX_WINDOW_M → window
  Also: _find_window_symbols checks dark|bright|dark pattern for CAD windows
  CURRENTLY: window symbols only checked on V-walls (bug — fix in Problem 4)

Pass 2: Inter-segment (collinear stubs)
  Group walls by fixed coordinate
  For adjacent stubs on same line, check the gap between them
  If bright span is door/window width → emit as inter-segment opening

Pass 3: T-junction (perpendicular stub meets outer wall)
  For each wall endpoint, find perpendicular walls within WALL_HALF_PX
  Scan the gap between endpoint and outer wall face
  EDGE_GUARD prevents image-edge artifacts from becoming doors
```

---

## Key Lessons

1. **Never use a fixed pixel constant for border crop** — floor plans vary in size and DPI
2. **Extend, don't trim, at corners** — raster endpoints have ±0.15m drift; trimming creates gaps
3. **Y-axis flip is required** in any code that converts world coordinates to image coordinates
4. **PPM must come from the image** — hardcoding 100 blocks everything else
5. **Thin walls need narrower scan bands** — SCAN_HALF must adapt to wall thickness
6. **Window symbols must be checked on BOTH H and V walls** — not just V
7. **T-junction doors live in the dead zone** between outer wall body and inner wall endpoint
8. **Consecutive run, not span** — span overestimates when the wall body has any gaps
