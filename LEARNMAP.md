# LEARNMAP — Floor Plan 3D Visualizer

> What we've learned through trial and error. Read before making any change.
> This is the accumulated knowledge of 5+ sessions of debugging.

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

### Dynamic Border Crop
- `_detect_border_crop()`: scans inward from each edge until green coverage < 10%
- For fp4.png (3px green bar): returns ~17px → outer walls at y=24 are preserved ✓
- For large screenshots (18px green bar): returns ~26px → same behavior ✓
- Safety cap: never crops > 8% of smaller image dimension

### Green Mask Suppression
- `GREEN_DIFF=22`: (G-R > 22) AND (G-B > 22) correctly identifies annotation pixels
- Dilate by 4px catches fringe anti-aliased green pixels

### Wall Segmentation Range
- WALL_LO=82, WALL_HI=148 (gray range for wall pixels)
- ROOM_LO=153, ROOM_HI=244 (gray range for room floor pixels)
- The gap between WALL_HI=148 and ROOM_LO=153 is intentional

### Hough + Merge Pipeline
- MERGE_GAP=60: bridges doorway gaps (~50px at 33ppm or ~25px at 65ppm)
- MIN_WALL_PX=40: filters noise/arc artifacts without losing short real wall stubs
- COORD_GROUP_TOL=11: groups parallel double-edges into one centerline
- DEDUP_DIST=20: collapses both faces of a thick wall into single centerline

### Wall Splitting at Junctions (`_split_walls_at_junctions`)
- Cuts wall A wherever wall B's body physically reaches it
- Critical gate: `t_b * Lb ∈ [-reach, Lb+reach]` where `reach = wb.thickness/2 + 0.15m`
  This handles T-junctions (inner wall terminates at outer wall face) correctly
  without the old hardcoded `t_b ∈ [0.10..0.90]` which broke on non-standard plans
- Each sub-wall inherits all parent attributes (thickness, height, layer, paired, confidence)
- Must run BEFORE opening grouping — openings reference wall_idx which changes after split

### Opening Re-projection After Wall Splitting (`_reproj_openings`)
After `_split_walls_at_junctions`, wall indices change. Two fields need updating:
- `wall_idx`: which sub-wall the opening belongs to
- `t_center`: fraction along the new (shorter) sub-wall

**What does NOT need updating:**
- `t_start`, `t_end`: these don't exist on the `Opening` dataclass.
  `split_wall_at_openings()` in `opening_detector.py` derives them internally from
  `t_center ± (width/2) / wall_length`. Never try to dc_replace these.

**The window-pushed-to-side bug (fixed):**
- Root cause: `_reproj_openings` v1 only updated `wall_idx`, left `t_center` as a fraction
  of the original full-length wall. On the shorter sub-wall, `t_center=0.85` means
  85% along a 5m sub-wall = 4.25m, but 85% of the original 10m wall meant 8.5m.
- Fix: recompute `t_center` from the opening's world position `(op.x, op.y)` projected
  onto the matched sub-wall's [0..1] range.

**The TypeError crash (fixed):**
- `_reproj_openings` v2 tried to `dc_replace(op, t_start=..., t_end=...)` which don't
  exist as fields on the Opening dataclass → `TypeError: unexpected keyword argument`
- Fix: only `dc_replace(op, wall_idx=best_idx, t_center=round(best_t, 4))`

### Door Leaf Positioning Fix
Old bug: `_door_leaf()` passed `rot_y` (Three.js Y-rotation angle) to `math.cos/sin`
for world-space positioning. This is wrong — Three.js rotation_y ≠ world-space angle.
Fix: use wall unit vector `(ux, uy)` and perpendicular `(nx, ny)` directly from the wall
start/end points. Swing travel direction rotated using the wall's own coordinate frame.

### Relative Thickness Classification (viewer)
Old approach: hardcoded threshold values broke for floor plans with different PPM.
Fix: use median of all wall thicknesses as the split point.
`outerThreshold = medianThick * 1.15` — 15% above median handles raster measurement noise.
This works at any PPM/scale without calibration.

### Procedural Textures (no external files)
All textures are generated via `HTMLCanvasElement` + 2D canvas API → `THREE.CanvasTexture`.
Advantages:
- No HTTP requests, no CORS issues, no missing texture errors
- Works completely offline
- Textures scale correctly via `tex.repeat.set(w/tileSize, h/tileSize)`
- Can be regenerated at any resolution by changing the `size` parameter to `makeTex()`

### Blueprint ↔ Realistic Mode Switching
Key pattern: store the original blueprint material in a WeakMap keyed on the mesh
(`_blueprintMats`) before any mode switch. On switch back, restore from this map.
This avoids having to rebuild the entire model to restore the original appearance.
WeakMap is correct here because mesh objects are garbage-collected when the model is
cleared, so there's no memory leak from keeping references.

### Wall Selection via Raycasting
```javascript
raycaster.setFromCamera(mouse, camera);
const hits = raycaster.intersectObjects(wallMeshes);
```
Selection highlight: add a `THREE.LineSegments(EdgesGeometry, orange material)` as a
separate object positioned/rotated identically to the wall, scaled by 1.002 to avoid
z-fighting. Store it in `mesh.userData._selectionOutline` so it can be removed on deselect.

---

## What Failed (and Why)

### Fixed BORDER_CROP_PX=52 ← DON'T USE
- Was calibrated for 670px screenshots
- For fp4.png (355px), cut off outer walls at y=24-32

### _compute_corner_trims (geometry_builder) ← REVERTED
- Concept: trim wall endpoints at junctions to prevent overlap
- Problem: perp tolerance of 0.12m was too tight for raster endpoints (drift ±0.15m)
- Result: walls shorter than needed, creating gaps at junctions
- **Replacement: `_compute_wall_extensions()` extending outward instead**

### SimpleDraw Parser ← REMOVED
- Caused regressions: 25 inner walls detected for simple plans
- The main raster pipeline handles both formats adequately when PPM is correct

### Hardcoded pixels_per_meter=100 ← FIXED
- `main.py` had `default=100.0` → all PPM-relative thresholds calculated against wrong scale
- Fix: `default=0.0`, viewer sends `&pixels_per_meter=0`

### t_start / t_end on Opening ← DON'T ADD
- These fields do NOT exist on the Opening dataclass
- `split_wall_at_openings()` in `opening_detector.py` derives them from `t_center` internally
- Adding them via dc_replace causes `TypeError: Opening.__init__() got an unexpected keyword argument`
- This exact mistake was made twice — do not repeat it

### Hardcoded wall thickness thresholds in viewer ← REPLACED
- Values like `if (thick > 0.25)` break completely for floor plans at different scales
- Replaced with median-relative: `outerThreshold = medianThick * 1.15`

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

**This Y-flip is required in:**
- `_measure_wall_thicknesses()` when computing perpendicular direction in image space
- Any code converting world angle/direction to image pixel scan direction
- `_door_leaf()` — fixed by using wall unit vector instead of rot_y angle

---

## Opening Dataclass (Critical Reference)

```python
@dataclass
class Opening:
    wall_idx:   int         # wall index (-1 = freestanding)
    t_center:   float       # position along wall [0,1]
    width:      float       # opening width in metres
    kind:       str         # "door" | "window"
    x:          float = 0.0 # world X position
    y:          float = 0.0 # world Y position
    angle:      float = 0.0 # wall angle (radians)
    swing_side: str   = 'right'
```

Fields that do NOT exist (never add to dc_replace): `t_start`, `t_end`

The only fields `_reproj_openings` should ever dc_replace: `wall_idx`, `t_center`

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
│  │  │  Window: ▓░░░▓ pattern     │ │  │  ← detected by symbol scan (V-walls only, bug)
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

The ratio 9:6 = 1.5:1 is consistent across all images of this type.
`medianThick * 1.15` correctly separates these two groups.

---

## Key Lessons (ordered by importance)

1. **Never hardcode pixel-space thresholds** — everything must be PPM-relative
2. **Opening dataclass has no t_start/t_end** — dc_replace only wall_idx and t_center
3. **Wall splitting must happen before opening grouping** — indices change after split
4. **t_center must be recomputed after split** — it was a fraction of the old wall length
5. **Y-axis flip is required** in any world → image coordinate conversion
6. **Extend, don't trim, at corners** — raster endpoints have ±0.15m drift; trimming creates gaps
7. **Store blueprint materials before switching modes** — WeakMap on mesh is the right pattern
8. **Procedural textures beat external files** — no CORS, no missing assets, works offline
9. **rot_y ≠ world angle** — for world-space vector math, derive (ux,uy) from start/end points
10. **Window symbols must be checked on BOTH H and V walls** — currently a known bug
