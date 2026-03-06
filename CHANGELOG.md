# CHANGELOG — Floor Plan 3D Visualizer

All notable changes to this project, in reverse chronological order.

---

## [Session 5] — Viewer UX + Material System

### Added
- **Blueprint / Realistic mode toggle** — right-side collapsible panel with `◀ ▶` toggle button
- **6 procedural wall textures** (no external files): Plaster, Brick, Concrete, Wood, Marble, White Tile — all generated via HTML Canvas 2D API → `THREE.CanvasTexture`
- **6 procedural floor textures**: Tile, Parquet, Marble, Concrete, Stone, Carpet
- **Material scope**: Apply to ALL WALLS or SELECT individual walls via raycasting (orange outline on selected)
- **Custom solid colour picker** for walls
- **Tile size slider** (0.3m–4m) — controls texture repeat scale
- **APPLY / RESET ALL** buttons
- **Amber REALISTIC mode indicator** badge on canvas (pulsing dot) — replaces the old light-background approach
- **`_split_walls_at_junctions()`** in `geometry_builder.py` — cuts wall A at every T/+ junction with wall B using reach-based gate (`t_b * Lb ∈ [-reach, Lb+reach]`)
- **`_assign_room_ids()`** in `geometry_builder.py` — tags every wall box with nearest room centroid
- **`mesh.userData.room_id`** stored on every wall mesh for future per-room filtering

### Fixed
- **Door leaf positioning** (`_door_leaf()`) — was using `rot_y` angle in `math.cos/sin` for world-space offset (wrong). Now uses wall unit vector `(ux, uy)` and perpendicular `(nx, ny)` directly
- **Window pushed to side** — `_reproj_openings()` v1 only updated `wall_idx`, leaving `t_center` as a fraction of the original unsplit wall. Fixed by re-projecting `op.x/op.y` onto the new sub-wall
- **TypeError crash on `_reproj_openings`** — v2 tried `dc_replace(op, t_start=..., t_end=...)` which don't exist on the `Opening` dataclass. Fixed to only replace `wall_idx` and `t_center`
- **Relative wall thickness classification** in viewer — replaced hardcoded pixel thresholds with `medianThick * 1.15` so it works at any PPM/scale
- **`file://` protocol detection** — now shows a clear toast explaining you must serve via HTTP, instead of silently failing
- **`AbortSignal.timeout()` compatibility** — replaced with manual `AbortController` + `setTimeout` for older browser support

### Changed
- Mode + Materials panel moved from left sidebar to collapsible **right panel**
- Room labels removed from 3D scene and legend panel
- Plaster texture darkened (`#d4c9b8` → `#9a8e80`) — much easier on the eyes
- Concrete texture darkened (`#8a8a82` → `#606058`) with added form-board lines and aggregate speckle
- Default realistic mode now applies **plaster texture** to all walls (was flat white `0xd4c9b8`)

### Known Issues
- Window sill/header parent wall lookup uses XZ proximity (≤0.8m) because backend doesn't send `wall_idx` on window dicts — accurate in most cases but may fail on very dense plans

---

## [Session 4] — Wall Splitting + Opening Fix

### Added
- `_split_walls_at_junctions(walls)` — T-junction and + junction wall splitting
- `_reproj_openings(openings, walls)` — re-projects opening positions after wall split

### Fixed
- Walls were not split at room junctions — inner wall ran through outer wall as one box
- Opening `wall_idx` was stale after splitting

---

## [Session 3] — Thickness Visual + PPM Fix

### Fixed
- `wall_detector.py`: removed thickness blend — raw measured values used directly (outer ≈ 0.271m, inner ≈ 0.180m)
- `raster_parser.py`: dedup radius made PPM-relative (`max(12, int(ppm * 0.5))`)
- `main.py`: `pixels_per_meter` default changed from `100.0` to `0.0` (auto-detect)
- Viewer: relative thickness classification (median-based) for outer vs inner wall visual distinction

---

## [Session 2] — Raster Parser Overhaul

### Added
- `_detect_border_crop()` — dynamic green frame detection, replaces fixed `BORDER_CROP_PX`
- Inter-segment opening detection (Pass 2) — gaps between collinear wall stubs
- T-junction opening detection (Pass 3) — doors between perpendicular wall endpoints
- `EDGE_GUARD=20` — prevents tick marks at image corners from becoming phantom doors

### Fixed
- Fixed `BORDER_CROP_PX=52` removing outer walls from small floor plans
- Removed SimpleDraw parser (caused regressions)

---

## [Session 1] — Initial Working Build

### Added
- FastAPI backend with `/upload`, `/process`, `/health` endpoints
- `raster_parser.py` — PNG/JPG/PDF → ParsedGeometry via Hough line detection
- `wall_detector.py` — Segment list → Wall list with thickness measurement
- `opening_detector.py` — DXF arc/line → door/window positions
- `room_detector.py` — Connected component room detection
- `geometry_builder.py` — Wall/Opening/Room → Three.js JSON
- `index_v2.html` — Three.js viewer with orbit camera, blueprint aesthetic
- Green tick mark PPM detection (`_detect_ppm()`)
- Autocrop for UI screenshots (`_autocrop()`)
