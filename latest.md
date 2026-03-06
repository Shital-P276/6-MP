# latest.md — Session Changes Log
> Floor Plan 3D Visualizer — Changes made in this session (2026-03-07)

---

## 1. Duplicate Door Dedup Fix (`raster_parser.py`)

**Problem:** The same door was being detected multiple times at T-junctions.
Pass 2 (inter-segment) and Pass 3 (T-junction) both scan the same gap from
different directions — the V-wall endpoint scan and the H-wall endpoint scan
report positions that differ by roughly the wall half-thickness (~8–15px),
which slipped through the old hardcoded 10px dedup radius.

**Fix:** Replaced the hardcoded `10px` dedup radius in both Pass 2 and Pass 3
with `DEDUP_RADIUS_PX = max(12, int(ppm * 0.5))` — approximately 0.5m at any
image scale. This correctly merges duplicates at the same junction while still
keeping genuinely separate nearby doors distinct.

**Files changed:** `raster_parser.py` (Pass 2 dedup, Pass 3 dedup — both
V-wall endpoint scan and H-wall endpoint scan loops)

---

## 2. Wall Thickness Blend Removed (`wall_detector.py`)

**Problem:** `measure_wall_thicknesses()` was blending 75% measured + 25%
default thickness. The pipeline sets `default_thickness = min(0.40, 20/ppm)`,
which for fp4.png (ppm=33) is 0.40m. This pulled every wall toward 0.40m —
a correctly-measured thin inner wall (0.10m) became 0.175m, and an outer wall
(0.27m) became 0.30m, collapsing the visible difference between them.

**Fix:** Removed the blend entirely. The measured pixel width is now converted
to metres and clamped directly to `[THICK_MIN_M=0.05, THICK_MAX_M=0.55]`.
Also removed the now-unused `THICK_BLEND` constant and `default_thickness`
parameter from `measure_wall_thicknesses()`.

**Confirmed correct:** Standalone test on fp4.png shows outer walls = 0.271m,
inner partition walls = 0.180m — the 1.5:1 ratio is preserved.

**Files changed:** `wall_detector.py`

---

## 3. Wall Splitting at Room Junctions (`geometry_builder.py`)

**Problem:** Long outer walls spanning multiple rooms were rendered as a single
3D box. When textures are applied per wall box, a single long wall stretches
its texture across all rooms it borders — causing texture bleed.

**Fix:** Added `_split_walls_at_junctions(walls)` which runs after
`_trim_wall_corners`. For each wall, it collects all perpendicular wall
endpoints within `SNAP_TOL = 0.35m` of the wall's axis and inserts cut points
at those t-values (0..1 along the wall). Each resulting sub-segment preserves
the original wall's thickness, height, layer, and confidence.

Example: a 20m top outer wall bordered by 3 interior V-walls at x=5, x=10,
x=15 becomes 4 segments of 5m each — one per room face. Verified by unit test.

**Files changed:** `geometry_builder.py`

---

## 4. Per-Room Wall Colour + Thickness Visual in Viewer (`index_v2.html`)

**Problem 1:** All walls rendered in identical dark blue `0x1a2535` regardless
of which room they belong to, making it impossible to visually distinguish room
boundaries or tell thick outer walls from thin inner walls.

**Problem 2 (root cause of "thickness not showing"):** The viewer was using
the same single `wallMat` instance for all walls — even though `dimensions.depth`
was correct in the JSON, there was no visual differentiation.

**Fix:** Replaced the single shared `wallMat` with per-wall material generation:
- `room_color` (added by `_assign_room_ids` in geometry_builder) is blended
  75% dark base + 25% room hue, keeping the architectural aesthetic while making
  room ownership visible.
- Wall thickness drives emissive brightness: outer walls (thick, ratio≈1.0) glow
  noticeably brighter than inner partitions (thin, ratio≈0.6).
- Wireframe edge colour matches the room accent colour instead of flat cyan.
- `mesh.userData` stores `{ room_id, room_color, thickness }` for future
  texture slot assignment without any further backend changes.

**Files changed:** `index_v2.html`

---

## 5. Room ID Assignment (`geometry_builder.py`)

**New function:** `_assign_room_ids(wall_boxes, rooms)` — after all wall boxes
are built, each box is tagged with the nearest room's `id` and `color` (by
centroid distance in Three.js XZ space). This gives the viewer a stable
`room_id` per box that can drive texture selection, material override, or
future UI room-selection highlighting.

**Fields added to every wall box in the JSON response:**
- `room_id` — string id matching a room in `model.rooms`
- `room_color` — hex colour string (e.g. `"#2d6a9f"`)

**Files changed:** `geometry_builder.py`

---

## 6. Opening Re-projection After Wall Splitting (`geometry_builder.py`)

**Problem:** The original `build()` grouped openings by `wall_idx` before any
splitting. After `_split_walls_at_junctions` re-indexes walls, the old
`wall_idx` values point to wrong or nonexistent walls, so doors/windows would
go missing or appear on the wrong segment.

**Fix:** After splitting, all non-freestanding openings are re-projected onto
the new wall list. Each opening's world position `(op.x, op.y)` is projected
onto every wall segment; the wall with the closest perpendicular projection
(within `0 ≤ t ≤ 1`) wins. Freestanding openings (`wall_idx == -1`) are
excluded from this pass and handled by the existing freestanding block.

**Files changed:** `geometry_builder.py`

---

## Files Delivered This Session

| File | Change type |
|------|-------------|
| `raster_parser.py` | Bugfix — dedup radius ppm-relative |
| `wall_detector.py` | Bugfix — removed thickness blend |
| `geometry_builder.py` | Feature — wall splitting + room_id assignment + opening re-projection |
| `index_v2.html` | Feature — per-room colour, thickness-based brightness, room_id in userData |

---

## Known Remaining Issues (not fixed this session)

- **Door pushed into wall** — minor geometry offset, deferred (low priority for MVP)
- **Window symbols only detected on V-walls** — H-wall windows missed (Problem 4 from PROJECTMAP)
- **Corner half-cut edges** — `_compute_wall_extensions()` not yet added (Problem 5)
- **Thin wall door detection** — dynamic `SCAN_HALF` not yet implemented (Problem 2)
