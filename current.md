# current.md — Active Work Status
> Floor Plan 3D Visualizer — What we're working on right now

---

## What We're Currently Doing

We are working on **`index_v2.html`** — the Three.js viewer — and the backend
pipeline together. The immediate focus is:

1. Getting walls to **look visually distinct** (thick outer vs thin inner)
2. **Breaking walls at room boundaries** so future per-room textures don't bleed
3. Fixing **duplicate door detection** at T-junctions

The viewer file is `viewer/index_v2.html`. Backend is a FastAPI server running
on `localhost:8000`. The test image we're using is **`fp4.png`** (640×355px,
L-shaped plan, 5 rooms: R1–R5, PPM≈33).

---

## Active Issues & Current Status

### 🔴 Issue A — All Walls Look the Same Thickness (PARTIALLY FIXED)

**What's happening:** In the 3D viewer every wall renders as the same visual
thickness even though outer walls (0.27m) and inner partitions (0.18m) are
physically different in the JSON.

**Root cause we found:** Two separate things were wrong:
- `wall_detector.py` was blending measured thickness 75% with a hardcoded
  `default_thickness = 0.40m`, which collapsed inner walls from 0.18m → 0.175m
  and outer walls from 0.27m → 0.30m. The ratio was being destroyed.
- `index_v2.html` was using one shared `wallMat` for all walls so even if
  `dimensions.depth` was different per wall, nothing made it visually obvious.

**What we fixed:**
- Removed the blend in `wall_detector.py` — raw measured value now used directly
- Updated `index_v2.html` to generate per-wall materials where thick walls get
  a brighter emissive value than thin ones

**Still needs testing:** We haven't seen a fresh render yet to confirm it looks
right. The thickness measurement itself IS proven correct (outer=0.271m,
inner=0.180m confirmed by standalone pixel scan test).

**If it still looks flat:** Check that the server is running the updated
`wall_detector.py` — if uvicorn is caching an old `.pyc` file the fix won't
show. Restart uvicorn after deploying.

---

### 🔴 Issue B — Duplicate Doors at T-Junctions (FIXED, NEEDS VISUAL CONFIRM)

**What's happening:** The debug image shows multiple orange `D` dots stacked
at the same door location. In the 3D view this means 2–3 door frames on top
of each other at every interior T-junction.

**Root cause:** Pass 2 (inter-segment scan) and Pass 3 (T-junction scan) both
detect the same gap, but their reported pixel positions differ by ~the wall
half-thickness (8–15px). The dedup check used a hardcoded `10px` radius which
was too small to catch these near-duplicates.

**What we fixed:** Changed dedup radius from `10px` to `max(12, int(ppm * 0.5))`
in both Pass 2 and Pass 3 in `raster_parser.py`. At ppm=33 this is ~16px,
enough to merge duplicates without merging genuinely separate close doors.

**Status:** Code fix is in. Needs a fresh debug image run to confirm orange
dot count reduced.

---

### 🟡 Issue C — Walls Not Split at Room Boundaries (FIXED IN CODE)

**What's happening:** A long outer wall spanning e.g. R1+R2+R3 is one single
`BoxGeometry` in Three.js. When you apply a texture to it, the texture stretches
across all three rooms. You can't give each room its own wall finish.

**What we built:**
- `_split_walls_at_junctions(walls)` in `geometry_builder.py` — cuts each wall
  wherever a perpendicular wall's endpoint lands on it (within 0.35m tolerance).
  The 20m top wall with 3 interior V-walls at x=5, x=10, x=15 correctly becomes
  4 × 5m segments. Verified by unit test.
- `_assign_room_ids(wall_boxes, rooms)` — tags every wall box JSON with
  `room_id` and `room_color` so the viewer knows which room each box belongs to.
- Opening re-projection — after splitting re-indexes walls, all door/window
  openings are re-matched to their correct new segment by world-position
  projection.

**Status:** Logic tested and working. Needs a full render test to confirm doors
still appear in the right places after the re-projection.

---

### 🟡 Issue D — Door Pushed Into Wall (ON HOLD)

**What's happening:** Screenshot 1 from this session shows the door leaf
geometry overlapping/sinking into the wall box rather than sitting flush with
the wall face.

**Cause:** The door leaf positioning in `geometry_builder.py` uses the wall
centerline. The leaf should be offset by `thickness/2` toward the room side,
which requires knowing the `swing_side` (left/right). The swing side is
detected but the offset calculation may not be applying it correctly for all
wall orientations.

**Decision:** Deferred — it's a cosmetic issue and doesn't block the MVP.
Will revisit after texturing is working.

**Quick fix idea when we return to it:** In `wall_to_boxes()` find the
`_door_leaf()` call and add `+ thick/2 * side_sign` to the position offset
perpendicular to the wall.

---

### 🟠 Issue E — Window Symbols Only Detected on V-Walls (KNOWN, NOT STARTED)

**What's happening:** The CAD window symbol scanner (Pass 1b in raster_parser)
only runs on vertical walls. Horizontal walls with the same double-line window
symbol are missed.

**Fix is simple:** The `_find_window_symbols` call inside the H-wall loop in
Pass 1 of `_detect_openings_from_image` is commented out or missing. Just
duplicate the V-wall call for H-walls, swapping the orientation parameter.

**Priority:** Low for MVP — windows are detected via gap scan (Pass 1a) as
structural openings even if the symbol isn't recognised.

---

### 🟠 Issue F — Corner Gap Edges (KNOWN, NOT STARTED)

**What's happening:** Where two walls meet at an L-corner (e.g. top-left of
the building), there's a small triangular gap visible in top-down view because
both wall endpoints stop at the wall centerline rather than extending to fill
the corner.

**Previous attempt:** `_compute_corner_trims()` was tried and reverted — it
trimmed inward and made gaps worse (documented in LEARNMAP).

**Correct approach:** `_compute_wall_extensions()` — extend each wall endpoint
outward by `other_wall.thickness / 2` at every L/T junction. This is the
opposite direction from what was tried before. Extend outward, not trim inward.

**Priority:** Medium — noticeable in top-down view, less visible in perspective.

---

## Texturing Plan (Next Big Feature)

This is the real reason we're breaking walls at room boundaries. Here's the
architecture we discussed:

### Backend — no changes needed
The `room_id` and `room_color` fields are now in every wall box JSON. The
backend doesn't need to know about texture URLs.

### Viewer — what needs adding to `index_v2.html`

**Step 1: Texture loader**
```javascript
const texLoader = new THREE.TextureLoader();
const roomTextures = {};  // room_id → THREE.Texture

function loadRoomTexture(roomId, url) {
  texLoader.load(url, tex => {
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    roomTextures[roomId] = tex;
    reapplyTextures();
  });
}
```

**Step 2: Texture repeat based on wall size**

This is critical — without correct repeat, a 5m wall and a 1m wall will show
the same number of tile repetitions, making the texture look stretched on one
and squashed on the other.

```javascript
function applyTextureToMesh(mesh, tex) {
  const w = mesh.geometry.parameters.width;   // wall length
  const h = mesh.geometry.parameters.height;  // wall height
  const TILE_SIZE = 1.0;  // metres per texture tile
  tex.repeat.set(w / TILE_SIZE, h / TILE_SIZE);
  mesh.material.map = tex;
  mesh.material.needsUpdate = true;
}
```

**Step 3: UI for room texture assignment**
A small panel in the sidebar — room list with a colour swatch and a texture
picker button. When a texture is chosen for a room, all wall meshes where
`mesh.userData.room_id === roomId` get re-materialised.

**Step 4: `mesh.userData` is already set**
We added this in the current session — every wall mesh already has:
```javascript
mesh.userData = { room_id: "room_1", room_color: "#2d6a9f", thickness: 0.27 }
```
So Step 3 can iterate `wallMeshes.filter(m => m.userData.room_id === id)` — no
further backend changes needed.

---

## Key Files Right Now

| File | What's actively changing |
|------|--------------------------|
| `viewer/index_v2.html` | Per-room colours, thickness brightness, userData — **this session** |
| `backend/app/core/geometry_builder.py` | Wall splitting, room_id assign, opening re-projection — **this session** |
| `backend/app/core/raster_parser.py` | Dedup radius fix — **this session** |
| `backend/app/core/wall_detector.py` | Thickness blend removed — **this session** |

---

## Test Checklist Before Declaring MVP Ready

- [ ] `fp4.png` renders with visually thicker outer walls vs thin partitions
- [ ] No stacked duplicate door frames at T-junctions
- [ ] Top-down view shows walls correctly split at room boundaries (no single
      wall spanning the full building width)
- [ ] All 5 doors still appear after the wall-splitting + opening re-projection
- [ ] Room labels R1–R5 still appear at correct centroids
- [ ] Larger test image (672×670 screenshot) still processes without regression

---

## How to Run

```bash
# Backend
cd backend
uvicorn app.main:app --reload --port 8000

# Viewer
# Open viewer/index_v2.html directly in browser
# Or serve it: python3 -m http.server 3000
```

API params sent by viewer:
```
POST /process/{job_id}?scale=1&wall_height=3&wall_thickness=0.2&pixels_per_meter=0
```
`pixels_per_meter=0` is critical — tells backend to auto-detect PPM from the
green tick marks. Do NOT change this to a fixed value.

---

## Things to NOT Touch

- `dxf_parser.py` — stable, unchanged throughout project
- `opening_detector.py` — stable, only used for DXF input
- `room_detector.py` — stable, works correctly
- `main.py` — stable, `pixels_per_meter` default must stay `0.0`
- `BORDER_CROP_PX` hardcoded values — dynamic detection replaced these,
  do not re-introduce fixed pixel constants (see LEARNMAP)
