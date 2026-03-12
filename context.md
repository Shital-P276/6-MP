# Floor Plan 3D Visualizer — Complete Project Context

**For**: Claude (and human developers) working on further development  
**Last Updated**: March 12, 2026  
**Version**: v2.2  
**Repository**: https://github.com/Shital-P276/6-MP/tree/v2.2

---

## 📋 Quick Summary

**FloorViz** is a full-stack application that converts architectural floor plans (DXF, PNG, JPG, PDF) into interactive 3D models for visualization, analysis, and virtual tours.

| Aspect | Detail |
|--------|--------|
| **Backend** | FastAPI (Python 3.10+) running standalone or containerized |
| **Frontend** | Three.js renderer (no build step, vanilla HTML/JS) |
| **Key Challenge** | Accurately extract walls, doors, windows from raster (image) floor plans |
| **Current Focus** | ML-based improvements (SegFormer Phase 1, MuraNet Phase 2) |
| **Status** | Production: core pipeline stable; raster detection has edge cases |

---

## 🚀 Quick Start (5 minutes)

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Expected output:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

Visit `http://localhost:8000/docs` for Swagger API docs.

### Frontend Setup

```bash
cd viewer
python -m http.server 3000
```

Open **`http://localhost:3000/index_v2.html`** (must be HTTP, not `file://`).

You should see a green dot in the top-right header: **"API CONNECTED"** ✓

### Test with Sample Floor Plan

1. In the viewer, click **"Select Floor Plan"** or drag a file onto the upload zone
   - Use samples from: `sample_data/floor_plan.dxf`, `floor_plan1.dxf`, etc.
   - Or supply a PNG/JPG/PDF of a floor plan

2. Click **"PROCESS FLOOR PLAN"**

3. Wait ~2–10 seconds (depending on file size and complexity)

4. Interact with the 3D model:
   - **Left drag**: Orbit camera
   - **Right drag**: Pan camera
   - **Scroll**: Zoom
   - **Keys**: `1`/`2`/`3`/`4` (views), `S` (solid), `W` (wireframe), `F` (floor), `T` (tour)

---

## 🏗️ Architecture at a Glance

```
┌─────────────────────────────────────────────┐
│         THREE.JS VIEWER (Browser)           │
│          index_v2.html + Canvas             │
│  • Orbit controls, material system          │
│  • Blueprint/Realistic mode toggle          │
│  • Virtual room tours with hotspots         │
│  • Selection + raycasting for wall picking  │
└──────────────────┬──────────────────────────┘
                   │ REST API (JSON)
┌──────────────────▼──────────────────────────┐
│       FASTAPI SERVER (Python, port 8000)    │
│  • /upload         → store file + job_id    │
│  • /process/{id}   → run pipeline           │
│  • /result/{id}    → return BuildingModel   │
│  • /debug/{id}     → return debug images    │
└──────────────────┬──────────────────────────┘
                   │ orchestrates
┌──────────────────▼──────────────────────────┐
│         PROCESSING PIPELINE (core/)         │
│  For DXF:                                   │
│    dxf_parser → wall_detector →             │
│    opening_detector → room_detector →       │
│    geometry_builder                         │
│  For Raster (PNG/JPG/PDF):                 │
│    raster_parser (big complex module) →     │
│    same downstream steps                    │
└───────────────────────────────────────────┘
```

---

## 📁 Directory Structure

```
floorviz/
├── backend/
│   ├── app/
│   │   ├── main.py                    ← FastAPI routes, parameters
│   │   └── core/
│   │       ├── __init__.py
│   │       ├── pipeline.py            ← Orchestrates all detection steps
│   │       ├── dxf_parser.py          ← Parses DXF files
│   │       ├── raster_parser.py       ← IMAGE→WALLS (Hough, morphology) ★ COMPLEX
│   │       ├── wall_detector.py       ← Raw segments → Wall objects + properties
│   │       ├── opening_detector.py    ← DXF/raster doors & windows
│   │       ├── room_detector.py       ← Detect rooms from wall topology
│   │       └── geometry_builder.py    ← Walls → Three.js JSON geometry
│   ├── models/                        ← Cached processed BuildingModels (JSON)
│   ├── uploads/                       ← Temporary uploaded files
│   ├── tests/
│   │   └── test_pipeline.py           ← Integration tests
│   ├── debug_test.py                  ← Manual test entry point
│   ├── context.md                     ← Backend-focused documentation
│   ├── ClaudeGuide.md                 ← ML implementation roadmap
│   ├── ClaudeGuideExtra.md            ← Dataset details, model choices
│   └── requirements.txt                ← Python dependencies
├── viewer/
│   ├── index_v2.html                  ← Main Three.js application
│   ├── index_v7.html, index_backup.html  ← Legacy versions
│   └── Ez_viewer.html                 ← Minimal demo
├── sample_data/
│   ├── floor_plan.dxf, floor_plan1.dxf, etc.
│   ├── *.png
│   └── generate*.py                   ← Scripts to create synthetic test plans
├── dependencies/
│   └── poppler-25.12.0/               ← PDF rendering engine (for pdf2image)
├── PROJECTMAP.md                      ← Function-level breakdown of each module
├── LEARNMAP.md                        ← Hard-won lessons from 5+ debugging sessions
├── README.md                          ← User-facing quick start
├── CHANGELOG.md                       ← Git-style version history
└── instructions.md                    ← Custom Copilot instructions (optional)
```

---

## 🔄 Core Processing Pipeline

### 1. File Upload → Job Creation

**Endpoint**: `POST /upload`

```javascript
// Frontend
const formData = new FormData();
formData.append("file", file);
const response = await fetch("http://localhost:8000/upload", {
  method: "POST",
  body: formData,
});
const { job_id, filename, format, status } = await response.json();
// status = "uploaded"
// Next: POST /process/{job_id}
```

**Backend** (`main.py`):
- Validates file type (DXF, PNG, JPG, PDF)
- Checks file size (max 50 MB)
- Saves to `uploads/{job_id}.{ext}`
- Creates `jobs[job_id]` entry with metadata

### 2. Processing → Model Generation

**Endpoint**: `POST /process/{job_id}?scale=1&wall_height=3&...`

**Key Parameters**:
```
scale              = 1.0 (1 CAD unit = X meters)
auto_scale         = true (infer from coordinate magnitude)
wall_height        = 3.0 metres
wall_thickness     = 0.2 metres (fallback; overridden by measured)
pixels_per_meter   = 0.0 (0 = auto-detect from green tick marks)
pdf_dpi            = 200 (resolution for PDF→image conversion)
hough_threshold    = 50 (Hough line detection sensitivity)
hough_min_length   = 30 pixels (minimum line segment length)
```

**Critical Note**: `pixels_per_meter=0` is REQUIRED for image files. Setting it to a fixed value (e.g., 100) breaks scale-relative heuristics. Always auto-detect.

**Flow** (`pipeline.py`):

```python
def process(filepath: str, ...) -> PipelineResult:
    # 1. Detect file type
    if filepath.endswith('.dxf'):
        parsed = _parse_dxf(filepath)  # → ParsedGeometry
    else:  # PNG, JPG, PDF
        parsed = _parse_raster(filepath)  # → ParsedGeometry
    
    # 2. Common downstream pipeline
    walls = wall_detector.detect(parsed.segments)     # Raw Wall objects
    openings = opening_detector.detect(...)            # Doors/windows
    rooms = room_detector.detect(walls, openings)     # Room polygons
    
    # 3. Build 3D model
    model = geometry_builder.build(walls, openings, rooms)
    # → BuildingModel (serializable to JSON)
    
    # 4. Cache + return
    save_to_models/{job_id}.json
    return model
```

### 3. Result Retrieval → Viewer Display

**Endpoint**: `GET /result/{job_id}`

Returns `BuildingModel` JSON:
```json
{
  "version": "1.0",
  "unit": "m",
  "walls": [
    {
      "id": "wall_0",
      "positions": [[0,0,0], [5,0,0], ...],
      "wall_thickness": 0.2,
      "height": 3.0,
      "room_id": "room_1",
      "material": "plaster"
    }
  ],
  "doors": [
    {
      "id": "door_0",
      "position": [2.5, 0, 0],
      "rotation": 0,
      "width": 0.8,
      "height": 2.1,
      "swing": "left"
    }
  ],
  "rooms": [
    {
      "id": "room_0",
      "name": "HALL",
      "centroid": [5, 5],
      "area": 25.0,
      "polygon": [[0,0], [10,0], [10,10], [0,10]]
    }
  ]
}
```

**Viewer** (`index_v2.html`):
- Parses JSON
- Creates THREE.Mesh geometry for each wall
- Applies materials (textures or colors)
- Sets up lighting, camera, controls
- Renders + allows interaction

---

## 🔧 Key Components Deep Dive

### `raster_parser.py` ★ (The Complex One)

**Purpose**: Convert floor plan image → wall segments, door/window openings, room polygons

**Why It's Complex**:
- Floor plans have huge visual variation (hand-drawn, CAD, photos, Indian/Western styles)
- No standard color scheme
- Text overlays, dimension markers, scale annotations
- Thick walls, thin walls, curved walls, columns, double walls

**Pipeline** (10 steps):

```python
def parse(image_path: str) -> ParsedGeometry:
    # Step 1: Auto-crop UI chrome
    #   Detects floor plan panel in screenshot, crops to interior
    img = _autocrop(img)
    
    # Step 2: Detect scale (PPM)
    #   Scans top 5% for green tick marks → inter-gap distance → pixels/meter
    ppm = _detect_ppm(img)
    
    # Step 3: Detect border crop (green frame)
    #   Finds inner edge of annotation frame dynamically
    crop_px = _detect_border_crop(img)
    
    # Step 4: Segment pixels by brightness
    #   WALL: 82-148, ROOM: 153-244, GREEN: masked out
    wall_pixels, room_pixels = _segment(img)
    
    # Step 5: Skeletonize
    #   Morphological thinning → 1px centerlines for wall axes
    skeleton = _skeleton(wall_pixels)
    
    # Step 6: Detect walls (Hough + merge)
    #   HoughLinesP on skeleton → snap to H/V → merge collinear → dedup thick walls
    segments = _detect_walls(skeleton)
    
    # Step 7: Complete outer walls
    #   Add missing sides from bounding box
    segments = _complete_outer_walls(segments)
    
    # Step 8: Detect openings (doors/windows)
    #   Pass 1: brightness scan within segments (jamb gaps)
    #   Pass 2: collinear inter-segment gaps
    #   Pass 3: T-junctions (orphan doorway stubs)
    openings = _detect_openings_from_image(segments, wall_pixels)
    
    # Step 9: Detect rooms
    #   Fill (seal) doorway gaps → connected components
    rooms = _detect_rooms(wall_pixels, openings)
    
    return ParsedGeometry(segments, openings, rooms)
```

**Critical Constants** (v2.2 stable):

```python
BORDER_CROP_PX = 4              # minimum; overridden by _detect_border_crop()
MERGE_GAP = 60                  # max collinear gap to bridge (px)
MIN_WALL_PX = 40                # minimum segment length (filters noise)
COORD_GROUP_TOL = 11            # groups parallel H/V double-edges
DEDUP_DIST = 20                 # collapses both faces of thick wall
WALL_LO, WALL_HI = 82, 148      # grayscale range for wall pixels
ROOM_LO, ROOM_HI = 153, 244     # grayscale range for room floor
GREEN_DIFF = 22                 # (G-R > 22) AND (G-B > 22) → green
SCAN_HALF_PX = 14               # brightness sample band width
OPENING_BRIGHT = 148            # gap detection threshold
```

**Known Issues & Workarounds**:

| Issue | Root Cause | Fix | Status |
|-------|-----------|-----|--------|
| Thick walls detected twice | Raster capture has wall on both sides | DEDUP_DIST=20 merges them | ✅ Fixed v2.1 |
| Small doors missed | brightness scan band too narrow | Auto-scale based on PPM | 🔄 Planned |
| Curved walls → many small segments | Hough doesn't detect curves | Post-process arcs manually | 🔄 Phase 2 (SegFormer) |
| PPM detection fails on no-tick plans | Relies on green frame marks | Fallback PPM=65 (manual calibration) | ⚠️ Not ideal |
| Dimension text → false wall segments | Text pixel brightness overlaps WALL_LO-HI | Green mask + dilation helps, not perfect | ⚠️ Phase 2 (ML) |

**Recent Fixes** (Session 5):
- `_split_walls_at_junctions()` — cuts inner walls at T-junctions to prevent overlap
- `_reproj_openings()` — fixes opening positions referenced to split walls
- Door leaf positioning now uses wall unit vector, not Three.js rotation angles

### `wall_detector.py`

**Input**: `ParsedGeometry.segments` (list of segment dicts with endpoints, length)  
**Output**: `List[Wall]` with properties: thickness, height, layer, confidence, paired_wall_id

**Logic**:
```python
def detect(segments: List[dict]) -> List[Wall]:
    walls = []
    
    # 1. Create initial Wall objects from segments
    for seg in segments:
        wall = Wall(
            id=f"wall_{i}",
            x0, y0 = seg['start']
            x1, y1 = seg['end']
            length = seg['length']
            thickness = estimate_thickness_from_neighbors(seg)
            height = default_wall_height
            layer = "generic"
            confidence = "high"  # or estimated from HOG
        )
        walls.append(wall)
    
    # 2. Find double walls (paired walls running parallel)
    for i, w1 in enumerate(walls):
        for j, w2 in enumerate(walls[i+1:]):
            if parallel(w1, w2) and distance(w1, w2) < 0.5:
                w1.paired_wall_id = w2.id
                w2.paired_wall_id = w1.id
    
    return walls
```

**Status**: ✅ Stable. No recent changes needed.

### `opening_detector.py`

**Input**: DXF entities (for DXF) OR raster openings dict (from raster_parser)  
**Output**: `List[Opening]` with position (x, y), size (width, height), type (door/window), swing direction

**Logic** (simplified):
```python
def detect(walls: List[Wall], segments: List[dict] | List[Entity], **kwargs) -> List[Opening]:
    openings = []
    
    if is_dxf:
        # Extract DOOR/WINDOW entities
        for entity in dxf_entities:
            op = Opening(
                id=entity.dxf.name,
                x, y = entity.dxf.insert,
                width = entity.dxf.xscale,
                height = entity.dxf.yscale,
                type = "door" if ... else "window",
                swing = "in" | "out" | "left" | "right"
            )
            openings.append(op)
    else:
        # Raster: openings already detected by raster_parser
        # Now assign wall_idx and t_center using proximity matching
        for op in raster_openings:
            best_wall_idx, best_t = match_opening_to_wall(op, walls)
            op.wall_idx = best_wall_idx
            op.t_center = best_t  # [0..1] along wall
            openings.append(op)
    
    return openings
```

**Recent Fix** (Session 4):
- Added `_reproj_openings()` to update `wall_idx` and `t_center` after wall splitting at junctions

### `room_detector.py`

**Input**: Wall segments, opening positions  
**Output**: `List[Room]` with centroid, area, floor polygon, name (HALL, BEDROOM, etc.)

**Logic**:
```python
def detect(walls: List[Wall], openings: List[Opening]) -> List[Room]:
    rooms = []
    
    # 1. Fill doorway gaps in wall pixel mask
    filled_mask = _seal_openings(wall_pixels, openings)
    
    # 2. Connected components labeling
    labels = cv2.connectedComponents(filled_mask)[1]
    
    # 3. For each component, extract polygon + centroid + name
    for label_id in range(1, num_labels):
        room_pixels = labels == label_id
        polygon = _extract_polygon(room_pixels)
        centroid = _compute_centroid(room_pixels)
        area = cv2.contourArea(polygon)
        
        room = Room(
            id=f"room_{label_id}",
            name=_infer_room_type(area, polygon),  # HALL, BEDROOM, etc.
            centroid=centroid,
            area=area,
            polygon=polygon
        )
        rooms.append(room)
    
    return rooms
```

**Status**: ✅ Stable. Naming heuristic based on room area.

### `geometry_builder.py` ★ (Recently Refactored)

**Input**: `walls: List[Wall]`, `openings: List[Opening]`, `rooms: List[Room]`  
**Output**: `BuildingModel` (JSON-serializable) with wall boxes, door geometry, floor, materials

**Key Responsibilities**:

1. **Wall Splitting at Junctions** (`_split_walls_at_junctions`)
   - Detects T-junctions and + junctions using perp distance + parameter range checking
   - Cuts the intersected wall into multiple sub-walls
   - Each sub-wall inherits thickness, height, layer, etc.

2. **Opening Re-projection** (`_reproj_openings`)
   - After splitting, wall indices change
   - Updates opening `wall_idx` and `t_center` to reference new sub-walls
   - Computes `t_center` from opening's world position projected onto new wall

3. **Door Leaf Generation** (`_door_leaf`)
   - Creates a hinged rectangle geometry for door swing visualization
   - Uses wall unit vector and perpendicular to position correctly
   - (Not just Three.js rotation angles, which was a bug)

4. **Procedural Textures** (`_make_texture`)
   - All textures generated via HTMLCanvas 2D API → CanvasTexture
   - No external files, no HTTP requests, works offline
   - Includes concrete, plaster, brick, marble, tile, parquet, carpet

5. **Material System** (recent Session 5)
   - Blueprint mode: grayscale flat color (0xcccccc or 0x999999)
   - Realistic mode: applies procedural texture + lighting
   - Per-wall customization via raycasting + color picker
   - Stores original material in WeakMap before switching modes

**Status**: 🔄 Actively refined. Latest changes Session 5 (material system, wall splitting).

---

## 💡 Critical Design Patterns

### Pattern 1: Auto-Scale Detection (PPM)

**Problem**: Floor plans in images have no inherent unit. A 2000px wall could be 20m or 200m.

**Solution**: Green tick marks on the image border encode scale.
- Top 5% of image scanned for green pixels (G-R > 22, G-B > 22)
- Inter-gap distance (e.g., 130px) = distance between tick marks (e.g., 4m)
- PPM = gap_distance_px / distance_m = 130 / 4 = 32.5 px/m

**Fallback**: If no ticks detected, PPM = 65 (reasonable default for typical floor plans)

### Pattern 2: Wall Splitting at Junctions

**Problem**: Inner walls running perpendicular to outer walls were drawn as single long walls piercing through multiple walls, creating overlaps.

**Solution**: Detect junctions and split.
```
Before:        Wall A (outer)
  +-----------+
  |           |
  |--Wall B---+
  |
  +-----------+

After:    Wall A split at B's junction
  +-----------+
  |     A1    |
  |--Wall B---+
  |     A2    |
  +-----------+
  
  Wall A → [Wall A1, Wall A2]
  Wall B remains continuous
```

**Implementation**: For each pair (A, B), check if B's body crosses A perpendicularly at parameter `t_b`:
```python
if t_b * Lb in [-reach, Lb + reach] and B perp to A:
    split_A_at_junction(t_a)
reach = wall_thickness[B] / 2 + 0.15  # 0.15m = measurement tolerance
```

### Pattern 3: Opening Assignment After Splitting

**Problem**: Openings (doors/windows) reference `wall_idx`. After splitting, those indices are stale.

**Solution**: Re-project each opening onto the new wall topology.
```python
def _reproj_openings(openings, walls):
    for op in openings:
        # (op.x, op.y) is world position
        best_idx, best_t = find_best_wall(op.x, op.y, walls)
        op.wall_idx = best_idx
        op.t_center = best_t  # recompute from world position
```

### Pattern 4: Procedural Texture Generation (No External Files)

**Problem**: Viewing app needs textures. HTTP requests fail offline. CORS issues with external files.

**Solution**: Generate at runtime using HTMLCanvas 2D API:
```javascript
const canvas = new OffscreenCanvas(512, 512);
const ctx = canvas.getContext("2d");

// Draw repeating pattern
for (let x = 0; x < 512; x += tileSize) {
  // draw brick, concrete, marble, etc. as pixel art
}

const texture = new THREE.CanvasTexture(canvas);
texture.repeat.set(scale_x, scale_y);
material.map = texture;
```

All textures live in `index_v2.html`'s `makeTex()` function. Add new textures there.

### Pattern 5: Selection Highlighting via Raycasting

**Problem**: User clicks on a wall → highlight it, allow material/color change.

**Solution**: Three.js raycasting + WeakMap cache:
```javascript
// On mouse click
raycaster.setFromCamera(mouse, camera);
const hits = raycaster.intersectObjects(wallMeshes);
if (hits.length > 0) {
    const wallMesh = hits[0].object;
    
    // Add orange outline
    const outline = new THREE.LineSegments(
        new THREE.EdgesGeometry(wallMesh.geometry),
        new THREE.LineBasicMaterial({ color: 0xff9900 })
    );
    outline.scale.multiplyScalar(1.002);  // avoid z-fighting
    wallMesh.userData._selectionOutline = outline;
    scene.add(outline);
}
```

---

## ⚙️ API Reference

### POST /upload

Upload a floor plan file.

**Request**:
```
Content-Type: multipart/form-data
Body:
  file: <binary file data>
```

**Response** (200 OK):
```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "filename": "floor_plan.png",
  "format": ".png",
  "size_mb": 2.345,
  "status": "uploaded",
  "next": "POST /process/a1b2c3d4-..."
}
```

**Status Codes**:
- `200` — Success
- `400` — Unsupported format (not DXF/PNG/JPG/PDF)
- `413` — File too large (> 50 MB)

### POST /process/{job_id}

Process the uploaded file. Returns a 3D model.

**Query Parameters**:
```
scale              float  default=1.0        # 1 CAD unit = X meters
auto_scale         bool   default=true       # auto-infer scale
wall_height        float  default=3.0        # meters
wall_thickness     float  default=0.2        # meters
pixels_per_meter   float  default=0.0        # 0 = auto from tick marks
pdf_dpi            int    default=200        # PDF resolution
hough_threshold    int    default=50         # Hough vote threshold
hough_min_length   int    default=30         # min line segment (px)
```

**Response** (200 OK):
```json
{
  "job_id": "a1b2c3d4-...",
  "status": "completed",
  "processing_time_s": 5.234,
  "building": {
    "version": "1.0",
    "unit": "m",
    "walls": [...],
    "doors": [...],
    "windows": [...],
    "rooms": [...]
  }
}
```

**Status Codes**:
- `200` — Success
- `404` — Job ID not found
- `500` — Processing error (see response `error_message`)

### GET /result/{job_id}

Retrieve the cached processing result.

**Response** (200 OK):
```json
{
  "job_id": "a1b2c3d4-...",
  "building": { ... }
}
```

### GET /debug/{job_id}

Return debug images for the raster pipeline (for troubleshooting).

**Response** (200 OK, image/png):
```
Composite image showing:
- Original input
- Wall pixel mask
- Segments detected
- Openings detected
- Final rooms
```

Very useful for tuning raster_parser constants.

### GET /health

Check API status.

**Response** (200 OK):
```json
{
  "status": "ok",
  "version": "0.2.0",
  "supported_formats": [".dxf", ".png", ".jpg", ".pdf"]
}
```

---

## 📊 Data Structures

All data classes defined in `core/pipeline.py`. Key ones:

### ParsedGeometry
```python
@dataclass
class ParsedGeometry:
    segments: List[dict]          # raw wall segments from parser
    openings: List[dict]          # raw openings from parser
    rooms: List[dict]             # raw room polygons from parser
    ppm: float                    # pixels per meter (for raster)
    scale: float                  # CAD units to meters (for DXF)
```

### Wall
```python
@dataclass
class Wall:
    id: str
    x0: float; y0: float          # start point
    x1: float; y1: float          # end point
    length: float                 # computed
    thickness: float              # measured or default
    height: float                 # default 3.0m
    layer: str                    # DXF layer name
    confidence: str               # "high", "medium", "low"
    paired_wall_id: str | None    # ID of parallel double wall
```

### Opening
```python
@dataclass
class Opening:
    id: str
    x: float; y: float            # world position (center)
    width: float                  # or "z" dimension in DXF
    height: float                 # or "y" dimension in DXF
    type: str                     # "door" or "window"
    swing: str | None             # "left", "right", "in", "out"
    wall_idx: int                 # index in walls[] list
    t_center: float               # [0..1] position along wall
```

### Room
```python
@dataclass
class Room:
    id: str
    name: str                     # "BEDROOM 1", "HALL", etc.
    centroid: tuple[float, float]
    area: float
    polygon: List[tuple[float, float]]
```

### BuildingModel (JSON)
```json
{
  "version": "1.0",
  "unit": "m",
  "ppm": 32.5,
  "walls": [
    {
      "id": "wall_0",
      "positions": [...],
      "thickness": 0.2,
      "height": 3.0,
      "material": "plaster",
      "room_id": "room_1"
    }
  ],
  "doors": [...],
  "windows": [...],
  "rooms": [...]
}
```

---

## 🧪 Testing & Debugging

### Manual Testing

**File**: `backend/debug_test.py`

```bash
cd backend
python debug_test.py
```

Processes `sample_data/floor_plan.dxf` and saves result to `test_output.json`.

### Unit Tests

**File**: `backend/tests/test_pipeline.py`

```bash
cd backend
pytest tests/ -v
```

Tests:
- DXF parsing
- Raster parsing (Hough, wall detection)
- Building model generation
- API endpoints

### Debug Images

After processing, request debug images for raster files:

```bash
# After processing, get debug visualization
curl "http://localhost:8000/debug/{job_id}" > debug.png
```

Shows:
1. Input image (original)
2. Wall pixel mask (grayscale)
3. Hough line segments overlaid
4. Final detected walls (colored)
5. Openings (doors/windows) marked
6. Rooms (colored regions)

Use this to tune constants if wall detection is poor.

### Common Issues & Solutions

| Symptom | Likely Cause | Debug Steps |
|---------|--------------|-------------|
| No walls detected | PPM auto-detection failed | Check for green tick marks; set `pixels_per_meter` manually |
| Walls too thick | DEDUP_DIST too large | Reduce DEDUP_DIST in raster_parser.py |
| Doors/windows missed | SCAN_HALF_PX band too narrow | Increase or enable dynamic scaling |
| Floor plan cropped | BORDER_CROP_PX or _detect_border_crop issue | Check green frame detection; adjust manually |
| Outer walls incomplete | _complete_outer_walls not activating | Ensure wall_pixels are correct |
| Doors on wrong wall | Opening assignment failed | Check wall proximity in _reproj_openings |

---

## 🎯 Development Guidelines

### Code Style

- **Python**: PEP 8 (4 spaces, snake_case)
- **JavaScript**: Vanilla ES6+, no external libraries except Three.js
- **Comments**: Inline comments for complex logic; docstrings for functions
- **Type hints**: Python dataclasses with `@dataclass` decorator

### Adding a New Feature

1. **Design phase**:
   - Sketch data flow: what goes in, what comes out
   - Check if it modifies `ParsedGeometry`, `Wall`, `Opening`, `Room`, or `BuildingModel`
   - If not, you're adding a new detection step or viewer feature

2. **Implementation**:
   - Add function to appropriate module (or create new module in `core/`)
   - Use type hints for clarity
   - Write unit tests

3. **Integration**:
   - Update `pipeline.py` to call your new function
   - Update related detector (e.g., `wall_detector.py`, `room_detector.py`)
   - Test end-to-end with sample files

4. **Documentation**:
   - Add docstring to function
   - Update relevant section in `PROJECTMAP.md`, `LEARNMAP.md`, or `context.md`
   - Add entry to `CHANGELOG.md`

### Modifying Constants

Constants live in:
- `raster_parser.py`: BORDER_CROP_PX, MERGE_GAP, MIN_WALL_PX, SCAN_HALF_PX, etc.
- `geometry_builder.py`: Material colors, texture sizes
- `main.py`: API defaults, file size limits

**Before changing**:
1. Understand its effect (read comments and surrounding code)
2. Test on 3–5 sample files (DXF + raster)
3. Update `LEARNMAP.md` with reasoning
4. Record in `CHANGELOG.md`

### Commits

Useful git workflow:
```bash
# Before starting work
git checkout -b feature/wall-splitting

# Work on files
# ... edit code ...

# Commit
git add backend/app/core/geometry_builder.py
git commit -m "feat(geometry_builder): split walls at T-junctions

- Detects intersections using perp distance + t-parameter range
- Cuts intersected walls into multiple sub-walls
- Re-projects openings onto new wall topology
- Fixes overlapping wall meshes in 3D viewer"

# Push + create PR
git push origin feature/wall-splitting
```

---

## 📈 Roadmap & Future Work

### Phase 1: SegFormer Preprocessor (Weeks 1–2)

**Goal**: Improve raster parsing by using ML segmentation instead of brightness scanning.

**Approach**:
- Fine-tune SegFormer (NVIDIA) on CubiCasa5k dataset
- Outputs: wall_mask, door_mask, window_mask
- Feed wall_mask as cleaned input to existing raster_parser

**Why**: Removes dependency on color ranges (WALL_LO, WALL_HI), which break on unusual color schemes and text overlays.

**Status**: 🔄 Planned. See `backend/ClaudeGuide.md` for full ML roadmap.

### Phase 2: MuraNet End-to-End Replacement (Weeks 3–8)

**Goal**: Replace all of raster_parser.py with a learned model that directly outputs wall segments + openings.

**Architecture**:
- Encoder: Mix-Transformer (same as SegFormer)
- Decoder 1: Segmentation head → wall/door/window masks
- Decoder 2: Detection head → bounding boxes + swing directions
- Output: ParsedGeometry directly (bypassing Hough, brightness scanning, etc.)

**Advantage**: Generalizes to curved walls, unusual floor plans, non-CAD styles.

### Phase 3: Indian Plan Fine-Tuning (Ongoing)

**Goal**: Collect and annotate real Indian building floor plans; fine-tune MuraNet checkpoint.

**Characteristics**:
- Thick masonry walls (often 200mm+)
- Dimension annotations in Hindi
- Column markers
- Less standardized color schemes

### Phase 4: Virtual Tours (Current Session?)

**Status**: ✅ Complete (Session 5). Includes:
- Hotspots at room centroids
- Smooth camera fly-to animation
- HUD navigation bar with room list
- Minimap + crosshair
- Autoplay mode (4-second interval)

### Phase 5: VR Integration (Post v2.2)

**Goal**: Export to VR-ready formats (WebXR, Oculus SDK, HTC Vive).

**Approach**:
- Use three.js WebXR examples as template
- Hand controls (grab walls, doors; orient via head)
- Teleport locomotion (instead of smooth walk, which causes nausea)

---

## 📚 References & Resources

### Papers

- **SegFormer** (NVIDIA, 2021): "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers"
  - Model card: https://huggingface.co/nvidia/segformer-b2

- **CubiCasa5k** (2019): "CubiCasa5K: A Dataset and an Improved Multi-Task Model for Floorplan Image Analysis"
  - GitHub: https://github.com/CubiCasa/CubiCasa5k

- **ResPlan** (2025): Dataset with 17,000 residential floor plans
  - (Download link in `ClaudeGuideExtra.md`)

### Code Snippets

- Three.js raycasting: https://threejs.org/docs/#api/en/core/Raycaster
- HTMLCanvas textures: https://threejs.org/docs/#api/en/textures/CanvasTexture
- OpenCV Hough: https://docs.opencv.org/4.5.0/dd/d1a/group__imgproc__feature.html#gadf1dda2656e51e73ebcd3f12fac1f5b2

### External Tools

- **DXF Editor**: LibreCAD (free, open-source)
- **PDF Viewer with debug**: qpdf (CLI tool)
- **Image Editor**: GIMP (for creating synthetic test plans)
- **3D Viewer**: Three.js Playground https://threejs.org/manual/examples/threejs-inspector.html

---

## 🤝 Collaboration Notes

### For Claude (AI Assistant)

When working on this project:
1. **Always check LEARNMAP.md first** — it contains hard-won debugging knowledge
2. **Read PROJECTMAP.md** before modifying any core module
3. **Test on multiple file types**: at least one DXF + one image (PNG/JPG) + one PDF
4. **Never change PPM detection without understanding the consequences** — many heuristics depend on it
5. **When adding ML code**, follow structure in `ClaudeGuide.md` (Phase 1 → Phase 2 → Phase 3)

### For Human Developers

- Sync with `LEARNMAP.md` after each session — record what worked, what failed, why
- Keep `CHANGELOG.md` updated (guides future debugging)
- Use `PROJECTMAP.md` as the source of truth for module responsibilities
- Before proposing major changes (new parser, new ML model), discuss in context.md first

---

## 📞 Quick Reference

| Need | Location |
|------|----------|
| How to start backend | README.md, this file (Quick Start) |
| Module responsibilities | PROJECTMAP.md |
| Lessons from debugging | LEARNMAP.md |
| Constants & tuning | `raster_parser.py`, `geometry_builder.py` |
| ML implementation plan | `backend/ClaudeGuide.md`, `ClaudeGuideExtra.md` |
| Version history | CHANGELOG.md |
| API docs | `main.py` docstrings + Swagger at `/docs` |
| Test files | `backend/tests/`, `debug_test.py` |

---

**Last reviewed**: March 12, 2026  
**Next review**: After major feature completion or when LEARNMAP grows significantly
