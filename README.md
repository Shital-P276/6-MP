# FloorViz

FloorViz is a full-stack floor plan processing and visualization system that converts **2D architectural inputs** (DXF, raster images, or first-page PDF) into a structured **3D building model JSON** and renders it in a browser-based **Three.js viewer**. The backend (FastAPI) performs parsing, wall/room/opening detection, and geometry construction; the frontend provides upload, processing controls, multiple visualization modes, material editing, room-tour navigation, and VR entry points for interactive exploration.

---

## Core Features

- Multi-format floor plan ingestion:
  - Vector: `.dxf`
  - Raster: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`
  - Document: `.pdf` (first page rendered to image for processing)
- API-driven processing pipeline with per-job lifecycle (`upload` → `process` → `model retrieval`).
- Automated wall detection and pairing, room detection, opening (door/window) detection, and 3D geometry assembly.
- Raster/PDF debug image generation endpoint for visual verification of detected line structures.
- Browser viewer with:
  - Upload + process UI
  - Blueprint vs realistic rendering modes
  - Texture/material controls (global or selected-wall scope)
  - Layer visibility toggles (walls, wireframe, floor, doors/windows, labels)
  - Guided virtual room tour UX (hotspots, HUD, minimap, autoplay)
  - VR entry logic with WebXR + desktop fallback paths.

---

## High-Level Architecture

```text
[Client: viewer/index.html]
      |
      | HTTP (multipart upload + JSON APIs)
      v
[FastAPI app: backend/app/main.py]
      |
      v
[ProcessingPipeline: backend/app/core/pipeline.py]
      |
      +--> DXFParser (vector extraction)
      +--> RasterParser (image/PDF extraction + masks + openings)
      +--> WallDetector (segments -> wall objects)
      +--> RoomDetector (rooms from wall topology)
      +--> OpeningDetector (DXF/opening association)
      +--> GeometryBuilder (walls + rooms + openings -> model JSON)
      |
      v
[Persisted artifacts]
  /data/uploads/<job_id>.<ext>
  /data/models/<job_id>.json
```

The backend is stateless at the HTTP layer but stores in-memory job metadata in a process-local dictionary and persists uploaded/model files on disk under `/data/*` paths. If the process restarts, in-memory job state is lost even if files remain.

---

## Repository / File Structure

```text
FloorViz/
├── README.md
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   └── core/
│   │       ├── __init__.py
│   │       ├── dxf_parser.py
│   │       ├── raster_parser.py
│   │       ├── wall_detector.py
│   │       ├── room_detector.py
│   │       ├── opening_detector.py
│   │       ├── geometry_builder.py
│   │       └── pipeline.py
│   ├── tests/
│   │   └── test_pipeline.py
│   ├── models/                  # many generated model JSON snapshots
│   ├── debug_test.py
│   └── debug.png
├── viewer/
│   ├── index.html               # main current UI (Three.js app)
│   ├── index_v11.html
│   ├── index_v11_fixed.html
│   ├── index_v12.html
│   └── vr-check.html
├── sample_data/
│   ├── *.dxf
│   ├── img/*.png
│   └── generate*.py             # sample data generation helpers
├── dependencies/
│   └── poppler-25.12.0/         # bundled poppler binaries/assets
└── Automating_Spatial_Reality.pptx
```

Notes:
- `viewer/index.html` appears to be the actively evolved frontend; older `index_v11*` and `index_v12.html` are retained variants.
- `backend/models/` currently contains many previously generated output JSON files, indicating historical local processing runs.

---

## Backend Architecture

## API Server (`backend/app/main.py`)

`main.py` defines the FastAPI app, CORS middleware (`allow_origins=["*"]`), upload/model directories, in-memory job state, and route handlers.

### Runtime storage behavior

- Upload files are saved to: `/data/uploads/<uuid>.<ext>`
- Processed JSON models are saved to: `/data/models/<uuid>.json`
- Job status and metadata are stored in global dict `jobs: dict[str, dict]`

### Lifecycle model

Each uploaded plan is assigned `job_id` and transitions through statuses:
- `uploaded`
- `processing`
- `done` or `error`

There is no external queue/worker process; processing occurs synchronously in the request handling thread for `POST /process/{job_id}`.

---

## Processing Pipeline (`backend/app/core/pipeline.py`)

`ProcessingPipeline.run(filepath)` orchestrates the full conversion path.

### Format routing

- Vector (`.dxf`) → `DXFParser`
- Raster/PDF → `RasterParser`
- Optional SimpleDraw branch is referenced (`SimpleDrawParser`) via guarded imports; if unavailable, parser falls back to standard raster flow.

### Stages

1. **Parse source into `ParsedGeometry`**
2. **Fallback wall assignment** if no explicit wall layer is found
3. **Detect walls** via `WallDetector`
4. **Detect rooms** via `RoomDetector`
5. **Detect openings**
   - Raster path may use parser-provided `_raster_openings`
   - Otherwise uses `OpeningDetector`
6. **Build 3D model** with `GeometryBuilder`
7. **Return `PipelineResult`** including stats + warnings + metadata

### Output envelope

The pipeline returns a `PipelineResult` dataclass, serialized through `to_dict()`, containing:
- `success`
- `processing_time_ms`
- `warnings`
- `source_type`
- `applied_scale`
- `model` (when successful)
- `stats`:
  - `wall_segments`
  - `walls_detected`
  - `paired_walls`
  - `rooms_detected`
  - `doors_detected`
  - `windows_detected`

---

## Frontend Architecture

## Viewer (`viewer/index.html`)

The frontend is a **single self-contained HTML application** (markup, CSS, and JavaScript in one file) that loads Three.js and handles all UI + render logic client-side.

### Major frontend subsystems

- **Connectivity & notifications**
  - API health checks and status indicator
  - timestamped log entries
  - toast notifications
- **Upload/process panel**
  - file input + drag/drop
  - process controls (scale, wall height/thickness, etc.)
  - invokes backend `/upload` and `/process/{job_id}`
- **3D scene management**
  - camera, renderer, lights, grid, orbit controls
  - model instantiation from backend JSON response
- **View and layer toggles**
  - perspective/top/front/side views
  - solid/wire/floor/doors/windows/labels visibility
- **Material system**
  - procedural wall/floor textures
  - selection-aware material assignment
  - blueprint vs realistic mode switching
- **Tour system**
  - room hotspot generation
  - animated fly-to transitions
  - HUD navigation and autoplay
  - minimap and room progress state
- **VR subsystem**
  - desktop fallback controls
  - WebXR detection/session integration
  - session-specific navigation overlays

---

## Step-by-Step Request Flow

1. User opens viewer via HTTP server (not `file://`).
2. Viewer checks backend health (`GET /health`).
3. User uploads plan file.
4. Viewer sends multipart request to `POST /upload`.
5. Backend validates extension + size and stores file under `/data/uploads`.
6. Backend returns `job_id`.
7. Viewer sends `POST /process/{job_id}` with query params.
8. Backend instantiates `ProcessingPipeline` and runs conversion synchronously.
9. On success, backend stores resulting model JSON to `/data/models/<job_id>.json` and returns response payload.
10. Viewer builds Three.js meshes from returned `model` sections (`walls`, `rooms`, `doors`, `windows`, floor bounds, metadata).
11. User interacts with modes, materials, tour, and optional VR path.

---

## Key Modules and What They Do

## `backend/app/core/dxf_parser.py`

- Reads DXF via `ezdxf`.
- Converts `LINE`, `LWPOLYLINE`, `POLYLINE`, and `ARC` entities into normalized line segments.
- Approximates arcs into line chunks (`_arc_to_segments`).
- Classifies layer names into `WALL`, `DOOR`, `WINDOW`, `OTHER` by keyword matching (supports non-English variants).
- Collects `TEXT` / `MTEXT` for potential room labels.
- Computes global 2D bounds.

## `backend/app/core/raster_parser.py`

- Converts raster/PDF inputs to image arrays.
- Auto-crops to bright plan panel region.
- Detects format class (`cad` vs `simpledraw`) using brightness distribution heuristics.
- Applies threshold/morphology/line extraction logic for wall and room segmentation.
- Detects openings through intra-segment and inter-segment gap analysis with filters (junction and bounds checks).
- Produces `ParsedGeometry` with metadata fields used later by pipeline.

## `backend/app/core/wall_detector.py`

- Converts parsed segments into wall primitives with thickness/height and scale handling.
- Supports auto-scale behavior and raster-aware thickness defaults.
- Produces wall objects used by both opening association and geometry building.

## `backend/app/core/room_detector.py`

- Computes room polygons/centroids (and related room metadata) from wall segment arrangements.
- Results are passed into final model metadata + viewer tour/hotspot systems.

## `backend/app/core/opening_detector.py`

- Associates detected door/window candidates with walls.
- For raster sources, pipeline may bypass generic DXF-oriented detection and build openings from raster-derived metadata.

## `backend/app/core/geometry_builder.py`

- Translates walls/rooms/openings into viewer-friendly 3D JSON schema.
- Produces dimensions, positions, rotations, metadata counters, and structures consumed directly by frontend mesh constructors.

## `backend/app/core/pipeline.py`

- Integration coordinator for the whole backend transformation path.
- Owns source-type branching, warnings, stats, and error capture.

## `backend/tests/test_pipeline.py`

- Contains automated tests validating pipeline behavior.

## `sample_data/*.py`

- Utility scripts for generating or manipulating sample floor-plan input data.

---

## Key Functions / Classes

- `ProcessingPipeline.run(filepath)`
  - Core end-to-end execution entrypoint.
- `PipelineResult.to_dict()`
  - API-safe serialization including stats/warnings.
- `DXFParser.parse()`
  - DXF-to-`ParsedGeometry` conversion.
- `classify_layer(name)`
  - Semantic layer mapping logic.
- `RasterParser.parse(filepath)`
  - Raster/PDF-to-geometry conversion (called by pipeline).
- `GeometryBuilder.build(...)`
  - Final 3D model assembly.
- `upload_file(...)` / `process_file(...)` / `get_model(...)`
  - Main API route handlers.
- Frontend functions in `viewer/index.html`:
  - `checkAPI()` (health check)
  - `uploadAndProcess(file)` (request orchestration)
  - `buildModel(data)` (Three.js model construction)
  - `setMode(...)`, `applyBlueprintMode()`, `applyRealisticMode()` (render mode state)
  - `toggleTour()`, `activateTour()`, `_flyToRoom(...)` (tour flow)
  - `enterVR()`, `_vrRequestXRSession()` (VR flow)

---

## API Endpoints

Base URL (default local): `http://localhost:8000`

### `GET /health`
Returns service status, API version, and supported formats.

### `POST /upload`
Uploads a floor plan file.

- Content type: multipart/form-data
- Field: `file`
- Validation:
  - Extension must be in supported set
  - File size must be <= 50 MB
- Returns: `job_id`, file metadata, and next step hint

### `POST /process/{job_id}`
Processes a previously uploaded file.

Query parameters:
- `scale` (float, default `1.0`; `0` allowed for auto workflow)
- `auto_scale` (bool, default `true`)
- `wall_height` (float, default `3.0`)
- `wall_thickness` (float, default `0.2`)
- `pixels_per_meter` (float, default `0.0` = auto detect in raster parser)
- `pdf_dpi` (int, default `200`)
- `hough_threshold` (int, default `50`, legacy compatibility)
- `hough_min_length` (int, default `30`, legacy compatibility)
- `hough_max_gap` (int, default `15`, legacy compatibility)

Returns: `PipelineResult` JSON.

### `POST /process/{job_id}/debug-image`
Generates and returns a PNG debug overlay (line detections) for raster/PDF uploads.

### `GET /model/{job_id}`
Returns model/result for existing job.
- If status is `processing`, returns `{ "status": "processing" }`.
- If uploaded but not processed, returns 400.

### `GET /jobs`
Returns summary map of known jobs (`status`, filename, format).

### `DELETE /job/{job_id}`
Deletes job metadata and associated saved upload/model files if present.

---

## Data Model / File Handling / Storage

## In-memory state

- Global `jobs` dictionary in API process.
- Contains per-job status, source filepath, size, format, and result payload.

## Persistent files

- Uploaded sources: `/data/uploads`
- Serialized model results: `/data/models`

## Model output shape

The viewer expects nested JSON structures for walls/openings/rooms/floor/bounds/metadata. The backend populates these via `GeometryBuilder` and pipeline wrappers.

## Important persistence caveat

Because job metadata is in-memory, restarting the backend will orphan files under `/data/uploads` and `/data/models` unless you rebuild metadata externally.

---

## Environment Variables

No required environment variables are explicitly defined in the checked-in backend/frontend code paths inspected.

However, runtime assumptions include:
- Write access to `/data/uploads` and `/data/models`
- Presence of native/system dependencies for optional parsers (e.g., poppler for PDF conversion via `pdf2image`)

If you deploy in restricted containers, you may need to parameterize these paths and permissions even though current code hardcodes them.

---

## Local Development Setup

## Prerequisites

- Python 3.10+
- Browser with WebGL support
- For PDF processing: poppler binaries available to `pdf2image`
- Python packages from backend requirements (not listed in this README because `requirements.txt` was not included in inspected files)

## Run backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Run viewer

```bash
cd viewer
python -m http.server 3000
```

Open:
- Main viewer: `http://localhost:3000/index.html`

## Basic smoke test

1. Confirm `GET http://localhost:8000/health` returns `status: ok`.
2. Upload sample file from `sample_data/` in viewer.
3. Trigger process and verify model appears.
4. Optionally call `/process/{job_id}/debug-image` for raster/PDF diagnostic output.

---

## Deployment Setup

Current repository provides application code but **no explicit deployment manifests** (e.g., Dockerfile, Compose, Kubernetes manifests, CI deployment scripts) in the inspected file set.

Expected deployment pattern from code behavior:

1. Deploy FastAPI service with persistent writable volume mounted at `/data`.
2. Serve `viewer/index.html` as static file from any web server/CDN.
3. Ensure CORS policy in backend remains compatible with viewer origin.
4. Provide poppler binaries/system package where PDF ingestion is required.

For production hardening, you would likely add:
- external job store (DB/Redis) replacing in-memory `jobs`
- asynchronous worker queue for long-running processing
- auth/rate limiting/file scanning
- path configurability via environment variables
- reverse-proxy limits aligned to `MAX_FILE_SIZE_MB`

---

## Common Errors and Debugging Notes

## API-level issues

- **404 Job not found**
  - Cause: invalid `job_id`, backend restart (memory reset), or deleted job.
- **409 Already processing**
  - Cause: duplicate concurrent processing request for same job.
- **413 File too large**
  - Cause: upload exceeds 50 MB limit.
- **400 unsupported format**
  - Cause: extension outside `ALL_FORMATS`.

## Parsing/geometry issues

- Use `POST /process/{job_id}/debug-image` to visualize detected line structures for raster/PDF inputs.
- Review `warnings` array from pipeline response; parser and scale heuristics emit detailed warning strings.
- If wall layer classification fails in DXF, pipeline may force all non-empty geometry into walls and emit warning.

## Frontend rendering issues

- Ensure viewer is served over HTTP; avoid opening file directly via `file://`.
- Confirm API URL in viewer script matches backend host/port.
- Browser console logs and toast panel provide immediate error signals.

---

## Limitations and Future Improvements

Observed limitations/incompleteness from repository state:

- In-memory job state is not durable across restarts.
- Processing is synchronous inside request lifecycle (no queue/background worker separation).
- Hardcoded storage paths (`/data/uploads`, `/data/models`) reduce portability.
- Multiple viewer versions are present; only one is likely canonical, but no explicit versioning policy is documented.
- Pipeline references optional `SimpleDrawParser` module that is not present in `backend/app/core` file listing (import is guarded, so behavior falls back safely).
- Dependency locking and reproducible environment descriptors are not visible in inspected paths.

Potential next steps:
- Introduce persistent metadata store and worker queue.
- Add config layer for path/CORS/limits.
- Provide container/deployment assets.
- Consolidate viewer versions and document supported one explicitly.
- Expand automated tests beyond pipeline core.

---

## License / Credits

- No project license file was found in the inspected repository root.
- Third-party dependency assets are vendored under `dependencies/poppler-25.12.0` and include their own licensing files (e.g., `COPYING`).

If this repository is intended for redistribution, add an explicit top-level `LICENSE` and dependency attribution policy.
