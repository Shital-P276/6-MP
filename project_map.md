# Project Map for Claude (Floor Plan → 3D Visualizer)

This document is a **context index** for quickly navigating the repository and requesting only the files needed for a specific edit.

---

## 1) What this repository does

- Backend service (FastAPI) that ingests floor plans (DXF, image formats, PDF), detects architectural elements, and outputs a JSON scene model for Three.js.
- Browser viewer(s) in plain HTML/JS render walls/floors/doors/windows/rooms from API output.
- Includes generator scripts to synthesize sample floor plans and export DXF test data.

---

## 2) High-level directory map

```text
/workspace/6-MP
├── README.md                         # Setup + basic API usage
├── requirements.txt                  # Python dependencies (FastAPI, OpenCV, ezdxf, etc.)
├── instructions.md                   # Local run notes
├── todo.txt                          # Future ideas
├── project_map.md                    # (this file)
│
├── backend/
│   ├── app/
│   │   ├── main.py                   # FastAPI app + API routes
│   │   └── core/
│   │       ├── __init__.py           # Core exports
│   │       ├── pipeline.py           # Orchestrates parse→detect→build
│   │       ├── dxf_parser.py         # DXF geometry parsing/classification
│   │       ├── raster_parser.py      # Image/PDF wall/opening/room extraction
│   │       ├── wall_detector.py      # Segment pairing + wall inference
│   │       ├── opening_detector.py   # Door/window detection + wall splitting
│   │       ├── room_detector.py      # Room polygon detection/labeling
│   │       └── geometry_builder.py   # Final 3D JSON assembly
│   ├── tests/
│   │   └── test_pipeline.py          # Unit/integration tests for parsers/pipeline
│   ├── uploads/                      # Runtime input artifacts (many files)
│   └── models/                       # Runtime output JSON artifacts (many files)
│
├── viewer/
│   ├── index.html                    # Viewer version 1 (single-file app)
│   └── index_v2.html                 # Viewer version 2 (enhanced labels/legend)
│
├── sample_data/
│   ├── generate_sample.py            # Simple sample DXF generator
│   ├── generate.py                   # GUI-driven random plan generator
│   ├── generate_v2.py                # Alternate generator variant
│   ├── generate_floorplan_gui.py     # Extended GUI generator
│   └── *.dxf                         # Example generated plans
│
└── dependencies/
    ├── instructions.txt
    └── poppler-25.12.0/              # Bundled Poppler binaries/assets for PDF support
```

---

## 3) Runtime architecture and call flow

### A) Backend request flow

1. `POST /upload` in `backend/app/main.py`
   - Validates extension against `ALL_FORMATS`.
   - Writes uploaded file to `backend/uploads`.
   - Registers in-memory job metadata.

2. `POST /process/{job_id}` in `backend/app/main.py`
   - Creates `ProcessingPipeline(...)` from query params.
   - Calls `pipeline.run(filepath)`.

3. `ProcessingPipeline` in `backend/app/core/pipeline.py`
   - Chooses parser by extension:
     - DXF path → `DXFParser`.
     - Raster/PDF path → `RasterParser`.
   - Runs detection pipeline:
     - walls → rooms → openings → geometry build.
   - Returns `PipelineResult` with serializable model data.

4. If successful, model JSON is stored in `backend/models/{job_id}.json`.

5. `GET /model/{job_id}` returns saved or in-memory output.

### B) Viewer flow

- `viewer/index.html` and `viewer/index_v2.html`:
  - Upload file to `/upload`.
  - Trigger `/process/{job_id}` with settings (`scale`, wall dimensions, etc.).
  - Build Three.js meshes from returned model arrays (`walls`, `floors`, `doors`, `windows`, `rooms`).

---

## 4) Core backend modules (what to request for each type of change)

### `backend/app/main.py`
**Contains:** API routes, job registry, upload/process orchestration, debug image endpoint.

**Ask for this when:**
- Adding/changing endpoints or query params.
- Adjusting response schema/status codes.
- Changing job lifecycle or persistence behavior.

---

### `backend/app/core/pipeline.py`
**Contains:** `PipelineResult`, `ProcessingPipeline`, extension routing, end-to-end orchestration.

**Ask for this when:**
- Changing overall processing sequence.
- Adding new input format support.
- Modifying metadata returned by pipeline.

---

### `backend/app/core/dxf_parser.py`
**Contains:** DXF entity extraction (`LINE`, `LWPOLYLINE`, `POLYLINE`, `ARC`), layer classification, bounds/units extraction.

**Ask for this when:**
- DXF import issues.
- Layer naming/classification changes (`WALL`, `DOOR`, `WINDOW`, etc.).
- Geometry segmentation behavior for vector files.

---

### `backend/app/core/raster_parser.py`
**Contains:** image/PDF loading, wall mask extraction, line detection/merging, opening detection, room detection, debug image output.

**Ask for this when:**
- PNG/JPG/PDF quality or detection accuracy issues.
- Hough/opening/room detection parameter changes.
- PDF DPI/preprocessing behavior.

> Note: This is the largest and densest module; request it in chunks when possible.

---

### `backend/app/core/wall_detector.py`
**Contains:** segment geometry math, line pairing/merging, wall thickness estimation, `Wall` model.

**Ask for this when:**
- Wrong wall thickness/length.
- Double-line pairing issues.
- Collinearity/fragment merge tuning.

---

### `backend/app/core/opening_detector.py`
**Contains:** opening projection onto walls, opening type inference, wall splitting around openings.

**Ask for this when:**
- Doors/windows positioned wrong.
- Wall holes/void logic is incorrect.
- Door swing/window placement logic needs changes.

---

### `backend/app/core/room_detector.py`
**Contains:** room polygon detection, label matching, room metadata.

**Ask for this when:**
- Missing or mislabeled rooms.
- Room area/dimension calculations look wrong.

---

### `backend/app/core/geometry_builder.py`
**Contains:** conversion from semantic wall/opening/room objects to viewer-ready mesh dictionaries.

**Ask for this when:**
- 3D output schema changes.
- Doors/windows/floors/labels visual geometry needs tweaks.
- Metadata fields consumed by viewer need to be added/renamed.

---

## 5) Viewer modules

### `viewer/index.html`
- Single-file web UI + Three.js renderer.
- Upload/process controls and visualization toggles.

### `viewer/index_v2.html`
- Enhanced viewer variant: additional labels/legend and richer display controls.

**Ask for these when:**
- UI controls, rendering style, camera controls, or model-to-Three.js mapping need edits.

---

## 6) Tests and quality checks

### `backend/tests/test_pipeline.py`
- Covers parser behavior, wall geometry helpers, builder output structure, and pipeline-level error/success paths.

**Ask for this when:**
- Making backend logic changes that should be regression-tested.
- Understanding expected behavior before refactoring.

---

## 7) Data-heavy directories (usually skip unless needed)

- `backend/uploads/` and `backend/models/`: many generated/runtime artifacts.
- These are useful for debugging specific job outputs, but usually **not** needed for code edits.

When requesting files, prefer asking for:
- one problematic model JSON in `backend/models/<job_id>.json`, and
- corresponding source input in `backend/uploads/<job_id>.<ext>`
instead of all artifacts.

---

## 8) “Ask-for-files” playbook for Claude

Use this minimal request strategy:

1. **Feature/API edits**
   - Request: `backend/app/main.py`, `backend/app/core/pipeline.py`, and relevant test file.

2. **DXF parsing issues**
   - Request: `backend/app/core/dxf_parser.py`, `backend/app/core/wall_detector.py`, optionally `backend/tests/test_pipeline.py`.

3. **Raster/PDF detection issues**
   - Request: `backend/app/core/raster_parser.py`, `backend/app/core/pipeline.py`, sample failing input path.

4. **3D output schema/visual issues**
   - Request: `backend/app/core/geometry_builder.py` + one viewer file (`viewer/index_v2.html` usually).

5. **Room/opening semantics**
   - Request: `backend/app/core/room_detector.py`, `backend/app/core/opening_detector.py`, `geometry_builder.py`.

6. **End-to-end bug with unknown origin**
   - Start with: `main.py`, `pipeline.py`, one parser (`dxf_parser.py` or `raster_parser.py` based on file type), `geometry_builder.py`, and one failing artifact pair.

---

## 9) Quick operational notes

- API supports DXF + raster + PDF input formats.
- Processing params (scale, wall dimensions, Hough tuning, DPI) are query-string driven on `/process/{job_id}`.
- `/process/{job_id}/debug-image` exists for raster/PDF visual diagnostics.
- Viewer files are static HTML and expect backend at `localhost:8000` unless edited.

---

## 10) Suggested first files for orientation (if opening this repo cold)

1. `README.md`
2. `backend/app/main.py`
3. `backend/app/core/pipeline.py`
4. `backend/app/core/geometry_builder.py`
5. `viewer/index_v2.html`
6. `backend/tests/test_pipeline.py`

This sequence gives a fast understanding of **API contract → processing pipeline → output schema → frontend consumption → expected behavior tests**.
