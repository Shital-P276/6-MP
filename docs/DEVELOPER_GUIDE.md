# Project Deep Dive

## System Overview
FloorViz is a two-part system:
1. **Static frontend viewer** (`viewer/index.html`) that handles upload UX, invokes API endpoints, and renders interactive 3D scenes using Three.js.
2. **FastAPI backend** (`backend/app/main.py`) that accepts uploads, orchestrates a parsing/detection pipeline, and emits a normalized JSON model used by the viewer.

The backend supports both **vector** (DXF) and **raster/PDF** input paths. A unified `ProcessingPipeline` dispatches to the correct parser, then applies wall detection, room detection, opening detection, and geometry building to produce consistent output.

## High-Level Architecture Diagram

```text
┌──────────────────────────────┐
│ Browser (Netlify/static host)│
│ viewer/index_v12.html        │
└──────────────┬───────────────┘
               │ HTTP (JSON + multipart)
               ▼
┌──────────────────────────────┐
│ FastAPI backend              │
│ backend/app/main.py          │
│  - /upload                   │
│  - /process/{job_id}         │
│  - /model/{job_id}           │
└──────────────┬───────────────┘
               │ calls
               ▼
┌──────────────────────────────┐
│ ProcessingPipeline           │
│ backend/app/core/pipeline.py │
└──────┬────────────┬──────────┘
       │            │
       │            ├─ Raster/PDF parser (OpenCV/pdf2image)
       │            │    raster_parser.py
       │            │
       └─ DXF parser (ezdxf)
            dxf_parser.py
               │
               ▼
   wall_detector + room_detector + opening_detector
               │
               ▼
      geometry_builder -> BuildingModel JSON
               │
               ▼
        /data/models/{job_id}.json
```

## Complete Request Lifecycle
1. Frontend boots and checks backend via `GET /health`.
2. User uploads file; frontend sends multipart request to `POST /upload`.
3. Backend validates extension and file size (50 MB limit), writes file to `/data/uploads/{job_id}{ext}`, and stores job metadata in in-memory `jobs` dict.
4. Frontend calls `POST /process/{job_id}` with query parameters (`scale`, `wall_height`, `wall_thickness`, etc.).
5. Backend creates `ProcessingPipeline` instance and calls `run(filepath)`.
6. Pipeline detects file type:
   - DXF → `DXFParser`
   - Raster/PDF → `RasterParser` (PDF rendered to image first)
7. Parsed geometry is normalized into `ParsedGeometry`.
8. `WallDetector` pairs/infers wall objects.
9. `RoomDetector` infers rooms from wall graph/text labels.
10. Openings:
    - Raster path can use parser-derived openings
    - Otherwise `OpeningDetector` runs on geometry/walls
11. `GeometryBuilder` converts semantic entities into model primitives/metadata.
12. Backend serializes result, persists model JSON to `/data/models/{job_id}.json` when successful, updates job status, returns payload.
13. Frontend consumes payload and builds scene meshes, overlays, stats, and optional tour objects.

## Repository Structure Deep Dive
- `backend/app/main.py`: API composition, CORS policy, upload/process endpoints, job registry.
- `backend/app/core/pipeline.py`: central orchestration and format dispatch.
- `backend/app/core/dxf_parser.py`: DXF entity/layer parsing into normalized segments.
- `backend/app/core/raster_parser.py`: CV pipeline for image/PDF extraction and debug output.
- `backend/app/core/wall_detector.py`: segment pairing and wall object generation.
- `backend/app/core/room_detector.py`: room identification from detected geometry.
- `backend/app/core/opening_detector.py`: door/window inference (primarily geometric path).
- `backend/app/core/geometry_builder.py`: converts walls/rooms/openings to viewer-ready JSON.
- `backend/tests/test_pipeline.py`: tests for classification helpers, detector behavior, builder output, and pipeline error/smoke cases.
- `viewer/index_v12.html`: latest UI/UX and 3D renderer.
- `sample_data/`: DXF samples + generation scripts for synthetic floorplans.
- `dependencies/`: includes poppler-related resources and notes for PDF conversion dependency setup.

## Backend Deep Dive
### FastAPI App (`backend/app/main.py`)
- Registers permissive CORS middleware.
- Initializes storage directories at startup (`/data/uploads`, `/data/models`).
- Maintains `jobs: dict[str, dict]` in process memory.

### Endpoints
- `GET /health`: returns status, API version, supported formats.
- `POST /upload`: validates extension against `ALL_FORMATS`; reads full file bytes; enforces `MAX_FILE_SIZE_MB=50`; saves file and tracks metadata.
- `POST /process/{job_id}`: validates job state; instantiates `ProcessingPipeline` with request params; runs pipeline; persists JSON result if success.
- `POST /process/{job_id}/debug-image`: raster/PDF only; emits line-overlay debug PNG via `RasterParser.save_debug_image`.
- `GET /model/{job_id}`: returns result/status with guardrails for unprocessed jobs.
- `GET /jobs`: compact status index of all in-memory jobs.
- `DELETE /job/{job_id}`: removes job entry and deletes saved upload/model files.

### Concurrency / Runtime Behavior
- Endpoint handlers are mostly synchronous after upload.
- No background worker queue is present.
- `jobs` map is non-persistent and process-local.

## Frontend Deep Dive
`viewer/index_v12.html` is a monolithic UI containing:
- Layout/styling system (sidebar, control panels, log/toast overlays)
- API constant (`const API = 'http://localhost:8000'` by default)
- Connectivity check and diagnostics
- Upload/process orchestration
- Three.js scene setup (camera, lights, grid, render loop)
- Mesh creation from backend JSON
- Visibility and view-mode controls
- Procedural material generation and assignment
- Virtual tour subsystem (hotspots, minimap, nav HUD, autoplay)

### State & Event Flow
- Upload input/drag-drop triggers `uploadAndProcess(file)`.
- API response hands off to `buildModel(procData)`.
- Scene state arrays track wall/wire/floor/label/door/window meshes.
- UI toggles mutate boolean state and call visibility/material update handlers.

## API Documentation
### `GET /health`
- **Response:** `{status, version, supported_formats[]}`
- **Side effects:** none

### `POST /upload`
- **Request:** multipart form with `file`
- **Validation:** extension in `ALL_FORMATS`, size <= 50MB
- **Response:** `{job_id, filename, format, size_mb, status, next}`
- **Side effects:** writes upload file; inserts job record

### `POST /process/{job_id}`
- **Query params:**
  - `scale`, `auto_scale`, `wall_height`, `wall_thickness`
  - `pixels_per_meter`, `pdf_dpi`
  - legacy Hough tuning params
- **Response:** pipeline result (`success`, `warnings`, `model`, `stats`, etc.)
- **Side effects:** reads upload, computes geometry, may write model JSON

### `POST /process/{job_id}/debug-image`
- **Use case:** debug raster line detection
- **Response:** `image/png` file

### `GET /model/{job_id}`
- **Response:** processing status/result

### `GET /jobs`
- **Response:** job status dictionary

### `DELETE /job/{job_id}`
- **Response:** `{deleted: job_id}`
- **Side effects:** file cleanup

## Data Flow Analysis
- **Upload lifecycle:** browser file → in-memory bytes (`UploadFile.read`) → `/data/uploads`.
- **Processing lifecycle:** parser emits normalized segments + metadata; detectors enrich semantics; builder emits render model.
- **Persistence:** successful model snapshots are saved to `/data/models/{job_id}.json`.
- **Cleanup:** explicit via `DELETE /job/{job_id}` only; no TTL/cron cleanup is implemented.

## Processing Pipeline Analysis
`ProcessingPipeline.run()` stages:
1. Validate file existence and extension.
2. Parse:
   - DXF via `DXFParser`
   - Raster/PDF via `RasterParser` (plus optional SimpleDraw path if module available)
3. Fallback: if no wall layer detected, reclassify other segments as walls.
4. Determine wall thickness strategy (raster metadata-informed vs fallback default).
5. Detect walls (`WallDetector`).
6. Detect rooms (`RoomDetector`).
7. Detect openings (`_raster_openings` path or `OpeningDetector`).
8. Build output model (`GeometryBuilder`).
9. Return `PipelineResult` with warnings, timing, scale, and stats.

## Deployment Architecture
### Production Pattern
- Frontend: Netlify static hosting.
- Backend: Railway Python web service.
- Storage: Railway mounted volume at `/data`.

### Railway Backend Notes
- Start with `uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}`.
- `0.0.0.0` binding is required for ingress routing.
- Use `opencv-python-headless` to avoid headless libGL failures.
- Container filesystem is ephemeral; persistent volume is required for data durability.

### Frontend Hosting Notes
- No build step required for `viewer/index_v12.html`.
- API base URL is currently hardcoded and must be updated for production.

## Configuration and Environment Variables
- No first-class `.env` system is implemented in code.
- Runtime assumptions:
  - `PORT` provided by host platform (Railway)
  - writable `/data` volume with `/uploads` and `/models` subdirs

## Dependency Analysis
- **FastAPI/Uvicorn:** API server + ASGI runtime.
- **ezdxf/Shapely:** vector floorplan geometry parsing/operations.
- **opencv-python-headless/numpy/Pillow/pdf2image:** raster and PDF computer-vision pipeline.
- **python-multipart:** required for file upload handling.
- **Three.js:** client-side 3D scene graph/rendering.

## Error Handling and Debugging
- Backend returns structured errors for unsupported format, missing job, oversized upload, etc.
- Pipeline catches broad exceptions and returns traceback text in `error` field.
- Frontend surfaces failures in toast/log panels.

### Common Failure Modes
- API offline or unreachable (bad API constant, backend not running).
- PDF conversion failures if poppler dependencies are missing.
- Incorrect plan scale/thickness requiring query tuning.
- Memory pressure for very large raster/PDF files.

## Security Notes
- No authentication/authorization present.
- CORS allows all origins.
- Uploaded content is file-based and parsed server-side; production deployments should add stricter validation, quota, and rate limiting.

## Performance Considerations
- Processing is CPU-bound and synchronous; concurrent heavy jobs may degrade response times.
- Upload endpoint reads whole file into memory before persisting.
- In-memory `jobs` store may grow without bounds.

## Technical Debt / Risks
- Hardcoded frontend API URL.
- Lack of persistent metadata DB and background job system.
- Permissive CORS and no auth.
- Sparse API contract tests for raster/PDF edge cases.

## Suggested Improvements
### Short-term
- Add env-configurable frontend API URL.
- Add request validation limits and cleanup policy.
- Document expected PDF/poppler runtime setup per OS.

### Medium-term
- Move job state to persistent store (SQLite/Postgres/Redis).
- Add async task queue for long-running processing.
- Add structured logging and metrics.

### Production-grade
- Introduce auth, per-user quotas, rate limiting.
- Containerize with Docker and add CI/CD pipelines.
- Add observability (tracing, dashboards, alerting).

## Developer Onboarding
1. Install Python dependencies: `pip install -r requirements.txt`.
2. Run backend from `backend/` with Uvicorn.
3. Serve `viewer/` over HTTP and open `index_v12.html`.
4. Upload sample files from `sample_data/`.
5. Validate API via `/docs` and `/health`.
6. Run tests from repo root/backend context.

## Appendix
### Useful Commands
```bash
# Backend
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Frontend
cd viewer
python -m http.server 3000

# Tests
pytest backend/tests/test_pipeline.py

# Health check
curl http://localhost:8000/health
```

### Deployment-Oriented Commands
```bash
# Railway start command
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

### Troubleshooting Commands
```bash
# verify upload endpoint quickly
curl -F "file=@sample_data/sample_floorplan.dxf" http://localhost:8000/upload

# list active jobs
curl http://localhost:8000/jobs
```

## Known Uncertainties / Inconsistencies
- Repository contains multiple viewer versions (`index.html`, `index_v11*`, `index_v12.html`); this guide treats `index_v12.html` as primary because it is the most feature-rich current file.
- Existing root README references `index_v2.html`, which does not match the current `viewer/` filenames.
- Optional SimpleDraw path in pipeline is conditionally imported (`simpledraw_parser`) and not present in the inspected file list.
