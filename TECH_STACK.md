# Technical Stack Documentation

This document explains the project tech stack in practical terms: **what** each technology is, **where** it is used in this codebase, and **why** it is used.

---

## 1) Stack at a glance

| Layer | Core technologies |
|---|---|
| API/backend | Python, FastAPI, Uvicorn |
| CAD + geometry pipeline | ezdxf, custom geometry modules |
| Raster/PDF processing | OpenCV (`opencv-python`), NumPy, pdf2image, Poppler binaries |
| Frontend viewer | HTML/CSS/JavaScript, Three.js (CDN) |
| Testing | pytest |
| Data exchange | JSON model schema over REST |

---

## 2) Backend/API stack

### Python
- **What:** Main implementation language.
- **Where:** All backend logic under `backend/app/` and tests under `backend/tests/`.
- **Why:** Strong ecosystem for CAD/image processing and rapid API development.

### FastAPI
- **What:** Web framework for REST endpoints.
- **Where:** API app and routes in `backend/app/main.py` (`/health`, `/upload`, `/process/{job_id}`, `/model/{job_id}`, etc.).
- **Why:** Clean route definitions, typed query parameters, automatic OpenAPI docs, and good performance for this workload.

### Uvicorn
- **What:** ASGI server running the FastAPI app.
- **Where:** Declared in `requirements.txt` (`uvicorn[standard]==0.29.0`) and used in run instructions (`uvicorn app.main:app --reload --port 8000`).
- **Why:** Lightweight production-grade server commonly paired with FastAPI.

### CORS middleware
- **What:** Cross-origin policy middleware.
- **Where:** `CORSMiddleware` setup in `backend/app/main.py` with `allow_origins=["*"]`.
- **Why:** Lets the browser-based viewer call the backend during local development without origin blocking.

### python-multipart
- **What:** Multipart form-data parser dependency for file uploads.
- **Where:** Declared in `requirements.txt`; required by FastAPI `UploadFile = File(...)` endpoint in `backend/app/main.py`.
- **Why:** Enables DXF/image/PDF upload support via `/upload`.

---

## 3) Floorplan processing stack

### Orchestrator: ProcessingPipeline
- **What:** Main workflow coordinator for vector and raster inputs.
- **Where:** `backend/app/core/pipeline.py` (`ProcessingPipeline`, `PipelineResult`, format routing, wall/room/opening flow).
- **Why:** Centralizes business logic so API endpoints stay thin and processing steps remain modular.

### DXF parsing with ezdxf
- **What:** DXF reader and entity access library.
- **Where:** `backend/app/core/dxf_parser.py` imports and uses `ezdxf.readfile(...)`.
- **Why:** Reliable DXF parsing for CAD entities (`LINE`, `LWPOLYLINE`, `POLYLINE`, `ARC`) and layer-based classification.

### Geometry modules (custom)
- **What:** Internal algorithms for geometric interpretation and 3D model construction.
- **Where:**
  - Wall detection: `backend/app/core/wall_detector.py`
  - Room detection: `backend/app/core/room_detector.py`
  - Door/window opening detection: `backend/app/core/opening_detector.py`
  - 3D JSON generation: `backend/app/core/geometry_builder.py`
- **Why:** Domain-specific logic tailored for architectural floor plans and Three.js-friendly model output.

### OpenCV (`opencv-python`)
- **What:** Image processing + computer vision toolkit.
- **Where:** `backend/app/core/raster_parser.py` (cropping, thresholding, morphology, skeletonization, Hough lines, debug overlays).
- **Why:** Needed for robust line extraction from PNG/JPG/BMP/TIFF and rasterized PDF floor plans.

### NumPy
- **What:** Fast numeric array operations.
- **Where:** `backend/app/core/raster_parser.py` (`import numpy as np`) and image conversions.
- **Why:** Efficient pixel/array manipulation required by OpenCV workflows.

### pdf2image + Poppler
- **What:** PDF page rasterization pipeline.
- **Where:**
  - `pdf2image` usage in `backend/app/core/raster_parser.py` (`convert_from_path`).
  - Poppler binaries/vendor files under `dependencies/poppler-25.12.0/...`.
- **Why:** Converts PDF floor plans into images so they can be processed by the same raster pipeline as PNG/JPG.

### Pillow
- **What:** Python imaging dependency used by `pdf2image` outputs and image handling stack.
- **Where:** Declared in `requirements.txt`.
- **Why:** Supports image object interoperability in PDF/image conversion paths.

---

## 4) Frontend/viewer stack

### Plain HTML/CSS/JavaScript
- **What:** No framework frontend (single-file app style).
- **Where:** `viewer/index.html` and `viewer/index_v2.html`.
- **Why:** Keeps the viewer lightweight and easy to run directly in browser for local testing.

### Three.js (CDN)
- **What:** 3D rendering library.
- **Where:** Included via CDN script tag in both viewer files.
- **Why:** Renders walls/floors/openings from backend JSON as interactive 3D scene (camera, mesh, lighting, controls).

### Fetch API
- **What:** Browser HTTP client.
- **Where:** `viewer/index.html` and `viewer/index_v2.html` calls to backend endpoints (`/health`, `/upload`, `/process`).
- **Why:** Handles client-server workflow without additional frontend dependencies.

---

## 5) Testing and quality

### pytest
- **What:** Python test framework.
- **Where:** `backend/tests/test_pipeline.py`; test command documented in `README.md`.
- **Why:** Validates parser/pipeline behavior and regressions around success/failure flows.

---

## 6) Data and interface contracts

### JSON model contract for rendering
- **What:** Structured output containing walls/floors/metadata/statistics.
- **Where:**
  - Generated by pipeline/model builder (`backend/app/core/pipeline.py`, `backend/app/core/geometry_builder.py`).
  - Example schema documented in `README.md` under “3D Model JSON Format”.
- **Why:** Decouples backend computation from frontend rendering; viewer only consumes stable JSON.

### File-based job artifacts
- **What:** Uploaded source files and generated model/debug outputs.
- **Where:** `backend/uploads/` and `backend/models/`, managed by `backend/app/main.py`.
- **Why:** Simple persistence strategy for local/dev workflows and repeatable model retrieval.

---

## 7) Notable implementation choices

1. **Hybrid input support** (vector DXF + raster image/PDF) broadens compatibility with real-world floorplan sources.
2. **Modular core pipeline** separates parsing, detection, and model generation for maintainability.
3. **Framework-free viewer** reduces setup friction (open file + run API).
4. **Open CORS policy** is convenient for development; should be tightened for production.
5. **Some declared dependencies may be legacy/forward-looking** (e.g., `shapely` appears in requirements and test notes but is not currently imported in active core modules), so periodic dependency audits are recommended.

---

## 8) How everything fits together (runtime flow)

1. User uploads floorplan to FastAPI `/upload`.
2. FastAPI stores file and returns `job_id`.
3. User triggers `/process/{job_id}`.
4. Pipeline routes by extension:
   - `.dxf` → DXF parser path.
   - image/PDF → raster parser path.
5. Core detectors produce walls/rooms/openings.
6. Geometry builder creates 3D-ready JSON.
7. Viewer fetches and renders JSON with Three.js.

