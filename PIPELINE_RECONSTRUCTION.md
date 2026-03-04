# Pipeline Reconstruction (Image → Processing → Geometry → Rendering)

This document reconstructs the **current runtime pipeline** used by the repository, with exact file/function references.

## End-to-end diagram

```mermaid
flowchart TD
  A[Input floorplan file<br/>PNG/JPG/BMP/TIFF/PDF/DXF] --> B[FastAPI /upload<br/>backend/app/main.py::upload_file]
  B --> C[FastAPI /process/{job_id}<br/>backend/app/main.py::process_file]
  C --> D[ProcessingPipeline.run(filepath)<br/>backend/app/core/pipeline.py]

  D --> E{File extension}
  E -->|DXF| F[DXFParser.parse()<br/>backend/app/core/dxf_parser.py]
  E -->|Raster/PDF| G[RasterParser.parse()<br/>backend/app/core/raster_parser.py]

  G --> G1[_pdf_to_cv2/_load_cv2]
  G1 --> G2[_preprocess]
  G2 --> G3[_hough_lines]
  G3 --> G4[_px_to_segments -> ParsedGeometry.wall_segments]

  F --> H[ParsedGeometry]
  G4 --> H

  H --> I[WallDetector.detect()<br/>backend/app/core/wall_detector.py]
  I --> J[RoomDetector.detect()<br/>backend/app/core/room_detector.py]
  I --> K[OpeningDetector.detect()<br/>backend/app/core/opening_detector.py]

  I --> L[GeometryBuilder.build(..., rooms, openings)<br/>backend/app/core/geometry_builder.py]
  J --> L
  K --> L

  L --> M[BuildingModel.to_dict() JSON]
  M --> N[/process response + save to backend/models/{job_id}.json]

  N --> O[Viewer upload/process fetch flow<br/>viewer/index_v2.html::uploadAndProcess]
  O --> P[buildModel(data) creates Three.js meshes]
  P --> Q[Walls/Floor/Doors/Windows/Room labels rendered]
```

## Stage-by-stage mapping

### 1) Input/API stage
- `upload_file()` validates extension against `ALL_FORMATS`, stores bytes under `backend/uploads`, and tracks `job_id` in-memory. (`backend/app/main.py`)
- `process_file()` constructs `ProcessingPipeline(...)` from query parameters (including raster tuning knobs) and calls `pipeline.run(job["filepath"])`. (`backend/app/main.py`)

### 2) Core processing orchestration
- `ProcessingPipeline.run()` controls: format dispatch, fallback if no `wall_segments`, wall detection, room detection, opening detection, and geometry build.
- Routing:
  - DXF → `_parse_dxf()` → `DXFParser.parse()`.
  - Raster/PDF → `_parse_raster()` → `RasterParser.parse()`.

### 3) Image/PDF parsing (current implementation)
- `RasterParser.parse()`:
  1. load page/image (`_pdf_to_cv2` or `_load_cv2`),
  2. preprocess (`_preprocess`),
  3. detect lines (`_hough_lines`),
  4. convert pixels→meters (`_px_to_segments`).
- Current raster output populates `ParsedGeometry.wall_segments`; doors/windows are not semantically segmented in raster parser today.

### 4) Geometry interpretation
- `WallDetector.detect()`:
  - optional scale inference (`infer_scale`),
  - segment length filter,
  - optional raster fragment merge (`merge_collinear_fragments`) when `is_raster=True`,
  - parallel double-line pairing (`pair_double_lines`),
  - fallback single-line walls.
- `RoomDetector.detect()`:
  - rasterizes wall segments to grid (`_draw_line`),
  - flood-fills empty regions (`_flood_fill`),
  - removes border-connected exterior,
  - converts regions to typed/labeled `Room` objects.
- `OpeningDetector.detect()`:
  - extracts door points (`_extract_door_points`),
  - extracts window points (`_extract_window_points`),
  - projects openings onto nearest wall (`_find_wall` / `_project`),
  - wall intervals split later via `split_wall_at_openings`.

### 5) 3D geometry creation
- `GeometryBuilder.build()` merges wall + room + opening outputs into a `BuildingModel`:
  - wall solids from `wall_to_boxes`,
  - door descriptors including `_door_leaf`,
  - window descriptors including `_window_pieces`,
  - floor plane via `_floor`,
  - room labels via `room_to_label`,
  - metadata counts/length/bounds.

### 6) Rendering stage (frontend)
- `viewer/index_v2.html`:
  - `uploadAndProcess(file)` calls `/upload` then `/process/{job_id}`.
  - `buildModel(data)` reads `data.model` and creates Three.js objects:
    - walls (`THREE.BoxGeometry`),
    - floor (`THREE.PlaneGeometry`),
    - doors (leaf + frame meshes),
    - windows (sill/header/glass pieces),
    - room label sprites.
  - `initThree()` sets scene/camera/lights and the render loop (`renderer.render(scene, camera)`).

## Notes on current behavior
- The pipeline supports both vector and raster inputs, but raster semantic extraction is currently wall-line driven.
- In `ProcessingPipeline.run()`, `WallDetector` is currently instantiated without `is_raster=True`, so raster-only collinear merge is not activated through this path unless changed.
