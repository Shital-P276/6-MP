# Technical Analysis — 6-MP (Floor Plan → 3D)

## 1) Full directory tree

The complete repository tree (excluding `.git`) is provided in:

- `DIRECTORY_TREE.txt`

This includes:
- backend API and core processing modules
- frontend viewers
- sample floorplan generators
- uploaded/generated model artifacts
- vendored Poppler dependency files

## 2) Purpose of every major file

### Root
- `README.md`: project overview, API usage, architecture summary, and expected JSON output format.
- `requirements.txt`: Python dependencies (FastAPI, DXF parsing, CV/image/PDF stack).
- `instructions.md`: local run notes.
- `DIRECTORY_TREE.txt`: full file tree snapshot.
- `TECHNICAL_ANALYSIS.md`: this analysis.

### Backend API
- `backend/app/main.py`: FastAPI app entrypoint and endpoints for upload/process/debug/model retrieval. It configures processing query parameters and instantiates the processing pipeline.

### Core processing pipeline (`backend/app/core/`)
- `pipeline.py`: orchestrates end-to-end parsing → wall detection → room detection → opening detection → 3D geometry build.
- `dxf_parser.py`: parses DXF entities (`LINE`, `LWPOLYLINE`, `POLYLINE`, `ARC`) into normalized 2D segments and classifies by layer (`WALL`, `DOOR`, `WINDOW`, `OTHER`).
- `raster_parser.py`: parses raster/PDF floorplans via OpenCV preprocessing + Canny + probabilistic Hough transform, then converts pixel lines to meter-space segments.
- `wall_detector.py`: performs scale inference/application, optional collinear fragment merge (raster mode), double-line pairing, and single-line fallback wall generation.
- `room_detector.py`: rasterizes wall segments into a grid, flood-fills enclosed regions, infers room type/label/color, and optionally applies DXF text labels.
- `opening_detector.py`: extracts door/window candidates from classified segments, attaches openings to nearest wall, and splits wall intervals around openings.
- `geometry_builder.py`: converts walls/openings/rooms to Three.js-ready JSON (wall boxes, floor plane, room labels, door/window mesh descriptors).

### Data and tests
- `backend/tests/test_pipeline.py`: legacy unit/integration tests (currently out-of-sync with current module APIs/import names).
- `backend/uploads/*`: uploaded source inputs keyed by job UUID.
- `backend/models/*`: processed model JSON outputs keyed by job UUID.

### Viewer
- `viewer/index.html`: Three.js-based frontend with upload/process controls and wall/floor/room rendering.
- `viewer/index_v2.html`: extended viewer with explicit door/window rendering and layer toggles.

### Sample data and generators
- `sample_data/*.dxf`: sample floorplan inputs.
- `sample_data/generate_sample.py`: deterministic sample DXF generator with walls/doors/windows.
- `sample_data/generate.py`, `generate_v2.py`, `generate_floorplan_gui.py`, `generate_sample.py`: procedural/randomized floorplan generation utilities.

## 3) Current pipeline used to process floorplan images

For image/PDF inputs, the active runtime path is:

1. API `/upload` stores file and metadata (`main.py`).
2. API `/process/{job_id}` builds `ProcessingPipeline` with raster params (`pixels_per_meter`, `pdf_dpi`, Hough params).
3. `ProcessingPipeline.run()` detects extension and routes non-DXF files to `_parse_raster()`.
4. `_parse_raster()` builds `RasterParser` and calls `parse()`.
5. `RasterParser.parse()` runs:
   - load image / convert first PDF page,
   - grayscale + denoise + CLAHE + Otsu inverse threshold + morphological close,
   - Canny edge detection,
   - HoughLinesP line extraction,
   - pixel→meter conversion into `ParsedGeometry.wall_segments`.
6. `WallDetector.detect()` creates wall objects from segments (double-line pair if possible, else single-line default-thickness wall fallback).
7. `RoomDetector.detect()` finds enclosed spaces from wall segments via flood-fill grid.
8. `OpeningDetector.detect()` maps door/window segments onto nearest walls.
   - For pure raster mode today, door/window classes are usually empty because raster parser currently only populates wall segments.
9. `GeometryBuilder.build()` creates final JSON for walls/floor/rooms/doors/windows and metadata.
10. API returns JSON and persists to `backend/models/{job_id}.json`.

## 4) Where walls, doors, windows, and rooms are currently handled

- **Walls**:
  - parsed/classified in `dxf_parser.py` and `raster_parser.py`
  - converted to `Wall` objects in `wall_detector.py`
  - turned into renderable wall boxes in `geometry_builder.py`
- **Doors**:
  - DXF-door extraction/mapping in `opening_detector.py`
  - render descriptors (leaf/frame placeholders) emitted by `geometry_builder.py`
  - displayed in `viewer/index_v2.html`
- **Windows**:
  - DXF-window extraction/mapping in `opening_detector.py`
  - render descriptors (sill/header/glass) emitted by `geometry_builder.py`
  - displayed in `viewer/index_v2.html`
- **Rooms**:
  - room region detection + label inference in `room_detector.py`
  - room labels exported in `geometry_builder.py`
  - room labels rendered in viewer(s)

## 5) Any hardcoded geometry or coordinates

### Hardcoded geometric constants/heuristics
- `raster_parser.py`: Canny thresholds, Hough defaults, min line length, default `pixels_per_meter`.
- `wall_detector.py`: minimum wall length, pairing distance thresholds, angle tolerances, raster collinear merge tolerances.
- `opening_detector.py`: door/window width bounds, max wall distance, minimum output wall-piece length.
- `room_detector.py`: grid resolution/padding/min-region thresholds and room type heuristics by area/aspect ratio.
- `geometry_builder.py`: sill/window height and door leaf thickness constants.

### Hardcoded viewer/camera/render values
- `viewer/index.html` and `viewer/index_v2.html`:
  - camera defaults (FOV, near/far planes, initial position/target),
  - light colors/intensities,
  - grid size,
  - material colors/opacities,
  - API URL hardcoded to `http://localhost:8000`.

### Hardcoded sample coordinates
- `sample_data/generate_sample.py` creates explicit wall/door/window coordinates (e.g., outer rectangle 10x8m, fixed interior splits and fixed opening locations).

## 6) Which files must be modified to support automatic detection

To improve automatic object detection from image/PDF inputs, these are the key required changes:

1. `backend/app/core/raster_parser.py`
   - Add semantic detection beyond plain line extraction (door/window symbol detection, contour/shape analysis, CNN/segmentation integration, or post-Hough classification).
   - Emit `door_segments`/`window_segments` (or richer primitives), not only `wall_segments`.

2. `backend/app/core/pipeline.py`
   - Pass source mode to wall detector correctly (`is_raster=True` for image/PDF) so raster fragment merge path is actually used.
   - Potentially branch into different opening/room detectors for raster confidence handling.

3. `backend/app/core/opening_detector.py`
   - Extend extraction logic for raster-derived openings (current logic is primarily DXF-layer driven).
   - Add confidence modeling and symbol/noise rejection for image-derived candidates.

4. `backend/app/core/room_detector.py`
   - Improve robustness to wall gaps/noise common in raster plans (gap-closing, topology repair, adaptive grid resolution).

5. `backend/app/core/wall_detector.py`
   - Tune/extend raster wall merging and pairing for noisy detection outputs.

6. `backend/app/core/geometry_builder.py`
   - If new opening/room representations are introduced, update geometry serialization contracts.

7. `viewer/index_v2.html` (and/or `index.html`)
   - Render any new semantic outputs and confidence overlays for detected elements.

8. Optional API surface updates in `backend/app/main.py`
   - expose additional detection configuration parameters and debug endpoints.

## 7) Which files are responsible for rendering

### Backend-side rendering payload generation (model serialization)
- `backend/app/core/geometry_builder.py`: produces the JSON scene primitives consumed by Three.js.

### Frontend-side rendering (actual visualization)
- `viewer/index.html`: initializes Three.js scene/camera/lights, renders walls/floors/room labels.
- `viewer/index_v2.html`: same core renderer plus explicit door/window rendering and visibility toggles.

## Focus summary (image processing / geometry creation / rendering)

- **Image processing core**: `raster_parser.py` (OpenCV + Hough extraction).
- **Geometry creation core**: `wall_detector.py`, `opening_detector.py`, `room_detector.py`, `geometry_builder.py`.
- **Rendering core**: `viewer/index.html`, `viewer/index_v2.html` (Three.js), using `geometry_builder.py` output schema.
