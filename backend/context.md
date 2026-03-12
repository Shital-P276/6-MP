# FloorViz Backend - Project Context

**Project Overview**: FloorViz is a floor plan processing system that converts architectural drawings (DXF, images, PDFs) into 3D interactive models for visualization and analysis.

**Current Status**: Production-grade backend with REST API, supporting vector and raster floor plan formats.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Project Architecture](#project-architecture)
3. [Directory Structure](#directory-structure)
4. [Core Components](#core-components)
5. [API Documentation](#api-documentation)
6. [Data Structures](#data-structures)
7. [Processing Pipeline](#processing-pipeline)
8. [Configuration & Constants](#configuration--constants)
9. [Testing & Debugging](#testing--debugging)
10. [Development Workflow](#development-workflow)
11. [Known Limitations & Future Work](#known-limitations--future-work)

---

## Quick Start

### Running the Backend

```bash
# Start FastAPI server (assuming dependencies installed)
python app/core/main.py

# Server runs on http://localhost:8000
# API docs: http://localhost:8000/docs (Swagger UI)
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_pipeline.py -v
```

### Project Dependencies

Core packages (no requirements.txt currently):
- **fastapi** - REST API framework
- **ezdxf** - DXF file parsing and generation
- **opencv-python (cv2)** - Image processing, Hough line detection
- **pdf2image** - PDF to raster conversion
- **numpy** - Array operations
- **shapely** - Geometric operations (distance, intersection calculations)
- **pillow** - Image manipulation (PIL)

---

## Project Architecture

```
┌─────────────────┐
│   HTTP Client   │
└────────┬────────┘
         │ REST API
    ┌────▼──────────────────┐
    │  FastAPI Application   │
    │  (app/core/main.py)    │
    └────┬───────────────────┘
         │ orchestration
    ┌────▼──────────────────────┐
    │ ProcessingPipeline         │
    │ (app/core/pipeline.py)     │
    └────┬───────────────────────┘
         │
    ┌────┴────────────────────────────────────┐
    │                                         │
    └────┬─────────────────────────────────┬──┴─────┐
         │                                 │        │
    ┌────▼──────────┐    ┌────────────┐   │   ┌────▼──────────┐
    │  DXF Parser   │    │Raster Parser│   │   │ Wall Detector │
    │ (dxf_parser.py)│  │(raster...py) │   │   │(wall_det...py)│
    └────┬──────────┘    └────────────┘   │   └────┬──────────┘
         │                                 │        │
         └─────────────────────┬───────────┴────────┘
                          ┌────▼──────────────────┐
                          │  Room Detector        │
                          │(room_detector.py)     │
                          └────┬──────────────────┘
                               │
                          ┌────▼──────────────────┐
                          │ Opening Detector      │
                          │(opening_detector.py)  │
                          └────┬──────────────────┘
                               │
                          ┌────▼──────────────────────┐
                          │ Geometry Builder (3D)     │
                          │(geometry_builder.py)      │
                          └────┬──────────────────────┘
                               │ Three.js JSON
                          ┌────▼────────────────┐
                          │  Output Model File   │
                          │  (models/UUID.json)  │
                          └─────────────────────┘
```

---

## Directory Structure

```
d:\Projects\floorviz\backend\
│
├── app/                              # Main application package
│   ├── core/
│   │   ├── __init__.py              # Public API exports
│   │   ├── main.py                  # FastAPI REST API server (6 endpoints)
│   │   ├── pipeline.py              # ProcessingPipeline orchestrator
│   │   ├── dxf_parser.py            # DXF file parsing (AutoCAD)
│   │   ├── raster_parser.py         # Image/PDF parsing (Hough detect)
│   │   ├── wall_detector.py         # Wall detection & pairing logic
│   │   ├── geometry_builder.py      # 3D model generation (Three.js)
│   │   ├── room_detector.py         # Room detection (flood-fill)
│   │   └── opening_detector.py      # Door & window detection
│   └── __pycache__/
│
├── tests/
│   └── test_pipeline.py             # 20+ unit & integration tests
│
├── models/                          # Output 3D models (UUID-named .json)
├── uploads/                         # Temporary uploaded files
│
├── context.md                       # This file - project context
├── ClaudeGuide.md                   # ML implementation roadmap (~40KB)
├── ClaudeGuideExtra.md              # Extended ML planning notes (~23KB)
└── debug_test.py                    # Debugging/quick testing script
```

---

## Core Components

### 1. **FastAPI Server** (`app/core/main.py`)

**Purpose**: HTTP REST API for receiving floor plans and returning 3D models.

**Key Features**:
- CORS enabled for cross-origin requests
- File upload (50MB max, formats: DXF, PNG, JPG, BMP, TIFF, PDF)
- Job-based processing (async job tracking via UUID)
- Debug endpoints for visualization

**Main Classes**:
- `FloorVizAPI` - Main API server instance

---

### 2. **Processing Pipeline** (`app/core/pipeline.py`)

**Purpose**: Orchestrate all processing steps in a defined sequence.

**Key Classes**:
- `ProcessingPipeline` - Main orchestrator
- `PipelineResult` - Output dataclass with model, metadata, and success flags

**Key Methods**:
- `process(file_path, config)` - Execute complete pipeline

**Responsibilities**:
1. Load file (DXF or raster)
2. Parse input (delegate to DXFParser or RasterParser)
3. Detect walls
4. Detect rooms
5. Detect openings (doors/windows)
6. Generate 3D geometry
7. Return complete model

---

### 3. **DXF Parser** (`app/core/dxf_parser.py`)

**Purpose**: Extract geometry from AutoCAD DXF files with layer-based classification.

**Key Classes**:
- `DXFParser` - Main parser
- `ParsedGeometry` - Container for extracted segments
- `Segment` - Individual line segment with properties
- `Point2D` - 2D point (x, y)

**Supported Layers**:
- `WALL` - Wall centerlines or outlines
- `DOOR` - Door arcs with swing angles
- `WINDOW` - Window lines
- `OTHER` - Unclassified geometry

**Key Methods**:
- `parse(filepath)` → `ParsedGeometry` with segments grouped by layer
- `get_bounds()` - Returns bounding box (min_x, min_y, max_x, max_y)

**Output**:
- Segments (lines) classified by layer for downstream processing
- Bounds for spatial reference

---

### 4. **Raster Parser** (`app/core/raster_parser.py`)

**Purpose**: Extract geometry from images/PDFs using Hough line detection.

**Key Classes**:
- `RasterParser` - Main parser for raster images and PDFs

**Processing Steps**:
1. Convert PDF to image (if needed)
2. Detect wall lines using Hough transform (OpenCV)
3. Scan image for bright areas (potential openings)
4. Generate segments from detected lines
5. Estimate scale from user input or heuristics

**Key Methods**:
- `parse(filepath, pixels_per_meter, pdf_dpi)` → `ParsedGeometry`
- `get_debug_image()` - Return PNG with detected lines overlaid

**Tuning Constants**:
- `WALL_LO=82, WALL_HI=148` - CAD floor plan color range (BGR)
- `MIN_JAMB_M=0.50m` - Minimum door stub length
- `MAX_DOOR_M=1.40m`, `MAX_WINDOW_M=3.50m` - Opening size filters
- Hough: `threshold=7, min_len=12px, max_gap=60px`

**Limitations**:
- Hough detection requires good contrast between walls and background
- May struggle with complex/detailed floor plans
- Manual scale input recommended for accuracy

---

### 5. **Wall Detector** (`app/core/wall_detector.py`)

**Purpose**: Process raw segments into classified walls with thickness inference.

**Key Classes**:
- `WallDetector` - Main detector
- `Wall` - Detected wall with thickness, height, and properties

**Key Algorithm**:
1. **Filter tiny segments** - Remove geometry < 0.1m
2. **Parallel pairing** - Detect double-line walls (typical CAD drafting)
   - Find segments within 0.3m offset with similar angle
   - Average centerline
   - Infer thickness from separation
3. **Scale application** - Apply manual or auto-inferred scale
4. **Angle calculation** - Compute wall orientation

**Key Methods**:
- `detect(segments, scale, wall_height, wall_thickness)` → list of `Wall` objects
- `detect_parallel_walls(segments)` - Find double-line wall pairs
- `infer_scale(segments)` - Auto-detect scale from coordinate ranges (heuristic)

**Output**:
- `Wall` objects with:
  - `start`, `end` - 3D coordinates
  - `thickness` - Wall thickness (m)
  - `height` - Wall height (m)
  - `angle_deg` - Orientation (degrees)
  - `paired` - Boolean flag

---

### 6. **Room Detector** (`app/core/room_detector.py`)

**Purpose**: Identify enclosed regions (rooms) using flood-fill on rasterized walls.

**Key Classes**:
- `RoomDetector` - Main detector
- `Room` - Detected room with properties

**Algorithm**:
1. **Rasterize walls to 2D grid** - 5cm resolution (GRID_RES)
2. **Flood-fill** - Find connected empty cells (rooms)
3. **Classify by heuristics**:
   - Area thresholds (hallway <8m², bedroom 15-25m², kitchen 15-20m²)
   - Aspect ratio analysis
   - Layer hints (if available from DXF)

**Key Methods**:
- `detect(walls)` → list of `Room` objects
- Room properties: `id`, `centroid`, `area`, `width`, `depth`, `room_type`, `label`, `color`

---

### 7. **Opening Detector** (`app/core/opening_detector.py`)

**Purpose**: Identify doors and windows within walls.

**Key Classes**:
- `OpeningDetector` - Main detector
- `Opening` - Door or window with position and swing angle

**Processing**:
1. **Door detection**:
   - Parse door arcs from DXF (swing angles)
   - Calculate door frame center and opening width
   - Find nearest wall for placement
2. **Window detection**:
   - Identify window segments by layer or characteristics
   - Calculate dimensions and position
   - Find nearest wall

**Key Methods**:
- `detect(walls, segments)` → list of `Opening` objects
- Opening properties: `wall_idx`, `t_center` (position on wall), `width`, `kind` (door/window), `x`, `y`, `angle`, `swing_side`

---

### 8. **Geometry Builder** (`app/core/geometry_builder.py`)

**Purpose**: Convert detected architecture into Three.js JSON 3D model.

**Key Classes**:
- `GeometryBuilder` - Main 3D model generator
- `BuildingModel` - Output model representation

**Processing**:
1. **Wall generation**: Create box geometry for each wall
2. **Opening placement**: Split walls at door/window positions
3. **Door geometry**: Create frame + swing leaf
4. **Window voids**: Remove wall geometry at window positions
5. **Floor generation**: Create polygon from outer wall boundary
6. **Metadata**: Assemble statistics and bounds

**Output Format** (Three.js JSON):
```json
{
  "walls": [
    {
      "type": "box",
      "width": 4.5,
      "height": 2.8,
      "depth": 0.15,
      "position": [2.25, 1.4, 0],
      "rotation": [...],
      "userData": {"layer": "WALL", ...}
    }
  ],
  "floors": [
    {
      "points": [[0,0], [10,0], [10,8], [0,8]],
      "color": "#f0f0f0"
    }
  ],
  "rooms": [
    {
      "id": "room_1",
      "centroid": [2.5, 2.0],
      "area": 7.5,
      "room_type": "living",
      "color": "#ffb3ba"
    }
  ],
  "metadata": {
    "bounds": {"min": [0,0], "max": [10,8]},
    "wall_count": 12,
    "room_count": 4,
    "scale_m": 1.0,
    "height_m": 2.8
  }
}
```

---

## API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. **Health Check**
```
GET /health
```

**Response**:
```json
{
  "status": "ok",
  "supported_formats": ["dxf", "png", "jpg", "jpeg", "bmp", "tif", "tiff", "pdf"],
  "max_file_size_mb": 50
}
```

---

#### 2. **Upload Floor Plan**
```
POST /upload
Content-Type: multipart/form-data

Body:
  file: <floor_plan_file>
```

**Supported Formats**:
- DXF (AutoCAD vector)
- PNG, JPG, BMP, TIFF (raster images)
- PDF (multi-page documents)

**Response**:
```json
{
  "job_id": "abc123-def456-...",
  "filename": "floor_plan.dxf",
  "file_path": "/uploads/abc123.dxf",
  "status": "uploaded"
}
```

---

#### 3. **Process Floor Plan**
```
POST /process/{job_id}
Content-Type: application/json

Body:
{
  "scale": 1.0,
  "auto_scale": false,
  "wall_height": 2.8,
  "wall_thickness": 0.15,
  "pixels_per_meter": null,
  "pdf_dpi": 150
}
```

**Parameters**:
- `scale` (float, default: 1.0) - Manual scale factor (m/unit)
- `auto_scale` (bool, default: false) - Auto-detect scale from coordinates
- `wall_height` (float, default: 2.8) - Wall height in meters
- `wall_thickness` (float, default: 0.15) - Wall thickness in meters
- `pixels_per_meter` (float, optional) - For raster images (scale hint)
- `pdf_dpi` (int, default: 150) - PDF rendering DPI

**Response**:
```json
{
  "job_id": "abc123-def456-...",
  "status": "success",
  "model": { ... },  // Three.js JSON model
  "metadata": {
    "wall_count": 12,
    "room_count": 4,
    "total_area_m2": 85.5,
    "bounds": {
      "min": [0, 0],
      "max": [12.5, 10.2]
    }
  }
}
```

---

#### 4. **Debug Visualizer** (Raster Only)
```
POST /process/{job_id}/debug-image
```

**Response**: PNG image with detected Hough lines overlaid on original floor plan.

---

#### 5. **Retrieve Model**
```
GET /model/{job_id}
```

**Response**: Complete Three.js JSON 3D model.

---

#### 6. **List Jobs**
```
GET /jobs
```

**Response**:
```json
{
  "jobs": [
    {
      "job_id": "abc123-...",
      "filename": "floor_plan.dxf",
      "status": "processed",
      "created_at": "2026-03-12T10:30:00Z",
      "processed_at": "2026-03-12T10:30:15Z"
    }
  ],
  "total": 5
}
```

---

#### 7. **Delete Job**
```
DELETE /job/{job_id}
```

**Response**:
```json
{
  "job_id": "abc123-...",
  "status": "deleted",
  "message": "Job and associated files removed"
}
```

---

## Data Structures

### Geometric Primitives

**Point2D**
```python
@dataclass
class Point2D:
    x: float
    y: float
```

**Segment**
```python
@dataclass
class Segment:
    start: Point2D
    end: Point2D
    layer: str  # "WALL", "DOOR", "WINDOW", "OTHER"
    source_type: str  # "dxf", "raster", "simple_draw"
```

**ParsedGeometry**
```python
@dataclass
class ParsedGeometry:
    segments: list[Segment]
    bounds: tuple  # (min_x, min_y, max_x, max_y)
    text_labels: dict  # Map of text content to positions
    layer_summary: dict  # Count per layer
```

---

### Detected Features

**Wall**
```python
@dataclass
class Wall:
    start: tuple  # (x, y, z)
    end: tuple    # (x, y, z)
    thickness: float
    height: float
    paired: bool  # Double-line wall?
    angle_deg: float
```

**Room**
```python
@dataclass
class Room:
    id: str
    centroid: tuple  # (x, y)
    area: float  # m²
    width: float
    depth: float
    room_type: str  # "bedroom", "kitchen", "hallway", "living", "bathroom"
    label: str
    color: str  # Hex color for visualization
```

**Opening** (Door or Window)
```python
@dataclass
class Opening:
    wall_idx: int  # Index in walls list
    t_center: float  # Position along wall (0.0 to 1.0)
    width: float  # Opening width (m)
    kind: str  # "door", "window"
    x: float  # 3D position
    y: float
    angle: float  # Swing angle (doors)
    swing_side: str  # "left", "right"
```

---

### Output Model

**BuildingModel** (converted to JSON)
```python
{
    "walls": [
        {
            "type": "box",
            "width": float,
            "height": float,
            "depth": float,
            "position": [x, y, z],
            "rotation": [rx, ry, rz],
            "userData": {...}
        }
    ],
    "floors": [
        {
            "points": [[x, y], ...],
            "color": str
        }
    ],
    "rooms": [
        {
            "id": str,
            "centroid": [x, y],
            "area": float,
            "room_type": str,
            "color": str
        }
    ],
    "metadata": {
        "bounds": {"min": [x, y], "max": [x, y]},
        "wall_count": int,
        "room_count": int,
        "scale_m": float,
        "height_m": float
    }
}
```

---

## Processing Pipeline

### Complete Workflow (Sequence)

```
1. INPUT VALIDATION
   ├─ Check file exists
   ├─ Validate format (extension)
   └─ Check file size < 50MB

2. PARSING
   ├─ If DXF → DXFParser
   │  ├─ Load ezdxf file
   │  ├─ Extract WALL, DOOR, WINDOW layers
   │  └─ Convert to Segments
   │
   └─ If Raster/PDF → RasterParser
      ├─ Convert PDF to PNG (if needed)
      ├─ Detect walls via Hough transform
      ├─ Scan for openings (bright areas)
      └─ Generate Segments

3. WALL DETECTION
   ├─ Filter segments < 0.1m
   ├─ Attempt parallel wall pairing
   ├─ Apply scale (auto or manual)
   └─ Output Wall objects (centerlines + thickness)

4. ROOM DETECTION
   ├─ Rasterize walls to 5cm grid
   ├─ Flood-fill to find enclosed regions
   ├─ Classify rooms by area + aspect ratio
   └─ Output Room objects with colors + labels

5. OPENING DETECTION
   ├─ Extract door arcs + swing angles
   ├─ Identify windows by layer/size
   ├─ Match to nearest walls
   └─ Output Opening objects

6. 3D GEOMETRY GENERATION
   ├─ Create Three.js box geometry per wall
   ├─ Split walls at opening positions
   ├─ Add door frames + swing leaves
   ├─ Create window voids
   ├─ Generate floor polygon
   ├─ Assemble metadata
   └─ Output Three.js JSON

7. RETURN MODEL
   └─ Save JSON to models/UUID.json
```

### Processing Time Estimates

- **Small DXF** (1 floor, 50 walls): ~100-200ms
- **Large DXF** (5+ floors, 200+ walls): ~500-1000ms
- **Raster/PDF** (1 page): ~1-3s (includes Hough detection)
- **Multi-page PDF**: +500-800ms per page

---

## Configuration & Constants

### Wall Detection (`wall_detector.py`)

```python
FILTER_MIN_LENGTH = 0.1  # Minimum segment length (m)
PARALLEL_MAX_OFFSET = 0.3  # Max offset for parallel detection (m)
PARALLEL_MIN_ANGLE_SIM = 0.95  # Angle similarity threshold
CORNERSTONE_ANGLE_TOL = 5.0  # Corner angle tolerance (degrees)
```

### Room Detection (`room_detector.py`)

```python
GRID_RES = 0.05  # Rasterization resolution (m) - 5cm cells
HALLWAY_MAX_AREA = 8.0  # m²
BEDROOM_AREA = (15, 25)  # m² range
KITCHEN_AREA = (15, 20)  # m² range
MIN_ASPECT_RATIO = 0.3  # Width/length ratio
```

### Raster Parser (`raster_parser.py`)

```python
# Color detection for CAD floor plans (BGR)
WALL_LO = 82  # Dark blue (floor plan default)
WALL_HI = 148

# Hough line detection
HOUGH_THRESHOLD = 7
HOUGH_MIN_LEN = 12  # pixels
HOUGH_MAX_GAP = 60  # pixels
MERGE_DISTANCE = 30  # pixels

# Opening detection
MIN_JAMB_M = 0.50  # Minimum door stub length (m)
MAX_DOOR_M = 1.40  # Maximum door width (m)
MAX_WINDOW_M = 3.50  # Maximum window width (m)
```

### Geometry Builder (`geometry_builder.py`)

```python
DEFAULT_WALL_HEIGHT = 2.8  # meters
DEFAULT_WALL_THICKNESS = 0.15  # meters
DOOR_FRAME_DEPTH = 0.10  # meters
WINDOW_PROPORTION = 0.8  # 80% of opening width
```

### API Server (`main.py`)

```python
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
UPLOAD_DIR = "./uploads"
MODEL_OUTPUT_DIR = "./models"
ALLOWED_EXTENSIONS = {
    'dxf', 'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff', 'pdf'
}
PDF_DPI = 150  # Default PDF rendering resolution
```

---

## Testing & Debugging

### Running Tests

```bash
# Run all tests with verbose output
pytest tests/test_pipeline.py -v

# Run specific test class
pytest tests/test_pipeline.py::TestWallDetector -v

# Run with coverage
pytest tests/test_pipeline.py --cov=app/core --cov-report=html
```

### Test Categories

1. **Layer Classification** - DXF layer-based detection
2. **Wall Detection Functions**:
   - Angle calculation
   - Parallel wall detection
   - Wall pairing logic
3. **Geometry Builder** - Output format validation
4. **Integration Tests** - End-to-end pipeline on sample files
5. **Error Handling** - Invalid inputs, malformed files

### Debug Features

**Debug Image Generation** (Raster only):
```python
# After raster parsing, retrieve debug PNG
debug_image = raster_parser.get_debug_image()
# Shows Hough lines detected, overlaid on original
```

**Quick Test Script** (`debug_test.py`):
- Script for manual testing without HTTP API
- Useful for debugging specific file formats
- Can be run directly: `python debug_test.py`

### Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Walls not detected (raster) | Poor image contrast | Adjust HOUGH_THRESHOLD, ensure dark walls on light background |
| Scale off by 2x | Assumption of double-line walls | Disable auto_scale, provide manual scale |
| Rooms not detected | Walls not closed/continuous | Check wall geometry, use wall_detector debug output |
| Opening detection fails | Opening size outside thresholds | Adjust MAX_DOOR_M, MAX_WINDOW_M constants |
| PDF renders slowly | High DPI setting | Reduce pdf_dpi parameter (default 150) |

---

## Development Workflow

### Adding a New Processing Step

1. **Create module** (`app/core/new_processor.py`)
2. **Define dataclass** for input/output
3. **Implement main class** with `detect/process` method
4. **Update pipeline** (`pipeline.py`) to call new step
5. **Add tests** (`tests/test_pipeline.py`)
6. **Update API** if new parameters needed

### Adding a New API Endpoint

1. **Add route** in `main.py`:
   ```python
   @app.get("/new-endpoint/{job_id}")
   async def new_endpoint(job_id: str):
       # Implementation
       return {"result": ...}
   ```
2. **Test with Swagger UI** (http://localhost:8000/docs)
3. **Update documentation** (this file)

### Dependency Selection

For geometric operations:
- **shapely** - Line intersection, buffering, distance
- **numpy** - Array operations, grid processing
- **ezdxf** - DXF generation/parsing
- **cv2** - Image processing, Hough lines

Avoid:
- Heavy ML libraries (use raster_parser.py heuristics instead)
- Multiple coordinate systems (stick to DXF units → scale to meters)

### Code Style

- Use `@dataclass` for data structures
- Type hints on all function signatures
- Docstrings for public APIs
- Single responsibility per module
- Avoid mutable default arguments

---

## Known Limitations & Future Work

### Current Limitations

1. **Raster Parser**:
   - Hough detection requires good contrast (struggles with complex plans)
   - No automatic perspective correction (assumes orthogonal scan)
   - Manual scale often needed for accuracy

2. **Room Detection**:
   - Simple area-based heuristics (may misclassify unusual shapes)
   - No multi-floor support (assumed single-floor per file)
   - Text label extraction not implemented

3. **Opening Detection**:
   - DXF: Requires door arcs + DOOR layer
   - Raster: Limited to brightness scanning (no swing angle detection)

4. **Performance**:
   - No multi-threading (single-threaded processing)
   - Memory footprint scales with floor plan complexity
   - PDF multi-page processing done sequentially

### Future Enhancement Roadmap

**Phase 1: ML-Based Raster Processing** (documented in `ClaudeGuide.md`)
- Replace Hough detection with SegFormer semantic segmentation
- Automated wall/door/window classification from image
- Multi-floor support via page/annotated regions
- Dataset: CubiCasa5k + custom augmentation

**Phase 2: Advanced ML** (referenced in `ClaudeGuideExtra.md`)
- MuraNet end-to-end model (wall detection + room + openings)
- Fine-tuning on Indian floor plan dataset
- Real-time processing pipeline

**Phase 3: Extended Features**
- Multi-floor stacking (Z-level support)
- Text label extraction & room naming
- Furniture placement hints from floor plans
- Export to CAD formats (DWG, IFC)
- Batch processing API
- Web viewer integration

### Optimization Opportunities

1. **Caching**: Room detection grid rasterization could be cached for repeated queries
2. **Vectorization**: Wall pairing logic (currently O(n²)) could use spatial indexing
3. **Parallel Processing**: Multiple floor pages could be processed concurrently
4. **Geometry Simplification**: Merge adjacent walls with same thickness/height

---

## Quick Reference: Key Files Summary

| File | Responsibility | Key Class | Lines |
|------|-----------------|-----------|-------|
| `main.py` | REST API server | `FloorVizAPI` | ~300 |
| `pipeline.py` | Orchestration | `ProcessingPipeline` | ~200 |
| `dxf_parser.py` | DXF parsing | `DXFParser` | ~150 |
| `raster_parser.py` | Image/PDF parsing | `RasterParser` | ~400+ |
| `wall_detector.py` | Wall detection | `WallDetector` | ~250 |
| `room_detector.py` | Room detection | `RoomDetector` | ~200 |
| `opening_detector.py` | Door/window detection | `OpeningDetector` | ~150 |
| `geometry_builder.py` | 3D model generation | `GeometryBuilder` | ~300 |
| `test_pipeline.py` | Test suite | Various test classes | ~600+ |

---

## For AI Assistant (Claude) Development Notes

When continuing development on this project:

1. **Always check layer classification** - DXF uses layers (`WALL`, `DOOR`, `WINDOW`) to separate geometry types
2. **Scale is critical** - Manual scale input often required for raster images; auto-scale is a fallback
3. **Wall thickness matters** - Default 0.15m, but double-line walls compute thickness from offset
4. **Coordinate system**: DXF (X,Y) → Pipeline (X, Y, Z meters) → Three.js JSON (X, -Z, Y for 3D view)
5. **Room classification** is heuristic-based (area + aspect ratio) - may need refinement
6. **Testing is essential** - Use `pytest` with sample floor plans before deploying changes
7. **Performance bottleneck**: Raster processing (Hough detection) dominates for image-based floors
8. **No multi-threading currently** - Sequential processing is safe but could be optimized

---

**Last Updated**: 2026-03-12
**Status**: Production-grade with ML roadmap for enhancement
