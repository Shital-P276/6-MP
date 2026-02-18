# Floor Plan → 3D Visualizer — Backend

FastAPI backend that converts DXF floor plans into 3D model JSON for Three.js.

---

## Project Structure

```
floorplan-visualizer/
├── requirements.txt
├── sample_data/
│   └── generate_sample.py       ← Creates a test DXF file
└── backend/
    ├── app/
    │   ├── main.py              ← FastAPI app + endpoints
    │   └── core/
    │       ├── dxf_parser.py    ← DXF → raw geometry
    │       ├── wall_detector.py ← segments → Wall objects
    │       ├── geometry_builder.py ← Walls → 3D mesh data
    │       └── pipeline.py      ← Orchestrates all steps
    └── tests/
        └── test_pipeline.py     ← pytest test suite
```

---

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Generate a Sample DXF File

```bash
cd sample_data
python generate_sample.py
# → Creates sample_floorplan.dxf (simple 3-room layout)
```

---

## Run the API

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

API docs available at: **http://localhost:8000/docs**

---

## API Usage

### 1. Upload a DXF file
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@../sample_data/sample_floorplan.dxf"

# Returns:
# { "job_id": "abc-123", "status": "uploaded", ... }
```

### 2. Process it
```bash
curl -X POST "http://localhost:8000/process/abc-123?scale=1.0&wall_height=3.0"

# Returns full 3D model JSON:
# { "success": true, "model": { "walls": [...], "floors": [...] }, ... }
```

### 3. Retrieve the model later
```bash
curl http://localhost:8000/model/abc-123
```

### Query Parameters for `/process`
| Param | Default | Description |
|---|---|---|
| `scale` | `1.0` | 1 CAD unit = X meters. Use `0.001` if DXF is in mm |
| `wall_height` | `3.0` | Default wall height in meters |
| `wall_thickness` | `0.2` | Default wall thickness (used if auto-detect fails) |

---

## Run Tests

```bash
cd backend
pytest tests/ -v
```

---

## 3D Model JSON Format

```json
{
  "metadata": {
    "wall_count": 8,
    "total_wall_length": 36.4,
    "bounds": { "minx": 0, "miny": 0, "maxx": 10, "maxy": 8 }
  },
  "walls": [
    {
      "type": "box",
      "position": { "x": 5.0, "y": 1.5, "z": 0.0 },
      "dimensions": { "width": 10.0, "height": 3.0, "depth": 0.2 },
      "rotation_y": 0.0,
      "length": 10.0
    }
  ],
  "floors": [
    {
      "type": "floor",
      "position": { "x": 5.0, "y": 0.0, "z": 4.0 },
      "dimensions": { "width": 10.0, "depth": 8.0 }
    }
  ]
}
```

Each wall box maps directly to a Three.js `BoxGeometry`:
- `position` → `mesh.position.set(x, y, z)`
- `dimensions.width/height/depth` → `new BoxGeometry(width, height, depth)`
- `rotation_y` → `mesh.rotation.y = rotation_y`

---

## What's Next

- **Phase 1 complete**: DXF parsing, wall detection, 3D JSON output, REST API
- **Phase 2**: Three.js viewer that consumes this JSON
- **Phase 3**: Door/window detection, room labeling, GLTF export
