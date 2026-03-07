# How the API Works in This Project

This guide explains the backend API flow for the **Floor Plan 3D Visualizer** in a practical way.

---

## 1) Big picture (what happens end-to-end)

The API follows a **job-based pipeline**:

1. `POST /upload` → stores the input file and returns a `job_id`.
2. `POST /process/{job_id}` → runs parsing + wall/room/opening detection + 3D model generation.
3. `GET /model/{job_id}` → fetches the stored result for that job.

Extra helper endpoints:
- `GET /health` for connectivity check
- `GET /jobs` to list current in-memory jobs
- `DELETE /job/{job_id}` to remove uploaded/model files
- `POST /process/{job_id}/debug-image` for raster debug output

---

## 2) Endpoint-by-endpoint explanation

### `GET /health`
Use this first to confirm backend status.

Typical response:
```json
{
  "status": "ok",
  "version": "0.2.0",
  "supported_formats": [".bmp", ".dxf", ".jpeg", ".jpg", ".pdf", ".png", ".tif", ".tiff"]
}
```

---

### `POST /upload`
Uploads one floor plan file (`multipart/form-data`, field name = `file`).

What it does:
- Validates extension against supported formats.
- Enforces max file size (50 MB).
- Saves file in `uploads/` using a generated UUID as filename.
- Creates a job entry in in-memory `jobs` dictionary.

Typical success response:
```json
{
  "job_id": "<uuid>",
  "filename": "floor_plan1.dxf",
  "format": ".dxf",
  "size_mb": 0.341,
  "status": "uploaded",
  "next": "POST /process/<job_id>"
}
```

---

### `POST /process/{job_id}`
Runs the core pipeline and returns a full processing result.

Important query params:
- `scale` (default `1.0`) → CAD unit-to-meter scale; use `0` for auto detect.
- `auto_scale` (default `true`) → allow scale inference.
- `wall_height` (default `3.0`) → wall extrusion height.
- `wall_thickness` (default `0.2`) → fallback thickness.
- `pixels_per_meter` (default `0`) → raster scale; `0` triggers auto-detection.
- `pdf_dpi` (default `200`) → PDF rendering quality/speed trade-off.

Internally this endpoint:
1. Loads job file path.
2. Chooses parser by extension (`DXF` vs `raster/PDF`).
3. Detects walls.
4. Detects rooms.
5. Detects openings (doors/windows).
6. Builds final 3D model JSON.
7. Stores result in memory, and if success writes `models/<job_id>.json`.

If successful, response includes:
- `success: true`
- `processing_time_ms`
- `warnings` (very useful diagnostics)
- `source_type`
- `applied_scale`
- `model` (geometry for frontend)
- `stats` (wall/room/door/window counters)

---

### `GET /model/{job_id}`
Used to fetch result by job id after processing.

Behavior:
- If uploaded but not processed: returns 400 with message.
- If still processing: returns `{ "status": "processing" }`.
- Else returns stored processing result.

---

### `POST /process/{job_id}/debug-image`
Useful for raster/PDF inputs.

Returns a PNG with detected lines drawn on image to help tune detection quality.

---

### `GET /jobs` and `DELETE /job/{job_id}`
Operational endpoints:
- `GET /jobs` gives status summary of each in-memory job.
- `DELETE /job/{job_id}` removes tracked job + uploaded/model files from disk.

---

## 3) Core pipeline internals

The backend delegates work to `ProcessingPipeline`.

High-level processing order in `run()`:
1. Validate path + format.
2. Parse source:
   - DXF via `DXFParser`
   - image/PDF via `RasterParser` (or `SimpleDrawParser` if detected)
3. Fallback logic if no wall layer found.
4. `WallDetector` to produce wall objects and applied scale.
5. `RoomDetector` to infer rooms.
6. `OpeningDetector` or raster opening mapper for door/window openings.
7. `GeometryBuilder` to produce final `BuildingModel` returned to frontend.

---

## 4) How frontend uses the API

`viewer/index_v2.html` calls these endpoints in this order:
1. `GET /health` on load to show API connected/offline badge.
2. On upload action:
   - `POST /upload`
   - then `POST /process/{job_id}` with UI-configured params.
3. If processing succeeds, frontend immediately builds and renders model from returned JSON.

So in normal UI flow, frontend usually does **upload + process directly**, without separately calling `GET /model/{job_id}`.

---

## 5) cURL examples you can run

From project root:

### Health
```bash
curl -s http://localhost:8000/health
```

### Upload
```bash
curl -s -X POST http://localhost:8000/upload \
  -F "file=@sample_data/floor_plan1.dxf"
```

### Process (replace `<job_id>`)
```bash
curl -s -X POST "http://localhost:8000/process/<job_id>?scale=1&wall_height=3&wall_thickness=0.2&pixels_per_meter=0"
```

### Get model
```bash
curl -s http://localhost:8000/model/<job_id>
```

### Delete job
```bash
curl -s -X DELETE http://localhost:8000/job/<job_id>
```

---

## 6) Important implementation notes

- Job state is held in an in-memory dictionary (`jobs`), so restarting the backend clears runtime job metadata.
- CORS is fully open (`*`) to simplify browser integration during development.
- `/process/{job_id}` is implemented as **POST** in backend and frontend code.
- README and implementation are aligned: `/process/{job_id}` uses POST.

