"""
Floor Plan Visualizer API v2

New in v2:
- Image support: PNG, JPG, BMP, TIFF
- PDF support (requires poppler)
- Hough parameter tuning via query params
- Auto-scale detection
- Debug image endpoint for raster files
"""

import uuid
import os
import json
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.core.pipeline import ProcessingPipeline, ALL_FORMATS

app = FastAPI(
    title="Floor Plan Visualizer API v2",
    description="Converts DXF / PNG / JPG / PDF floor plans into 3D model data.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
MODELS_DIR = Path("models")
UPLOAD_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

jobs: dict[str, dict] = {}
MAX_FILE_SIZE_MB = 50


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "0.2.0", "supported_formats": sorted(ALL_FORMATS)}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALL_FORMATS:
        raise HTTPException(400, f"Unsupported format '{suffix}'. Supported: {sorted(ALL_FORMATS)}")

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(413, f"File too large ({size_mb:.1f}MB). Max: {MAX_FILE_SIZE_MB}MB")

    job_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{job_id}{suffix}"
    save_path.write_bytes(contents)

    jobs[job_id] = {
        "status": "uploaded",
        "filename": file.filename,
        "filepath": str(save_path),
        "size_mb": round(size_mb, 3),
        "format": suffix,
        "result": None,
    }

    return {
        "job_id": job_id,
        "filename": file.filename,
        "format": suffix,
        "size_mb": round(size_mb, 3),
        "status": "uploaded",
        "next": f"POST /process/{job_id}",
    }


@app.post("/process/{job_id}")
def process_file(
    job_id: str,
    # DXF / general
    scale: float = Query(default=1.0, description="1 CAD unit = X meters. Use 0 for auto-detect"),
    auto_scale: bool = Query(default=True, description="Auto-infer scale from coordinate magnitude"),
    wall_height: float = Query(default=3.0, description="Default wall height in meters"),
    wall_thickness: float = Query(default=0.2, description="Default wall thickness in meters"),
    # Raster / image params
    pixels_per_meter: float = Query(default=100.0, description="For images: how many pixels = 1 meter"),
    pdf_dpi: int = Query(default=200, description="DPI for PDF rendering (higher = more detail, slower)"),
    hough_threshold: int = Query(default=50, description="Hough line vote threshold (lower = more lines)"),
    hough_min_length: int = Query(default=30, description="Minimum Hough line length in pixels"),
    hough_max_gap: int = Query(default=15, description="Maximum gap in a Hough line in pixels"),
):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    if job["status"] == "processing":
        raise HTTPException(409, "Already processing")

    jobs[job_id]["status"] = "processing"

    actual_scale = scale if scale > 0 else 1.0
    actual_auto = auto_scale if scale == 0 else auto_scale

    pipeline = ProcessingPipeline(
        scale=actual_scale,
        auto_scale=actual_auto,
        wall_height=wall_height,
        wall_thickness=wall_thickness,
        pixels_per_meter=pixels_per_meter,
        pdf_dpi=pdf_dpi,
        hough_threshold=hough_threshold,
        hough_min_length=hough_min_length,
        hough_max_gap=hough_max_gap,
    )

    result = pipeline.run(job["filepath"])
    result_dict = result.to_dict()

    if result.success and result.model:
        model_path = MODELS_DIR / f"{job_id}.json"
        model_path.write_text(json.dumps(result_dict, indent=2))
        jobs[job_id]["model_path"] = str(model_path)

    jobs[job_id]["status"] = "done" if result.success else "error"
    jobs[job_id]["result"] = result_dict

    return result_dict


@app.post("/process/{job_id}/debug-image")
def debug_image(job_id: str, output: str = Query(default="debug.png")):
    """
    For raster uploads: returns a PNG with detected Hough lines drawn on the original image.
    Useful for tuning hough_threshold / hough_min_length.
    """
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    filepath = job["filepath"]
    suffix = Path(filepath).suffix.lower()

    if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pdf"}:
        raise HTTPException(400, "Debug image only available for raster/PDF uploads")

    try:
        from app.core.raster_parser import RasterParser
        parser = RasterParser()
        out_path = str(MODELS_DIR / f"{job_id}_debug.png")
        parser.save_debug_image(filepath, out_path)
        return FileResponse(out_path, media_type="image/png", filename="debug_lines.png")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/model/{job_id}")
def get_model(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job["status"] == "uploaded":
        raise HTTPException(400, "Not yet processed. Call POST /process/{job_id} first.")
    if job["status"] == "processing":
        return {"status": "processing"}
    return job["result"]


@app.get("/jobs")
def list_jobs():
    return {
        jid: {"status": j["status"], "filename": j.get("filename"), "format": j.get("format")}
        for jid, j in jobs.items()
    }


@app.delete("/job/{job_id}")
def delete_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs.pop(job_id)
    for key in ("filepath", "model_path"):
        p = job.get(key)
        if p and os.path.exists(p):
            os.remove(p)
    return {"deleted": job_id}
