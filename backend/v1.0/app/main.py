"""
Floor Plan Visualizer — FastAPI Backend

Endpoints:
    POST /upload          Upload a DXF file, returns a job ID
    POST /process/{id}    Process the uploaded file
    GET  /model/{id}      Retrieve the 3D model JSON
    GET  /health          Health check
"""

import uuid
import os
import json
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.pipeline import ProcessingPipeline

# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Floor Plan Visualizer API",
    description="Converts 2D DXF floor plans into 3D model data for web rendering.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Storage (in-memory + disk for MVP — replace with DB in production)
# ─────────────────────────────────────────────────────────────────────────────

UPLOAD_DIR = Path("uploads")
MODELS_DIR = Path("models")
UPLOAD_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# In-memory job store {job_id: {"status": ..., "filepath": ..., "result": ...}}
jobs: dict[str, dict] = {}

ALLOWED_EXTENSIONS = {".dxf"}
MAX_FILE_SIZE_MB = 50


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a DXF floor plan file.
    Returns a job_id to use for processing.
    """
    # Validate extension
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Accepted: {ALLOWED_EXTENSIONS}",
        )

    # Validate size
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max: {MAX_FILE_SIZE_MB} MB",
        )

    # Save to disk
    job_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{job_id}{suffix}"

    with open(save_path, "wb") as f:
        f.write(contents)

    jobs[job_id] = {
        "status": "uploaded",
        "filename": file.filename,
        "filepath": str(save_path),
        "size_mb": round(size_mb, 3),
        "result": None,
    }

    return {
        "job_id": job_id,
        "filename": file.filename,
        "size_mb": round(size_mb, 3),
        "status": "uploaded",
        "next": f"POST /process/{job_id}",
    }


@app.post("/process/{job_id}")
def process_file(
    job_id: str,
    scale: float = Query(default=1.0, description="Unit conversion: 1 CAD unit = X meters"),
    wall_height: float = Query(default=3.0, description="Default wall height in meters"),
    wall_thickness: float = Query(default=0.2, description="Default wall thickness in meters"),
):
    """
    Process an uploaded floor plan into a 3D model.

    Query params allow scale and geometry overrides.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    if job["status"] == "processing":
        raise HTTPException(status_code=409, detail="Already processing")

    # Run pipeline
    jobs[job_id]["status"] = "processing"

    pipeline = ProcessingPipeline(
        scale=scale,
        wall_height=wall_height,
        wall_thickness=wall_thickness,
    )

    result = pipeline.run(job["filepath"])
    result_dict = result.to_dict()

    if result.success and result.model:
        # Persist model JSON
        model_path = MODELS_DIR / f"{job_id}.json"
        with open(model_path, "w") as f:
            json.dump(result_dict, f, indent=2)
        jobs[job_id]["model_path"] = str(model_path)

    jobs[job_id]["status"] = "done" if result.success else "error"
    jobs[job_id]["result"] = result_dict

    return result_dict


@app.get("/model/{job_id}")
def get_model(job_id: str):
    """Retrieve the processed 3D model for a given job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    if job["status"] == "uploaded":
        raise HTTPException(status_code=400, detail="File not yet processed. POST /process/{job_id} first.")

    if job["status"] == "processing":
        return {"status": "processing", "message": "Still processing, try again shortly."}

    if job["result"] is None:
        raise HTTPException(status_code=500, detail="No result available.")

    return job["result"]


@app.get("/jobs")
def list_jobs():
    """List all jobs (dev/debug endpoint)."""
    return {
        jid: {
            "status": j["status"],
            "filename": j.get("filename"),
            "size_mb": j.get("size_mb"),
        }
        for jid, j in jobs.items()
    }


@app.delete("/job/{job_id}")
def delete_job(job_id: str):
    """Clean up a job and its files."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs.pop(job_id)

    for path_key in ("filepath", "model_path"):
        path = job.get(path_key)
        if path and os.path.exists(path):
            os.remove(path)

    return {"deleted": job_id}
