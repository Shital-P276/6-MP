# FloorViz (Floor Plan 3D Visualizer)

🌍 **Live Demo:** https://floorviz.netlify.app  
⚙️ **API Endpoint:** https://floorviz.up.railway.app

## Abstract / Overview
FloorViz converts 2D floor plans into interactive 3D browser models. Users upload DXF or raster plans (PNG/JPG/BMP/TIFF/PDF), the FastAPI backend extracts walls/rooms/openings, and the frontend renders the generated 3D geometry with camera controls, visualization modes, materials, and a room-based virtual tour. The project is intended for rapid architectural visualization, QA of plan geometry, and lightweight spatial walkthroughs without a desktop CAD toolchain.

## Features
- Multi-format floor plan ingestion (`.dxf`, `.png`, `.jpg/.jpeg`, `.bmp`, `.tif/.tiff`, `.pdf`)
- API pipeline for upload → process → retrieve model
- Computer-vision raster parsing (wall, room, door, window extraction)
- DXF parsing + wall pairing + opening/room inference
- 3D web rendering with Three.js (no frontend build step)
- Interactive controls (views, wireframe/solid/floor toggles, labels)
- Realistic/blueprint style modes + procedural material system
- Virtual tour mode with room hotspots, HUD navigation, autoplay, minimap
- Debug image endpoint for raster Hough/line-detection verification
- Persistent backend storage via Railway Volume (`/data/uploads`, `/data/models`)

## Screenshots / Demo
- Live product: https://floorviz.netlify.app
- Repository assets include sample images in `sample_data/img/` and `backend/debug.png` for parsing/debug output.

## Tech Stack

| Layer | Technology | Notes |
|---|---|---|
| Frontend | HTML/CSS/JavaScript, Three.js (CDN) | Single-file viewer app (`viewer/index_v12.html`) |
| Backend | FastAPI, Uvicorn | REST API + processing orchestration |
| Geometry/CAD | `ezdxf`, `shapely` | DXF parsing, geometry logic |
| CV/Imaging | `opencv-python-headless`, `numpy`, `Pillow`, `pdf2image` | Raster/PDF parsing and feature extraction |
| File Upload | `python-multipart` | Multipart form upload handling |
| Storage | Filesystem + Railway Volumes | Persistent data for uploads/models in production |
| Deployment | Netlify (frontend), Railway (backend) | Decoupled static UI + compute API |

## Architecture Summary
- **Frontend/backend communication:** The viewer calls backend endpoints directly over HTTP (`/health`, `/upload`, `/process/{job_id}`, etc.).
- **Processing flow:** Uploaded file is saved with a generated `job_id`, then processed by `ProcessingPipeline`, which dispatches DXF vs raster/PDF parsing, runs wall/room/opening detection, and builds a Three.js-friendly JSON model.
- **Request lifecycle:** Browser performs upload (multipart), then triggers processing with optional scale/geometry parameters, then renders returned model JSON.
- **Storage:** Backend writes uploaded source files and generated model JSON files to mounted storage paths.

## Repository Structure

```text
FloorViz/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app and endpoints
│   │   └── core/                   # Parsing + detection + geometry pipeline
│   ├── tests/                      # Unit/integration tests for pipeline components
│   └── models/                     # Generated model JSON artifacts (local/dev)
├── viewer/                         # Static frontend viewer variants (v12 is latest)
├── sample_data/                    # Sample DXF plans and generation scripts
├── dependencies/                   # Local poppler bundle/instructions
└── requirements.txt                # Python dependencies
```

## Installation

### Backend
```bash
cd backend
pip install -r ../requirements.txt
```

### Frontend
No package manager/build is required. Serve `viewer/` via any static HTTP server.

## Running Locally

1. **Start API**
```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

2. **Serve frontend**
```bash
cd viewer
python -m http.server 3000
```

3. Open:
- Viewer: `http://localhost:3000/index_v12.html`
- API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

> Note: `index_v12.html` currently defaults to `const API = 'http://localhost:8000'`. Update this constant for non-local environments.

## Environment Variables
The backend itself does not define custom `.env` variables in code, but relies on standard runtime variables and filesystem conventions:

| Variable | Required | Purpose |
|---|---|---|
| `PORT` | Production (Railway) | Bound by Uvicorn startup command (`${PORT:-8000}`) |

Operational paths are hardcoded in backend startup:
- `/data/uploads` for source uploads
- `/data/models` for output JSON models

In Railway, mount a persistent volume at `/data`.

## Deployment

### Railway (Backend)
- Root directory: typically `backend/` (so `requirements.txt` is discoverable)
- Start command:
```bash
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
```
- CORS is open (`allow_origins=["*"]`) for static frontend access.
- Use `opencv-python-headless` (already in `requirements.txt`) to avoid GUI-linked OpenCV failures in headless Linux containers.
- Configure Railway Volume mounted to `/data` for persistence.

### Netlify (Frontend)
- Static deployment (no build command)
- Publish directory should include `viewer/index_v12.html`
- Update frontend API constant to Railway URL for production



## API Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health + supported formats |
| `POST` | `/upload` | Upload floor plan and receive `job_id` |
| `POST` | `/process/{job_id}` | Run processing pipeline with tuning query params |
| `POST` | `/process/{job_id}/debug-image` | Generate line-overlay debug PNG (raster/PDF only) |
| `GET` | `/model/{job_id}` | Retrieve processing result or status |
| `GET` | `/jobs` | List in-memory jobs and statuses |
| `DELETE` | `/job/{job_id}` | Delete job + related stored files |

## Limitations
- Job state is kept in process memory; restart clears active job index.
- `allow_origins=["*"]` is permissive for production.
- Frontend API URL is hardcoded in `index_v12.html`.
- Processing is synchronous per request (no queue/worker separation).
- Large set of generated model artifacts in-repo suggests cleanup/retention policy is needed.

## Future Improvements
- Move to persistent job metadata store (DB/Redis)
- Configurable environment-driven API base URL in frontend
- Add auth/rate limiting and tighter CORS
- Add async/background processing queue and progress tracking
- Add Dockerfile + CI/CD workflows + formal release process
- Expand automated tests for raster/PDF pipelines and endpoint contracts

## Contributing
1. Fork and create a feature branch.
2. Keep changes focused and include tests where practical.
3. Validate backend endpoints and viewer behavior locally.
4. Submit PR with architecture/behavior notes and screenshots if UI changed.

## License
FloorBiz was originally developed as a college project. Third-party dependencies (like Poppler and Three.js) retain their respective licenses.
