# Floor Plan 3D Visualizer

Upload a floor plan image (PNG/JPG/PDF) or DXF file and get an interactive 3D model with textured walls, doors, and windows.

**Stack:** FastAPI (Python 3.10+) + Three.js (browser, no build step)
**Repo:** `https://github.com/Shital-P276/6-MP/tree/v2.2`

---

## Quick Start

### 1. Start the backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

You should see: `Uvicorn running on http://127.0.0.1:8000`

### 2. Serve the viewer

> ⚠️ **Do NOT open `index_v2.html` directly as a file.** Browsers block `fetch()` to localhost from `file://` URLs. You must serve it via HTTP.

```bash
# From the viewer/ folder:
python -m http.server 3000
```

Then open: **`http://localhost:3000/index_v2.html`**

The green dot in the top-right header should show **API CONNECTED**.

### 3. Upload a floor plan

Drop a PNG/JPG/PDF floor plan into the upload zone (or click to browse). Supported DXF files also work. Hit **PROCESS FLOOR PLAN**.

---

## Project Structure

```
project/
├── backend/
│   └── app/
│       ├── main.py                  ← FastAPI routes
│       └── core/
│           ├── pipeline.py          ← Orchestrates all steps
│           ├── raster_parser.py     ← Image → wall/door/window detection
│           ├── wall_detector.py     ← Segments → Wall objects
│           ├── opening_detector.py  ← DXF door/window detection
│           ├── room_detector.py     ← Room polygon detection
│           └── geometry_builder.py  ← Walls → Three.js JSON
└── viewer/
    └── index_v2.html                ← Self-contained Three.js viewer
```

---

## Viewer Controls

| Action | Control |
|--------|---------|
| Orbit | Left drag |
| Zoom | Scroll wheel |
| Pan | Right drag |
| Views | Keys `1` `2` `3` `4` or PERSP / TOP / FRONT / SIDE buttons |
| Toggle solid | `S` or SOLID button |
| Toggle wireframe | `W` or WIRE button |
| Toggle floor | `F` or FLOOR button |
| Toggle doors/windows | DOORS / WIN buttons |
| Process | `Space` |

---

## Modes

### ◈ Blueprint Mode (default)
Dark navy walls, cyan wireframes, dark grid background. The classic technical drawing look.

### ◉ Realistic Mode
Neutral white lighting, procedural textures on walls and floor. Toggle using the right panel.
An amber **REALISTIC** badge appears on the canvas when active.

---

## Material System (right panel)

Open/close the right panel with the `◀ ▶` button on its left edge.

**Wall Finish — 6 built-in textures:**
| Texture | Description |
|---------|-------------|
| PLASTER | Warm grey matte with surface variation |
| BRICK | Red/orange coursed brick with mortar lines |
| CONCRETE | Dark grey with form-board lines and aggregate |
| WOOD | Vertical timber panels with grain |
| MARBLE | Light stone with veining |
| W.TILE | White ceramic grid tiles |

**Floor Finish — 6 built-in textures:**
Tile · Parquet · Marble · Concrete · Stone · Carpet

**Scope:**
- `ALL WALLS` — applies chosen finish to every wall at once
- `SELECT` — click walls in the viewport to select them (orange outline), then apply to selected only

**Custom colour:** Colour picker for any solid wall colour.

**Tile size:** Controls how large each texture tile is (0.3m–4m).

All textures are procedurally generated — no external image files needed.

---

## API Reference

Base URL: `http://localhost:8000`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check — returns `{"status": "ok"}` |
| `/upload` | POST | Upload floor plan file, returns `job_id` |
| `/process/{job_id}` | GET | Process uploaded file, returns 3D model JSON |

**Process query params:**
```
scale=1.0              # unit scale (0 = auto-detect)
wall_height=3.0        # wall height in metres
wall_thickness=0.2     # fallback wall thickness
pixels_per_meter=0     # MUST be 0 — triggers auto-detection from image
```

---

## Known Limitations

| Issue | Severity | Status |
|-------|----------|--------|
| Window symbols only detected on vertical walls | Low | Open |
| Thin wall door detection (SCAN_HALF too wide) | Medium | Open |
| Phantom doors from tick marks near edges | Medium | Open |
| Wall thickness measurement accuracy | High | Open |
| Corner gap edges between perpendicular walls | High | Partial |

---

## Documentation Files

| File | Purpose |
|------|---------|
| `CHANGELOG.md` | Full history of every change by session |
| `LEARNMAP.md` | Hard-won lessons — what works, what broke, and why |
| `PROJECTMAP.md` | Full map of every file, function, data flow, and open problems |

---

## Troubleshooting

**API OFFLINE / not connecting**
1. Make sure uvicorn is running: `cd backend && uvicorn app.main:app --reload --port 8000`
2. Make sure you opened the viewer via HTTP (`http://localhost:3000`), not as `file://`
3. Open browser console and run: `fetch('http://localhost:8000/health').then(r=>r.json()).then(console.log).catch(console.error)`

**Walls look wrong / missing**
- Check the log panel in the viewer for PPM detection output
- Try uploading the raw floor plan PNG, not a screenshot of the 3D viewer

**Textures look too bright**
- Switch to Blueprint mode and back to Realistic to re-apply defaults
- Use the RESET ALL button and reapply your preferred texture

**Floor plan processes but shows no model**
- Check browser console for JS errors
- Verify the backend logs in the terminal for Python tracebacks
