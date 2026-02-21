# Floorplan 3D Project — Recommended Next Steps

## Where the project is right now

The backend already supports:
- Uploading DXF, image, and PDF files.
- Converting plans into walls/floors/rooms/doors/windows JSON for Three.js.
- Endpoints for upload/process/model retrieval.

That means the core pipeline exists and works as a strong prototype.

## Immediate priorities (next 1–2 weeks)

1. **Stabilize test suite first (highest priority).**
   - Current tests reference old function names/imports and are out of sync with the core modules.
   - Fixing this first gives us confidence for every next change.

2. **Fix raster-specific wall detection flow.**
   - `WallDetector` has raster-specific behavior, but the pipeline currently creates it without `is_raster=True` for raster/PDF inputs.
   - That likely reduces quality on noisy image floorplans.

3. **Define a versioned API/JSON contract.**
   - Freeze and document response schema (`model`, `metadata`, `stats`, warnings), then add contract tests.

4. **Add job persistence.**
   - Current in-memory `jobs` means process state is lost on restart.
   - Move job metadata into SQLite/Postgres and keep files in durable storage.

## Product roadmap (phased)

### Phase A — Reliability & quality gates
- Bring tests to green: parser, wall detection, rooms, openings, API integration.
- Add fixture corpus (clean CAD DXF, messy DXF, scanned plan PNG/PDF).
- Add baseline metrics: wall precision/recall proxy, room count match, runtime.

### Phase B — Better geometry accuracy
- Improve auto-scale with explicit use of DXF units + optional calibration reference.
- Refine room detection to reduce leakage through tiny gaps.
- Improve opening detection (door swing and window attachment confidence).

### Phase C — Viewer integration
- Build a minimal Three.js viewer consuming current JSON format.
- Add toggles for walls/rooms/openings and confidence overlays.
- Add a debug mode to inspect detected source segments and wall pairing.

### Phase D — Export and interoperability
- Add GLTF export endpoint.
- Add optional IFC-style metadata mapping per room/opening.
- Support a downloadable package (JSON + textures + glTF).

## Technical debt to address now

- Replace wildcard CORS with environment-driven allowed origins.
- Add input validation/rate limiting for upload/process endpoints.
- Add structured logging + request IDs for pipeline tracing.
- Add proper error classes (parse error vs detection error vs system error).

## Suggested sprint breakdown

### Sprint 1 (stability)
- Repair and modernize tests.
- Add CI pipeline (pytest + lint).
- Fix raster detector wiring.

### Sprint 2 (quality)
- Add evaluation fixtures and quality dashboard script.
- Tune room/opening heuristics against fixtures.

### Sprint 3 (user value)
- Ship first viewer + model inspection UI.
- Add persistent jobs and model history.

## Definition of done for “v1 usable”

- API handles DXF + raster plans reliably on a representative fixture set.
- Tests pass in CI with meaningful coverage for core pipeline.
- User can upload file, process, inspect in viewer, and download model.
- Basic persistence + operational logs are in place.
