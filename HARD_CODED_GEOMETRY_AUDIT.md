# Hardcoded Geometry Audit

This audit lists places where wall coordinates, room sizes, and manual opening geometry are hardcoded.

| File | Lines | Hardcoded item | Suggestion for automatic detection |
|---|---:|---|---|
| `sample_data/generate_sample.py` | 47-56 | Fixed wall coordinates for outer shell and interior partitions (`(0,0)->(10,0)`, `(6,0)->(6,5)`, etc.). | Replace with wall extraction from uploaded plan via `RasterParser.parse()` / `DXFParser.parse()` and infer topology from detected segments instead of scripted coordinates. |
| `sample_data/generate_sample.py` | 63-82 | Manually placed doors/windows with explicit coordinates and widths (`(2,0)->(2.9,0)`, `(4,0)->(5.5,0)`, etc.). | Detect openings from symbols/layers and project them onto nearest walls (`OpeningDetector.detect()`), rather than fixed coordinates in generation scripts. |
| `sample_data/generate.py` | 60-69, 72-76 | Hardcoded room size ranges per room type and default fallback size tuple. | Learn room scales from detected wall graph + constraints (or infer from annotations) and move these values to configurable model metadata instead of constants. |
| `sample_data/generate.py` | 130, 208-218 | Fixed hallway width (`1.2m`) and open-plan dimensions (`6-10m` x `4-7m`), plus forced room placement at `x=0,y=0`. | Compute layout from detected boundaries/adjacencies. For synthetic generation, parameterize in config and/or sample from dataset statistics. |
| `sample_data/generate.py` | 282-285 | Room rectangles converted directly to wall edges (`(rx,ry)->(rx+rw,ry)`, etc.): manual room polygons. | Replace rectangle assumptions with polygonization over detected wall segments, including non-rectangular rooms. |
| `sample_data/generate.py` | 311-349 | Manual annotation geometry offsets and opening placement (`+0.05` inset, `door_w=0.9`, `door_x=rx+rw*0.3`, `win_w=min(1.2,rw*0.4)`). | Use symbol detection/arc fitting for doors and detected window spans, then serialize with confidence from measured geometry. |
| `sample_data/generate_floorplan_gui.py` | 60-69, 72-76 | Same hardcoded room size ranges and defaults as generator variant. | Externalize to config + derive from detected plans (or training priors) to avoid static assumptions. |
| `sample_data/generate_floorplan_gui.py` | 130, 208-220 | Fixed hallway width/open-plan dimensions and deterministic placement anchors. | Infer corridor widths from wall spacing and detect large open zones from segmentation. |
| `sample_data/generate_floorplan_gui.py` | 269-272 | Walls drawn from room rectangle corners (manual geometry). | Build walls from detected segment network; maintain graph nodes/intersections instead of rectangle expansion. |
| `sample_data/generate_floorplan_gui.py` | 302-320 | Door/window placement heuristics with fixed width and ratio-based positions. | Use detected door arcs/window glyphs and nearest-wall projection from parsed data. |
| `sample_data/generate_v2.py` | 60-69, 72-76 | Hardcoded room size ranges/defaults. | Same: infer or parameterize from data, not fixed tuples in code. |
| `sample_data/generate_v2.py` | 130, 208-220 | Hardcoded hallway/open-plan dimensions and anchoring. | Replace with boundary-driven placement extracted from detected geometry. |
| `sample_data/generate_v2.py` | 282-285, 306-308 | Manual room polygons/outline offsets. | Use contour/polygon extraction from walls; avoid synthetic rectangle-only assumptions. |
| `sample_data/generate_v2.py` | 331-344 | Fixed door width, arc sweep, and window-size formula/position. | Detect openings from geometry and classify with confidence; parameterize fallback only when detection fails. |
| `backend/tests/test_pipeline.py` | 43-50 | Test fixture builds fixed 5x4 room with explicit wall coordinates and offsets. | Keep for deterministic tests, but add property-based/randomized fixtures generated from parsed real plans to validate detection robustness. |
| `backend/app/core/geometry_builder.py` | 21-23 | Hardcoded architectural constants (`SILL_HEIGHT=0.9`, `WIN_HEIGHT=1.2`, `DOOR_LEAF_T=0.05`). | Read opening height/depth from detected metadata or file annotations; keep constants as fallback defaults only. |
| `backend/app/core/opening_detector.py` | 27-32 | Hardcoded opening thresholds and distance limits (`MAX_WALL_DIST`, min/max door/window widths). | Make thresholds adaptive to inferred scale/wall thickness and tune via data-driven calibration. |
| `backend/app/core/room_detector.py` | 23-37 | Hardcoded grid resolution/padding/min-cell thresholds and room-type area/aspect heuristics. | Use adaptive resolution based on drawing scale and replace rule-based typing with learned or probabilistic classifier. |

## Notes
- Most **fixed coordinates/polygons/openings** are in `sample_data/*` generators.
- Runtime backend uses detection for real uploads, but still includes hardcoded fallback geometry thresholds and default dimensions.
