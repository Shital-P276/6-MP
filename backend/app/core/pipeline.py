"""Processing Pipeline — DXF / Image / PDF → 3D JSON."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import math, time

from .dxf_parser import DXFParser, ParsedGeometry, Segment, Point2D
from .wall_detector import WallDetector, Wall
from .geometry_builder import GeometryBuilder, BuildingModel
from .room_detector import RoomDetector
from .opening_detector import OpeningDetector, Opening

try:
    from .raster_parser import RasterParser
    RASTER_OK = True
except ImportError:
    RASTER_OK = False

RASTER_FORMATS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VECTOR_FORMATS  = {".dxf"}
PDF_FORMATS     = {".pdf"}
ALL_FORMATS     = RASTER_FORMATS | VECTOR_FORMATS | PDF_FORMATS


@dataclass
class PipelineResult:
    success: bool
    model:   BuildingModel | None = None
    geometry: ParsedGeometry | None = None
    walls:   list[Wall] | None = None
    processing_time_ms: float = 0.0
    error:   str | None = None
    warnings: list[str] = field(default_factory=list)
    source_type:   str = "unknown"
    applied_scale: float = 1.0

    def to_dict(self) -> dict:
        if not self.success:
            return {"success": False, "error": self.error, "warnings": self.warnings}
        return {
            "success": True,
            "processing_time_ms": round(self.processing_time_ms, 1),
            "warnings":    self.warnings,
            "source_type": self.source_type,
            "applied_scale": self.applied_scale,
            "model": self.model.to_dict() if self.model else None,
            "stats": {
                "wall_segments":    len(self.geometry.wall_segments) if self.geometry else 0,
                "walls_detected":   len(self.walls) if self.walls else 0,
                "paired_walls":     sum(1 for w in self.walls if w.paired) if self.walls else 0,
                "rooms_detected":   self.model.metadata.get("room_count", 0) if self.model else 0,
                "doors_detected":   self.model.metadata.get("door_count", 0) if self.model else 0,
                "windows_detected": self.model.metadata.get("window_count", 0) if self.model else 0,
            },
        }


class ProcessingPipeline:
    def __init__(
        self,
        scale: float = 1.0,
        auto_scale: bool = True,
        wall_height: float = 3.0,
        wall_thickness: float = 0.2,
        pixels_per_meter: float = 0.0,   # 0 = auto-detect from image
        pdf_dpi: int = 200,
        # Legacy Hough params (ignored in raster v4, kept for API compat)
        hough_threshold: int = 50,
        hough_min_length: int = 30,
        hough_max_gap: int = 15,
    ):
        self.scale            = scale
        self.auto_scale       = auto_scale
        self.wall_height      = wall_height
        self.wall_thickness   = wall_thickness
        self.pixels_per_meter = pixels_per_meter
        self.pdf_dpi          = pdf_dpi

    def run(self, filepath: str) -> PipelineResult:
        t0 = time.perf_counter()
        warnings: list[str] = []

        try:
            path = Path(filepath)
            if not path.exists():
                return PipelineResult(success=False,
                                      error=f"File not found: {filepath}")

            suffix = path.suffix.lower()
            if suffix not in ALL_FORMATS:
                return PipelineResult(success=False,
                    error=f"Unsupported format '{suffix}'. "
                          f"Supported: {sorted(ALL_FORMATS)}")

            # ── Parse ─────────────────────────────────────────────────────────
            is_raster = suffix not in VECTOR_FORMATS
            raster_parser = None   # saved so we can pull metadata later

            if suffix in VECTOR_FORMATS:
                source_type = "dxf"
                geometry = self._parse_dxf(filepath, warnings)
            else:
                source_type = "pdf" if suffix in PDF_FORMATS else "raster"
                geometry, raster_parser = self._parse_raster(filepath, warnings)

            if geometry is None:
                return PipelineResult(
                    success=False, error="Parsing failed", warnings=warnings,
                    processing_time_ms=(time.perf_counter()-t0)*1000)

            # ── Fallback: no WALL layer ────────────────────────────────────────
            if not geometry.wall_segments:
                all_other = (geometry.door_segments + geometry.window_segments
                             + geometry.other_segments)
                if all_other:
                    warnings.append(
                        "No WALL layer found — treating all segments as walls.")
                    geometry.wall_segments = all_other
                    geometry.door_segments = []
                    geometry.window_segments = []
                    geometry.other_segments = []
                else:
                    warnings.append("No geometry detected in file.")

            # ── Detect walls ──────────────────────────────────────────────────
            # For raster: use wall thickness extracted from pixel measurements
            # (wall_thickness from UI is used as fallback only)
            detected_wall_thick = self.wall_thickness
            if raster_parser is not None:
                meta = geometry.metadata_extra
                ppm  = meta.get("pixels_per_meter", 65)
                # Wall thickness from image: we measured ~20px at 65ppm ≈ 0.31m
                # Use a reasonable architectural default: 0.25m exterior, 0.1m interior
                # For raster we use the ppm-derived thickness
                detected_wall_thick = round(20.0 / max(ppm, 1), 3)
                detected_wall_thick = max(0.1, min(0.5, detected_wall_thick))

            detector = WallDetector(
                scale=self.scale,
                auto_scale=self.auto_scale if not is_raster else False,
                default_thickness=detected_wall_thick,
                default_height=self.wall_height,
                is_raster=is_raster,
            )
            walls = detector.detect(geometry)
            applied_scale = detector.applied_scale

            if applied_scale != 1.0 and self.auto_scale and self.scale == 1.0:
                warnings.append(
                    f"Auto-scale applied: 1 unit = {applied_scale}m. "
                    f"Override with ?scale=X if incorrect.")

            if not walls:
                warnings.append("Wall detection produced no results.")

            # ── Detect rooms ──────────────────────────────────────────────────
            text_labels = getattr(geometry, 'text_labels', None)
            rooms = RoomDetector().detect(geometry.wall_segments,
                                          text_labels=text_labels)

            # ── Detect openings ───────────────────────────────────────────────
            if is_raster and hasattr(geometry, '_raster_openings'):
                # Use openings extracted directly from the image pixels —
                # much more accurate than the DXF-oriented arc detector
                openings = self._build_raster_openings(
                    geometry._raster_openings, walls,
                    geometry._fp_h, geometry._ppm)
            else:
                openings = OpeningDetector().detect(geometry, walls)

            # ── Build 3D model ────────────────────────────────────────────────
            model = GeometryBuilder().build(
                walls, bounds=geometry.bounds,
                rooms=rooms, openings=openings,
                wall_height=self.wall_height,
            )

            return PipelineResult(
                success=True,
                model=model,
                geometry=geometry,
                walls=walls,
                processing_time_ms=(time.perf_counter() - t0) * 1000,
                warnings=warnings,
                source_type=source_type,
                applied_scale=applied_scale,
            )

        except Exception as e:
            import traceback
            return PipelineResult(
                success=False, error=str(e) + "\n" + traceback.format_exc(),
                warnings=warnings,
                processing_time_ms=(time.perf_counter() - t0) * 1000,
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _parse_dxf(self, filepath, warnings):
        try:
            return DXFParser(filepath).parse()
        except Exception as e:
            warnings.append(f"DXF parse error: {e}")
            return None

    def _parse_raster(self, filepath, warnings):
        if not RASTER_OK:
            warnings.append(
                "opencv-python not installed — raster/PDF parsing unavailable.")
            return None, None
        try:
            parser = RasterParser(
                pixels_per_meter=self.pixels_per_meter,
                pdf_dpi=self.pdf_dpi,
            )
            geo = parser.parse(filepath)
            meta = geo.metadata_extra
            ppm  = meta.get("pixels_per_meter", 65)
            warnings.append(
                f"Raster v5: {meta['total_walls']} walls, "
                f"{meta['doors_detected']} doors, "
                f"{meta['windows_detected']} windows, "
                f"{meta['rooms_detected']} rooms detected "
                f"at {ppm:.0f} px/m ({meta['ppm_source']}). "
                f"Cropped from {meta['original_size']} → {meta['cropped_size']}."
            )
            return geo, parser
        except Exception as e:
            import traceback
            warnings.append(f"Raster parse error: {e}\n{traceback.format_exc()}")
            return None, None

    def _build_raster_openings(self, img_openings, walls, fp_h, ppm):
        """
        Convert pixel-space opening dicts from the raster parser into Opening objects.

        Normal (intra-segment) openings: matched to nearest wall, width clipped.
        Inter-segment openings (t_center == -1.0): placed between two collinear wall
          stubs. These are returned with wall_idx=-1 so geometry_builder creates a
          freestanding door/window dict from world coordinates directly.
        """
        from .opening_detector import Opening, _project

        openings = []
        world_h  = fp_h / ppm

        for op in img_openings:
            ox = op["x"]
            oy = world_h - op["y"]   # Y-flip to match wall coords

            is_inter = op.get("t_center", 0) == -1.0

            if is_inter:
                # ── Inter-segment: freestanding opening between two collinear stubs ──
                # Find the wall angle (use any collinear wall nearby)
                orient = op.get("orient", "H")
                tol    = 0.20   # metres
                angle  = 0.0
                for wall in walls:
                    on_line = (
                        (orient == 'H' and abs(wall.start.y - oy) < tol)
                        or (orient == 'V' and abs(wall.start.x - ox) < tol)
                    )
                    if on_line:
                        angle = math.atan2(wall.end.y - wall.start.y,
                                           wall.end.x - wall.start.x)
                        break

                openings.append(Opening(
                    wall_idx   = -1,          # sentinel: freestanding
                    t_center   = -1.0,
                    width      = op["width_m"],
                    kind       = op["kind"],
                    x=ox, y=oy,
                    angle      = angle,
                    swing_side = op.get("swing_side", "right"),
                ))

            else:
                # ── Normal intra-segment opening ──────────────────────────────
                best_wi, best_t, best_dist = None, 0.0, 1.5

                for wi, wall in enumerate(walls):
                    t, _, _, dist = _project(
                        ox, oy,
                        wall.start.x, wall.start.y,
                        wall.end.x,   wall.end.y,
                    )
                    if dist < best_dist:
                        best_dist = dist
                        best_t    = t
                        best_wi   = wi

                if best_wi is None:
                    continue

                wall = walls[best_wi]
                openings.append(Opening(
                    wall_idx   = best_wi,
                    t_center   = best_t,
                    width      = min(op["width_m"], wall.length * 0.92),
                    kind       = op["kind"],
                    x=ox, y=oy,
                    angle=math.atan2(
                        wall.end.y - wall.start.y,
                        wall.end.x - wall.start.x,
                    ),
                    swing_side = op.get("swing_side", "right"),
                ))

        return openings
