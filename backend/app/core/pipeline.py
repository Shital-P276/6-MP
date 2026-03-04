"""Processing Pipeline — image/vector input to floorplan model and renderer JSON.

Architecture stages:
image -> preprocessing -> wall detection -> wall graph generation ->
door detection -> window detection -> room extraction -> dimension scaling ->
floorplan data model -> renderer
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import time

from .dxf_parser import DXFParser, ParsedGeometry
from .wall_detector import WallDetector, Wall
from .geometry_builder import GeometryBuilder, BuildingModel
from .room_detector import RoomDetector
from .room_graph_detector import GraphRoomDetector
from .opening_detector import OpeningDetector
from .door_detector import DoorDetector, Door
from .window_detector import WindowDetector, Window
from .dimension_scaler import DimensionScaler
from .dimension_detector import DimensionDetector
from .wall_graph import generate_wall_graph, WallGraph
from .floorplan_model import FloorPlan

try:
    from .raster_parser import RasterParser
    RASTER_OK = True
except ImportError:
    RASTER_OK = False

RASTER_FORMATS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VECTOR_FORMATS = {".dxf"}
PDF_FORMATS = {".pdf"}
ALL_FORMATS = RASTER_FORMATS | VECTOR_FORMATS | PDF_FORMATS


@dataclass
class PipelineResult:
    success: bool
    model: BuildingModel | None = None
    geometry: ParsedGeometry | None = None
    walls: list[Wall] | None = None
    wall_graph: WallGraph | None = None
    floorplan: FloorPlan | None = None
    processing_time_ms: float = 0.0
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    source_type: str = "unknown"
    applied_scale: float = 1.0

    def to_dict(self) -> dict:
        if not self.success:
            return {"success": False, "error": self.error, "warnings": self.warnings}
        return {
            "success": True,
            "processing_time_ms": round(self.processing_time_ms, 1),
            "warnings": self.warnings,
            "source_type": self.source_type,
            "applied_scale": self.applied_scale,
            "model": self.model.to_dict() if self.model else None,
            "floorplan": self.floorplan.to_dict() if self.floorplan else None,
            "stats": {
                "wall_segments": len(self.geometry.wall_segments) if self.geometry else 0,
                "walls_detected": len(self.walls) if self.walls else 0,
                "graph_nodes": len(self.wall_graph.nodes) if self.wall_graph else 0,
                "graph_edges": len(self.wall_graph.edges) if self.wall_graph else 0,
                "rooms_detected": self.model.metadata.get("room_count", 0) if self.model else 0,
                "doors_detected": self.model.metadata.get("door_count", 0) if self.model else 0,
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
        pixels_per_meter: float = 100.0,
        pdf_dpi: int = 200,
        hough_threshold: int = 50,
        hough_min_length: int = 30,
        hough_max_gap: int = 15,
    ):
        self.scale = scale
        self.auto_scale = auto_scale
        self.wall_height = wall_height
        self.wall_thickness = wall_thickness
        self.pixels_per_meter = pixels_per_meter
        self.pdf_dpi = pdf_dpi
        self.hough_threshold = hough_threshold
        self.hough_min_length = hough_min_length
        self.hough_max_gap = hough_max_gap

    def run(self, filepath: str) -> PipelineResult:
        t0 = time.perf_counter()
        warnings: list[str] = []

        try:
            path = Path(filepath)
            if not path.exists():
                return PipelineResult(success=False, error=f"File not found: {filepath}")

            suffix = path.suffix.lower()
            if suffix not in ALL_FORMATS:
                return PipelineResult(success=False,
                                      error=f"Unsupported format '{suffix}'. Supported: {sorted(ALL_FORMATS)}")

            # 1) image/vector input + preprocessing
            if suffix in VECTOR_FORMATS:
                source_type = "dxf"
                geometry = self._parse_dxf(filepath, warnings)
            else:
                source_type = "pdf" if suffix in PDF_FORMATS else "raster"
                geometry = self._parse_raster(filepath, warnings)

            if geometry is None:
                return PipelineResult(success=False, error="Parsing failed",
                                      warnings=warnings,
                                      processing_time_ms=(time.perf_counter() - t0) * 1000)

            # Layer fallback
            geometry = self._fallback_wall_segments(geometry, warnings)

            # 2) dimension detection + scaling
            detected_scale = None
            scale_source = "heuristic"
            scale_evidence = None
            if source_type in {"raster", "pdf"} and self.auto_scale and self.scale == 1.0:
                dim_detector = DimensionDetector()
                detected_scale, scale_evidence = dim_detector.detect_scale(filepath)
                if detected_scale:
                    scale_source = "ocr-dimension"
                    warnings.append(f"Dimension OCR scale applied: 1 px = {detected_scale:.6f}m")
                elif scale_evidence and scale_evidence.get("status") not in {"ok"}:
                    warnings.append(f"Dimension OCR unavailable: {scale_evidence.get('status')}")

            scaler = DimensionScaler(
                scale=detected_scale if detected_scale is not None else self.scale,
                auto_scale=self.auto_scale if detected_scale is None else False,
            )
            geometry = scaler.apply(geometry)
            applied_scale = scaler.applied_scale
            if detected_scale is None and applied_scale != 1.0 and self.auto_scale and self.scale == 1.0:
                warnings.append(
                    f"Auto-scale applied: 1 unit = {applied_scale}m. "
                    f"Override with ?scale=X if incorrect."
                )

            # 3) initialize shared FloorPlan structure
            floorplan = FloorPlan(
                source_type=source_type,
                scale=applied_scale,
                scale_source=scale_source,
                scale_evidence=scale_evidence,
                bounds=geometry.bounds,
            )

            # 4) wall detection -> floorplan.walls
            wall_detector = WallDetector(
                scale=1.0,
                auto_scale=False,
                default_thickness=self.wall_thickness,
                default_height=self.wall_height,
                is_raster=(source_type in {"raster", "pdf"}),
            )
            wall_detector.detect_into(
                floorplan,
                geometry,
                image_path=filepath if source_type in {"raster", "pdf"} else None,
            )
            if not floorplan.walls:
                warnings.append(
                    "Wall detection produced no results. "
                    "The file may have single-line walls — try uploading as-is, "
                    "they'll be detected with default thickness."
                )

            # 5) wall graph generation -> floorplan.wall_graph
            floorplan.wall_graph = generate_wall_graph(floorplan.walls)

            # 6) door detection -> floorplan.doors
            opening_detector = OpeningDetector()
            if source_type in {"raster", "pdf"}:
                DoorDetector(pixels_per_meter=self.pixels_per_meter).detect_into(
                    floorplan,
                    image_path=filepath,
                )
            else:
                floorplan.doors = [Door.from_opening(o) for o in opening_detector.detect_doors(geometry, floorplan.walls)]

            # 7) window detection -> floorplan.windows
            if source_type in {"raster", "pdf"}:
                WindowDetector(pixels_per_meter=self.pixels_per_meter).detect_into(
                    floorplan,
                    geometry=geometry,
                    image_path=filepath,
                )
            else:
                floorplan.windows = []
                for op in opening_detector.detect_windows(geometry, floorplan.walls):
                    w = Window.from_opening(op, floorplan.walls)
                    if w is not None:
                        floorplan.windows.append(w)

            # 8) room extraction from wall graph cycles -> floorplan.rooms
            GraphRoomDetector().detect_into(floorplan)
            if not floorplan.rooms:
                # fallback for sparse/noisy graphs
                text_labels = getattr(geometry, "text_labels", None)
                floorplan.rooms = RoomDetector().detect(geometry.wall_segments, text_labels=text_labels)

            # 9) renderer payload
            model = GeometryBuilder().build_from_floorplan(floorplan, wall_height=self.wall_height)

            return PipelineResult(
                success=True,
                model=model,
                geometry=geometry,
                walls=floorplan.walls,
                wall_graph=floorplan.wall_graph,
                floorplan=floorplan,
                processing_time_ms=(time.perf_counter() - t0) * 1000,
                warnings=warnings,
                source_type=source_type,
                applied_scale=applied_scale,
            )

        except Exception as e:
            return PipelineResult(
                success=False, error=str(e), warnings=warnings,
                processing_time_ms=(time.perf_counter() - t0) * 1000,
            )

    def _fallback_wall_segments(self, geometry: ParsedGeometry, warnings: list[str]) -> ParsedGeometry:
        if geometry.wall_segments:
            return geometry
        all_other = geometry.door_segments + geometry.window_segments + geometry.other_segments
        if all_other:
            warnings.append(
                "No WALL layer found — treating all segments as walls. "
                "Name your layer 'WALL' for better accuracy."
            )
            geometry.wall_segments = all_other
            geometry.door_segments = []
            geometry.window_segments = []
            geometry.other_segments = []
        else:
            warnings.append("No geometry detected in file.")
        return geometry

    def _parse_dxf(self, filepath, warnings):
        try:
            return DXFParser(filepath).parse()
        except Exception as e:
            warnings.append(f"DXF parse error: {e}")
            return None

    def _parse_raster(self, filepath, warnings):
        if not RASTER_OK:
            warnings.append("opencv-python not installed — raster/PDF parsing unavailable.")
            return None
        try:
            parser = RasterParser(
                pixels_per_meter=self.pixels_per_meter,
                pdf_dpi=self.pdf_dpi,
                hough_threshold=self.hough_threshold,
                hough_min_length=self.hough_min_length,
                hough_max_gap=self.hough_max_gap,
            )
            geo = parser.parse(filepath)
            if not geo.wall_segments:
                warnings.append(
                    "No lines detected in image. Try lowering hough_threshold or hough_min_length."
                )
            else:
                warnings.append(
                    f"Raster mode: {len(geo.wall_segments)} lines detected. "
                    "Set pixels_per_meter for accurate scale."
                )
            return geo
        except Exception as e:
            warnings.append(f"Raster parse error: {e}")
            return None
