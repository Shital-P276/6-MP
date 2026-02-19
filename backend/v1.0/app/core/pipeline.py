"""
Processing Pipeline — orchestrates the full DXF → 3D conversion.

Usage:
    pipeline = ProcessingPipeline(scale=1.0)
    result = pipeline.run("path/to/plan.dxf")
    print(result.model.to_json())
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import time

from .dxf_parser import DXFParser, ParsedGeometry
from .wall_detector import WallDetector, Wall
from .geometry_builder import GeometryBuilder, BuildingModel


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    success: bool
    model: BuildingModel | None = None
    geometry: ParsedGeometry | None = None
    walls: list[Wall] | None = None
    processing_time_ms: float = 0.0
    error: str | None = None
    warnings: list[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

    def to_dict(self) -> dict:
        if not self.success:
            return {"success": False, "error": self.error}

        return {
            "success": True,
            "processing_time_ms": round(self.processing_time_ms, 1),
            "warnings": self.warnings,
            "model": self.model.to_dict() if self.model else None,
            "stats": {
                "segments_parsed": (
                    len(self.geometry.wall_segments) +
                    len(self.geometry.door_segments) +
                    len(self.geometry.window_segments) +
                    len(self.geometry.other_segments)
                ) if self.geometry else 0,
                "wall_segments": len(self.geometry.wall_segments) if self.geometry else 0,
                "walls_detected": len(self.walls) if self.walls else 0,
            }
        }


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class ProcessingPipeline:
    def __init__(
        self,
        scale: float = 1.0,
        wall_height: float = 3.0,
        wall_thickness: float = 0.2,
    ):
        """
        Args:
            scale: unit conversion factor (e.g. 0.001 if DXF is in mm → meters)
            wall_height: default wall height in meters
            wall_thickness: default wall thickness in meters
        """
        self.scale = scale
        self.wall_height = wall_height
        self.wall_thickness = wall_thickness

    def run(self, filepath: str) -> PipelineResult:
        start = time.perf_counter()
        warnings = []

        try:
            path = Path(filepath)
            if not path.exists():
                return PipelineResult(success=False, error=f"File not found: {filepath}")

            suffix = path.suffix.lower()
            if suffix != ".dxf":
                return PipelineResult(success=False, error=f"Unsupported format: {suffix}. Only DXF supported in MVP.")

            # ── Step 1: Parse ────────────────────────────────────────────────
            parser = DXFParser(filepath)
            geometry = parser.parse()

            if not geometry.wall_segments and not geometry.other_segments:
                warnings.append("No geometry found. File may be empty or use unsupported entity types.")

            if not geometry.wall_segments:
                # Fall back: treat ALL segments as walls if no WALL layer found
                warnings.append(
                    "No WALL layer found. Treating all segments as walls. "
                    "Consider naming your layer 'WALL' for better results."
                )
                geometry.wall_segments = (
                    geometry.door_segments +
                    geometry.window_segments +
                    geometry.other_segments
                )
                geometry.door_segments = []
                geometry.window_segments = []
                geometry.other_segments = []

            # ── Step 2: Detect walls ─────────────────────────────────────────
            detector = WallDetector(
                scale=self.scale,
                default_thickness=self.wall_thickness,
                default_height=self.wall_height,
            )
            walls = detector.detect(geometry)

            if not walls:
                warnings.append("Wall detection produced no results. The geometry may be too complex or use unsupported patterns.")

            # ── Step 3: Build 3D model ───────────────────────────────────────
            builder = GeometryBuilder()
            model = builder.build(walls, bounds=geometry.bounds)

            elapsed = (time.perf_counter() - start) * 1000

            return PipelineResult(
                success=True,
                model=model,
                geometry=geometry,
                walls=walls,
                processing_time_ms=elapsed,
                warnings=warnings,
            )

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return PipelineResult(
                success=False,
                error=str(e),
                processing_time_ms=elapsed,
                warnings=warnings,
            )
