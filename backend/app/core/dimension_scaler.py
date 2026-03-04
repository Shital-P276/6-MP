"""Dimension scaling stage for parsed geometry."""

from __future__ import annotations

from .dxf_parser import ParsedGeometry, Segment, Point2D
from .wall_detector import infer_scale


class DimensionScaler:
    def __init__(self, scale: float = 1.0, auto_scale: bool = True):
        self.scale = scale
        self.auto_scale = auto_scale
        self.applied_scale = scale

    def fit(self, geometry: ParsedGeometry) -> float:
        if self.auto_scale and self.scale == 1.0:
            self.applied_scale = infer_scale(geometry.bounds)
        else:
            self.applied_scale = self.scale
        return self.applied_scale

    def apply(self, geometry: ParsedGeometry) -> ParsedGeometry:
        s = self.fit(geometry)
        if s == 1.0:
            return geometry

        def _scale_segments(segments: list[Segment]) -> list[Segment]:
            return [
                Segment(
                    start=Point2D(seg.start.x * s, seg.start.y * s),
                    end=Point2D(seg.end.x * s, seg.end.y * s),
                    layer=seg.layer,
                    source_type=seg.source_type,
                )
                for seg in segments
            ]

        geometry.wall_segments = _scale_segments(geometry.wall_segments)
        geometry.door_segments = _scale_segments(geometry.door_segments)
        geometry.window_segments = _scale_segments(geometry.window_segments)
        geometry.other_segments = _scale_segments(geometry.other_segments)

        if geometry.bounds:
            geometry.bounds = {k: v * s for k, v in geometry.bounds.items()}

        if geometry.text_labels:
            geometry.text_labels = [(x * s, y * s, t) for x, y, t in geometry.text_labels]

        return geometry
