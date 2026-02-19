"""
Tests for the Floor Plan Visualizer backend.

Run: pytest tests/ -v
(Requires ezdxf and shapely to be installed)
"""

import sys
import os
import math
import tempfile
import pytest

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.dxf_parser import DXFParser, ParsedGeometry, classify_layer, Segment, Point2D
from app.core.wall_detector import WallDetector, Wall, _are_parallel, _segment_angle
from app.core.geometry_builder import GeometryBuilder, wall_to_box
from app.core.pipeline import ProcessingPipeline


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_segment(x1, y1, x2, y2, layer="WALL") -> Segment:
    return Segment(start=Point2D(x1, y1), end=Point2D(x2, y2), layer=layer)


def make_wall(x1, y1, x2, y2, thickness=0.2, height=3.0) -> Wall:
    return Wall(start=Point2D(x1, y1), end=Point2D(x2, y2), thickness=thickness, height=height)


def create_simple_dxf(filepath: str):
    """Create a minimal DXF file for testing."""
    import ezdxf
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    doc.layers.add("WALL", color=7)
    wall_attr = {"layer": "WALL"}
    # Simple 5x4 room with double-line walls
    msp.add_line((0, 0), (5, 0), dxfattribs=wall_attr)
    msp.add_line((0, 0.2), (5, 0.2), dxfattribs=wall_attr)  # parallel pair
    msp.add_line((5, 0), (5, 4), dxfattribs=wall_attr)
    msp.add_line((5.2, 0), (5.2, 4), dxfattribs=wall_attr)
    msp.add_line((0, 4), (5, 4), dxfattribs=wall_attr)
    msp.add_line((0, 4.2), (5, 4.2), dxfattribs=wall_attr)
    msp.add_line((0, 0), (0, 4), dxfattribs=wall_attr)
    msp.add_line((-0.2, 0), (-0.2, 4), dxfattribs=wall_attr)
    doc.saveas(filepath)


# ─────────────────────────────────────────────────────────────────────────────
# Layer classification tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLayerClassification:
    def test_wall_layer(self):
        assert classify_layer("WALL") == "WALL"
        assert classify_layer("wall") == "WALL"
        assert classify_layer("A-WALL") == "WALL"
        assert classify_layer("WALLS_EXTERIOR") == "WALL"

    def test_door_layer(self):
        assert classify_layer("DOOR") == "DOOR"
        assert classify_layer("door") == "DOOR"
        assert classify_layer("DOORS") == "DOOR"

    def test_window_layer(self):
        assert classify_layer("WINDOW") == "WINDOW"
        assert classify_layer("WINDOWS") == "WINDOW"
        assert classify_layer("fenetre") == "WINDOW"

    def test_unknown_layer(self):
        assert classify_layer("FURNITURE") == "OTHER"
        assert classify_layer("0") == "OTHER"
        assert classify_layer("DIMENSIONS") == "OTHER"


# ─────────────────────────────────────────────────────────────────────────────
# Wall detector tests
# ─────────────────────────────────────────────────────────────────────────────

class TestWallDetector:
    def test_segment_angle_horizontal(self):
        seg = make_segment(0, 0, 5, 0)
        assert abs(_segment_angle(seg) - 0.0) < 1e-6

    def test_segment_angle_vertical(self):
        seg = make_segment(0, 0, 0, 5)
        assert abs(_segment_angle(seg) - 90.0) < 1e-6

    def test_parallel_segments_horizontal(self):
        a = make_segment(0, 0, 5, 0)
        b = make_segment(0, 0.2, 5, 0.2)
        assert _are_parallel(a, b)

    def test_non_parallel_segments(self):
        a = make_segment(0, 0, 5, 0)
        b = make_segment(0, 0, 0, 5)
        assert not _are_parallel(a, b)

    def test_wall_pairing(self):
        geo = ParsedGeometry()
        geo.wall_segments = [
            make_segment(0, 0, 5, 0),
            make_segment(0, 0.2, 5, 0.2),
        ]
        detector = WallDetector()
        walls = detector.detect(geo)
        assert len(walls) == 1
        wall = walls[0]
        assert abs(wall.thickness - 0.2) < 0.05
        assert abs(wall.length - 5.0) < 0.1

    def test_single_segment_fallback(self):
        """Unpaired segment should produce wall with default thickness."""
        geo = ParsedGeometry()
        geo.wall_segments = [make_segment(0, 0, 10, 0)]
        detector = WallDetector(default_thickness=0.2)
        walls = detector.detect(geo)
        assert len(walls) == 1
        assert walls[0].thickness == 0.2

    def test_wall_length(self):
        wall = make_wall(0, 0, 3, 4)  # 3-4-5 triangle
        assert abs(wall.length - 5.0) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Geometry builder tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGeometryBuilder:
    def test_wall_to_box_basic(self):
        wall = make_wall(0, 0, 5, 0, thickness=0.2, height=3.0)
        box = wall_to_box(wall)
        assert box is not None
        assert box["type"] == "box"
        assert abs(box["dimensions"]["width"] - 5.0) < 0.01
        assert abs(box["dimensions"]["height"] - 3.0) < 0.01
        assert abs(box["dimensions"]["depth"] - 0.2) < 0.01

    def test_wall_to_box_position(self):
        wall = make_wall(0, 0, 4, 0)
        box = wall_to_box(wall)
        assert abs(box["position"]["x"] - 2.0) < 0.01  # midpoint x
        assert abs(box["position"]["y"] - 1.5) < 0.01  # height/2

    def test_builder_output_structure(self):
        walls = [make_wall(0, 0, 5, 0), make_wall(5, 0, 5, 4)]
        builder = GeometryBuilder()
        model = builder.build(walls)
        assert "walls" in model.to_dict()
        assert "metadata" in model.to_dict()
        assert model.metadata["wall_count"] == 2

    def test_builder_with_bounds(self):
        walls = [make_wall(0, 0, 10, 0)]
        builder = GeometryBuilder()
        bounds = {"minx": 0, "miny": 0, "maxx": 10, "maxy": 8}
        model = builder.build(walls, bounds=bounds)
        assert len(model.floors) == 1
        floor = model.floors[0]
        assert floor["type"] == "floor"


# ─────────────────────────────────────────────────────────────────────────────
# Integration: pipeline on real DXF file
# ─────────────────────────────────────────────────────────────────────────────

class TestPipeline:
    def test_pipeline_missing_file(self):
        pipeline = ProcessingPipeline()
        result = pipeline.run("nonexistent.dxf")
        assert not result.success
        assert "not found" in result.error.lower()

    def test_pipeline_wrong_extension(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"fake")
            path = f.name
        pipeline = ProcessingPipeline()
        result = pipeline.run(path)
        assert not result.success
        os.unlink(path)

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("ezdxf"),
        reason="ezdxf not installed"
    )
    def test_pipeline_with_dxf(self, tmp_path):
        dxf_path = str(tmp_path / "test.dxf")
        create_simple_dxf(dxf_path)

        pipeline = ProcessingPipeline(scale=1.0)
        result = pipeline.run(dxf_path)

        assert result.success
        assert result.model is not None
        assert len(result.model.walls) > 0

        model_dict = result.model.to_dict()
        assert "walls" in model_dict
        assert model_dict["metadata"]["wall_count"] > 0

    def test_pipeline_result_dict_on_error(self):
        pipeline = ProcessingPipeline()
        result = pipeline.run("bad_file.dxf")
        d = result.to_dict()
        assert d["success"] is False
        assert "error" in d
