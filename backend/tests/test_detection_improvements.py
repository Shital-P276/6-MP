import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from app.core.dxf_parser import classify_layer, Point2D
from app.core.wall_detector import Wall
from app.core.opening_detector import OpeningDetector


def test_classify_dimension_layers_as_ignored():
    assert classify_layer("A-DIMS") == "IGNORE"
    assert classify_layer("dimension_text") == "IGNORE"
    assert classify_layer("MEASUREMENTS") == "IGNORE"


def test_opening_wall_match_is_not_over_permissive():
    walls = [
        Wall(start=Point2D(0, 0), end=Point2D(5, 0), thickness=0.2),
    ]
    detector = OpeningDetector()
    # too far from wall to be a plausible opening
    assert detector._find_wall(2.5, 1.0, walls) is None
    # close enough should still match
    assert detector._find_wall(2.5, 0.2, walls) is not None
