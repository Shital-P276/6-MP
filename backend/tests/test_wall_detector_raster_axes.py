import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.wall_detector import _filter_to_dominant_axes


def test_filter_to_dominant_axes_drops_diagonal_noise():
    lines = [
        (0, 0, 100, 0),
        (0, 10, 100, 10),
        (0, 0, 0, 80),
        (20, 0, 20, 80),
        (5, 5, 50, 45),
        (60, 10, 20, 55),
        (15, 50, 80, 20),
    ]

    filtered = _filter_to_dominant_axes(lines, tolerance_deg=10)

    assert len(filtered) == 4
    assert (5, 5, 50, 45) not in filtered
    assert (60, 10, 20, 55) not in filtered
    assert (15, 50, 80, 20) not in filtered


def test_filter_to_dominant_axes_fallback_when_too_aggressive():
    lines = [
        (0, 0, 100, 30),
        (0, 10, 100, 45),
        (0, 20, 100, 60),
        (0, 30, 100, 75),
        (0, 40, 100, 90),
    ]

    filtered = _filter_to_dominant_axes(lines, tolerance_deg=2)

    assert filtered == lines
