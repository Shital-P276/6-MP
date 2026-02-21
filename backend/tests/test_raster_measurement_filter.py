import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

cv2 = pytest.importorskip("cv2")
import numpy as np

from app.core.raster_parser import _preprocess, _hough_lines, _filter_lines_with_wall_mask


def test_measurement_like_thin_lines_are_filtered():
    img = np.full((220, 220, 3), 255, dtype=np.uint8)

    # thick structural wall (dark)
    cv2.line(img, (20, 120), (200, 120), (20, 20, 20), 10)
    # thin dimension line near border
    cv2.line(img, (10, 20), (210, 20), (0, 180, 0), 1)

    binary_lines, wall_mask = _preprocess(img)
    raw_lines = _hough_lines(binary_lines, threshold=25, min_length=40, max_gap=8)
    kept, filtered_out = _filter_lines_with_wall_mask(raw_lines, wall_mask)

    assert len(raw_lines) > 0
    assert filtered_out > 0
    # at least one strong structural line should survive
    assert len(kept) > 0
