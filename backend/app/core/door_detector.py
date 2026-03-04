"""Door detection from wall graph gaps + arc/circle evidence on edge maps."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math

from .dxf_parser import Point2D
from .opening_detector import Opening
from .wall_detector import Wall
from .wall_graph import WallGraph

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class Door:
    hinge_point: Point2D
    radius: float
    opening_angle: float
    wall_reference: int
    confidence: float = 0.5

    def to_dict(self) -> dict:
        return {
            "hinge_point": {"x": round(self.hinge_point.x, 4), "y": round(self.hinge_point.y, 4)},
            "radius": round(self.radius, 4),
            "opening_angle": round(self.opening_angle, 4),
            "wall_reference": self.wall_reference,
            "confidence": round(self.confidence, 3),
        }

    def to_opening(self, walls: list[Wall]) -> Opening | None:
        if self.wall_reference < 0 or self.wall_reference >= len(walls):
            return None
        wall = walls[self.wall_reference]
        L = wall.length
        if L < 1e-6:
            return None
        dx = wall.end.x - wall.start.x
        dy = wall.end.y - wall.start.y
        t = ((self.hinge_point.x - wall.start.x) * dx + (self.hinge_point.y - wall.start.y) * dy) / (L * L)
        t = max(0.0, min(1.0, t))
        return Opening(
            wall_idx=self.wall_reference,
            t_center=t,
            width=min(self.radius, L * 0.9),
            kind="door",
            x=self.hinge_point.x,
            y=self.hinge_point.y,
            angle=math.atan2(dy, dx),
        )

    @classmethod
    def from_opening(cls, opening: Opening) -> "Door":
        return cls(
            hinge_point=Point2D(opening.x, opening.y),
            radius=opening.width,
            opening_angle=math.pi / 2,
            wall_reference=opening.wall_idx,
            confidence=0.6,
        )


class DoorDetector:
    def __init__(
        self,
        pixels_per_meter: float = 100.0,
        min_gap_m: float = 0.6,
        max_gap_m: float = 1.5,
    ):
        self.pixels_per_meter = pixels_per_meter
        self.min_gap_m = min_gap_m
        self.max_gap_m = max_gap_m

    def detect(
        self,
        wall_graph: WallGraph,
        walls: list[Wall],
        image_path: str | None = None,
    ) -> list[Door]:
        candidates = self._detect_gaps_from_graph(wall_graph, walls)
        if not candidates:
            return []

        edge_map, gray = self._edge_map(image_path)
        doors = []
        for hinge, gap_len, wall_idx in candidates:
            radius = max(self.min_gap_m, min(gap_len, self.max_gap_m))
            angle = self._estimate_opening_angle(hinge, radius, wall_idx, walls)
            conf = 0.45

            if edge_map is not None and gray is not None:
                ok, detected_radius = self._arc_near_gap(edge_map, gray, hinge, radius)
                if ok:
                    radius = detected_radius
                    conf = 0.8

            doors.append(
                Door(
                    hinge_point=hinge,
                    radius=radius,
                    opening_angle=angle,
                    wall_reference=wall_idx,
                    confidence=conf,
                )
            )
        return doors

    def _detect_gaps_from_graph(self, wall_graph: WallGraph, walls: list[Wall]) -> list[tuple[Point2D, float, int]]:
        """Find candidate wall gaps from graph leaves that are near-collinear."""
        if not wall_graph or not wall_graph.nodes or not wall_graph.edges:
            return []

        leaves = [nid for nid, eids in wall_graph.adjacency.items() if len(eids) == 1]
        if len(leaves) < 2:
            return []

        candidates = []
        used_pairs = set()
        for i, n1 in enumerate(leaves):
            for n2 in leaves[i + 1:]:
                if (n1, n2) in used_pairs or (n2, n1) in used_pairs:
                    continue
                a = wall_graph.nodes[n1]
                b = wall_graph.nodes[n2]
                gap_len = math.hypot(b.x - a.x, b.y - a.y)
                if gap_len < self.min_gap_m or gap_len > self.max_gap_m:
                    continue

                e1 = wall_graph.edges[wall_graph.adjacency[n1][0]]
                e2 = wall_graph.edges[wall_graph.adjacency[n2][0]]
                if e1.wall_index >= len(walls) or e2.wall_index >= len(walls):
                    continue
                w1 = walls[e1.wall_index]
                w2 = walls[e2.wall_index]

                ang = self._angle_diff(w1.angle_deg, w2.angle_deg)
                if ang > 15.0:
                    continue

                # Reference wall: choose longer segment
                ref_idx = e1.wall_index if w1.length >= w2.length else e2.wall_index
                hinge = Point2D((a.x + b.x) / 2, (a.y + b.y) / 2)
                candidates.append((hinge, gap_len, ref_idx))
                used_pairs.add((n1, n2))

        return candidates

    @staticmethod
    def _angle_diff(a, b):
        d = abs(a - b) % 360
        if d > 180:
            d = 360 - d
        return d

    def _edge_map(self, image_path: str | None):
        if not image_path or not CV2_AVAILABLE:
            return None, None
        suffix = Path(image_path).suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            return None, None

        img = cv2.imread(image_path)
        if img is None:
            return None, None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            5,
        )
        edges = cv2.Canny(binary, 50, 150)
        edges = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )
        return edges, gray

    def _arc_near_gap(self, edge_map, gray, hinge: Point2D, radius_m: float) -> tuple[bool, float]:
        """Use HoughCircles and contour fallback to find an arc near a gap."""
        h, w = edge_map.shape[:2]
        cx = int(round(hinge.x * self.pixels_per_meter))
        cy = int(round(h - hinge.y * self.pixels_per_meter))

        r_px = max(5, int(round(radius_m * self.pixels_per_meter)))
        pad = int(r_px * 1.5)
        x0, x1 = max(0, cx - pad), min(w, cx + pad)
        y0, y1 = max(0, cy - pad), min(h, cy + pad)
        if x1 <= x0 or y1 <= y0:
            return False, radius_m

        roi_edges = edge_map[y0:y1, x0:x1]
        roi_gray = gray[y0:y1, x0:x1]

        circles = cv2.HoughCircles(
            roi_gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(8, r_px // 2),
            param1=120,
            param2=15,
            minRadius=max(4, int(r_px * 0.5)),
            maxRadius=max(8, int(r_px * 1.5)),
        )
        if circles is not None:
            c = circles[0][0]
            rr = float(c[2]) / self.pixels_per_meter
            return True, rr

        contours, _ = cv2.findContours(roi_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        best_r = None
        for cnt in contours:
            if len(cnt) < 8:
                continue
            (x, y), rr = cv2.minEnclosingCircle(cnt)
            if rr < max(4, r_px * 0.45) or rr > max(8, r_px * 1.6):
                continue
            if best_r is None or rr > best_r:
                best_r = rr
        if best_r is not None:
            return True, float(best_r) / self.pixels_per_meter

        return False, radius_m


    def detect_into(self, floorplan, image_path: str | None = None):
        floorplan.doors = self.detect(
            wall_graph=floorplan.wall_graph,
            walls=floorplan.walls,
            image_path=image_path,
        )
        return floorplan
    @staticmethod
    def _estimate_opening_angle(hinge: Point2D, radius: float, wall_idx: int, walls: list[Wall]) -> float:
        if wall_idx < 0 or wall_idx >= len(walls):
            return math.pi / 2
        w = walls[wall_idx]
        # Keep as relative swing amplitude; default quarter turn.
        return math.pi / 2
