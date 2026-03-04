"""Wall graph generation from detected wall centerlines."""

from __future__ import annotations
from dataclasses import dataclass, field
import math

from .dxf_parser import Point2D
from .wall_detector import Wall


@dataclass
class GraphNode:
    id: int
    x: float
    y: float


@dataclass
class GraphEdge:
    id: int
    start_node: int
    end_node: int
    wall_index: int
    length: float
    thickness: float


@dataclass
class WallGraph:
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    adjacency: dict[int, list[int]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "nodes": [{"id": n.id, "x": round(n.x, 4), "y": round(n.y, 4)} for n in self.nodes],
            "edges": [
                {
                    "id": e.id,
                    "start_node": e.start_node,
                    "end_node": e.end_node,
                    "wall_index": e.wall_index,
                    "length": round(e.length, 4),
                    "thickness": round(e.thickness, 4),
                }
                for e in self.edges
            ],
            "adjacency": self.adjacency,
        }


def _quantize(x: float, y: float, tol: float) -> tuple[int, int]:
    return (int(round(x / tol)), int(round(y / tol)))


def generate_wall_graph(walls: list[Wall], snap_tolerance: float = 0.05) -> WallGraph:
    """Build a topological graph by snapping wall endpoints to shared nodes."""
    graph = WallGraph()
    node_by_key: dict[tuple[int, int], int] = {}

    def get_node_id(p: Point2D) -> int:
        key = _quantize(p.x, p.y, snap_tolerance)
        if key in node_by_key:
            return node_by_key[key]
        node_id = len(graph.nodes)
        node_by_key[key] = node_id
        graph.nodes.append(GraphNode(id=node_id, x=p.x, y=p.y))
        graph.adjacency[node_id] = []
        return node_id

    for wall_index, wall in enumerate(walls):
        s_id = get_node_id(wall.start)
        e_id = get_node_id(wall.end)
        if s_id == e_id:
            continue

        edge_id = len(graph.edges)
        length = math.hypot(wall.end.x - wall.start.x, wall.end.y - wall.start.y)
        graph.edges.append(
            GraphEdge(
                id=edge_id,
                start_node=s_id,
                end_node=e_id,
                wall_index=wall_index,
                length=length,
                thickness=wall.thickness,
            )
        )
        graph.adjacency[s_id].append(edge_id)
        graph.adjacency[e_id].append(edge_id)

    return graph
