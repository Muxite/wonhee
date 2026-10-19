"""Minimal JSON-serializable graph data model for depgraph."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Node:
    id: str
    path: str
    loc: int
    language: str


@dataclass
class Edge:
    from_: str
    to: str


@dataclass
class Graph:
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "nodes": [{"id": n.id, "path": n.path, "loc": n.loc, "language": n.language} for n in self.nodes],
            "edges": [{"from": e.from_, "to": e.to} for e in self.edges],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Graph":
        nodes = [Node(id=n["id"], path=n["path"], loc=n["loc"], language=n["language"]) for n in data["nodes"]]
        edges = [Edge(from_=e["from"], to=e["to"]) for e in data["edges"]]
        return cls(nodes=nodes, edges=edges)

    def stats(self) -> dict:
        total_loc = sum(n.loc for n in self.nodes)
        by_language: dict[str, int] = {}
        for n in self.nodes:
            by_language[n.language] = by_language.get(n.language, 0) + 1
        return {
            "file_count": len(self.nodes),
            "total_loc": total_loc,
            "edge_count": len(self.edges),
            "by_language": dict(sorted(by_language.items(), key=lambda kv: -kv[1])),
        }
