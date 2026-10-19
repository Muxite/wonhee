"""Cut a dependency graph down to the files around a focus: the focus files plus every file within N import hops
in either direction. Generalises the Ledger render (2026-10-13), where a whole-repo graph of webRAG would have
looked like every other Euglena picture."""
from __future__ import annotations

from pathspec import PathSpec

from wonhee.edit.viz.graph import Graph


def neighbourhood(graph: Graph, focus: PathSpec, hops: int = 1) -> Graph:
    keep = {n.id for n in graph.nodes if focus.match_file(n.path)}
    if not keep:
        return Graph()
    adj: dict[str, set[str]] = {}
    for e in graph.edges:
        adj.setdefault(e.from_, set()).add(e.to)
        adj.setdefault(e.to, set()).add(e.from_)
    frontier = set(keep)
    for _ in range(max(0, hops)):
        frontier = {m for n in frontier for m in adj.get(n, ())} - keep
        keep |= frontier
    return Graph(nodes=[n for n in graph.nodes if n.id in keep],
                 edges=[e for e in graph.edges if e.from_ in keep and e.to in keep])
