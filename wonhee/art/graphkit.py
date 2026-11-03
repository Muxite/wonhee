"""graphkit: shared helpers for the graph / DAG / structure effects.

Contents
  Graph + loaders     as_graph(data), from_repo(path), from_ts(path), sbm(), random_layered()
  Layouts             layered() (Sugiyama-style), force(), tree_layout(), matrix_order(), communities()
  Routing             flow_path(), ortho_path(), bundle_curve(), catmull()
  Raster              Canvas + Batch (anti-aliased cv2 masks composited in palette colours), glow()
  Node styles         draw_node(): box, pill, dot, ring, square
  Colour              tone(), group_tone(): palette-driven, one hue family, contrast from lightness

Everything is deterministic given an ``rng`` (numpy Generator). No hubs: the
layouts spread nodes over the whole frame and the draw helpers take a ``view``
rectangle larger than the frame so structure bleeds off the edges.
"""
from __future__ import annotations

import ast
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

import cv2
import numpy as np

# =====================================================================================================================
# Graph model and loaders
# =====================================================================================================================


@dataclass
class Graph:
    ids: list[str]
    edges: np.ndarray                   # (m, 2) int, source -> target
    group: np.ndarray                   # (n,) int group / cluster id
    weight: np.ndarray                  # (n,) float node weight (size)
    eweight: np.ndarray                 # (m,) float edge weight
    directed: bool = True
    group_names: list[str] = field(default_factory=list)
    layer: np.ndarray | None = None     # optional fixed layer per node

    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def m(self) -> int:
        return len(self.edges)

    def degree(self) -> np.ndarray:
        d = np.zeros(self.n)
        for a, b in self.edges:
            d[a] += 1; d[b] += 1
        return d

    def subgraph(self, keep: np.ndarray) -> "Graph":
        keep = np.asarray(keep, int)
        remap = -np.ones(self.n, int); remap[keep] = np.arange(len(keep))
        mask = (remap[self.edges[:, 0]] >= 0) & (remap[self.edges[:, 1]] >= 0) if self.m else np.zeros(0, bool)
        return Graph([self.ids[i] for i in keep], remap[self.edges[mask]].reshape(-1, 2), self.group[keep], self.weight[keep],
                     self.eweight[mask], self.directed, self.group_names, None if self.layer is None else self.layer[keep])


def _build(ids: list[str], pairs: list[tuple[str, str]], groups: dict[str, Any] | None = None,
           weights: dict[str, float] | None = None, ew: dict[tuple[str, str], float] | None = None, directed: bool = True) -> Graph:
    idx = {k: i for i, k in enumerate(ids)}
    seen, E, W = set(), [], []
    for a, b in pairs:
        if a == b or a not in idx or b not in idx:
            continue
        key = (a, b) if directed else tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key); E.append((idx[a], idx[b])); W.append(float((ew or {}).get((a, b), 1.0)))
    gl = [str((groups or {}).get(k, "")) for k in ids]
    names = sorted(set(gl)); gi = np.array([names.index(g) for g in gl], int)
    w = np.array([float((weights or {}).get(k, 1.0)) for k in ids])
    return Graph(ids, np.array(E, int).reshape(-1, 2), gi, w, np.array(W), directed, names)


def _nodeid(x) -> str:
    if isinstance(x, dict):
        for k in ("id", "name", "key", "label"):
            if k in x:
                return str(x[k])
    return str(x)


def as_graph(data: Any, directed: bool = True) -> Graph:
    """Accept any of these and return a Graph (else ValueError):

    * ``Graph``
    * ``{"nodes": [{"id", "group"?, "weight"?} | "id"], "edges"|"links": [{"source","target","weight"?} | [a, b] | {"from","to"}]}``
      (also FlowSpec-like ``{"nodes":[{"id","group","layer"}],"edges":[{"from","to"}]}``; ``layer`` is kept)
    * module -> list of imports: ``{"a.py": ["b.py", "c.py"]}``  (path-like keys group by top directory)
    * networkx-like adjacency: ``{"a": {"b": {"weight": 2}}}`` or ``{"a": {"b": 2}}``
    * a list of ``(source, target)`` pairs
    * a euglena-style trace ``{"events": [{"type": "job_created", "jobId", "parentJobId"}]}``
    * an object with ``nodes`` / ``edges`` attributes or ``to_dict`` (networkx ``node_link_data`` also works)
    """
    if isinstance(data, Graph):
        return data
    if hasattr(data, "adj") and hasattr(data, "nodes") and not isinstance(data, dict):   # a real networkx graph
        data = {"nodes": [{"id": str(n), **{k: v for k, v in d.items() if k in ("group", "weight")}} for n, d in data.nodes(data=True)],
                "edges": [{"source": str(a), "target": str(b), "weight": d.get("weight", 1.0)} for a, b, d in data.edges(data=True)]}
    if isinstance(data, (list, tuple)):
        pairs = [(_nodeid(a), _nodeid(b)) for a, b in data]
        ids = list(dict.fromkeys([p for pr in pairs for p in pr]))
        return _build(ids, pairs, directed=directed)
    if not isinstance(data, dict):
        raise ValueError("unsupported graph data: expected dict, list of pairs, or Graph")
    if "events" in data and isinstance(data["events"], list):
        ids, pairs, groups = [], [], {}
        for e in data["events"]:
            if e.get("type") == "job_created":
                j = e["jobId"]; ids.append(j); groups[j] = (e.get("job") or {}).get("kind", "")
                if e.get("parentJobId"):
                    pairs.append((e["parentJobId"], j))
        return _build(list(dict.fromkeys(ids)), pairs, groups, directed=directed)
    ekey = "edges" if "edges" in data else ("links" if "links" in data else None)
    if "nodes" in data and ekey:
        nodes = data["nodes"]
        ids = [_nodeid(n) for n in nodes]
        groups = {_nodeid(n): n.get("group", n.get("type", n.get("subtype", ""))) for n in nodes if isinstance(n, dict)}
        weights = {_nodeid(n): n.get("weight", 1.0) for n in nodes if isinstance(n, dict)}
        pairs, ew = [], {}
        for e in data[ekey]:
            if isinstance(e, dict):
                a = e.get("source", e.get("from", e.get("src", e.get("u"))))
                b = e.get("target", e.get("to", e.get("dst", e.get("v"))))
                a, b = _nodeid(a), _nodeid(b)
                ew[(a, b)] = float(e.get("weight", 1.0))
            else:
                a, b = _nodeid(e[0]), _nodeid(e[1])
            pairs.append((a, b))
        for a, b in pairs:                      # edges may mention nodes not listed
            for k in (a, b):
                if k not in groups and k not in ids:
                    ids.append(k)
        g = _build(ids, pairs, groups, weights, ew, directed)
        if any(isinstance(n, dict) and "layer" in n for n in nodes):
            lay = {_nodeid(n): n.get("layer", 0) for n in nodes if isinstance(n, dict)}
            g.layer = np.array([int(lay.get(k, 0)) for k in ids])
        return g
    # adjacency dict: module -> imports (list) or neighbour -> attrs (dict)
    ids = list(data.keys()); pairs, ew = [], {}
    for a, nb in data.items():
        if isinstance(nb, dict):
            for b, attr in nb.items():
                pairs.append((str(a), str(b)))
                ew[(str(a), str(b))] = float(attr.get("weight", 1.0)) if isinstance(attr, dict) else float(attr) if isinstance(attr, (int, float)) else 1.0
        elif isinstance(nb, (list, tuple, set)):
            pairs += [(str(a), _nodeid(b)) for b in nb]
        else:
            continue
    for a, b in pairs:
        if b not in ids:
            ids.append(b)
    groups = {k: (k.replace("\\", "/").split("/")[0] if "/" in k else (k.split(".")[0] if "." in k and not k.endswith((".py", ".ts", ".tsx", ".js")) else "")) for k in ids}
    return _build(ids, pairs, groups, None, ew, directed)


_IMPORT_PY = (ast.Import, ast.ImportFrom)


def from_repo(path: str, max_nodes: int = 160, exts=(".py", ".ts", ".tsx", ".js", ".jsx"), skip=("node_modules", ".git", "__pycache__", ".venv", "venv", "dist", "build", ".next", "tests", "test")) -> Graph:
    """Build a module dependency Graph from a repo's import structure.

    Python: ``import x`` / ``from x import y`` resolved to files inside the repo. TS/JS: relative
    ``from './x'`` and ``@/x`` (-> src/x). Nodes are repo-relative files, group = top directory
    (second level for ``src/``-style roots), weight = bytes. If more than ``max_nodes`` files exist
    the best-connected ``max_nodes`` are kept. Edges point importer -> imported.
    """
    path = os.path.abspath(path)
    files = []
    for root, dirs, fs in os.walk(path):
        dirs[:] = sorted(d for d in dirs if d not in skip and not d.startswith("."))
        for f in sorted(fs):
            if f.endswith(exts) and not f.endswith((".d.ts", ".test.ts", ".test.tsx")):
                files.append(os.path.relpath(os.path.join(root, f), path).replace(os.sep, "/"))
    fset = set(files)
    py_mod = {}
    for f in files:
        if f.endswith(".py"):
            m = f[:-3].replace("/", ".")
            py_mod[m[:-9] if m.endswith(".__init__") else m] = f
    def resolve_py(mod: str, cur: str, level: int):
        if level:
            base = cur.split("/")[:-level] if not cur.endswith("__init__.py") else cur.split("/")[:-level]
            full = ".".join(base + ([mod] if mod else []))
        else:
            full = mod
        parts = full.split(".")
        while parts:
            k = ".".join(parts)
            if k in py_mod:
                return py_mod[k]
            parts.pop()
        return None
    pairs = []
    for f in files:
        try:
            src = open(os.path.join(path, f), encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        if f.endswith(".py"):
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        t = resolve_py(a.name, f, 0)
                        if t: pairs.append((f, t))
                elif isinstance(node, ast.ImportFrom):
                    base = node.module or ""
                    t = resolve_py(base, f, node.level)
                    if t: pairs.append((f, t))
                    for a in node.names:                      # from pkg import submodule
                        t2 = resolve_py(((base + ".") if base else "") + a.name, f, node.level)
                        if t2: pairs.append((f, t2))
        else:
            for spec in re.findall(r"""(?:from|import|require\()\s*['"]([^'"]+)['"]""", src):
                if spec.startswith("."):
                    tgt = os.path.normpath(os.path.join(os.path.dirname(f), spec)).replace(os.sep, "/")
                elif spec.startswith("@/"):
                    tgt = "src/" + spec[2:]
                else:
                    continue
                for cand in [tgt] + [tgt + e for e in exts] + [tgt + "/index" + e for e in exts]:
                    if cand in fset:
                        pairs.append((f, cand)); break
    def grp(f):
        p = f.split("/")
        return p[1] if len(p) > 2 and p[0] in ("src", "app", "lib", "backend", "packages") else p[0] if len(p) > 1 else "."
    weights = {f: max(1, os.path.getsize(os.path.join(path, f))) for f in files}
    g = _build(files, pairs, {f: grp(f) for f in files}, weights)
    deg = g.degree()
    keep = np.where(deg > 0)[0]
    if len(keep) > max_nodes:
        keep = keep[np.argsort(-deg[keep], kind="stable")[:max_nodes]]
        keep.sort()
    return g.subgraph(keep) if len(keep) != g.n else g


def from_ts(path: str) -> Graph:
    """Parse a TypeScript data module of the muksite projectViz kind.

    Understands ``entity('id', 'SUBTYPE', ...)`` calls, ``{ id: 'x', type: 'y' ...}`` node literals,
    FlowSpec nodes ``{ id: 'x', layer: 0, ... group: 'g'`` and edges ``{ source, target }`` / ``{ from, to }``.
    """
    src = open(path, encoding="utf-8").read()
    nodes: dict[str, str] = {}
    for m in re.finditer(r"entity\(\s*'([^']+)'\s*,\s*'([^']+)'", src):
        nodes[m.group(1)] = m.group(2)
    for m in re.finditer(r"\{\s*id:\s*'([^']+)'\s*,([^}]*?)(?:\}|\n)", src):
        body = m.group(2)
        t = re.search(r"\b(?:group|type)\s*:\s*'([^']+)'", body)
        nodes.setdefault(m.group(1), t.group(1) if t else "")
    for m in re.finditer(r"\{\s*id:\s*'([^']+)'\s*,\s*layer:\s*(\d+)", src):
        nodes.setdefault(m.group(1), "")
    lay = {m.group(1): int(m.group(2)) for m in re.finditer(r"\{\s*id:\s*'([^']+)'\s*,\s*layer:\s*(\d+)", src)}
    pairs = [(m.group(1), m.group(2)) for m in re.finditer(r"source:\s*'([^']+)'\s*,\s*target:\s*'([^']+)'", src)]
    pairs += [(m.group(1), m.group(2)) for m in re.finditer(r"from:\s*'([^']+)'\s*,\s*to:\s*'([^']+)'", src)]
    for a, b in pairs:
        nodes.setdefault(a, ""); nodes.setdefault(b, "")
    g = _build(list(nodes), pairs, nodes)
    if lay:
        g.layer = np.array([lay.get(k, 0) for k in g.ids])
    return g


def from_json(path: str) -> Graph:
    return as_graph(json.load(open(path)))


def communities(g: Graph, rng: np.random.Generator, iters: int = 12) -> np.ndarray:
    """Label propagation on the undirected graph; returns group ids (compact 0..k-1)."""
    n = g.n
    nb = [[] for _ in range(n)]
    for a, b in g.edges:
        nb[a].append(b); nb[b].append(a)
    lab = np.arange(n)
    for _ in range(iters):
        changed = 0
        for v in rng.permutation(n):
            if not nb[v]:
                continue
            c = np.bincount(lab[nb[v]])
            best = np.flatnonzero(c == c.max())
            new = lab[v] if lab[v] in best else best[rng.integers(len(best))]
            changed += new != lab[v]; lab[v] = new
        if not changed:
            break
    _, inv = np.unique(lab, return_inverse=True)
    return inv


def sbm(rng: np.random.Generator, n: int = 90, k: int = 9, p_in: float = 0.22, p_out: float = 0.006, directed: bool = False) -> Graph:
    """Stochastic block model: k clusters, dense inside, sparse between (synthetic stand-in for real data)."""
    grp = np.sort(rng.integers(0, k, n))
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < (p_in if grp[i] == grp[j] else p_out):
                pairs.append((f"n{i}", f"n{j}"))
    ids = [f"n{i}" for i in range(n)]
    g = _build(ids, pairs, {f"n{i}": int(grp[i]) for i in range(n)}, {i: 0 for i in ids}, directed=directed)
    g.weight = 0.5 + rng.random(n) * 1.5
    return g


def random_layered(rng: np.random.Generator, widths: list[int], extra: float = 0.5, skip: float = 0.08, lanes: int = 4) -> Graph:
    """Synthetic layered DAG: widths[i] nodes in layer i, nearest-neighbour edges plus extras and a few long (skip) edges.
    Group = a horizontal band (lane). ``layer`` is fixed on the result."""
    ids, layer, grp, fr = [], [], [], []
    for l, w in enumerate(widths):
        for j in range(w):
            ids.append(f"L{l}.{j}"); layer.append(l); f = (j + 0.5) / w; fr.append(f); grp.append(min(lanes - 1, int(f * lanes)))
    layer, fr = np.array(layer), np.array(fr)
    pairs = []
    for l in range(len(widths) - 1):
        A, B = np.where(layer == l)[0], np.where(layer == l + 1)[0]
        has_parent = set()
        for a in A:
            order = B[np.argsort(np.abs(fr[B] - fr[a]) + rng.random(len(B)) * 0.08)]
            tg = [order[0]]
            if rng.random() < extra and len(B) > 1:
                tg.append(order[1 if rng.random() < 0.8 else min(2, len(B) - 1)])
            for t in tg:
                pairs.append((ids[a], ids[t])); has_parent.add(t)
        for b in B:
            if b not in has_parent:
                a = A[np.argmin(np.abs(fr[A] - fr[b]))]; pairs.append((ids[a], ids[b]))
        if l + 2 < len(widths) and rng.random() < skip * 4:
            C = np.where(layer == l + 2)[0]
            for _ in range(int(rng.integers(1, 3))):
                pairs.append((ids[rng.choice(A)], ids[rng.choice(C)]))
    g = _build(ids, pairs, {k: grp[i] for i, k in enumerate(ids)})
    g.layer = layer
    return g


# =====================================================================================================================
# Layouts
# =====================================================================================================================

@dataclass
class Layered:
    x: np.ndarray                 # (N,) layer coordinate 0..1 (real + dummy nodes)
    y: np.ndarray                 # (N,) 0..1 across the layer
    layer: np.ndarray
    n_real: int
    chains: list[list[int]]       # per edge (aligned with kept edges): node indices incl. dummies
    edge_ids: list[int]           # index into g.edges for each chain
    nlayers: int

    def real_xy(self) -> np.ndarray:
        return np.stack([self.x[:self.n_real], self.y[:self.n_real]], 1)


def acyclic_edges(g: Graph) -> list[int]:
    """Indices of edges kept after removing back edges found by DFS (so cyclic imports still lay out)."""
    n = g.n
    out = [[] for _ in range(n)]
    for k, (a, b) in enumerate(g.edges):
        out[a].append((b, k))
    state = np.zeros(n, int); drop = set()
    for s in range(n):
        if state[s]:
            continue
        stack = [(s, 0)]; state[s] = 1
        while stack:
            v, i = stack.pop()
            if i < len(out[v]):
                stack.append((v, i + 1))
                w, k = out[v][i]
                if state[w] == 1:
                    drop.add(k)
                elif state[w] == 0:
                    state[w] = 1; stack.append((w, 0))
            else:
                state[v] = 2
    return [k for k in range(g.m) if k not in drop and g.edges[k][0] != g.edges[k][1]]


def longest_path(g: Graph, weight: bool = False) -> list[int]:
    """The critical path: longest chain in the acyclic part of the graph (node count, or summed node weight)."""
    keep = acyclic_edges(g)
    n = g.n
    outs = [[] for _ in range(n)]; indeg = np.zeros(n, int)
    for k in keep:
        a, b = g.edges[k]; outs[a].append(b); indeg[b] += 1
    order, q = [], [i for i in range(n) if indeg[i] == 0]
    while q:
        v = q.pop(); order.append(v)
        for w in outs[v]:
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
    best = np.zeros(n); prev = -np.ones(n, int)
    wv = g.weight if weight else np.ones(n)
    for v in order:
        best[v] = max(best[v], wv[v]) if prev[v] < 0 and best[v] == 0 else best[v]
        for w in outs[v]:
            if best[v] + wv[w] > best[w]:
                best[w] = best[v] + wv[w]; prev[w] = v
    v = int(np.argmax(best)); path = [v]
    while prev[v] >= 0:
        v = int(prev[v]); path.append(v)
    return path[::-1]


def layered(g: Graph, rng: np.random.Generator, sweeps: int = 14, relax: float = 0.55, compress: bool = False) -> Layered:
    """Sugiyama-style layout: cycle removal, longest-path layering (or ``g.layer``), dummy nodes so long edges
    become bundles of per-layer waypoints, barycentre/median crossing reduction (down/up sweeps), then
    neighbour-average coordinate relaxation blended with even spacing so each layer spans the whole height."""
    keep = acyclic_edges(g)
    n = g.n
    if g.layer is not None:
        lay = g.layer.astype(int).copy()
    else:
        lay = np.zeros(n, int)
        outs = [[] for _ in range(n)]; indeg = np.zeros(n, int)
        for k in keep:
            a, b = g.edges[k]; outs[a].append(b); indeg[b] += 1
        q = [i for i in range(n) if indeg[i] == 0]
        while q:
            v = q.pop()
            for w in outs[v]:
                lay[w] = max(lay[w], lay[v] + 1); indeg[w] -= 1
                if indeg[w] == 0:
                    q.append(w)
    # edges must go forward
    kept = []
    for k in keep:
        a, b = g.edges[k]
        if lay[b] > lay[a]:
            kept.append(k)
        elif lay[b] < lay[a]:
            kept.append(k)               # drawn reversed below
    nl = int(lay.max()) + 1
    layer = list(lay); chains = []; eids = []
    for k in kept:
        a, b = (int(g.edges[k][0]), int(g.edges[k][1]))
        if lay[a] > lay[b]:
            a, b = b, a
        chain = [a]
        for l in range(lay[a] + 1, lay[b]):
            layer.append(l); chain.append(len(layer) - 1)
        chain.append(b); chains.append(chain); eids.append(k)
    N = len(layer); layer = np.array(layer)
    up = [[] for _ in range(N)]; dn = [[] for _ in range(N)]
    for ch in chains:
        for u, v in zip(ch[:-1], ch[1:]):
            dn[u].append(v); up[v].append(u)
    # initial order: by group then random
    key0 = np.zeros(N)
    key0[:n] = g.group + rng.random(n) * 0.9
    key0[n:] = rng.random(N - n) * (g.group.max() + 1 if n else 1)
    pos = np.zeros(N)
    members = [np.where(layer == l)[0] for l in range(nl)]
    for l in range(nl):
        o = members[l][np.argsort(key0[members[l]], kind="stable")]
        members[l] = o; pos[o] = np.arange(len(o)) / max(len(o), 1)

    def sweep(rng_l, nbrs):
        for l in rng_l:
            ms = members[l]
            if len(ms) < 2:
                continue
            k = np.array([np.median(pos[nbrs[v]]) if nbrs[v] else pos[v] for v in ms])
            o = ms[np.argsort(k + 1e-6 * pos[ms], kind="stable")]
            members[l] = o; pos[o] = np.arange(len(o)) / len(o)
    for s in range(sweeps):
        sweep(range(1, nl), up)
        sweep(range(nl - 2, -1, -1), dn)
    # coordinates: even spacing then relaxation toward neighbour mean
    y = np.zeros(N)
    for l in range(nl):
        ms = members[l]; y[ms] = (np.arange(len(ms)) + 0.5) / len(ms)
    even = y.copy()
    for it in range(8):
        tgt = y.copy()
        for v in range(N):
            nb = up[v] + dn[v]
            if nb:
                tgt[v] = y[nb].mean()
        for l in range(nl):
            ms = members[l]
            if len(ms) < 2:
                y[ms] = tgt[ms]; continue
            gap = 0.5 / len(ms)
            t = tgt[ms] - np.arange(len(ms)) * gap
            a = np.maximum.accumulate(t); b = np.minimum.accumulate(t[::-1])[::-1]
            y[ms] = (a + b) / 2 + np.arange(len(ms)) * gap
            lo, hi = y[ms].min(), y[ms].max()
            if hi - lo > 1e-6:
                y[ms] = (y[ms] - lo) / (hi - lo)
            y[ms] = 0.03 + 0.94 * y[ms]
    y = relax * y + (1 - relax) * even
    for l in range(nl):                  # lone nodes: jitter so they do not form a spine
        ms = members[l]
        if len(ms) == 1:
            y[ms] = 0.15 + 0.7 * rng.random()
    x = layer / max(nl - 1, 1)
    return Layered(x.astype(float), y, layer, n, chains, eids, nl)


def force(g: Graph, rng: np.random.Generator, iters: int = 160, aspect: float = 1.5, anchors: np.ndarray | None = None,
          pull: float = 0.0, spread: float = 0.55) -> np.ndarray:
    """Fruchterman-Reingold in a box [0,aspect] x [0,1]. ``anchors`` (n,2) with ``pull`` > 0 adds a spring to a per-node
    anchor (used to place clusters on a grid so no hub forms). After the run the axes are partly rank-equalised
    (``spread``) so density is even over the frame."""
    n = g.n
    p = rng.random((n, 2)) * [aspect, 1.0]
    if anchors is not None:
        p = anchors + rng.normal(0, 0.03, (n, 2))
    k = math.sqrt(aspect / max(n, 1)) * 0.9
    t = 0.12
    E = g.edges
    for i in range(iters):
        d = p[:, None, :] - p[None, :, :]
        dist = np.sqrt((d ** 2).sum(-1)) + 1e-4
        rep = (k * k / dist ** 2)[..., None] * d
        disp = rep.sum(1)
        if len(E):
            dd = p[E[:, 0]] - p[E[:, 1]]
            dl = np.sqrt((dd ** 2).sum(-1)) + 1e-4
            f = (dl / k)[:, None] * dd
            np.add.at(disp, E[:, 0], -f); np.add.at(disp, E[:, 1], f)
        if anchors is not None and pull:
            disp += (anchors - p) * pull * n / 12
        ln = np.sqrt((disp ** 2).sum(-1)) + 1e-9
        p += disp / ln[:, None] * np.minimum(ln, t)[:, None]
        t *= 0.985
        p[:, 0] = np.clip(p[:, 0], 0, aspect); p[:, 1] = np.clip(p[:, 1], 0, 1)
    if spread:
        for ax, span in ((0, aspect), (1, 1.0)):
            r = np.argsort(np.argsort(p[:, ax])) / max(n - 1, 1) * span
            p[:, ax] = (1 - spread) * (p[:, ax] - p[:, ax].min()) / max(np.ptp(p[:, ax]), 1e-6) * span + spread * r
    return p


def grid_anchors(rng: np.random.Generator, k: int, aspect: float = 1.5, jitter: float = 0.35) -> np.ndarray:
    """k cluster centres on a jittered grid covering the frame (Poisson-ish, never centred)."""
    gx = max(1, int(round(math.sqrt(k * aspect)))); gy = max(1, int(math.ceil(k / gx)))
    cells = [(i, j) for i in range(gx) for j in range(gy)]
    rng.shuffle(cells)
    out = []
    for i, j in cells[:k]:
        out.append(((i + 0.5 + jitter * (rng.random() - 0.5) * 2) / gx * aspect, (j + 0.5 + jitter * (rng.random() - 0.5) * 2) / gy))
    return np.array(out)


def tree_layout(parent: list[int], depth_first: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Tidy tree: x = in-order leaf index (0..1), y = depth (0..1). parent[root] = -1."""
    n = len(parent)
    kids = [[] for _ in range(n)]
    root = 0
    for v, p in enumerate(parent):
        if p < 0: root = v
        else: kids[p].append(v)
    depth = np.zeros(n); x = np.zeros(n); cnt = [0]
    def rec(v, d):
        depth[v] = d
        if not kids[v]:
            x[v] = cnt[0]; cnt[0] += 1
        else:
            for c in kids[v]:
                rec(c, d + 1)
            x[v] = np.mean([x[c] for c in kids[v]])
    import sys
    sys.setrecursionlimit(10000)
    rec(root, 0)
    return x / max(cnt[0] - 1, 1), depth / max(depth.max(), 1)


def matrix_order(g: Graph, rng: np.random.Generator | None = None) -> np.ndarray:
    """Block ordering: group (or detected community) first, inside a group reverse Cuthill-McKee, so dense blocks
    sit on the diagonal."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import reverse_cuthill_mckee
    grp = g.group if len(set(g.group.tolist())) > 1 else communities(g, rng or np.random.default_rng(0))
    n = g.n
    A = coo_matrix((np.ones(g.m), (g.edges[:, 0], g.edges[:, 1])), shape=(n, n)).tocsr() if g.m else coo_matrix((n, n)).tocsr()
    A = ((A + A.T) > 0).astype(int).tocsr()
    order = []
    for c in np.unique(grp):
        idx = np.where(grp == c)[0]
        if len(idx) > 2:
            sub = A[idx][:, idx]
            idx = idx[reverse_cuthill_mckee(sub.tocsr())]
        order += idx.tolist()
    return np.array(order, int)


# =====================================================================================================================
# Edge routing
# =====================================================================================================================

def bezier(p0, p1, p2, p3, n: int = 20) -> np.ndarray:
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def s_curve(p, q, horizontal: bool = True, n: int = 22, k: float = 0.5) -> np.ndarray:
    """S-shaped edge with tangents along the layer axis."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    if horizontal:
        d = (q[0] - p[0]) * k
        return bezier(p, p + [d, 0], q - [d, 0], q, n)
    d = (q[1] - p[1]) * k
    return bezier(p, p + [0, d], q - [0, d], q, n)


def flow_path(pts: np.ndarray, n: int = 18, k: float = 0.5, horizontal: bool = True) -> np.ndarray:
    """Chain of S-curves through waypoints (dummy nodes of a long edge) = a smooth bundle-friendly route."""
    segs = [s_curve(pts[i], pts[i + 1], horizontal, n, k) for i in range(len(pts) - 1)]
    return np.concatenate([segs[0]] + [s[1:] for s in segs[1:]])


def ortho_path(p, q, frac: float = 0.5, r: float = 0.0, horizontal: bool = True) -> np.ndarray:
    """Orthogonal (Manhattan) edge: along, across, along. ``r`` rounds the corners."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    if horizontal:
        mx = p[0] + (q[0] - p[0]) * frac
        pts = np.array([p, [mx, p[1]], [mx, q[1]], q])
    else:
        my = p[1] + (q[1] - p[1]) * frac
        pts = np.array([p, [p[0], my], [q[0], my], q])
    return pts if r <= 0 else catmull(pts, 8, 0.0)


def bundle_curve(p, q, c, beta: float = 0.8, n: int = 24) -> np.ndarray:
    """Holten-style bundle: quadratic through a control pulled toward ``c`` (e.g. a shared gate or the ring centre)."""
    p, q, c = np.asarray(p, float), np.asarray(q, float), np.asarray(c, float)
    mid = (p + q) / 2
    ctrl = mid + (c - mid) * beta
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * p + 2 * (1 - t) * t * ctrl + t ** 2 * q


def catmull(pts: np.ndarray, n: int = 8, tension: float = 0.5) -> np.ndarray:
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return pts
    P = np.vstack([pts[0], pts, pts[-1]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(pts[-1][None])
    return np.concatenate(out)


# =====================================================================================================================
# Colour
# =====================================================================================================================

def tone(pal, t: float) -> np.ndarray:
    """0 = canvas, 0.5 = structure, 1 = accent (linear in sRGB between role colours; contrast = lightness)."""
    t = float(np.clip(t, 0, 1))
    if t < 0.5:
        return pal.canvas + (pal.structure - pal.canvas) * (t / 0.5)
    return pal.structure + (pal.accent - pal.structure) * ((t - 0.5) / 0.5)


def group_tone(pal, gi: int, ng: int, lo: float = 0.36, hi: float = 0.62) -> np.ndarray:
    """Per-group colour: same hue family, lightness varies; three-colour palettes also use structure2."""
    f = (gi % 5) / 4.0
    c = tone(pal, lo + (hi - lo) * f)
    if pal.structure2 is not None and gi % 2 == 1:
        c = c + (pal.structure2 - c) * 0.55
    return c


# =====================================================================================================================
# Raster: anti-aliased masks composited into a float canvas
# =====================================================================================================================

SH = 4
_K = 1 << SH


def _ip(pts) -> np.ndarray:
    return np.round(np.asarray(pts, float) * _K).astype(np.int32)


def m_line(m: np.ndarray, pts, w: float) -> None:
    pts = np.asarray(pts, float)
    if len(pts) < 2:
        return
    cv2.polylines(m, [_ip(pts)], False, 255, max(1, int(round(w))), cv2.LINE_AA, SH)


def m_disc(m, c, r: float) -> None:
    cv2.circle(m, tuple(int(v) for v in np.round(np.asarray(c, float) * _K)), max(1, int(round(r * _K))), 255, -1, cv2.LINE_AA, SH)


def m_ring(m, c, r: float, w: float) -> None:
    cv2.circle(m, tuple(int(v) for v in np.round(np.asarray(c, float) * _K)), max(1, int(round(r * _K))), 255, max(1, int(round(w))), cv2.LINE_AA, SH)


def rrect_pts(x0, y0, x1, y1, r: float, n: int = 5) -> np.ndarray:
    r = max(0.0, min(r, (x1 - x0) / 2, (y1 - y0) / 2))
    if r < 0.5:
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
    out = []
    for cx, cy, a0 in ((x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)):
        for a in np.linspace(a0, a0 + 90, n):
            out.append((cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a))))
    return np.array(out)


def m_rrect(m, x0, y0, x1, y1, r: float = 0.0, w: int = -1) -> None:
    pts = _ip(rrect_pts(x0, y0, x1, y1, r))
    if w < 0:
        cv2.fillPoly(m, [pts], 255, cv2.LINE_AA, SH)
    else:
        cv2.polylines(m, [pts], True, 255, max(1, int(w)), cv2.LINE_AA, SH)


def m_poly(m, pts, fill: bool = True, w: float = 1) -> None:
    if fill:
        cv2.fillPoly(m, [_ip(pts)], 255, cv2.LINE_AA, SH)
    else:
        cv2.polylines(m, [_ip(pts)], True, 255, max(1, int(round(w))), cv2.LINE_AA, SH)


def arc_pts(c, r: float, a0: float, a1: float, n: int = 12) -> np.ndarray:
    a = np.linspace(a0, a1, n)
    return np.stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)], 1)


def sector_pts(c, r0: float, r1: float, a0: float, a1: float, n: int = 10) -> np.ndarray:
    return np.concatenate([arc_pts(c, r1, a0, a1, n), arc_pts(c, r0, a1, a0, n)])


class Canvas:
    """Float RGB canvas. Shapes are drawn as 8-bit anti-aliased masks (``Batch``) and composited with a colour."""

    def __init__(self, size, bg):
        self.W, self.H = int(size[0]), int(size[1])
        self.img = np.empty((self.H, self.W, 3), np.float32)
        self.img[:] = np.asarray(bg, np.float32)

    def mask(self) -> np.ndarray:
        return np.zeros((self.H, self.W), np.uint8)

    def paint(self, mask: np.ndarray, color, alpha: float = 1.0) -> None:
        x, y, w, h = cv2.boundingRect(mask)
        if w == 0 or h == 0:
            return
        m = mask[y:y + h, x:x + w].astype(np.float32) * (alpha / 255.0)
        reg = self.img[y:y + h, x:x + w]
        reg += (np.asarray(color, np.float32) - reg) * m[..., None]

    def glow(self, mask: np.ndarray, color, sigma: float, gain: float = 0.6) -> None:
        """Soft halo: blurred mask blended toward ``color`` (works on dark and light canvases)."""
        s = max(1.0, sigma)
        f = 1 if s < 6 else 4
        small = cv2.resize(mask, (self.W // f, self.H // f), interpolation=cv2.INTER_AREA) if f > 1 else mask
        b = cv2.GaussianBlur(small.astype(np.float32) / 255.0, (0, 0), s / f)
        if f > 1:
            b = cv2.resize(b, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        a = np.clip(b * gain, 0, 1)[..., None]
        self.img += (np.asarray(color, np.float32) - self.img) * a

    def out(self) -> np.ndarray:
        return np.clip(self.img, 0, 1).astype(np.float32)


class Batch:
    """Collect shapes per colour, then flush in one composite per colour (keeps painting O(colours))."""

    def __init__(self, cv: Canvas):
        self.cv = cv; self.m: dict[tuple, tuple[np.ndarray, np.ndarray, float]] = {}

    def mask(self, color, alpha: float = 1.0, z: int = 0) -> np.ndarray:
        """Mask for (colour, alpha, z). Higher ``z`` is painted later (details over fills)."""
        c = np.asarray(color, np.float32)
        key = (z, tuple(np.round(c * 40).astype(int)), round(alpha, 2))
        if key not in self.m:
            self.m[key] = (self.cv.mask(), c, alpha)
        return self.m[key][0]

    def flush(self) -> None:
        for key in sorted(self.m, key=lambda k: k[0]):
            m, c, a = self.m[key]
            self.cv.paint(m, c, a)
        self.m = {}


# =====================================================================================================================
# Node styles + layered drawing
# =====================================================================================================================

NODE_STYLES = ("box", "pill", "dot", "ring", "square")


def draw_node(b: Batch, pal, style: str, c, size: float, col, hot: bool = False, canvas_col=None, aspect: float = 1.6) -> None:
    """Draw one node. ``size`` = half-height-ish unit in px. box/pill: filled body with an inner label bar;
    dot: solid disc; ring: outlined disc; square: filled rounded square."""
    x, y = c
    cc = pal.canvas if canvas_col is None else canvas_col
    fillc = pal.accent if hot else col
    if style in ("box", "pill"):
        hw, hh = size * aspect, size
        r = hh * (0.95 if style == "pill" else 0.28)
        m_rrect(b.mask(fillc), x - hw, y - hh, x + hw, y + hh, r)
        inner = cc + (fillc - cc) * (0.25 if not hot else 0.0)
        if hot:
            inner = pal.canvas + (pal.accent - pal.canvas) * 0.35
        if style == "box":
            m_rrect(b.mask(inner, 1.0, 1), x - hw * 0.62, y - hh * 0.2, x + hw * 0.62, y + hh * 0.2, hh * 0.12)
        else:
            m_disc(b.mask(inner, 1.0, 1), (x - hw + hh, y), hh * 0.38); m_disc(b.mask(inner, 1.0, 1), (x + hw - hh, y), hh * 0.38)
    elif style == "dot":
        m_disc(b.mask(fillc), c, size * (1.25 if hot else 1.0))
    elif style == "ring":
        r = size * (1.15 if hot else 1.0)
        m_disc(b.mask(cc, 1.0, 1), c, r); m_ring(b.mask(fillc, 1.0, 2), c, r * 0.82, max(2, size * 0.42))
    else:
        h = size
        m_rrect(b.mask(fillc), x - h, y - h, x + h, y + h, h * 0.22)


def layered_to_px(lay: Layered, view: tuple[float, float, float, float], flip: bool = False) -> np.ndarray:
    """Map normalised layout to px inside ``view`` = (x0, y0, x1, y1) (may exceed the frame => bleed)."""
    x0, y0, x1, y1 = view
    px = x0 + lay.x * (x1 - x0)
    py = y0 + lay.y * (y1 - y0)
    return np.stack([px, py], 1)
