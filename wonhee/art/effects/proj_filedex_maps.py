"""FileDex project effects: the two pictures a semantic file search is made of.

proj-filedex-treemap: squarified treemap of a REAL file tree (every file = a rounded tile, area ~ bytes**0.6, nested by
  directory, lightness by directory), tiling the whole frame. One query's result files get accent tiles joined by a thin
  glowing path (the "light path" of a search).
proj-filedex-embedding: the embedding-space view: one point per real file placed by its TF-IDF/SVD position, relaxed so the
  points spread evenly over the frame, links to each file's nearest neighbours by similarity, and the query's result path.

ctx.data = {"files": [{"path","size","x","y"}], "knn": [[j,..] per file], "result": [file index, ...]} (see prep_data.py of
the filedex card project). Without data a small synthetic tree is used."""
from __future__ import annotations

import numpy as np

from .. import graphkit as gk
from ..registry import effect


def _synthetic(rng):
    n = 160
    files = [{"path": f"d{rng.integers(0, 6)}/s{rng.integers(0, 4)}/f{i}.py", "size": int(np.exp(rng.normal(7, 1))),
              "x": float(rng.normal()), "y": float(rng.normal())} for i in range(n)]
    knn = [list(rng.choice(n, 3, replace=False)) for _ in range(n)]
    return {"files": files, "knn": knn, "result": [list(rng.choice(n, 6, replace=False)) for _ in range(3)]}


def _squarify(items, x, y, w, h, out):
    """items: [(weight, payload)] sorted desc, weights sum to w*h. Appends (payload, x0, y0, x1, y1)."""
    items = list(items)
    while items:
        short = min(w, h)
        row = [items[0]]; s = items[0][0]
        def worst(r, s):
            mx, mn = max(a for a, _ in r), min(a for a, _ in r)
            return max(short * short * mx / (s * s), s * s / (short * short * mn))
        i = 1
        while i < len(items):
            r2 = row + [items[i]]; s2 = s + items[i][0]
            if worst(r2, s2) <= worst(row, s):
                row, s = r2, s2; i += 1
            else:
                break
        items = items[i:]
        t = s / short
        pos = 0.0
        for a, p in row:
            L = a / t
            if w >= h:
                out.append((p, x, y + pos, x + t, y + pos + L))
            else:
                out.append((p, x + pos, y, x + pos + L, y + t))
            pos += L
        if w >= h:
            x += t; w -= t
        else:
            y += t; h -= t


def _tree(files):
    root = {}
    for i, f in enumerate(files):
        node = root
        parts = f["path"].split("/")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = i
    return root


def _weight(node, files):
    if isinstance(node, int):
        return max(files[node]["size"], 200) ** 0.6
    return sum(_weight(c, files) for c in node.values())


def _layout(node, files, x0, y0, x1, y1, pad, depth, out, top):
    kids = sorted(node.items(), key=lambda kv: -_weight(kv[1], files))
    tot = sum(_weight(v, files) for _, v in kids)
    area = (x1 - x0) * (y1 - y0)
    items = [(_weight(v, files) / tot * area, (k, v)) for k, v in kids]
    cells = []
    _squarify(items, x0, y0, x1 - x0, y1 - y0, cells)
    for (k, v), a, b, c, d in cells:
        t = top if depth > 0 else k
        if isinstance(v, int):
            out.append((v, a, b, c, d, t))
        else:
            p = pad * (0.7 if depth else 0.0) if (c - a) > pad * 4 and (d - b) > pad * 4 else 0
            _layout(v, files, a + p, b + p, c - p, d - p, pad, depth + 1, out, t)


@effect("proj-filedex-treemap", family="feature", tags=["filedex", "treemap", "all-over", "real-data", "file-tree"],
        params={"gap": (0.0030, "tile gap, fraction of width"), "bleed": (0.03, "layout overscan beyond the frame"),
                "round": (0.22, "tile corner radius, fraction of tile"), "paths": (1, "draw the query result path (0 = tiles only)"), "dim": (1.0, "tile lightness multiplier toward canvas")})
def proj_filedex_treemap(ctx):
    W, H = ctx.size; pal = ctx.palette; rng = ctx.rng
    d = ctx.data if isinstance(ctx.data, dict) and "files" in ctx.data else _synthetic(rng)
    files = d["files"]
    bl = float(ctx.p("bleed"))
    X0, Y0, X1, Y1 = -bl * W, -bl * H, (1 + bl) * W, (1 + bl) * H
    out = []
    # order top-level groups deterministically: seed shuffles which siblings sit where only via weights; real layout otherwise.
    _layout(_tree(files), files, X0, Y0, X1, Y1, W * 0.004, 0, out, "")
    tops = sorted({o[5] for o in out}); ng = max(len(tops), 1)
    cv = gk.Canvas((W, H), pal.canvas)
    gap = max(2.0, W * float(ctx.p("gap")))
    b = gk.Batch(cv)
    cen = {}
    for i, a, bb, c, dd, t in out:
        cen[i] = ((a + c) / 2, (bb + dd) / 2)
        if c < 0 or dd < 0 or a > W or bb > H:
            continue
        # lightness: directory family base, per-file jitter (a deterministic hash of the path)
        h = (hash_str(files[i]["path"]) % 997) / 997.0
        base = 0.30 + 0.20 * ((tops.index(t)) % 4) / 3.0
        col = gk.tone(pal, float(ctx.p("dim")) * np.clip(base + 0.12 * (h - 0.5) * 2, 0.22, 0.62))
        r = min(c - a, dd - bb) * float(ctx.p("round"))
        gk.m_rrect(b.mask(col, 1.0, 0), a + gap / 2, bb + gap / 2, c - gap / 2, dd - gap / 2, r)
        ext = files[i]["path"].rsplit(".", 1)[-1]
        if ext in ("md", "yaml", "json", "toml", "txt") and min(c - a, dd - bb) > W * 0.02:
            s = min(c - a, dd - bb) * 0.34
            gk.m_rrect(b.mask(gk.tone(pal, 0.14), 1.0, 1), a + gap + s * 0.5, bb + gap + s * 0.5, c - gap - s * 0.5, dd - gap - s * 0.5, r * 0.5, 2)
    b.flush()
    if not int(ctx.p("paths")):
        return cv.out()
    rect = {o[0]: o for o in out}
    hb = gk.Batch(cv); hm = cv.mask(); any_ = False
    for res in _groups(d):
        pts = [np.array(cen[i]) for i in res]
        order = _chain(pts)
        path = np.array([pts[j] for j in order])
        for j in order:
            o = rect[res[j]]
            gk.m_rrect(hb.mask(pal.accent, 1.0, 2), o[1] + gap / 2, o[2] + gap / 2, o[3] - gap / 2, o[4] - gap / 2, min(o[3] - o[1], o[4] - o[2]) * float(ctx.p("round")))
            gk.m_rrect(hm, o[1], o[2], o[3], o[4], 0)
        line = gk.catmull(path, 10)
        gk.m_line(hb.mask(pal.canvas, 1.0, 3), line, W * 0.0085)
        gk.m_line(hb.mask(gk.tone(pal, 1.0), 1.0, 4), line, W * 0.0040)
        gk.m_line(hm, line, W * 0.006); any_ = True
    if any_:
        hb.flush(); cv.glow(hm, pal.accent, W * 0.010, 0.5)
    return cv.out()


def _groups(d):
    r = d.get("result", [])
    if r and not isinstance(r[0], (list, tuple)):
        r = [r]
    return [[int(i) for i in g] for g in r]


def _chain(pts):
    order = [int(np.argmin([p[0] for p in pts]))]
    left = set(range(len(pts))) - set(order)
    while left:
        j = min(left, key=lambda k: np.hypot(*(pts[k] - pts[order[-1]])))
        order.append(j); left.remove(j)
    return order


def hash_str(s):
    h = 2166136261
    for ch in s:
        h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return h


def _relax(P, rng, iters=220):
    """Spread points evenly in the unit-aspect box (Lloyd-like repulsion with a pull toward the SVD position)."""
    P0 = P.copy(); n = len(P)
    for it in range(iters):
        D = P[:, None] - P[None]
        r2 = (D ** 2).sum(-1) + 1e-4
        F = (D / r2[..., None] ** 1.0 * (r2 < 0.06)[..., None]).sum(1)
        P = P + 0.0016 * F / (1 + np.abs(F).max()) * 8 + 0.02 * (P0 - P)
        P = P.clip(-0.05, 1.05 * np.array([1.0, 1.0]))
    return P


@effect("proj-filedex-embedding", family="graph", tags=["filedex", "embedding", "all-over", "real-data", "similarity"],
        params={"links": (3, "nearest neighbours linked per file"), "spread": (1.0, "0 = raw SVD positions, 1 = fully relaxed")})
def proj_filedex_embedding(ctx):
    W, H = ctx.size; pal = ctx.palette; rng = ctx.rng
    d = ctx.data if isinstance(ctx.data, dict) and "files" in ctx.data else _synthetic(rng)
    files = d["files"]; n = len(files)
    P = np.array([[f["x"], f["y"]] for f in files], float)
    # rank-uniform marginals first (removes the dense core of an SVD blob), then relax
    for k in range(2):
        P[:, k] = (np.argsort(np.argsort(P[:, k])) + 0.5) / n
    P[:, 0] *= W / H
    sp = float(ctx.p("spread"))
    asp = W / H
    Q = P.copy() / np.array([asp, 1.0])
    Q = _relax(Q, rng) * sp + Q * (1 - sp)
    px = np.stack([(-0.04 + Q[:, 0] * 1.08) * W, (-0.04 + Q[:, 1] * 1.08) * H], 1)
    cv = gk.Canvas((W, H), pal.canvas)
    tops = sorted({f["path"].split("/")[0] + "/" + f["path"].split("/")[1] for f in files})
    eb = gk.Batch(cv)
    nl = int(ctx.p("links"))
    for i, nb in enumerate(d["knn"]):
        for j in nb[:nl]:
            if np.hypot(*(px[i] - px[j])) < W * 0.13:
                gk.m_line(eb.mask(gk.tone(pal, 0.33), 1.0), np.array([px[i], px[j]]), max(2.0, W * 0.0028))
    eb.flush()
    nbch = gk.Batch(cv)
    for i, f in enumerate(files):
        t = f["path"].split("/")[0] + "/" + f["path"].split("/")[1]
        col = gk.tone(pal, 0.40 + 0.16 * (tops.index(t) % 4) / 3.0)
        r = W * (0.0055 + 0.0035 * min(2.0, (f["size"] / 4000.0) ** 0.5))
        gk.m_disc(nbch.mask(pal.canvas, 1.0, 0), px[i], r * 1.5)
        gk.m_disc(nbch.mask(col, 1.0, 1), px[i], r)
    nbch.flush()
    hb = gk.Batch(cv); hm = cv.mask(); any_ = False
    for res in _groups(d):
        pts = [px[i] for i in res]
        order = _chain(pts)
        path = np.array([pts[j] for j in order])
        line = gk.catmull(path, 10)
        gk.m_line(hb.mask(gk.tone(pal, 0.75), 1.0, 1), line, W * 0.0050)
        gk.m_line(hm, line, W * 0.006); any_ = True
        for j in order:
            gk.m_disc(hb.mask(pal.accent, 1.0, 2), pts[j], W * 0.0105); gk.m_disc(hm, pts[j], W * 0.011)
            gk.m_ring(hb.mask(gk.tone(pal, 0.8), 1.0, 2), pts[j], W * 0.0185, max(2, W * 0.0025))
    if any_:
        hb.flush(); cv.glow(hm, pal.accent, W * 0.012, 0.55)
    return cv.out()
