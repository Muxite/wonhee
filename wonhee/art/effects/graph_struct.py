"""Data-structure family: B-tree, binary heap, hash table with chains, skip list. Each is drawn as a big all-over diagram that
bleeds off the frame (the root of a tree is off-frame left; a hash table's bucket column and a skip list's ends are cropped) with a
thin accent path (search / lookup) crossing the whole frame.

``ctx.data`` (optional, all structures): {"keys": [numbers]} to build the structure from real keys (inserted in order);
without it seeded random keys are used.
"""
from __future__ import annotations

import math

import numpy as np

from .. import graphkit as gk
from ..registry import effect


def _keys(ctx, n):
    rng = ctx.rng
    d = ctx.data if isinstance(ctx.data, dict) else {}
    if d.get("keys"):
        k = list(d["keys"])
        return k
    return list(rng.permutation(n * 6)[:n] + 1)


def _hot_edge_paint(cv, pal, hot_mask, ew, glow=0.5):
    cv.paint(hot_mask, pal.accent, 1.0)
    cv.glow(hot_mask, pal.accent, ew * 2.2, glow)


# ------------------------------------------------------------------------------------------------------------------
@effect("graph-btree", family="graph", tags=["tree", "graph", "all-over", "btree", "structure"],
        params={"fanout": (3, "children per node"), "levels": (3, "visible levels (root is off-frame left)"), "keys_per_node": (3, "key cells per node")})
def graph_btree(ctx):
    """B-tree levels as columns of key-slot nodes (filled slots = used keys), S-curve child links, a thin accent search path root to leaf across the frame; the root sits off-frame left. ctx.data: {"keys"}."""
    W, H = ctx.size; pal = ctx.palette; rng = ctx.rng
    f = int(ctx.p("fanout")); L = int(ctx.p("levels")); kk = int(ctx.p("keys_per_node"))
    cv = gk.Canvas((W, H), pal.canvas)
    # build levels: level 0 off-frame root with 2 children, then f-ary
    fans = [f] * (L - 1) + [2]            # last level: 2 children per node keeps the leaf column readable
    counts = [f]
    for i in range(1, L):
        counts.append(counts[-1] * fans[i])
    rows = [np.arange(c) for c in counts]
    colx = [W * (0.12 + 0.34 * i) for i in range(L)]
    colx = [x + W * 0.0 for x in colx]
    # leaves also get a trailing level of slots to the right (leaf chain cropped)
    pos = []
    ys = []
    for i, c in enumerate(counts):
        step = H / c
        pos.append([(colx[i], (j + 0.5) * step) for j in range(c)])
    nw = W * 0.22; nh = lambda c: min(H / c * 0.70, W * 0.1)
    b = gk.Batch(cv); hot = cv.mask(); ew = max(3.0, W * 0.0045)
    # hot path: random leaf, ancestors by integer division
    leaf = int(rng.integers(0, counts[-1])); hpath = [leaf]
    for i in range(L - 1, 0, -1):
        hpath.append(hpath[-1] // fans[i])
    hpath = hpath[::-1]
    # edges
    root_pos = (-W * 0.14, H * (0.35 + 0.3 * rng.random()))
    for j, (x, y) in enumerate(pos[0]):
        pts = gk.s_curve(root_pos, (x - nw * 0.5, y), True, 20)
        (gk.m_line(hot, pts, ew * 1.7) if hpath[0] == j else gk.m_line(b.mask(gk.tone(pal, 0.34)), pts, ew))
    for i in range(1, L):
        for j, (x, y) in enumerate(pos[i]):
            par = pos[i - 1][j // fans[i]]
            pts = gk.s_curve((par[0] + nw * 0.5, par[1]), (x - nw * 0.5, y), True, 20)
            if hpath[i] == j and hpath[i - 1] == j // fans[i]:
                gk.m_line(hot, pts, ew * 1.7)
            else:
                gk.m_line(b.mask(gk.tone(pal, 0.34)), pts, ew)
    # leaf continuation stubs to the right (leaf chain, cropped)
    for j, (x, y) in enumerate(pos[-1]):
        gk.m_line(b.mask(gk.tone(pal, 0.28)), np.array([[x + nw * 0.5, y], [W + nw, y]]), ew)
    b.flush()
    # nodes
    for i in range(L):
        c = counts[i]; h = nh(c)
        for j, (x, y) in enumerate(pos[i]):
            ishot = hpath[i] == j
            used = int(rng.integers(max(1, kk - 1), kk + 1)) if i < L - 1 else int(rng.integers(1, kk + 1))
            frame_col = pal.accent if ishot else gk.tone(pal, 0.40 + 0.04 * i)
            gk.m_rrect(b.mask(frame_col, 1.0, 0), x - nw / 2, y - h / 2, x + nw / 2, y + h / 2, h * 0.22)
            slot = nw / kk
            for s in range(kk):
                sx0 = x - nw / 2 + s * slot + h * 0.12; sx1 = x - nw / 2 + (s + 1) * slot - h * 0.12
                full = s < used
                if ishot:
                    col = pal.canvas + (pal.accent - pal.canvas) * (0.55 if full else 0.15)
                else:
                    col = gk.tone(pal, 0.60 if full else 0.08) if full else pal.canvas
                gk.m_rrect(b.mask(col, 1.0, 1), sx0, y - h * 0.32, sx1, y + h * 0.32, h * 0.12)
    b.flush()
    _hot_edge_paint(cv, pal, hot, ew)
    return cv.out()


# ------------------------------------------------------------------------------------------------------------------
@effect("graph-heap", family="graph", tags=["tree", "graph", "all-over", "heap", "structure"],
        params={"levels": (6, "tree levels (the root is off-frame)"), "show_array": (False, "array strip of node values across the bottom edge")})
def graph_heap(ctx):
    """Binary min-heap as a left-to-right tree: dot tone = value (heap order makes lightness rise with depth), links are S-curves; the sift-down path is the thin accent line. ctx.data: {"keys"}."""
    W, H = ctx.size; pal = ctx.palette; rng = ctx.rng
    L = int(ctx.p("levels"))
    n = 2 ** L - 1
    keys = sorted(_keys(ctx, n))[:n]
    # heap order = sorted by array index guarantees property; shuffle among equal depths
    vals = np.zeros(n)
    arr = np.sort(rng.random(n))
    for lv in range(L):
        a, b_ = 2 ** lv - 1, 2 ** (lv + 1) - 1
        seg = arr[a:b_].copy(); rng.shuffle(seg); arr[a:b_] = seg
    # repair to a valid heap by sifting
    for i in range(n // 2 - 1, -1, -1):
        j = i
        while True:
            l, r = 2 * j + 1, 2 * j + 2; m = j
            if l < n and arr[l] < arr[m]: m = l
            if r < n and arr[r] < arr[m]: m = r
            if m == j: break
            arr[j], arr[m] = arr[m], arr[j]; j = m
    vals = arr
    cv = gk.Canvas((W, H), pal.canvas)
    first = 1  # first drawn level index (levels above are off-frame)
    vis = list(range(first, L))
    colx = {lv: W * (0.05 + 0.90 * (k / (len(vis) - 1))) for k, lv in enumerate(vis)}
    def P(i):
        lv = int(math.floor(math.log2(i + 1))); j = i - (2 ** lv - 1)
        return (colx.get(lv, None), (j + 0.5) / (2 ** lv) * H, lv)
    ew = max(3.0, W * 0.0042); b = gk.Batch(cv); hot = cv.mask()
    # sift-down path
    path = [0]
    # start at the level-2 node
    i = int(rng.integers(3, 7))
    path = [i]
    while 2 * i + 1 < n:
        i = 2 * i + 1 + int(rng.integers(0, 2)); path.append(i)
    hp = set(zip(path[:-1], path[1:]))
    rad = lambda lv: min(H / (2 ** lv) * 0.40, W * 0.04)
    for i in range(n):
        lv = int(math.floor(math.log2(i + 1)))
        if lv < first:
            continue
        x, y, _ = P(i)
        for c in (2 * i + 1, 2 * i + 2):
            if c >= n: continue
            xc, yc, _ = P(c)
            pts = gk.s_curve((x, y), (xc, yc), True, 18)
            if (i, c) in hp: gk.m_line(hot, pts, ew * 1.6)
            else: gk.m_line(b.mask(gk.tone(pal, 0.30)), pts, ew * (1.0 if lv < L - 2 else 0.8))
        if lv == first:                     # parents are off-frame: stub from the left edge
            par = (i - 1) // 2; yp = (par + 0.5 - (2 ** (lv - 1) - 1)) / (2 ** (lv - 1)) * H
            pts = gk.s_curve((-W * 0.1, yp), (x, y), True, 18)
            gk.m_line(b.mask(gk.tone(pal, 0.30)), pts, ew)
    b.flush()
    inpath = set(path)
    for i in range(n):
        lv = int(math.floor(math.log2(i + 1)))
        if lv < first: continue
        x, y, _ = P(i); r = rad(lv)
        if i in inpath:
            gk.m_disc(b.mask(pal.accent, 1.0, 2), (x, y), r * 1.2)
        else:
            gk.m_disc(b.mask(pal.canvas, 1.0, 0), (x, y), r * 1.3)
            gk.m_disc(b.mask(gk.tone(pal, 0.66 - 0.42 * float(vals[i])), 1.0, 1), (x, y), r)
    b.flush()
    _hot_edge_paint(cv, pal, hot, ew)
    if ctx.p("show_array"):
        bw = W / 64.0; y0 = H - bw * 1.1
        for k in range(min(n, 64)):
            gk.m_rrect(b.mask(gk.tone(pal, 0.66 - 0.42 * float(vals[k])), 1.0, 1), k * bw + 2, y0, (k + 1) * bw - 2, H + bw, bw * 0.2)
        b.flush()
    return cv.out()


# ------------------------------------------------------------------------------------------------------------------
@effect("graph-hash", family="graph", tags=["structure", "graph", "all-over", "hash", "lanes"],
        params={"buckets": (15, "bucket rows"), "load": (9.0, "mean chain length")})
def graph_hash(ctx):
    """Separate-chaining hash table: a bucket column (cropped at the left) and a linked chain per row; cell tone = hash value, chain ends with a null bar; one lookup chain is the accent. ctx.data: {"keys"}."""
    W, H = ctx.size; pal = ctx.palette; rng = ctx.rng
    nb = int(ctx.p("buckets")); lh = H / (nb - 0.2)
    cv = gk.Canvas((W, H), pal.canvas)
    b = gk.Batch(cv); hot = cv.mask(); ew = max(3.0, W * 0.0042)
    cw = lh * 1.2; gapx = cw * 0.42
    hotb = int(rng.integers(0, nb))
    # bucket column
    for r in range(nb):
        y0 = -lh * 0.1 + r * lh
        gk.m_rrect(b.mask(gk.tone(pal, 0.22 + 0.06 * (r % 2)), 1.0, 0), -W * 0.02, y0 + lh * 0.08, W * 0.075, y0 + lh * 0.92, lh * 0.15)
        gk.m_rrect(b.mask(gk.tone(pal, 0.45), 1.0, 1), W * 0.045, y0 + lh * 0.25, W * 0.065, y0 + lh * 0.75, lh * 0.1)
    for r in range(nb):
        y0 = -lh * 0.1 + r * lh; yc = y0 + lh / 2
        n = int(rng.integers(max(2, int(ctx.p("load") * 0.5)), int(ctx.p("load") * 1.5) + 1)) + (3 if r == hotb else 0)
        x = W * 0.075
        ishot = r == hotb
        col_link = pal.accent if ishot else gk.tone(pal, 0.34)
        for i in range(n):
            xc0 = x + gapx; xc1 = xc0 + cw
            if xc0 > W + cw: break
            lm = hot if ishot else b.mask(col_link, 1.0, 0)
            gk.m_line(lm, np.array([[x, yc], [xc0, yc]]), ew * (1.6 if ishot else 1.0))
            v = float(rng.random())
            if ishot:
                gk.m_rrect(b.mask(pal.accent, 1.0, 1), xc0, y0 + lh * 0.12, xc1, y0 + lh * 0.88, lh * 0.18)
                gk.m_rrect(b.mask(pal.canvas + (pal.accent - pal.canvas) * 0.35, 1.0, 2), xc0 + cw * 0.18, y0 + lh * 0.36, xc1 - cw * 0.18, y0 + lh * 0.64, lh * 0.08)
            else:
                gk.m_rrect(b.mask(gk.tone(pal, 0.30 + 0.34 * v), 1.0, 1), xc0, y0 + lh * 0.12, xc1, y0 + lh * 0.88, lh * 0.18)
                gk.m_rrect(b.mask(pal.canvas + (gk.tone(pal, 0.30 + 0.34 * v) - pal.canvas) * 0.25, 1.0, 2), xc0 + cw * 0.18, y0 + lh * 0.36, xc1 - cw * 0.18, y0 + lh * 0.64, lh * 0.08)
            x = xc1
        # null terminator
        if x < W:
            gk.m_line(b.mask(col_link if not ishot else pal.accent, 1.0, 0) if not ishot else hot, np.array([[x, yc], [x + gapx * 0.6, yc]]), ew)
            gk.m_line(b.mask(col_link, 1.0, 0) if not ishot else hot, np.array([[x + gapx * 0.6, yc - lh * 0.22], [x + gapx * 0.6, yc + lh * 0.22]]), ew)
    b.flush()
    _hot_edge_paint(cv, pal, hot, ew)
    return cv.out()


# ------------------------------------------------------------------------------------------------------------------
@effect("graph-skiplist", family="graph", tags=["structure", "graph", "all-over", "skiplist"],
        params={"levels": (7, "levels"), "towers": (26, "towers across"), "p": (0.58, "promotion probability")})
def graph_skiplist(ctx):
    """Skip list: towers of random height joined by express lanes per level; the accent is the classic search path, a staircase from the top-left to the bottom-right across the whole frame. ctx.data: ignored."""
    W, H = ctx.size; pal = ctx.palette; rng = ctx.rng
    L = int(ctx.p("levels")); T = int(ctx.p("towers")); p = float(ctx.p("p"))
    cv = gk.Canvas((W, H), pal.canvas)
    dx = W / (T - 1.2); lh = H / (L - 0.2)
    h = np.minimum(L, 1 + rng.geometric(1 - p, T) - 1 + 0)
    h = np.clip(h, 1, L); h[0] = L; h[-1] = L
    X = lambda t: -dx * 0.1 + t * dx
    Y = lambda lv: H - (lv + 0.5) * lh * 0.98 + lh * 0.1
    b = gk.Batch(cv); hot = cv.mask(); ew = max(3.0, W * 0.0045)
    # search path for a target tower: classic descent from top-left
    target = int(rng.integers(T * 2 // 3, T - 1))
    path = []; cur = 0; lv = L - 1
    path.append((cur, lv))
    nxt = lambda c, l: next((t for t in range(c + 1, T) if h[t] > l), None)
    while True:
        n = nxt(cur, lv)
        if n is not None and n <= target:
            cur = n; path.append((cur, lv))
        elif lv > 0:
            lv -= 1; path.append((cur, lv))
        else:
            break
        if cur == target and lv == 0: break
    hp = set()
    for a, c in zip(path[:-1], path[1:]): hp.add((a, c))
    # horizontal express lanes
    for lv in range(L):
        ts = [t for t in range(T) if h[t] > lv]
        for a, c in zip(ts[:-1], ts[1:]):
            if ((a, lv), (c, lv)) in hp: gk.m_line(hot, np.array([[X(a), Y(lv)], [X(c), Y(lv)]]), ew * 1.7)
            else: gk.m_line(b.mask(gk.tone(pal, 0.26 + 0.03 * lv)), np.array([[X(a), Y(lv)], [X(c), Y(lv)]]), ew)
    for t in range(T):
        for lv in range(1, h[t]):
            if ((t, lv), (t, lv - 1)) in hp: gk.m_line(hot, np.array([[X(t), Y(lv)], [X(t), Y(lv - 1)]]), ew * 1.7)
            else: gk.m_line(b.mask(gk.tone(pal, 0.30)), np.array([[X(t), Y(lv)], [X(t), Y(lv - 1)]]), ew)
    b.flush()
    s = min(dx, lh) * 0.36
    onpath = {a for a in path}
    for t in range(T):
        for lv in range(h[t]):
            c = (X(t), Y(lv)); ish = (t, lv) in onpath
            if ish:
                gk.m_rrect(b.mask(pal.accent, 1.0, 2), c[0] - s * 1.15, c[1] - s * 1.15, c[0] + s * 1.15, c[1] + s * 1.15, s * 0.3)
            else:
                v = 0.40 + 0.04 * lv
                gk.m_rrect(b.mask(gk.tone(pal, v + 0.14), 1.0, 1), c[0] - s, c[1] - s, c[0] + s, c[1] + s, s * 0.3)
                gk.m_rrect(b.mask(pal.canvas + (gk.tone(pal, v + 0.14) - pal.canvas) * 0.25, 1.0, 2), c[0] - s * 0.45, c[1] - s * 0.45, c[0] + s * 0.45, c[1] + s * 0.45, s * 0.1)
    b.flush()
    _hot_edge_paint(cv, pal, hot, ew)
    return cv.out()
