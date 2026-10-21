"""Prototype 2: the structures redone on palette roles (canvas / structure / edge / accent / ink).

Accent is reserved for the hot path or top few percent of elements (about 10%), structure carries most of the ink, and
edges are dimmer than vertices. Counts follow a detail factor derived from the CSS display width.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from structures import Cv
from art_proto import field, fingerprint  # noqa: F401
from wonhee.edit import fx
from wonhee.edit.viz import palette as pal


def hx(c):
    return tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))


def detail(css_w):
    return .42 if css_w < 320 else .72 if css_w < 700 else 1.0


class Cv2(Cv):
    def __init__(self, W, H):
        super().__init__(W, H)
        self.hot = []

    def poly(self, pts, col, a=255):
        self.d.polygon([self._p(p) for p in pts], fill=col + (a,))

    def hotdot(self, p, r, col):
        self.hot.append((p, r))
        self.dot(p, r, col)


def sample(fp, rng, i, n):
    wk = fp.get("weeks")
    if wk is not None and len(wk) > 1:
        return float(np.interp(i / max(n - 1, 1), np.linspace(0, 1, len(wk)), wk))
    return float(rng.random())


# ---- drawers: (cv, rng, fp, R, k) ----------------------------------------------------------------------------------
def graph(cv, rng, fp, R, k, data=None):
    W, H, u = cv.W, cv.H, cv.u
    n = int(np.clip(W * H / (u * u * 100) * .30 * k, 14, 110))
    m = 3 * u
    d_min = math.sqrt((W - 2 * m) * (H - 2 * m) / n) * .66
    P = []
    for _ in range(n * 40):
        p = (rng.uniform(m, W - m), rng.uniform(m, H - m))
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 > d_min ** 2 for q in P):
            P.append(p)
        if len(P) >= n:
            break
    P = np.array(P); n = len(P)
    D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
    E = set()
    for i in range(n):
        for j in np.argsort(D[i])[1:4]:
            E.add((min(i, int(j)), max(i, int(j))))
    deg = np.zeros(n)
    for i, j in E:
        deg[i] += 1; deg[j] += 1
    hubs = set(np.argsort(-deg)[:max(2, n // 10)].tolist())
    for i, j in E:
        mid = ((P[i, 0] + P[j, 0]) / 2 + rng.normal(0, u * .7), (P[i, 1] + P[j, 1]) / 2 + rng.normal(0, u * .7))
        hot = i in hubs and j in hubs
        cv.bez(tuple(P[i]), mid, tuple(P[j]), R["accent"] if hot else R["edge"], .45 if hot else .32)
    for i in range(n):
        r = .75 + .2 * deg[i] + .9 * sample(fp, rng, i, n)
        (cv.hotdot if i in hubs else cv.dot)(tuple(P[i]), r * 1.25 if i in hubs else r, R["accent"] if i in hubs else R["structure"])


def _treegeo(cv, rng, depth, heap, horiz, m):
    W, H = cv.W, cv.H
    long_, short = (W, H) if horiz else (H, W)
    nodes = []

    def build(l):
        i = len(nodes); nodes.append({"l": l, "kids": [], "par": None})
        if l < depth - 1:
            for _ in range(2):
                if heap or rng.random() < .82 or not nodes[i]["kids"]:
                    c = build(l + 1); nodes[c]["par"] = i; nodes[i]["kids"].append(c)
        return i
    build(0)
    leaves = [i for i, n in enumerate(nodes) if not n["kids"]]
    for r, i in enumerate(leaves):
        nodes[i]["c"] = (r + .5) / len(leaves)
    for i in reversed(range(len(nodes))):
        if nodes[i]["kids"]:
            nodes[i]["c"] = float(np.mean([nodes[c]["c"] for c in nodes[i]["kids"]]))
    xy = (lambda p: p) if horiz else (lambda p: (p[1], p[0]))
    pos = {i: xy((m + (long_ - 2 * m) * n["l"] / (depth - 1), m + (short - 2 * m) * n["c"])) for i, n in enumerate(nodes)}
    return nodes, pos, leaves


def tree(cv, rng, fp, R, k, heap=False, data=None):
    W, H, u = cv.W, cv.H, cv.u
    horiz = W >= H
    long_, short = (W, H) if horiz else (H, W)
    m = 4 * u
    depth = int(np.clip(round(long_ / (u * 17) * (k ** .5)), 3, 7))
    if heap:
        depth = max(3, min(int(np.clip(round(long_ / (u * 20)), 3, 5)), int(math.log2(max(short / (u * 7), 2))) + 1))
    nodes, pos, leaves = _treegeo(cv, rng, depth, heap, horiz, m)
    hot = set(); i = leaves[int(rng.integers(0, len(leaves)))]
    while i is not None:
        hot.add(i); i = nodes[i]["par"]
    for i, n in enumerate(nodes):
        for c in n["kids"]:
            a, b = pos[i], pos[c]; mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            p1 = (mid[0], a[1]) if horiz else (a[0], mid[1]); p2 = (mid[0], b[1]) if horiz else (b[0], mid[1])
            h = i in hot and c in hot
            cv.curve([a, p1, p2, b], R["accent"] if h else R["edge"], .55 if h else .38)
    for i, n in enumerate(nodes):
        p = pos[i]; s = sample(fp, rng, i, len(nodes)); col = R["accent"] if i in hot else R["structure"]
        if heap:
            w = (1.4 + 1.2 * s) * 1.4
            cv.box(p[0] - w * 1.4 * u, p[1] - w * u, p[0] + w * 1.4 * u, p[1] + w * u, fill=col, r=.5)
            cv.box(p[0] - w * .8 * u, p[1] - .2 * w * u, p[0] + w * .8 * u, p[1] + .2 * w * u, fill=R["canvas"], r=.2)
        else:
            cv.dot(p, 1.0 + 1.3 * s, col)


def dag(cv, rng, fp, R, k, data=None):
    W, H, u = cv.W, cv.H, cv.u
    horiz = W >= H
    long_, short = (W, H) if horiz else (H, W)
    m = 5 * u
    L = int(np.clip(round(long_ / (u * 18) * (k ** .4)), 3, 9))
    layers = []
    for l in range(L):
        n = int(np.clip(rng.integers(2, 6) * short / (u * 60) * (.5 + k / 2) + 1, 2, 7))
        layers.append([(m + (long_ - 2 * m) * l / (L - 1), m + (short - 2 * m) * (j + .5) / n + rng.normal(0, u * .8)) for j in range(n)])
    xy = (lambda p: p) if horiz else (lambda p: (p[1], p[0]))
    crit = [int(rng.integers(0, len(layers[l]))) for l in range(L)]
    for l in range(L - 1):
        for j, p in enumerate(layers[l]):
            for t in {int(rng.integers(0, len(layers[l + 1]))) for _ in range(1 + int(rng.random() < .5))} | ({crit[l + 1]} if crit[l] == j else set()):
                hot = crit[l] == j and crit[l + 1] == t
                cv.s_curve(xy(p), xy(layers[l + 1][t]), R["accent"] if hot else R["edge"], .65 if hot else .3, horizontal=horiz)
    for l, lay in enumerate(layers):
        for j, p in enumerate(lay):
            q = xy(p); hot = crit[l] == j; s = sample(fp, rng, l * 7 + j, L * 7)
            hw, hh = 2.2 * u * (.7 + s * .5), 1.1 * u
            cv.box(q[0] - hw, q[1] - hh, q[0] + hw, q[1] + hh, fill=R["accent"] if hot else R["canvas"], line=R["accent"] if hot else R["structure"], lw=.4, r=.5)


def radial(cv, rng, fp, R, k, data=None):
    W, H, u = cv.W, cv.H, cv.u
    cx, cy = W / 2, H / 2; rx, ry = W / 2 - 4 * u, H / 2 - 4 * u
    n_mid = int(np.clip(W * H / (u * u * 100) * .12 * k, 6, 24)); n_out = int(np.clip(W * H / (u * u * 100) * .36 * k, 18, 84))
    mid = [(cx + rx * .42 * math.cos(a), cy + ry * .42 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, n_mid, endpoint=False)]
    out = [(cx + rx * (.86 + .14 * rng.random()) * math.cos(a), cy + ry * (.86 + .14 * rng.random()) * math.sin(a)) for a in np.linspace(0, 2 * math.pi, n_out, endpoint=False)]
    hot_j = int(rng.integers(0, n_mid))
    for i, o in enumerate(out):
        j = int(round(i * n_mid / n_out)) % n_mid
        cv.bez(o, (cx + (o[0] - cx) * .45, cy + (o[1] - cy) * .45), mid[j], R["accent"] if j == hot_j else R["edge"], .3 if j == hot_j else .2)
    for j, p in enumerate(mid):
        cv.bez(p, ((p[0] + cx) / 2 + rng.normal(0, u), (p[1] + cy) / 2 + rng.normal(0, u)), (cx, cy), R["accent"] if j == hot_j else R["edge"], .5 if j == hot_j else .35)
    for i, o in enumerate(out):
        j = int(round(i * n_mid / n_out)) % n_mid
        cv.dot(o, .55 + .6 * sample(fp, rng, i, n_out), R["accent"] if j == hot_j else R["structure"])
    for j, p in enumerate(mid):
        (cv.hotdot if j == hot_j else cv.dot)(p, 1.5, R["accent"] if j == hot_j else R["structure"])
    cv.hotdot((cx, cy), 3.0, R["accent"])


def btree(cv, rng, fp, R, k, data=None):
    W, H, u = cv.W, cv.H, cv.u
    horiz = W > H * 1.6
    A, B = (H, W) if horiz else (W, H); m = 4 * u
    L = int(np.clip(round(B / (u * 28)), 2, 4)); fan = int(np.clip(round(A / (u * 28) * (k ** .5)), 2, 5)); keys = 4
    blocks = [[.5]] + [[(i + .5) / c for i in range(c)] for c in [min(fan ** l, int(A / (u * 11))) for l in range(1, L)]]
    pos = lambda c, l: (m + (A - 2 * m) * c, m + (B - 2 * m) * (l + .5) / L)
    xy = (lambda p: (p[1], p[0])) if horiz else (lambda p: p)
    path = [int(rng.integers(0, len(b))) for b in blocks]
    for l in range(L - 1):
        for i, c in enumerate(blocks[l]):
            for kk in range(len(blocks[l + 1])):
                if int(kk * len(blocks[l]) / len(blocks[l + 1])) == i:
                    hot = path[l] == i and path[l + 1] == kk
                    cv.s_curve(xy(pos(c, l)), xy(pos(blocks[l + 1][kk], l + 1)), R["accent"] if hot else R["edge"], .5 if hot else .3, horizontal=horiz)
    for l in range(L):
        for i, c in enumerate(blocks[l]):
            p = pos(c, l); w = min((A - 2 * m) / len(blocks[l]) * .86, u * 7 * keys); h = 5 * u; hot = path[l] == i
            for kk in range(keys):
                x0 = p[0] - w / 2 + w * kk / keys; x1 = p[0] - w / 2 + w * (kk + 1) / keys
                a = xy((x0, p[1] - h / 2)); b = xy((x1 - .3, p[1] + h / 2))
                full = sample(fp, rng, i * 3 + kk + l * 31, 97) > .35
                col = R["accent"] if hot else R["structure"]
                cv.box(min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]), fill=col if full else None, line=col, lw=.3, r=.3)


def hashtable(cv, rng, fp, R, k, data=None):
    W, H, u = cv.W, cv.H, cv.u
    horiz = W >= H; A, B = (H, W) if horiz else (W, H); m = 4 * u
    nb = int(np.clip(round((A - 2 * m) / (u * 6.4) * (k ** .4)), 4, 16))
    xy = (lambda a, b: (b, a)) if horiz else (lambda a, b: (a, b))
    max_chain = int(np.clip((B - 2 * m - 6 * u) / (u * 9), 2, 9))
    chains = [int(rng.integers(0, max_chain + 1)) if rng.random() < .8 else 0 for _ in range(nb)]
    hot = int(np.argmax(chains))
    for i in range(nb):
        a = m + (A - 2 * m) * (i + .5) / nb
        cv.box(*(xy(a - 2.4 * u, m) + xy(a + 2.4 * u, m + 4.8 * u)), fill=R["ink"], r=.4)
        prev = xy(a, m + 4.8 * u)
        for kk in range(chains[i]):
            q = xy(a + rng.normal(0, .3 * u), m + 8 * u + (kk + 1) * (B - 2 * m - 8 * u) / (max_chain + .3))
            cv.line(prev, q, R["accent"] if i == hot else R["edge"], .4 if i == hot else .3)
            cv.box(q[0] - 2.5 * u, q[1] - 1.6 * u, q[0] + 2.5 * u, q[1] + 1.6 * u, fill=R["accent"] if i == hot else R["structure"], r=.4)
            prev = q


def skiplist(cv, rng, fp, R, k, data=None):
    W, H, u = cv.W, cv.H, cv.u
    horiz = W >= H; A, B = (W, H) if horiz else (H, W); m = 4 * u
    L = int(np.clip(round((B - 2 * m) / (u * 13)), 3, 8)); n = int(np.clip(round((A - 2 * m) / (u * 10) * (k ** .5)), 6, 22))
    xy = (lambda a, b: (a, b)) if horiz else (lambda a, b: (b, a))
    hs = [1 + sum(rng.random() < .5 ** kk for kk in range(1, L)) for _ in range(n)]; hs[0] = hs[-1] = L
    target = int(rng.integers(n // 2, n))
    for lv in range(L):
        b = m + (B - 2 * m) * (1 - lv / max(L - 1, 1)); idx = [i for i in range(n) if hs[i] > lv]
        for a_i, b_i in zip(idx, idx[1:]):
            hot = b_i <= target
            cv.line(xy(m + (A - 2 * m) * a_i / (n - 1), b), xy(m + (A - 2 * m) * b_i / (n - 1), b), R["accent"] if (hot and lv == L - 1) else R["edge"], .4)
        for i in idx:
            q = xy(m + (A - 2 * m) * i / (n - 1), b)
            cv.box(q[0] - 2 * u, q[1] - 2 * u, q[0] + 2 * u, q[1] + 2 * u, fill=R["accent"] if i == target else R["structure"], r=.5)


def ring(cv, rng, fp, R, k, data=None):
    W, H, u = cv.W, cv.H, cv.u; m = 3 * u
    cell = max(2.6 * u, min(W, H) / (9 * (.55 + k * .45)))
    cn, rn = max(4, int((W - 2 * m) / (cell * 1.2))), max(3, int((H - 2 * m) / (cell * 1.2)))
    cw, ch = (W - 2 * m) / cn, (H - 2 * m) / rn; total = cn * rn
    head = int(rng.integers(0, total)); size = int(total * rng.uniform(.35, .6)); filled = {(head + kk) % total for kk in range(size)}
    order = []
    for r in range(rn):
        order += [(r, c) for c in (range(cn) if r % 2 == 0 else range(cn - 1, -1, -1))]
    for idx, (r, c) in enumerate(order):
        x0, y0 = m + c * cw, m + r * ch; on = idx in filled
        edge_cell = idx in (head, (head + size - 1) % total)
        s = sample(fp, rng, idx, total)
        cv.box(x0 + cw * .08, y0 + ch * .08, x0 + cw * .92, y0 + ch * .92, fill=(R["accent"] if edge_cell else R["structure"]) if on else None,
               line=R["accent"] if edge_cell else (R["structure"] if on else R["edge"]), lw=.3, r=.5, a=255 if on else 200)


def matrix(cv, rng, fp, R, k, data=None):
    W, H, u = cv.W, cv.H, cv.u; m = 3 * u
    cell = max(1.6 * u, min(W, H) / (22 * (.5 + k * .5)))
    nx, ny = max(6, int((W - 2 * m) / cell)), max(4, int((H - 2 * m) / cell))
    cw, ch = (W - 2 * m) / nx, (H - 2 * m) / ny
    blocks = sorted(rng.integers(0, max(nx, ny), 3).tolist()); hot_block = int(rng.integers(0, 4))
    for j in range(ny):
        for i in range(nx):
            gi = sum(i * ny / nx >= b for b in blocks); gj = sum(j >= b for b in blocks)
            same = gi == gj; near = abs(i * ny / nx - j) < 1.2
            pr = max(.62 if same else .06, .9 if near else 0)
            c = (m + (i + .5) * cw, m + (j + .5) * ch)
            if rng.random() < pr:
                s = sample(fp, rng, i + j * nx, nx * ny)
                cv.dot(c, (.28 + .24 * s) * min(cw, ch) / u, R["accent"] if (same and gj == hot_block) else R["structure"])
            else:
                cv.dot(c, .12 * min(cw, ch) / u, R["edge"], 150)


def curves(cv, rng, fp, R, k, data=None):
    """Streamgraph from real series: stacked bands rescaled to fill the box height at every x."""
    W, H, u = cv.W, cv.H, cv.u
    series = data if data is not None else [np.abs(np.sin(np.linspace(0, 6, 200) * (1 + i * .7) + i)) + .2 for i in range(4)]
    xs = np.linspace(0, W, 160)
    S = np.array([np.interp(xs, np.linspace(0, W, len(s)), cv2.GaussianBlur(np.asarray(s, np.float32)[None, :], (0, 0), max(1, len(s) / 90))[0]) for s in series])
    S = (S - S.min(1, keepdims=True)) / np.maximum(S.max(1, keepdims=True) - S.min(1, keepdims=True), 1e-6) + .35  # comparable bands
    S = S / S.sum(0, keepdims=True)
    base = np.zeros_like(xs); hot = 1
    cols = [R["structure"], R["accent"], R["edge"], R["structure"]]
    alphas = [255, 255, 255, 170]
    for bi, s in enumerate(S):
        top = base + s
        pts = [(x, y * H) for x, y in zip(xs, base)] + [(x, y * H) for x, y in zip(xs[::-1], top[::-1])]
        gap = .15 * u / H
        cv.poly(pts, R["accent"] if bi == hot else R["structure"] if bi != 2 else R["edge"], alphas[bi])
        cv.curve([(x, y * H) for x, y in zip(xs, top)], R["canvas"], .35)
        base = top


def cascade(cv, rng, fp, R, k, data=None):
    """Sankey-style split (after mirage branch.py): one bar splits level after level; the hot branch is accent."""
    W, H, u = cv.W, cv.H, cv.u
    horiz = W >= H; L = int(np.clip(round((W if horiz else H) / (u * 24) * (k ** .5)), 3, 6)); m = 3 * u
    bar = 3 * u; gap = 1.0 * u
    nodes = [[(0.0, 1.0, 0)]]  # (start, size, parent)
    for l in range(1, L):
        lvl = []
        for pi, (s0, sz, _) in enumerate(nodes[-1]):
            parts = rng.dirichlet(np.ones(int(rng.integers(2, 4))) * 4) * sz  # alpha 4: no sliver splits
            c = s0
            for p in parts:
                lvl.append((c, float(p), pi)); c += float(p)
        nodes.append(lvl)
    hot = [int(rng.integers(0, len(nodes[-1])))]
    for l in range(L - 1, 0, -1):
        hot.append(nodes[l][hot[-1]][2])
    hot = hot[::-1]  # hot[l] = index at level l
    ext = (H if horiz else W) - 2 * m; lon = (W if horiz else H) - 2 * m
    def box(l, s0, sz):
        x = m + (lon - bar) * l / (L - 1)
        a, b = m + ext * s0, m + ext * (s0 + sz)
        return ((x, a, x + bar, b) if horiz else (a, x, b, x + bar))
    for l in range(1, L):
        for i, (s0, sz, pi) in enumerate(nodes[l]):
            ps0, psz, _ = nodes[l - 1][pi]
            # ribbon from the parent's matching slice to this node
            off = sum(n[1] for n in nodes[l] if n[2] == pi and n is not nodes[l][i] and nodes[l].index(n) < i)
            x0 = m + (lon - bar) * (l - 1) / (L - 1) + bar; x1 = m + (lon - bar) * l / (L - 1)
            a0, a1 = m + ext * (ps0 + off), m + ext * (ps0 + off + sz); b0, b1 = m + ext * s0, m + ext * (s0 + sz)
            isb = hot[l] == i and hot[l - 1] == pi
            P = [(x0, a0), (x1, b0), (x1, b1), (x0, a1)] if horiz else [(a0, x0), (b0, x1), (b1, x1), (a1, x0)]
            n = 14; top = [((1 - t) * x0 + t * x1, (1 - (3 * t * t - 2 * t ** 3)) * a0 + (3 * t * t - 2 * t ** 3) * b0) for t in np.linspace(0, 1, n)]
            bot = [((1 - t) * x0 + t * x1, (1 - (3 * t * t - 2 * t ** 3)) * a1 + (3 * t * t - 2 * t ** 3) * b1) for t in np.linspace(0, 1, n)][::-1]
            poly = top + bot if horiz else [(y, x) for x, y in top + bot]
            cv.poly(poly, R["accent"] if isb else R["edge"], 170 if isb else 90)
    for l in range(L):
        for i, (s0, sz, _) in enumerate(nodes[l]):
            x0, y0, x1, y1 = box(l, s0, max(sz - .004, .004))
            cv.box(x0, y0, x1, y1, fill=R["accent"] if hot[l] == i else R["structure"], r=.3)


DRAW = {"graph": graph, "tree": tree, "heap": lambda *a, **kw: tree(*a, heap=True, **kw), "dag": dag, "radial": radial, "btree": btree,
        "hash": hashtable, "skip": skiplist, "ring": ring, "matrix": matrix, "curves": curves, "cascade": cascade}


# ---- bases: the canvas, and nothing muddier -----------------------------------------------------------------------
def base_image(kind, R, W, H, seed):
    cv_ = np.array(hx(R["canvas"])[::-1], np.float32)
    im = np.empty((H, W, 3), np.float32); im[:] = cv_
    ed = np.array(hx(R["edge"])[::-1], np.float32)
    if kind == "grid":
        step = max(8, round(min(W, H) / 12))
        for x in range(0, W, step): im[:, x] = im[:, x] * .9 + ed * .1
        for y in range(0, H, step): im[y] = im[y] * .9 + ed * .1
    elif kind == "wash":  # one hue, a soft pool of structure colour at most 6% from the canvas
        rng = np.random.default_rng(seed); cx, cy = rng.uniform(.25, .75) * W, rng.uniform(.3, .7) * H
        yy, xx = np.mgrid[:H, :W].astype(np.float32)
        g = np.exp(-(((xx - cx) / (W * .55)) ** 2 + ((yy - cy) / (H * .7)) ** 2))[..., None]
        im = im * (1 - .10 * g) + np.array(hx(R["structure"])[::-1], np.float32) * (.10 * g)
    elif kind == "halftone":  # monochrome screen: structure dots on canvas, driven by a smooth field
        f = field({"weeks": None, "langs": [], "slug": "x"}, W, H, seed, calm=.9)
        g = (f * 255).astype(np.uint8); src = np.dstack([g, g, g])
        im = fx.halftone(src, ink=R["structure"], paper=R["canvas"], amount=.85, angle=30, cell=max(5, round(max(W, H) / 70))).astype(np.float32)
        im = im * .55 + cv_ * .45  # keep it a texture under the structure
    return im.clip(0, 255).astype(np.uint8)


def render(structure, base, fp, W, H, seed, domain, polarity, harmony="mono", css_w=None, data=None, grain=0.0):
    R = pal.for_domain(domain, polarity, harmony).__dict__
    k = detail(css_w if css_w is not None else W / 2)
    rng = np.random.default_rng(seed)
    cv = Cv2(W, H)
    RC = {kk: hx(v) for kk, v in R.items()}
    DRAW[structure](cv, rng, fp, RC, k, data=data)
    img = base_image(base, R, W, H, seed)
    out = Image.fromarray(img[..., ::-1]).convert("RGBA")
    if cv.hot and base in ("halo", "glow", "wash", "grid", "flat", "halftone") and harmony == "split":
        g = Image.new("RGBA", (W, H), (0, 0, 0, 0)); gd = ImageDraw.Draw(g)
        ac = RC["accent"]
        for (p, r) in cv.hot:
            rr = r * cv.u * 5
            gd.ellipse([p[0] - rr, p[1] - rr, p[0] + rr, p[1] + rr], fill=ac + (90,))
        out.alpha_composite(g.filter(ImageFilter.GaussianBlur(cv.u * 4)))
    out.alpha_composite(cv.flat())
    arr = np.array(out.convert("RGB"))[..., ::-1]
    return fx.grain(arr, grain, seed) if grain else arr


def nightcall_series(path="/home/muk/projects/nightcall/results/18h_run.log"):
    """Real PPO series from the SB3 tables in the 18h training log."""
    keys = {"approx_kl": [], "clip_fraction": [], "fps": [], "ep_rew_mean": [], "explained_variance": []}
    for line in Path(path).read_text(errors="ignore").splitlines():
        m = re.match(r"\|\s+(\w+)\s+\|\s+(-?[\d.e+-]+)\s+\|", line)
        if m and m.group(1) in keys:
            keys[m.group(1)].append(float(m.group(2)))
    out = [np.array(v) for v in keys.values() if len(v) > 20]
    return out
