"""Vertex-edge and data-structure art for the `art` prototype.

Every drawer lays itself out from the box's own W and H (counts and spacing come from the aspect), so the figure
fills the box at any ratio. Sizes are in units of u = min(W, H) / 100, so line weights scale with the box and DPR.
"""
from __future__ import annotations

import colorsys
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw

from art_proto import fingerprint, hues, field, stops, STYLES  # noqa: F401
from wonhee.edit import fx


def rgb(h, s, l):
    r, g, b = colorsys.hls_to_rgb((h % 360) / 360, l, s)
    return (int(r * 255), int(g * 255), int(b * 255))


def palette(fp, dark, override=None):
    if override:
        return override[1] if dark else override[0]
    h0, h1, h2 = hues(fp)
    if dark:
        return dict(bg=rgb(h0 + 215, .35, .07), ink=rgb(h0, .2, .9), a=rgb(h0, .7, .64), b=rgb(h1, .65, .62),
                    c=rgb(h2 + 120, .65, .64), mute=rgb(h0 + 215, .25, .24))
    return dict(bg=rgb(h0, .22, .94), ink=rgb(h0 + 215, .5, .14), a=rgb(h0, .62, .42), b=rgb(h1, .55, .46),
                c=rgb(h2 + 120, .6, .52), mute=rgb(h0, .15, .78))


class Cv:
    """A supersampled RGBA layer addressed in output pixels."""

    def __init__(self, W, H):
        self.W, self.H = W, H
        self.ss = 3 if W * H < 700_000 else 2 if W * H < 3_000_000 else 1
        self.u = min(W, H) / 100
        self.im = Image.new("RGBA", (W * self.ss, H * self.ss), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.im)

    def _p(self, p):
        return (p[0] * self.ss, p[1] * self.ss)

    def line(self, p, q, col, w=.5, a=255):
        self.d.line([self._p(p), self._p(q)], fill=col + (a,), width=max(1, round(w * self.u * self.ss)))

    def curve(self, pts, col, w=.5, a=255):
        self.d.line([self._p(p) for p in pts], fill=col + (a,), width=max(1, round(w * self.u * self.ss)), joint="curve")

    def bez(self, p0, p1, p2, col, w=.5, a=255, n=28):
        pts = [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
                (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1]) for t in np.linspace(0, 1, n)]
        self.curve(pts, col, w, a)

    def s_curve(self, p, q, col, w=.5, a=255, horizontal=True, n=28):
        pts = []
        for t in np.linspace(0, 1, n):
            e = t * t * (3 - 2 * t)
            pts.append((p[0] + (q[0] - p[0]) * (e if not horizontal else t), p[1] + (q[1] - p[1]) * (t if not horizontal else e)))
        self.curve(pts, col, w, a)

    def dot(self, p, r, col, a=255, ring=None):
        x, y = self._p(p)
        R = r * self.u * self.ss
        self.d.ellipse([x - R, y - R, x + R, y + R], fill=col + (a,), outline=None if ring is None else ring + (255,),
                       width=max(1, round(.35 * self.u * self.ss)))

    def box(self, x0, y0, x1, y1, fill=None, line=None, lw=.35, r=.6, a=255):
        X0, Y0 = self._p((x0, y0)); X1, Y1 = self._p((x1, y1))
        self.d.rounded_rectangle([X0, Y0, X1, Y1], radius=r * self.u * self.ss, fill=None if fill is None else fill + (a,),
                                 outline=None if line is None else line + (255,), width=max(1, round(lw * self.u * self.ss)))

    def flat(self):
        return self.im.resize((self.W, self.H), Image.LANCZOS)


def sample(fp, rng, i, n):
    wk = fp.get("weeks")
    if wk is not None and len(wk) > 1:
        return float(np.interp(i / max(n - 1, 1), np.linspace(0, 1, len(wk)), wk))
    return float(rng.random())


def lang_cols(fp, pal):
    """Cluster colours: the repo's own languages when known, else the project palette."""
    from art_proto import hash_hue
    cols = [pal["a"], pal["b"], pal["c"]]
    return cols


# --- drawers ------------------------------------------------------------------------------------------------------
def constellation(cv, rng, fp, pal):
    W, H, u = cv.W, cv.H, cv.u
    n = int(np.clip(W * H / (u * u * 100) * .30, 22, 120))
    m = 3 * u
    d_min = math.sqrt((W - 2 * m) * (H - 2 * m) / n) * .62
    pts = []
    for _ in range(n * 40):
        p = (rng.uniform(m, W - m), rng.uniform(m, H - m))
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 > d_min ** 2 for q in pts):
            pts.append(p)
        if len(pts) >= n:
            break
    P = np.array(pts)
    n = len(P)
    D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
    cen = P[rng.choice(n, 3, replace=False)]
    grp = np.argmin(np.hypot(P[:, None, 0] - cen[None, :, 0], P[:, None, 1] - cen[None, :, 1]), 1)
    cols = [pal["a"], pal["b"], pal["c"]]
    edges = set()
    for i in range(n):
        for j in np.argsort(D[i])[1:4]:
            edges.add((min(i, int(j)), max(i, int(j))))
    for _ in range(max(2, n // 10)):  # a few bridges across clusters
        i, j = rng.integers(0, n, 2)
        if i != j and D[i, j] < max(W, H) * .5:
            edges.add((min(i, int(j)), max(i, int(j))))
    deg = np.zeros(n)
    for i, j in edges:
        deg[i] += 1; deg[j] += 1
    for i, j in edges:
        c = cols[grp[i]] if grp[i] == grp[j] else pal["mute"]
        mid = ((P[i, 0] + P[j, 0]) / 2 + rng.normal(0, u * .8), (P[i, 1] + P[j, 1]) / 2 + rng.normal(0, u * .8))
        cv.bez(tuple(P[i]), mid, tuple(P[j]), c, .28, 200 if grp[i] == grp[j] else 120)
    for i in range(n):
        s = sample(fp, rng, i, n)
        cv.dot(tuple(P[i]), .7 + .22 * deg[i] + 1.1 * s, cols[grp[i]], ring=pal["bg"])


def tree(cv, rng, fp, pal, heap=False):
    W, H, u = cv.W, cv.H, cv.u
    horiz = W >= H
    long_, short = (W, H) if horiz else (H, W)
    m = 4 * u
    depth = int(np.clip(round(long_ / (u * 17)), 3, 7)) if not heap else int(np.clip(round(long_ / (u * 20)), 3, 5))
    depth = min(depth, int(math.log2(max(short / (u * 7), 2))) + 1) if heap else depth
    depth = max(depth, 3)
    nodes = []  # (level, leaf_rank or None, children)

    def build(level):
        i = len(nodes)
        nodes.append({"l": level, "kids": []})
        if level < depth - 1:
            for _ in range(2):
                if heap or rng.random() < .82 or not nodes[i]["kids"]:
                    nodes[i]["kids"].append(build(level + 1))
        return i
    root = build(0)
    leaves = [i for i, n in enumerate(nodes) if not n["kids"]]
    for r, i in enumerate(leaves):
        nodes[i]["c"] = (r + .5) / len(leaves)

    def place(i):
        n = nodes[i]
        if n["kids"]:
            for k in n["kids"]:
                place(k)
            n["c"] = float(np.mean([nodes[k]["c"] for k in n["kids"]]))
    place(root)
    pos = {i: ((m + (long_ - 2 * m) * n["l"] / (depth - 1)), (m + (short - 2 * m) * n["c"])) for i, n in enumerate(nodes)}
    xy = (lambda p: p) if horiz else (lambda p: (p[1], p[0]))
    for i, n in enumerate(nodes):
        for k in n["kids"]:
            a, b = xy(pos[i]), xy(pos[k])
            mid = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            p1 = (mid[0], a[1]) if horiz else (a[0], mid[1])
            p2 = (mid[0], b[1]) if horiz else (b[0], mid[1])
            cv.curve([a, p1, p2, b], pal["mute"] if rng.random() > .3 else pal["a"], .4)
    cols = [pal["a"], pal["b"], pal["c"]]
    for i, n in enumerate(nodes):
        p = xy(pos[i])
        s = sample(fp, rng, i, len(nodes))
        w = (1.4 + 1.6 * s) * (1.5 if heap else 1)
        if heap:
            cv.box(p[0] - w * 1.4 * u, p[1] - w * u, p[0] + w * 1.4 * u, p[1] + w * u, fill=cols[n["l"] % 3], r=.5)
            cv.box(p[0] - w * .8 * u, p[1] - .22 * w * u, p[0] + w * .8 * u, p[1] + .22 * w * u, fill=pal["bg"], r=.2)
        else:
            cv.dot(p, w, cols[n["l"] % 3] if n["kids"] else pal["ink"], ring=pal["bg"])


def dag(cv, rng, fp, pal):
    W, H, u = cv.W, cv.H, cv.u
    horiz = W >= H
    long_, short = (W, H) if horiz else (H, W)
    m = 5 * u
    L = int(np.clip(round(long_ / (u * 18)), 3, 9))
    layers = []
    for l in range(L):
        k = int(np.clip(rng.integers(2, 6) * short / (u * 60) + 1, 2, 7))
        layers.append([(m + (long_ - 2 * m) * l / (L - 1), m + (short - 2 * m) * (j + .5) / k + rng.normal(0, u * .8)) for j in range(k)])
    xy = (lambda p: p) if horiz else (lambda p: (p[1], p[0]))
    crit = [int(rng.integers(0, len(layers[0])))]
    for l in range(1, L):
        crit.append(int(rng.integers(0, len(layers[l]))))
    for l in range(L - 1):
        for j, p in enumerate(layers[l]):
            tgts = {int(rng.integers(0, len(layers[l + 1]))) for _ in range(1 + int(rng.random() < .5))}
            if crit[l] == j:
                tgts.add(crit[l + 1])
            for t in tgts:
                hot = crit[l] == j and crit[l + 1] == t
                cv.s_curve(xy(p), xy(layers[l + 1][t]), pal["a"] if hot else pal["mute"], .7 if hot else .3, horizontal=horiz)
        if l + 2 < L and rng.random() < .6:  # a skip edge over a layer
            j = int(rng.integers(0, len(layers[l]))); t = int(rng.integers(0, len(layers[l + 2])))
            cv.s_curve(xy(layers[l][j]), xy(layers[l + 2][t]), pal["b"], .3, 160, horizontal=horiz)
    for l, lay in enumerate(layers):
        for j, p in enumerate(lay):
            q = xy(p); hot = crit[l] == j
            s = sample(fp, rng, l * 7 + j, L * 7)
            cv.box(q[0] - 2.2 * u * (.7 + s * .5), q[1] - 1.1 * u, q[0] + 2.2 * u * (.7 + s * .5), q[1] + 1.1 * u, fill=pal["a"] if hot else pal["bg"],
                   line=pal["a"] if hot else pal["ink"], lw=.4, r=.5)


def radial(cv, rng, fp, pal):
    W, H, u = cv.W, cv.H, cv.u
    cx, cy = W / 2, H / 2
    rx, ry = W / 2 - 4 * u, H / 2 - 4 * u  # an ellipse fitted to the box
    n_in, n_mid, n_out = 1, int(np.clip(W * H / (u * u * 100) * .12, 8, 26)), int(np.clip(W * H / (u * u * 100) * .38, 24, 90))
    mid = [(cx + rx * .42 * math.cos(a), cy + ry * .42 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, n_mid, endpoint=False)]
    out = []
    for a in np.linspace(0, 2 * math.pi, n_out, endpoint=False):
        r = .86 + .14 * rng.random()
        out.append((cx + rx * r * math.cos(a + .03), cy + ry * r * math.sin(a + .03)))
    cols = [pal["a"], pal["b"], pal["c"]]
    for i, o in enumerate(out):
        j = int(round(i * n_mid / n_out)) % n_mid
        cv.bez(o, (cx + (o[0] - cx) * .45, cy + (o[1] - cy) * .45), mid[j], cols[j % 3], .22, 150)
        if rng.random() < .25:
            k = int(rng.integers(0, n_mid)); cv.bez(o, (cx + (o[0] - cx) * .3, cy + (o[1] - cy) * .3), mid[k], pal["mute"], .18, 120)
    for j, p in enumerate(mid):
        cv.bez(p, ((p[0] + cx) / 2 + rng.normal(0, u), (p[1] + cy) / 2 + rng.normal(0, u)), (cx, cy), cols[j % 3], .4)
    for i, o in enumerate(out):
        j = int(round(i * n_mid / n_out)) % n_mid
        cv.dot(o, .55 + .6 * sample(fp, rng, i, n_out), cols[j % 3], 230)
    for j, p in enumerate(mid):
        cv.dot(p, 1.5, cols[j % 3], ring=pal["bg"])
    cv.dot((cx, cy), 3.2, pal["ink"], ring=pal["a"])


def btree(cv, rng, fp, pal):
    W, H, u = cv.W, cv.H, cv.u
    horiz = W > H * 1.6
    A, B = (H, W) if horiz else (W, H)  # B is the axis the levels run along
    m = 4 * u
    L = int(np.clip(round(B / (u * 28)), 2, 4))
    fan = int(np.clip(round(A / (u * 28)), 2, 5))
    keys = 4
    blocks = [[(0.5, 1)]]
    for l in range(1, L):
        cnt = min(fan ** l, int(A / (u * 11)))
        blocks.append([((i + .5) / cnt, 1) for i in range(cnt)])
    pos = lambda c, l: (m + (A - 2 * m) * c, m + (B - 2 * m) * (l + .5) / L)
    xy = (lambda p: (p[1], p[0])) if horiz else (lambda p: p)
    cols = [pal["a"], pal["b"], pal["c"], pal["ink"]]
    bw = lambda l: min((A - 2 * m) / len(blocks[l]) * .86, u * 7 * keys)
    for l in range(L - 1):
        for i, (c, _) in enumerate(blocks[l]):
            for k in range(len(blocks[l + 1])):
                if int(k * len(blocks[l]) / len(blocks[l + 1])) == i:
                    a = xy(pos(c, l)); b = xy(pos(blocks[l + 1][k][0], l + 1))
                    cv.s_curve((a[0], a[1]), (b[0], b[1]), pal["mute"], .35, horizontal=horiz)
    for l in range(L):
        for i, (c, _) in enumerate(blocks[l]):
            p = pos(c, l)
            w = bw(l); h = 5 * u
            for k in range(keys):
                x0 = p[0] - w / 2 + w * k / keys; x1 = p[0] - w / 2 + w * (k + 1) / keys
                a = xy((x0, p[1] - h / 2)); b = xy((x1 - .3, p[1] + h / 2))
                s = sample(fp, rng, i * 3 + k + l * 31, 97)
                cv.box(min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]), fill=cols[l % 3] if s > .35 else pal["bg"], line=cols[l % 3], lw=.3, r=.3)


def hashtable(cv, rng, fp, pal):
    W, H, u = cv.W, cv.H, cv.u
    horiz = W >= H
    A, B = (H, W) if horiz else (W, H)  # A: bucket axis, B: chain axis
    m = 4 * u
    nb = int(np.clip(round((A - 2 * m) / (u * 6.4)), 4, 16))
    xy = (lambda a, b: (b, a)) if horiz else (lambda a, b: (a, b))
    cols = [pal["a"], pal["b"], pal["c"]]
    max_chain = int(np.clip((B - 2 * m - 6 * u) / (u * 9), 2, 9))
    for i in range(nb):
        a = m + (A - 2 * m) * (i + .5) / nb
        cv.box(*(xy(a - 2.4 * u, m) + xy(a + 2.4 * u, m + 4.8 * u)), fill=pal["ink"], r=.4)
        chain = int(rng.integers(0, max_chain + 1)) if rng.random() < .8 else 0
        prev = xy(a, m + 4.8 * u)
        for k in range(chain):
            b = m + 8 * u + (k + 1) * (B - 2 * m - 8 * u) / (max_chain + .3)
            q = xy(a + rng.normal(0, .3 * u), b)
            cv.line(prev, q, pal["mute"], .3)
            s = sample(fp, rng, i * 9 + k, nb * 9)
            cv.box(q[0] - 2.5 * u, q[1] - 1.6 * u, q[0] + 2.5 * u, q[1] + 1.6 * u, fill=cols[(i + k) % 3], r=.4, a=int(150 + 100 * s))
            prev = q


def skiplist(cv, rng, fp, pal):
    W, H, u = cv.W, cv.H, cv.u
    horiz = W >= H
    A, B = (W, H) if horiz else (H, W)
    m = 4 * u
    L = int(np.clip(round((B - 2 * m) / (u * 13)), 3, 8))
    n = int(np.clip(round((A - 2 * m) / (u * 10)), 6, 22))
    xy = (lambda a, b: (a, b)) if horiz else (lambda a, b: (b, a))
    cols = [pal["a"], pal["b"], pal["c"], pal["ink"]]
    heights = [1 + sum(rng.random() < .5 ** (k) for k in range(1, L)) for _ in range(n)]
    heights[0] = heights[-1] = L
    for lv in range(L):
        b = m + (B - 2 * m) * (1 - lv / max(L - 1, 1))
        idx = [i for i in range(n) if heights[i] > lv]
        for a_i, b_i in zip(idx, idx[1:]):
            pa, pb = xy(m + (A - 2 * m) * a_i / (n - 1), b), xy(m + (A - 2 * m) * b_i / (n - 1), b)
            cv.line(pa, pb, cols[lv % 3], .35, 220)
        for i in idx:
            q = xy(m + (A - 2 * m) * i / (n - 1), b)
            cv.box(q[0] - 2.1 * u, q[1] - 2.1 * u, q[0] + 2.1 * u, q[1] + 2.1 * u, fill=cols[lv % 3], r=.5)
    for i in range(n):  # towers
        a = m + (A - 2 * m) * i / (n - 1)
        cv.line(xy(a, m + (B - 2 * m) * (1 - (heights[i] - 1) / max(L - 1, 1))), xy(a, m + (B - 2 * m)), pal["mute"], .15, 120)


def ringbuffer(cv, rng, fp, pal):
    W, H, u = cv.W, cv.H, cv.u
    m = 3 * u
    cell = max(2.6 * u, min(W, H) / 9)
    cols_n = max(4, int((W - 2 * m) / (cell * 1.2))); rows_n = max(3, int((H - 2 * m) / (cell * 1.2)))
    cw, ch = (W - 2 * m) / cols_n, (H - 2 * m) / rows_n
    total = cols_n * rows_n
    head = int(rng.integers(0, total)); size = int(total * rng.uniform(.35, .6))
    filled = {(head + k) % total for k in range(size)}
    cols = [pal["a"], pal["b"], pal["c"]]
    order = []
    for r in range(rows_n):
        rng_cols = range(cols_n) if r % 2 == 0 else range(cols_n - 1, -1, -1)  # a serpentine: the buffer, unrolled
        order += [(r, c) for c in rng_cols]
    for idx, (r, c) in enumerate(order):
        x0, y0 = m + c * cw, m + r * ch
        on = idx in filled
        s = sample(fp, rng, idx, total)
        cv.box(x0 + cw * .08, y0 + ch * .08, x0 + cw * .92, y0 + ch * .92, fill=cols[int(s * 2.99)] if on else None,
               line=pal["mute"] if not on else cols[int(s * 2.99)], lw=.3, r=.5)
    for a, b in zip(order, order[1:]):
        pa = (m + a[1] * cw + cw / 2, m + a[0] * ch + ch / 2); pb = (m + b[1] * cw + cw / 2, m + b[0] * ch + ch / 2)
        cv.line(pa, pb, pal["mute"], .12, 90)
    for tag, idx in (("h", head), ("t", (head + size) % total)):
        r, c = order[idx]
        cv.dot((m + c * cw + cw / 2, m + r * ch + ch / 2), 1.1, pal["ink"], ring=pal["bg"])


def matrix(cv, rng, fp, pal):
    W, H, u = cv.W, cv.H, cv.u
    m = 3 * u
    cell = max(1.6 * u, min(W, H) / 22)
    nx, ny = max(6, int((W - 2 * m) / cell)), max(4, int((H - 2 * m) / cell))
    cw, ch = (W - 2 * m) / nx, (H - 2 * m) / ny
    blocks = sorted(rng.integers(0, max(nx, ny), 3).tolist())
    cols = [pal["a"], pal["b"], pal["c"]]
    for j in range(ny):
        for i in range(nx):
            gi = sum(i * ny / nx >= b for b in blocks); gj = sum(j >= b for b in blocks)
            same = gi == gj
            near = abs(i * ny / nx - j) < 1.2
            pr = .62 if same else .06
            pr = max(pr, .9 if near else 0)
            if rng.random() < pr:
                s = sample(fp, rng, i + j * nx, nx * ny)
                r = (.28 + .24 * s) * min(cw, ch) / u
                cv.dot((m + (i + .5) * cw, m + (j + .5) * ch), r, cols[gj % 3] if same else pal["ink"], 235)
            else:
                cv.dot((m + (i + .5) * cw, m + (j + .5) * ch), .12 * min(cw, ch) / u, pal["mute"], 200)


DRAW = {"graph": constellation, "tree": tree, "heap": lambda *a: tree(*a, heap=True), "dag": dag, "radial": radial,
        "btree": btree, "hash": hashtable, "skip": skiplist, "ring": ringbuffer, "matrix": matrix}


# --- bases and combos --------------------------------------------------------------------------------------------
def base_image(kind, fp, W, H, seed, pal, dark):
    bg = np.array(pal["bg"][::-1], np.uint8)
    flat = np.empty((H, W, 3), np.uint8); flat[:] = bg
    if kind == "flat":
        return flat
    if kind == "grid":  # faint graph paper
        im = flat.copy(); step = max(8, round(min(W, H) / 12))
        mc = np.array(pal["mute"][::-1], np.float32)
        for x in range(0, W, step): im[:, x] = (im[:, x] * .85 + mc * .15).astype(np.uint8)
        for y in range(0, H, step): im[y] = (im[y] * .85 + mc * .15).astype(np.uint8)
        return im
    f = field(fp, W, H, seed, calm=.8)
    a, b, c = (np.array(pal[k][::-1], np.float32) for k in "abc")
    bgf = np.array(pal["bg"][::-1], np.float32)
    mix = lambda t: (bgf * (1 - t) + t * x for x in [0])
    if kind in ("wash", "halftone"):
        col = stops(f, [bgf, bgf * .6 + b * .4, bgf * .35 + a * .65] if not dark else [bgf, bgf * .5 + b * .5, a * .8])
        if kind == "halftone":
            col = fx.halftone(col, ink="#%02x%02x%02x" % pal["a"], paper="#%02x%02x%02x" % pal["bg"], amount=.9, angle=30, cell=max(5, round(max(W, H) / 70)))
        return col
    if kind == "glow":
        return fx.bloom(stops(f ** 1.5, [bgf, bgf * .4 + a * .6, bgf * .2 + c * .8]), .5, .5)
    return flat


def render(structure, base, fp, W, H, seed, dark=False, override=None, grain=.1):
    pal = palette(fp, dark, override)
    rng = np.random.default_rng(seed)
    im = base_image(base, fp, W, H, seed, pal, dark)
    cv = Cv(W, H)
    DRAW[structure](cv, rng, fp, pal)
    layer = cv.flat()
    out = Image.fromarray(im[..., ::-1]).convert("RGBA")
    out.alpha_composite(layer)
    arr = np.array(out.convert("RGB"))[..., ::-1]
    return fx.grain(arr, grain, seed) if grain else arr
