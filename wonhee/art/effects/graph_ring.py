"""Ring-buffer family: a circular buffer drawn as cells, oldest to newest, with one accent head per buffer.

State of a cell: free (outline only), body (filled, lightness rising tail -> head), head (accent). Five layouts:
snake grid, wheel arcs (centre off the frame), lanes, log spiral (centre off the frame), gauge wall.
``ctx.data`` (optional): {"values": [0..1, ...]} = fill fraction per buffer (ring / lane / gauge, cycled); {"head": [..]} head
position fraction per buffer. Without data they are seeded random.
"""
from __future__ import annotations

import math

import numpy as np

from .. import graphkit as gk
from ..registry import effect


def _vals(ctx, rng, n, lo=0.35, hi=0.8):
    d = ctx.data if isinstance(ctx.data, dict) else {}
    v = list(d.get("values", [])); h = list(d.get("head", []))
    fills = [float(v[i % len(v)]) if v else rng.uniform(lo, hi) for i in range(n)]
    heads = [float(h[i % len(h)]) if h else rng.random() for i in range(n)]
    return fills, heads


def run_state(i, head, size, N):
    off = (head - i) % N
    if off >= size:
        return "free", 0.0
    return ("head" if off == 0 else "body"), 1 - off / max(size - 1, 1)


class Cells:
    """Collects cells then paints: free outlines, body fills by lightness bucket, head accent (+ glow)."""

    def __init__(self, cv, pal, outline_w):
        self.cv, self.pal, self.ow = cv, pal, outline_w
        self.items = []

    def add(self, pts, state, t, inset=True):
        self.items.append((np.asarray(pts, float), state, t, inset))

    def body_col(self, t):
        p = self.pal
        c = gk.tone(p, 0.27 + 0.40 * t)
        if p.structure2 is not None:
            c = c + (p.structure2 - c) * 0.35 * (1 - t)
        return c

    def paint(self, glow=0.0):
        pal, cv = self.pal, self.cv
        b = gk.Batch(cv); hm = cv.mask()
        for pts, st, t, inset in self.items:
            if st == "free":
                gk.m_poly(b.mask(gk.tone(pal, 0.24), 1.0, 0), pts, fill=False, w=self.ow)
        for pts, st, t, inset in self.items:
            if st == "body":
                col = self.body_col(round(t * 7) / 7)
                gk.m_poly(b.mask(col, 1.0, 1), pts)
                if inset:
                    c = pts.mean(0); q = c + (pts - c) * 0.55
                    gk.m_poly(b.mask(pal.canvas + (col - pal.canvas) * 0.25, 1.0, 2), q)
            elif st == "head":
                gk.m_poly(b.mask(pal.accent, 1.0, 3), pts)
                gk.m_poly(hm, pts)
                c = pts.mean(0); q = c + (pts - c) * 0.5
                gk.m_poly(b.mask(pal.canvas + (pal.accent - pal.canvas) * 0.35, 1.0, 4), q)
        b.flush()
        if glow:
            cv.glow(hm, pal.accent, self.ow * 3.0, glow)


def _quad(x0, y0, x1, y1, r=0.0):
    return gk.rrect_pts(x0, y0, x1, y1, r)


def _rot_quad(c, w, h, ang, r=0.0):
    pts = gk.rrect_pts(-w, -h, w, h, r)
    ca, sa = math.cos(ang), math.sin(ang)
    return np.stack([c[0] + pts[:, 0] * ca - pts[:, 1] * sa, c[1] + pts[:, 0] * sa + pts[:, 1] * ca], 1)


def _fit(ctx):
    W, H = ctx.size
    return W, H, ctx.palette, ctx.rng


@effect("graph-ring-snake", family="graph", tags=["ring", "graph", "all-over", "grid", "buffer"],
        params={"cols": (13, "cells across"), "fill": (0.0, "fraction of cells in use (0 = random 0.45-0.65)")})
def ring_snake(ctx):
    """Boustrophedon grid ring buffer: ~100 cells in snake order, tail faint, head accent, free cells outlined. Bleeds off the bottom. ctx.data: {"values","head"}."""
    W, H, pal, rng = _fit(ctx)
    cv = gk.Canvas((W, H), pal.canvas)
    cols = int(ctx.p("cols")); cw = W / (cols - 0.15); rows = int(math.ceil(H / cw)) + 1
    N = cols * rows
    fills, heads = _vals(ctx, rng, 1, 0.45, 0.65)
    f = float(ctx.p("fill")) or fills[0]
    head = int(heads[0] * N); size = int(N * f)
    order = []
    for r in range(rows):
        order += [(r, c) for c in (range(cols) if r % 2 == 0 else range(cols - 1, -1, -1))]
    ox = -cw * 0.1; oy = -cw * rng.uniform(0.1, 0.5)
    cells = Cells(cv, pal, max(3, W * 0.004))
    gap = cw * 0.07
    for idx, (r, c) in enumerate(order):
        st, t = run_state(idx, head, size, N)
        x, y = ox + c * cw, oy + r * cw
        cells.add(_quad(x + gap, y + gap, x + cw - gap, y + cw - gap, cw * 0.16), st, t)
    cells.paint(0.5)
    return cv.out()


def _arc_cells(ctx, cv, pal, rng, cx, cy, r_in, r_out, thick, gap, nr, cells: Cells, fills, heads, cell_len=1.25, shape="sector"):
    W, H = cv.W, cv.H
    for ri in range(nr):
        r1 = r_out - ri * (thick + gap); r0 = r1 - thick
        if r0 < 0:
            break
        rm = (r0 + r1) / 2
        n = max(8, int(round(2 * math.pi * rm / (thick * cell_len))))
        da = 2 * math.pi / n
        vis = []
        for i in range(n):
            a0 = i * da; am = a0 + da / 2
            x, y = cx + rm * math.cos(am), cy + rm * math.sin(am)
            if -thick < x < W + thick and -thick < y < H + thick:
                vis.append(i)
        if not vis:
            continue
        N = len(vis); head = int(heads[ri % len(heads)] * N); size = max(3, int(N * fills[ri % len(fills)]))
        for k, i in enumerate(vis):
            st, t = run_state(k, head, size, N)
            a0 = i * da + da * 0.07; a1 = (i + 1) * da - da * 0.07
            cells.add(gk.sector_pts((cx, cy), r0, r1, a0, a1, 6), st, t)


@effect("graph-ring-wheel", family="graph", tags=["ring", "graph", "all-over", "wheel", "buffer"],
        params={"rings": (7, "concentric buffers"), "thick": (0.075, "ring thickness / frame width")})
def ring_wheel(ctx):
    """Concentric ring buffers of a huge wheel whose hub is off the frame: cells sweep across the frame as stacked arcs. One head per ring. ctx.data: {"values","head"}."""
    W, H, pal, rng = _fit(ctx)
    cv = gk.Canvas((W, H), pal.canvas)
    thick = W * float(ctx.p("thick")); gap = thick * 0.14
    nr = int(ctx.p("rings"))
    sgn = 1 if ctx.seed % 2 == 0 else -1
    cx = (-0.12 if sgn > 0 else 1.12) * W + rng.uniform(-0.04, 0.04) * W
    cy = 1.12 * H + rng.uniform(-0.04, 0.04) * H
    far = math.hypot(max(cx, W - cx), cy)
    near = math.hypot(cx - min(max(cx, 0), W), cy - H)
    r_out = far + thick * 0.3
    nr = max(nr, int((far - near) / (thick + gap)) + 1)
    fills, heads = _vals(ctx, rng, nr, 0.4, 0.8)
    cells = Cells(cv, pal, max(3, W * 0.004))
    _arc_cells(ctx, cv, pal, rng, cx, cy, 0, r_out, thick, gap, nr, cells, fills, heads)
    cells.paint(0.5)
    return cv.out()


@effect("graph-ring-lanes", family="graph", tags=["ring", "graph", "all-over", "lanes", "buffer"],
        params={"lanes": (8, "buffers stacked top to bottom")})
def ring_lanes(ctx):
    """Parallel tape lanes, each its own ring buffer with its own head, lanes staggered so cells bleed off the sides; one lane carries the accent head. ctx.data: {"values","head"}."""
    W, H, pal, rng = _fit(ctx)
    cv = gk.Canvas((W, H), pal.canvas)
    L = int(ctx.p("lanes")); lh = H / (L - 0.2)
    fills, heads = _vals(ctx, rng, L, 0.3, 0.8)
    cells = Cells(cv, pal, max(3, W * 0.004))
    hot = int(rng.integers(0, L))
    for ln in range(L):
        cw = lh * float(rng.choice([1.25, 1.5, 1.8])); n = int(W / cw) + 3
        off = -cw * rng.uniform(0.2, 1.0)
        head = int(heads[ln] * n); size = max(3, int(n * fills[ln]))
        y0 = -lh * 0.1 + ln * lh; gap = lh * 0.08
        for i in range(n):
            st, t = run_state(i, head, size, n)
            if st == "head" and ln != hot:
                st = "body"
            x = off + i * cw
            cells.add(_quad(x + gap, y0 + gap, x + cw - gap, y0 + lh - gap, lh * 0.18), st, t)
    cells.paint(0.5)
    return cv.out()


@effect("graph-ring-spiral", family="graph", tags=["ring", "graph", "all-over", "spiral", "buffer"],
        params={"arms": (1, "spiral arms"), "pitch": (0.20, "radial gain per turn (fraction of frame height)")})
def ring_spiral(ctx):
    """Buffer laid along a growing spiral whose centre sits off the frame corner; cells are tangent-aligned squares that grow with radius, so detail is spread. ctx.data: {"values","head"}."""
    W, H, pal, rng = _fit(ctx)
    cv = gk.Canvas((W, H), pal.canvas)
    cx = (-0.06 if ctx.seed % 2 == 0 else 1.06) * W; cy = 1.08 * H
    far = math.hypot(max(cx, W - cx), cy)
    k = float(ctx.p("pitch")) * H / (2 * math.pi)    # r = r0 + k * theta
    r0 = H * 0.10
    pts = []; theta = 0.0; sizes = []
    while True:
        r = r0 + k * theta
        if r > far + 40:
            break
        arc = max(r * 0.001, 1e-3)
        thick = k * 2 * math.pi * 0.86           # radial pitch -> cell size
        cell = thick
        dth = cell * 1.08 / max(r, 1.0)
        theta += dth
        c = (cx + r * math.cos(theta), cy + r * math.sin(theta))
        if -cell < c[0] < W + cell and -cell < c[1] < H + cell:
            pts.append((c, theta, cell))
    N = len(pts)
    fills, heads = _vals(ctx, rng, 1, 0.45, 0.8)
    head = int(heads[0] * N); size = int(N * fills[0])
    cells = Cells(cv, pal, max(3, W * 0.004))
    for i, (c, th, cell) in enumerate(pts):
        st, t = run_state(i, head, size, N)
        cells.add(_rot_quad(c, cell * 0.44, cell * 0.44, th + math.pi / 2, cell * 0.1), st, t)
    cells.paint(0.5)
    return cv.out()


@effect("graph-ring-gauges", family="graph", tags=["ring", "graph", "all-over", "gauge", "buffer"],
        params={"cols": (4, "gauges across"), "segments": (18, "segments per gauge")})
def ring_gauges(ctx):
    """Wall of C-shaped segment gauges on a staggered grid, each a ring buffer at its own fill; accent heads are scattered evenly. ctx.data: {"values","head"}."""
    W, H, pal, rng = _fit(ctx)
    cv = gk.Canvas((W, H), pal.canvas)
    cols = int(ctx.p("cols")); d = W / (cols - 0.3)
    rows = int(math.ceil(H / (d * 0.86))) + 2
    ns = int(ctx.p("segments"))
    cells = Cells(cv, pal, max(3, W * 0.004))
    fills, heads = _vals(ctx, rng, cols * rows, 0.25, 0.95)
    g = 0
    for r in range(rows):
        for c in range(cols + 1):
            cx = (c + (0.5 if r % 2 else 0.0)) * d - d * 0.1; cy = -d * 0.1 + r * d * 0.86
            if cx < -d or cx > W + d or cy < -d or cy > H + d:
                continue
            R = d * 0.47; th = R * 0.36
            f = fills[g % len(fills)]; hd = heads[g % len(heads)]; g += 1
            span = math.radians(300); a_start = math.radians(120)
            size = max(2, int(ns * f)); head = int(size - 1) if size else 0
            show_head = (r + 2 * c) % 2 == 0
            for i in range(ns):
                a0 = a_start + span * i / ns + 0.035; a1 = a_start + span * (i + 1) / ns - 0.035
                st, t = ("free", 0.0) if i >= size else (("head" if i == size - 1 and show_head else "body"), i / max(size - 1, 1))
                cells.add(gk.sector_pts((cx, cy), R - th, R, a0, a1, 5), st, t, inset=False)
    cells.paint(0.5)
    return cv.out()
