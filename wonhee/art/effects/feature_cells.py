"""Cell-based edits of a source: halftone/line screens, superpixel mosaic,
adaptive quadtree mosaic, tiled detail crops. Each cell carries the local tone
of the real image, so detail is spread evenly by construction."""
from __future__ import annotations

import cv2
import numpy as np

from .. import features as F
from ..registry import effect
from .feature_common import prep, with_common, mixc, flat, finish, grade


@effect("feature-halftone-screen", family="feature", tags=["source", "halftone", "dots", "all-over", "print"],
        params=with_common({"cells": (72, "cells across the frame"), "angle": (35, "screen angle degrees"),
                            "shape": ("dot", "dot | line | diamond"), "accent": (0.7, "0..1 small accent dots in the brightest tones")}))
def halftone_screen(ctx):
    """Rotated halftone screen of the real tone: dots (or engraving lines /
    diamonds) swell where the source is bright (dark palette) or dark (light
    palette). Equalised tone means even sky and flat fills carry dots, so the
    frame is covered. Best on photos and glows; on flat dashboards it prints a
    moire of the layout blocks."""
    P = prep(ctx)
    s = P.s
    H, W = P.H, P.W
    cell = W / float(ctx.p("cells"))
    a = np.deg2rad(float(ctx.p("angle")))
    ink_s = F.blur(P.ink, cell * 0.35)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u = xx * np.cos(a) + yy * np.sin(a)
    v = -xx * np.sin(a) + yy * np.cos(a)
    iu, iv = np.floor(u / cell), np.floor(v / cell)
    cu, cv_ = (iu + 0.5) * cell, (iv + 0.5) * cell
    sx = np.clip((cu * np.cos(a) - cv_ * np.sin(a)).astype(np.int32), 0, W - 1)
    sy = np.clip((cu * np.sin(a) + cv_ * np.cos(a)).astype(np.int32), 0, H - 1)
    ink = np.clip(ink_s[sy, sx], 0, 1)
    shape = str(ctx.p("shape"))
    du, dv = u - cu, v - cv_
    if shape == "line":
        r = cell * 0.5 * ink * 1.05
        d = np.abs(dv)
    elif shape == "diamond":
        r = cell * 0.5 * np.sqrt(ink) * 1.3
        d = np.abs(du) + np.abs(dv)
    else:
        r = cell * 0.5 * np.sqrt(ink) * 1.08
        d = np.hypot(du, dv)
    aa = 1.0 * s + 0.6
    m = np.clip((r - d) / aa + 0.5, 0, 1)
    paper = mixc(flat(P, P.pal.canvas), P.body(0.10 + 0.12 * P.t), 0.6)
    out = mixc(paper, P.body(0.55 + 0.45 * P.t), m)
    if float(ctx.p("accent")) > 0:
        a2 = a + np.deg2rad(30)
        c2 = cell * 0.5
        u2 = xx * np.cos(a2) + yy * np.sin(a2); v2 = -xx * np.sin(a2) + yy * np.cos(a2)
        d2 = np.hypot((u2 / c2) % 1 - 0.5, (v2 / c2) % 1 - 0.5) * c2
        hi = np.clip((ink_s - 0.78) / 0.22, 0, 1)
        m2 = np.clip((c2 * 0.5 * np.sqrt(hi) * 0.8 - d2) / aa + 0.5, 0, 1) * (hi > 0.05)
        out = mixc(out, P.pal.accent, m2 * float(ctx.p("accent")))
    return finish(P, out, 0.3, ctx.seed)


@effect("feature-superpixel-mosaic", family="feature", tags=["source", "superpixels", "mosaic", "all-over", "flat"],
        params=with_common({"n": (420, "superpixel count"), "levels": (7, "tone levels in the palette"), "seam": (2.0, "seam width px at 1500 wide"),
                            "compact": (14.0, "SLIC compactness (higher = blockier)"), "color": ("palette", "palette | graded (OKLCH-graded source colours)")}))
def superpixel_mosaic(ctx):
    """SLIC-like superpixels of the source, each flooded with its mean tone
    posterised through the palette, separated by canvas seams, with a few
    accent cells at the extreme highlights. The subject keeps its real region
    boundaries (they are the cell edges). 'graded' keeps source colour
    variety inside one hue family. Best on photos, heatmaps, glows."""
    P = prep(ctx)
    s = P.s
    lab = F.superpixels(P.img, n=int(ctx.p("n")), compactness=float(ctx.p("compact")))
    tm = F.label_means(P.t, lab)
    lv = int(ctx.p("levels"))
    q = np.round(tm * (lv - 1)) / (lv - 1)
    if ctx.p("color") == "graded":
        cm = F.label_means(P.img, lab)
        col = grade(P, tone=q[lab], img=cm[lab])
        out = col
    else:
        out = P.body(0.08 + 0.92 * q)[lab]
        out = mixc(out, P.pal.accent, (q[lab] >= 0.999) * 0.55)
    seam = F.label_edges(lab, max(1, int(round(float(ctx.p("seam")) * s))))
    out = mixc(out, np.asarray(P.pal.canvas) * 0.9, seam * 0.92)
    return finish(P, out, 0.25, ctx.seed)


@effect("feature-quadtree-mosaic", family="feature", tags=["source", "mosaic", "quadtree", "all-over", "adaptive"],
        params=with_common({"min_cell": (14, "smallest cell px at 1500 wide"), "max_cell": (96, "largest cell px at 1500 wide"),
                            "split": (0.07, "tone-variance threshold to subdivide (lower = more small cells)"),
                            "levels": (6, "palette tone levels"), "gap": (1.2, "gutter px at 1500 wide")}))
def quadtree_mosaic(ctx):
    """Adaptive mosaic: cells are large where the source is calm and subdivide
    where it carries detail, so edges and text blocks of the real image show
    as clusters of small tiles. Flat colour per cell, gutters in canvas.
    Reads the structure of screenshots and dashboards best (cells snap to its
    blocks), also good on photos."""
    P = prep(ctx)
    s = P.s
    H, W = P.H, P.W
    mn = max(3, int(round(float(ctx.p("min_cell")) * s)))
    mx = max(mn * 2, int(round(float(ctx.p("max_cell")) * s)))
    mx = int(2 ** round(np.log2(mx))); mn = int(2 ** round(np.log2(mn)))
    Hp, Wp = int(np.ceil(H / mx) * mx), int(np.ceil(W / mx) * mx)
    t = cv2.copyMakeBorder(P.t, 0, Hp - H, 0, Wp - W, cv2.BORDER_REFLECT)
    q = np.zeros((Hp, Wp), np.float32)
    lines = np.zeros((Hp, Wp), np.float32)
    lv = int(ctx.p("levels"))
    thr = float(ctx.p("split"))
    gap = max(1, int(round(float(ctx.p("gap")) * s)))
    size = mx
    active = np.ones((Hp // mx, Wp // mx), bool)
    while True:
        by, bx = Hp // size, Wp // size
        blk = t.reshape(by, size, bx, size)
        mean = blk.mean((1, 3)); std = blk.std((1, 3))
        split = (std > thr) & (size > mn)
        # a block draws itself if it is active and not split (or at min size)
        draw = active & ~split
        for iy, ix in zip(*np.nonzero(draw)):
            qv = np.round(mean[iy, ix] * (lv - 1)) / (lv - 1)
            y0, x0 = iy * size, ix * size
            q[y0:y0 + size, x0:x0 + size] = qv
            lines[y0:y0 + gap, x0:x0 + size] = 1; lines[y0:y0 + size, x0:x0 + gap] = 1
        if size <= mn:
            break
        active = np.kron(active & split, np.ones((2, 2), bool))
        size //= 2
    q = q[:H, :W]; lines = lines[:H, :W]
    out = P.body(0.08 + 0.92 * q)
    out = mixc(out, P.pal.accent, (q >= 0.999) * 0.5)
    out = mixc(out, np.asarray(P.pal.canvas) * 0.9, lines * 0.9)
    return finish(P, out, 0.25, ctx.seed)


@effect("feature-detail-tiles", family="feature", tags=["source", "tiles", "crops", "all-over", "contact-sheet"],
        params=with_common({"grid": ("4x3", "tiles as COLSxROWS"), "gap": (6, "gutter px at 1500 wide"), "frac": (0.22, "crop width as a fraction of the source"),
                            "grade": (True, "re-key tiles to the palette hue (OKLCH)"), "mix_zoom": (True, "alternate zoom levels across tiles")}))
def detail_tiles(ctx):
    """The source is tiled into an even grid of its own most detailed crops
    (non-overlapping windows chosen by detail energy, not by centre), each
    graded into the palette hue: a contact sheet of the original's richest
    parts, kaleidoscope-free, no repetition. Needs a source with detail in
    several places (photos, dashboards, graphs); a flat source yields near
    copies."""
    P = prep(ctx)
    s = P.s
    W, H = P.W, P.H
    cols, rows = (int(v) for v in str(ctx.p("grid")).lower().split("x"))
    gap = max(1, int(round(float(ctx.p("gap")) * s)))
    boxes = F.tile_grid((W, H), cols, rows, gap)
    n = len(boxes)
    # candidate crops from the FULL source (not the frame-cropped one)
    full = np.asarray(ctx.source if ctx.source is not None else P.img, np.float32)[..., :3]
    asp = (boxes[0][2] - boxes[0][0]) / (boxes[0][3] - boxes[0][1])
    fr = float(ctx.p("frac"))
    crops = F.detail_crops(full, n, asp, fr * 1.6) if not ctx.p("mix_zoom") else (
        F.detail_crops(full, n // 2 + n % 2, asp, fr * 1.9) + F.detail_crops(full, n // 2, asp, fr * 1.1))
    crops = crops[:n]
    out = flat(P, np.asarray(P.pal.canvas) * 0.9)
    for k, (bx, crop) in enumerate(zip(boxes, crops)):
        x0, y0, x1, y1 = bx
        c = full[crop[1]:crop[3], crop[0]:crop[2]]
        c = cv2.resize(np.ascontiguousarray(c), (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA if c.shape[1] > x1 - x0 else cv2.INTER_CUBIC)
        if ctx.p("grade"):
            Lc = F.luminance(c)
            tc = F.stretch(F.local_equalize(Lc, 2.5, 4, float(ctx.p("local"))), 1, 99)
            c = grade(P, tone=tc, img=c, hue=None)
        out[y0:y1, x0:x1] = np.clip(c, 0, 1)
    return finish(P, out, 0.25, ctx.seed)
