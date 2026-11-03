"""dots-bitmatrix: a data matrix as dots. Cell value drives dot size AND colour
(dim -> structure -> accent); zero cells are dim 'off' dots. The matrix is
tiled edge to edge (with a gap) so the whole frame carries data and there is
no centre. Default demo data: a block-structured adjacency matrix (stochastic
block model, rows sorted by block) with one accent block.

ctx.data: a 2D array (any size; benchmark score matrix, adjacency matrix,
confusion matrix). Values are rescaled 0..1 robustly; NaN = off. Large
matrices are block-averaged to ``max_cells``. Dict form
{"matrix": arr, "mask": bool arr} is accepted. ctx.source unused.
Param ``layout``: 'repeat' tiles the matrix across the frame, 'fit' fits one copy
(leaving dim off-dots around it)."""
from __future__ import annotations

import numpy as np

from .. import dotkit as dk
from ..registry import effect


def _sbm(rng, n=30):
    k = int(rng.integers(8, 12))
    cuts = np.sort(rng.choice(np.arange(3, n - 2), k - 1, replace=False))
    lab = np.zeros(n, int)
    for c in cuts:
        lab[c:] += 1
    same = lab[:, None] == lab[None, :]
    base = rng.random((n, n))
    block_p = 0.12 + rng.random((k, k)) * 0.22
    p = np.where(same, 0.75, block_p[lab][:, lab])
    p = np.maximum(p, block_p[lab][:, lab].T) if True else p
    m = (base < p).astype(np.float32) * (0.45 + 0.55 * rng.random((n, n)))
    m = np.maximum(m, m.T * (rng.random((n, n)) < 0.7))
    hot = int(rng.integers(0, k))
    idx = np.nonzero(lab == hot)[0]
    m[np.ix_(idx, idx)] = np.where(m[np.ix_(idx, idx)] > 0, 0.85 + 0.15 * rng.random((len(idx), len(idx))), 0)
    return m


@effect("dots-bitmatrix", family="dots", tags=["dots", "matrix", "data", "adjacency", "heatmap", "all-over"],
        params={"cell": (0.024, "dot pitch, fraction of width"), "layout": ("repeat", "repeat | fit"), "gap": (2, "gap cells between tiles"),
                "shape": ("round", "round | square | diamond"), "max_cells": (80, "block-average larger matrices to this many rows/cols")})
def dots_bitmatrix(ctx):
    W, H = ctx.size
    rng = ctx.rng
    P = ctx.palette
    d = ctx.data
    mask = None
    if isinstance(d, dict):
        mask = d.get("mask"); d = d.get("matrix")
    m = np.asarray(d, np.float32) if d is not None else _sbm(rng)
    mc = int(ctx.p("max_cells"))
    if max(m.shape) > mc:
        f = int(np.ceil(max(m.shape) / mc))
        r, c = (m.shape[0] // f) * f, (m.shape[1] // f) * f
        m = np.nanmean(m[:r, :c].reshape(r // f, f, c // f, f), axis=(1, 3))
    nan = ~np.isfinite(m)
    mm = np.where(nan, 0, m)
    lo, hi = (np.percentile(mm[~nan], [1, 99]) if (~nan).any() else (0, 1))
    mm = np.clip((mm - lo) / (hi - lo + 1e-9), 0, 1)
    mm[nan] = 0
    R, C = mm.shape
    p = float(ctx.p("cell")) * W
    cols, rows = int(W / p) + 1, int(H / p) + 1
    gap = int(ctx.p("gap"))
    if str(ctx.p("layout")) == "fit":
        p = min(W / (C + 1), H / (R + 1)); cols, rows = int(W / p) + 1, int(H / p) + 1
        grid_m = np.zeros((rows, cols), np.float32)
        r0, c0 = (rows - R) // 2, (cols - C) // 2
        grid_m[max(r0, 0):max(r0, 0) + R, max(c0, 0):max(c0, 0) + C] = mm[:rows, :cols]
    else:
        tile = np.pad(mm, ((0, gap), (0, gap)))
        ro, co = int(rng.integers(0, R)), int(rng.integers(0, C))
        grid_m = np.tile(tile, (rows // tile.shape[0] + 2, cols // tile.shape[1] + 2))
        grid_m = np.roll(np.roll(grid_m, -ro, 0), -co, 1)[:rows, :cols]
    g = dk.grid(rows, cols, ctx.size, pitch=p, origin=(0, 0))
    off = P.canvas + (P.structure - P.canvas) * 0.22
    ramp = dk.palette_ramp(P)
    return dk.render_dots(grid_m, ctx.size, g, bg=P.canvas, ramp=ramp, shape=str(ctx.p("shape")), off=off, off_r=0.13, threshold=0.02,
                          r_min=0.2, r_max=0.5, gamma=0.7, normalize=False)
