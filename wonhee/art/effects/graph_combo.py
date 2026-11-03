"""Compositions: a DAG over a dot matrix, a DAG over a physics field (wave-ripple, via compose), and the critical path."""
from __future__ import annotations

import math

import cv2
import numpy as np

from .. import graphkit as gk
from ..registry import effect
from .graph_dag import draw_dag, pick_graph, view_for, _widths, pm


def _halftone_dots(cv, pal, rng, step, tone_lo=0.18, tone_hi=0.34):
    """Dot matrix whose dot radius follows a smooth sum-of-sines field: dense and sparse patches spread over the whole frame."""
    W, H = cv.W, cv.H
    nx, ny = int(W / step) + 2, int(H / step) + 2
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
    f = np.zeros((ny, nx))
    for _ in range(4):
        kx, ky = rng.uniform(0.06, 0.22, 2) * rng.choice([-1, 1], 2); f += np.sin(kx * X + ky * Y + rng.random() * 6.28)
    f = (f - f.min()) / (np.ptp(f) + 1e-9)
    b = gk.Batch(cv)
    for iy in range(ny):
        for ix in range(nx):
            v = f[iy, ix]
            r = step * (0.10 + 0.34 * v)
            gk.m_disc(b.mask(gk.tone(pal, tone_lo + (tone_hi - tone_lo) * round(v * 4) / 4), 1.0, 0), (ix * step, iy * step), r)
    b.flush()


@effect("graph-dag-dots", family="graph", tags=["dag", "graph", "all-over", "dot-matrix", "composition"],
        params={"layers": (11, "layers"), "step": (0.03, "dot pitch / frame width"), "max_nodes": (140, "cap for real data")})
def dag_dots(ctx):
    """A lattice DAG with canvas-coloured edge casings laid over a halftone dot matrix (dot size follows a smooth field, so density varies evenly across the frame). ctx.data: a graph."""
    W, H = ctx.size; pal = ctx.palette; rng = ctx.rng
    cv = gk.Canvas((W, H), pal.canvas)
    _halftone_dots(cv, pal, np.random.default_rng(ctx.seed + 5), W * float(ctx.p("step")))
    g = pick_graph(ctx, lambda r: gk.random_layered(r, _widths(r, int(ctx.p("layers")), 4, 8), 0.5))
    lay = gk.layered(g, rng); px = gk.layered_to_px(lay, view_for(ctx, W, H, 0.05))
    draw_dag(cv, pal, g, lay, px, node="dot", node_size=1.0, edge_w=1.1, glow=0.5, node_tone=(0.46, 0.70), casing=pal.canvas,
             edge_tone_rng=(0.40, 0.56), path_mode=pm(ctx))
    return cv.out()


@effect("graph-dag-field", family="graph", tags=["dag", "graph", "all-over", "physics", "composition"],
        params={"layers": (8, "layers"), "field": (0.8, "how strongly the wave field shows (0..1)"), "max_nodes": (140, "cap for real data")})
def dag_field(ctx):
    """A box-node DAG over a wave-ripple interference field (rendered with the physics effect via compose and pulled toward the canvas), edges and nodes cased in canvas colour. ctx.data: a graph."""
    from ..compose import render_effect
    W, H = ctx.size; pal = ctx.palette; rng = ctx.rng
    field = render_effect("wave-ripple", size=(W, H), seed=ctx.seed + 3, palette=pal, params={"steps": 200, "sources": 6})
    k = float(ctx.p("field"))
    cv = gk.Canvas((W, H), pal.canvas)
    lum = field.mean(2); lum = (lum - np.percentile(lum, 2)) / (np.percentile(lum, 98) - np.percentile(lum, 2) + 1e-9)
    lum = np.clip(lum, 0, 1)
    lut = np.stack([gk.tone(pal, 0.10 + 0.30 * k * t) for t in np.linspace(0, 1, 64)])     # one hue family, lightness only
    cv.img[:] = lut[(lum * 63).astype(int)]
    g = pick_graph(ctx, lambda r: gk.random_layered(r, _widths(r, int(ctx.p("layers")), 3, 6), 0.45))
    lay = gk.layered(g, rng); px = gk.layered_to_px(lay, view_for(ctx, W, H, 0.05))
    draw_dag(cv, pal, g, lay, px, node="box", node_size=0.95, edge_w=1.2, glow=0.6, node_tone=(0.45, 0.68), casing=pal.canvas,
             edge_tone_rng=(0.46, 0.62), path_mode=pm(ctx))
    return cv.out()


@effect("graph-critical-path", family="graph", tags=["dag", "graph", "all-over", "critical-path", "accent"],
        params={"layers": (18, "layers"), "wmin": (8, "min width"), "wmax": (13, "max width"), "max_nodes": (260, "cap for real data")})
def critical_path(ctx):
    """A quiet, dense lattice DAG (everything dim) with ONE thin glowing critical path that weaves through every layer and the whole frame height. Synthetic: max-swing path; real data: longest path (graphkit.longest_path). ctx.data: a graph."""
    W, H = ctx.size; pal = ctx.palette; rng = ctx.rng
    cv = gk.Canvas((W, H), pal.canvas)
    g = pick_graph(ctx, lambda r: gk.random_layered(r, _widths(r, int(ctx.p("layers")), int(ctx.p("wmin")), int(ctx.p("wmax"))), 0.65))
    lay = gk.layered(g, rng); px = gk.layered_to_px(lay, view_for(ctx, W, H, 0.03))
    draw_dag(cv, pal, g, lay, px, node="dot", node_size=0.85, edge_w=0.8, glow=0.7, node_tone=(0.34, 0.46), edge_tone_rng=(0.22, 0.32),
             hot_w=1.8, path_mode=pm(ctx))
    return cv.out()
