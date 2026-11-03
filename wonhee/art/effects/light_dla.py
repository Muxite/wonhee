"""Diffusion-limited aggregation: crystal / coral growth from many seeds."""
from __future__ import annotations

import cv2
import numpy as np

from ..registry import effect
from . import light_common as lc


@effect("dla-crystal", family="light", tags=["dla", "fractal", "physics", "all-over", "simulation", "crystal"],
        params={"seeds": (34, "number of crystal seeds on a jittered grid"),
                "cells": (300, "simulation grid width"),
                "fill": (0.1, "share of grid cells that stick"),
                "stick": (0.9, "sticking probability on contact (lower = bushier, smoother)"),
                "exposure": (1.0, "exposure")})
def dla_crystal(ctx):
    """Diffusion-limited aggregation. Thousands of random walkers diffuse on a
    periodic lattice (vectorised) and freeze on touching a crystal; every
    frozen cell remembers the cell it hit (parent) and its arrival time.
    Drawn as light: young branch tips burn bright, old trunks dim and thicken
    with the sub-tree mass. Seeds sit on a jittered grid so the crystals tile
    the frame. data: points are the seeds; source: luminance sets the local
    sticking probability (bright image areas crystallise more)."""
    W, H = ctx.size
    rng = ctx.rng
    sc = lc.scale(ctx)
    gw = int(ctx.p("cells")); gh = int(round(gw * H / W))
    dp = lc.data_points(ctx)
    seeds = (dp[0] / [W, H] * [gw, gh]) if dp is not None else lc.jitter_grid(rng, int(ctx.p("seeds")), gw, gh, 0.6)
    lum = lc.source_lum(ctx)
    stick_map = np.full((gh, gw), float(ctx.p("stick")), np.float32)
    if lum is not None:
        stick_map = np.clip(stick_map * (0.3 + 1.4 * cv2.resize(lum, (gw, gh), interpolation=cv2.INTER_AREA)), 0.05, 1)
    occ = np.zeros((gh, gw), bool); near = np.zeros((gh, gw), bool)
    idm = -np.ones((gh, gw), np.int32)
    parent = []; pos = []

    def add(x, y, par):
        idm[y, x] = len(pos); occ[y, x] = True
        parent.append(par); pos.append((x, y))
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            near[(y + dy) % gh, (x + dx) % gw] = True

    for x, y in seeds:
        add(int(x) % gw, int(y) % gh, -1)
    target = int(ctx.p("fill") * gw * gh)
    nw = 900
    wx = rng.integers(0, gw, nw); wy = rng.integers(0, gh, nw)
    dxs = np.array([1, -1, 0, 0]); dys = np.array([0, 0, 1, -1])
    steps = 0
    while len(pos) < target and steps < 90000:
        steps += 1
        r = rng.integers(0, 4, nw)
        wx = (wx + dxs[r]) % gw; wy = (wy + dys[r]) % gh
        hit = near[wy, wx] & ~occ[wy, wx]
        if hit.any():
            for i in np.nonzero(hit)[0]:
                x, y = int(wx[i]), int(wy[i])
                if occ[y, x] or rng.random() > stick_map[y, x]:
                    continue
                nb = [idm[(y + dy) % gh, (x + dx) % gw] for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
                nb = [q for q in nb if q >= 0]
                add(x, y, int(nb[rng.integers(len(nb))]))
                wx[i] = rng.integers(0, gw); wy[i] = rng.integers(0, gh)
    parent = np.array(parent); pos = np.array(pos, np.float64)
    n = len(pos)
    desc = np.ones(n)
    for i in range(n - 1, -1, -1):
        if parent[i] >= 0:
            desc[parent[i]] += desc[i]
    cell = np.array([W / gw, H / gh])
    P = (pos + 0.5) * cell + rng.normal(0, 0.15, pos.shape) * cell
    has = np.nonzero(parent >= 0)[0]
    a, b = P[has], P[parent[has]]
    ok = np.hypot(*(a - b).T) < 3 * cell.max()
    has = has[ok]; a, b = a[ok], b[ok]
    age = has / max(n - 1, 1)
    young = np.zeros((H, W), np.float32); old = np.zeros((H, W), np.float32)
    lc.splat_segments(young, a, b, 0.25 + 1.3 * age ** 1.5, step=0.7)
    lc.splat_segments(old, a, b, (np.sqrt(desc[has]) / np.sqrt(desc).max()) * 2.5, step=0.7)
    old = cv2.GaussianBlur(old, (0, 0), 2.0 * sc)
    # tip beads: leaves (no children) glow
    child = np.zeros(n, bool); child[parent[has]] = True
    leaf = np.nonzero(~child & (parent >= 0))[0]
    beads = np.zeros((H, W), np.float32)
    lc.splat(beads, P[leaf, 0], P[leaf, 1], 1.6)
    beads = cv2.GaussianBlur(beads, (0, 0), 1.6 * sc)
    I = young * 0.8 + old * 1.3 + beads * 3.0
    return lc.develop(ctx, I, exposure=1.1 * ctx.p("exposure"), glow=0.8, glow_sigmas=(3, 9, 26), base=lc.ground(ctx))
