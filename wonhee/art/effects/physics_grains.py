"""Grain growth by curvature flow (Merriman-Bence-Osher threshold dynamics):
random multi-phase domains coarsen into a foam of rounded grains, as in
polycrystal metals and soap froth. Grains shaded by distance to the boundary."""
from __future__ import annotations

import cv2
import numpy as np

from ..registry import effect
from .physics_common import fit, lookup, sim_size, source_lum


@effect("physics-grains", family="physics", tags=["coarsening", "physics", "all-over", "foam", "simulation"],
        params={"phases": (7, "number of grain phases (lightness levels)"), "steps": (55, "coarsening iterations (more = bigger, rounder grains)"),
                "sigma": (1.7, "diffusion width per step (sim px)"), "lines": (0.55, "grain-boundary line strength")})
def grains(ctx):
    """source: luminance quantised into phases is the initial condition, so the grain map coarsens the image."""
    W, H = ctx.size
    rng = ctx.rng
    sw, sh = sim_size(ctx, 360)
    q = int(ctx.p("phases"))
    lab = rng.integers(0, q, (sh, sw))
    lum = source_lum(ctx, sw, sh)
    if lum is not None:
        qs = np.quantile(lum, np.linspace(0, 1, q + 1)[1:-1])
        lab = np.digitize(lum, qs)
        lab = np.where(rng.random((sh, sw)) < 0.25, rng.integers(0, q, (sh, sw)), lab)
    sg = float(ctx.p("sigma"))
    probs = None
    for it in range(int(ctx.p("steps"))):
        probs = np.stack([cv2.GaussianBlur((lab == k).astype(np.float32), (0, 0), sg, borderType=cv2.BORDER_REFLECT) for k in range(q)])
        lab = probs.argmax(0)
    # smooth boundaries at full res: upsample probability maps then argmax
    big = np.stack([cv2.resize(cv2.GaussianBlur((lab == k).astype(np.float32), (0, 0), 1.0), (W, H), interpolation=cv2.INTER_CUBIC) for k in range(q)])
    L = big.argmax(0)
    edge = np.zeros((H, W), bool)
    edge[:-1] |= L[:-1] != L[1:]
    edge[:, :-1] |= L[:, :-1] != L[:, 1:]
    dist = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 3)
    levels = rng.permutation(q) / max(1, q - 1)
    base = 0.14 + 0.52 * levels[L]
    scale = W / sw * 5.0
    shade = 1 - np.exp(-dist / scale)
    t = base * (0.75 + 0.25 * shade) + 0.10 * shade
    ln = cv2.GaussianBlur(edge.astype(np.float32), (0, 0), 0.9)
    t = t + float(ctx.p("lines")) * np.clip(ln * 2.2, 0, 1) * (0.9 - t)
    return lookup(np.clip(t, 0, 1), ctx.palette).astype(np.float32)
