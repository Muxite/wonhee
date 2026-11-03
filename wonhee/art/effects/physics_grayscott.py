"""Gray-Scott reaction-diffusion (U + 2V -> 3V, V -> P): the Turing-pattern
zoo of spots, mitosis, worms and labyrinths. Dense seeding fills the frame."""
from __future__ import annotations

import cv2
import numpy as np

from ..registry import effect
from .physics_common import fit, lookup, source_lum

_REGIMES = {"spots": (0.0367, 0.0649), "worms": (0.039, 0.058), "maze": (0.029, 0.057), "mitosis": (0.0367, 0.0649 - 0.0002),
            "coral": (0.0545, 0.062), "solitons": (0.030, 0.0620), "chaos": (0.026, 0.0510)}


@effect("physics-grayscott", family="physics", tags=["reaction", "physics", "all-over", "turing", "simulation"],
        params={"regime": ("maze", "spots | worms | maze | mitosis | coral | solitons | chaos"),
                "steps": (7000, "time steps"), "cell": (4.2, "sim cell size in px per 1000 px frame width (bigger = finer pattern)"),
                "feed": (0.0, "override feed rate f (0 = regime default)"), "kill": (0.0, "override kill rate k")})
def grayscott(ctx):
    """source: luminance modulates the feed rate, so the pattern changes regime across the image
    (spots in dark areas, labyrinth in bright, ...)."""
    W, H = ctx.size
    rng = ctx.rng
    sw = int(min(W, max(120, 1000 / max(0.5, float(ctx.p("cell"))) * 1.35)))
    sh = max(8, round(sw * H / W))
    f0, k0 = _REGIMES.get(str(ctx.p("regime")), _REGIMES["maze"])
    if float(ctx.p("feed")) > 0:
        f0 = float(ctx.p("feed"))
    if float(ctx.p("kill")) > 0:
        k0 = float(ctx.p("kill"))
    U = np.ones((sh, sw), np.float32)
    V = np.zeros((sh, sw), np.float32)
    n = max(30, sw * sh // 350)
    for _ in range(n):
        x, y = rng.integers(0, sw - 4), rng.integers(0, sh - 4)
        U[y:y + 4, x:x + 4] = 0.5
        V[y:y + 4, x:x + 4] = 0.25 + 0.05 * rng.random()
    V += 0.01 * rng.random((sh, sw)).astype(np.float32)
    lum = source_lum(ctx, sw, sh)
    F = np.full((sh, sw), f0, np.float32) if lum is None else (f0 + 0.012 * (lum - lum.mean()) / (lum.std() + 1e-6) * 0.5).astype(np.float32)
    kern = np.array([[.05, .2, .05], [.2, -1, .2], [.05, .2, .05]], np.float32)
    Du, Dv = 0.2097, 0.105
    for _ in range(int(ctx.p("steps"))):
        lu = cv2.filter2D(U, -1, kern, borderType=cv2.BORDER_REFLECT)
        lv = cv2.filter2D(V, -1, kern, borderType=cv2.BORDER_REFLECT)
        uvv = U * V * V
        U += Du * lu - uvv + F * (1 - U)
        V += Dv * lv + uvv - (F + k0) * V
    t = np.clip(V / max(0.12, np.percentile(V, 99.5)), 0, 1)
    t = 0.04 + 0.96 * t ** 0.85
    return fit(lookup(fit(t, W, H), ctx.palette), W, H)
