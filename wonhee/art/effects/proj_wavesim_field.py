"""WaveSim's real update rule as a picture.

Same scheme as WaveSim's CUDA kernel (src/wave/calculate_wave_cuda.cu):
    next = 2*now - prev + speed * (up + down + left + right - 4*now)
with speed = 0.4 and fixed-zero border cells (so waves reflect off the frame,
like the app). Point sources are driven with sines, as in the app. The final
field is mapped through the original image's violet body / warm crest ramp
(`violet`) or WaveSim's own blue-black-red `colorize` (`diverging`)."""
from __future__ import annotations

import numpy as np

from ..palette import oklch_to_rgb
from ..registry import effect
from .physics_common import fit, get_points, sim_size

# (t, L, C, h) stops of the violet-glow ramp: deep violet -> purple -> magenta -> hot coral -> pale warm crest
_VIOLET = [(0.0, 0.17, 0.07, 305), (0.30, 0.30, 0.15, 308), (0.50, 0.42, 0.20, 320),
           (0.70, 0.60, 0.23, 350), (0.86, 0.76, 0.17, 25), (1.0, 0.90, 0.11, 80)]


def _ramp(kind: str, hue: float, n: int = 512) -> np.ndarray:
    if kind == "diverging":  # WaveSim colorize(): blue (neg) -> black (0) -> red (pos), lifted off pure black
        t = np.linspace(-1, 1, n)[:, None]
        neg = np.array([0.10, 0.22, 0.95]); pos = np.array([0.98, 0.18, 0.12]); mid = np.array([0.05, 0.03, 0.10])
        a = np.clip(np.abs(t), 0, 1)
        return (mid * (1 - a) + np.where(t < 0, neg, pos) * a).astype(np.float32)
    shift = hue - 305.0
    ts = np.array([s[0] for s in _VIOLET])
    cols = np.array([oklch_to_rgb(L, C, (h + shift * (0.6 if h < 340 else 0.25)) % 360) for _, L, C, h in _VIOLET])
    x = np.linspace(0, 1, n)
    return np.stack([np.interp(x, ts, cols[:, i]) for i in range(3)], 1).astype(np.float32)


@effect("proj-wavesim-field", family="physics", tags=["wave", "physics", "all-over", "simulation", "wavesim"],
        params={"sources": (7, "point sources when no data"), "steps": (1400, "time steps"), "speed": (0.4, "WaveSim speed (<= 0.5)"),
                "wavelength": (0.045, "wavelength (fraction of frame height)"), "gain": (1.5, "tanh gain"),
                "ramp": ("violet", "violet | diverging"), "hue": (305.0, "violet ramp body hue"),
                "energy": (0.35, "mix of local wave energy into the glow")})
def wavesim_field(ctx):
    """data: (x, y[, w]) sources in 0..1. Exact WaveSim stencil, zero border."""
    W, H = ctx.size
    rng = ctx.rng
    sw, sh = sim_size(ctx, 480)
    xs, ys, ws = get_points(ctx, int(ctx.p("sources")), rng)
    sx = np.clip((xs * sw).astype(int), 2, sw - 3); sy = np.clip((ys * sh).astype(int), 2, sh - 3)
    speed = float(ctx.p("speed"))
    lam = float(ctx.p("wavelength")) * sh
    base = np.sqrt(speed) / lam                      # cycles per step
    freqs = base * (1 + 0.12 * (rng.random(len(sx)) - 0.5) * 2)
    ph = rng.random(len(sx)) * 6.283
    prev = np.zeros((sh + 2, sw + 2), np.float32); now = prev.copy()
    steps = int(ctx.p("steps"))
    en = np.zeros((sh, sw), np.float32)
    for t in range(steps):
        c = now[1:-1, 1:-1]
        nxt = np.zeros_like(now)
        nxt[1:-1, 1:-1] = 2 * c - prev[1:-1, 1:-1] + speed * (now[:-2, 1:-1] + now[2:, 1:-1] + now[1:-1, :-2] + now[1:-1, 2:] - 4 * c)
        for x, y, w, f, p in zip(sx, sy, ws, freqs, ph):
            nxt[y + 1, x + 1] = w * np.sin(2 * np.pi * f * t + p) * min(1.0, t / 60)   # setCell: sine source
        prev, now = now, nxt
        if t > steps - 80:
            en += now[1:-1, 1:-1] ** 2
    u = now[1:-1, 1:-1]
    # sources are point singularities: clip them out so the far field sets the scale
    s = np.percentile(np.abs(u), 93) + 1e-9
    f = np.tanh(float(ctx.p("gain")) * u / s)
    e = np.sqrt(en / 80); e = np.tanh(e / (np.percentile(e, 90) + 1e-9))
    kind = str(ctx.p("ramp"))
    ramp = _ramp(kind, float(ctx.p("hue")))
    if kind == "diverging":
        t01 = 0.5 + 0.5 * f
    else:
        m = float(ctx.p("energy"))
        t01 = np.clip((1 - m) * (0.5 + 0.5 * f) + m * e * 0.9, 0, 1)
    idx = np.clip(t01 * (len(ramp) - 1), 0, len(ramp) - 1)
    i0 = np.minimum(idx.astype(int), len(ramp) - 2); fr = (idx - i0)[..., None]
    img = ramp[i0] * (1 - fr) + ramp[i0 + 1] * fr
    return fit(img, W, H)
