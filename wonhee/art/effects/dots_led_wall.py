"""dots-led-wall: an LED display wall. A square grid of round LEDs (dim 'off'
dots everywhere, lit ones with a glow) filled edge to edge with horizontal
bands of scrolling data: level-meter bars, hex ticker text, sparklines, bit
runs. Every band is equally busy, so there is no focal point.

ctx.data (optional): a list of numeric series (one per bar/spark band, cycled)
or a str / list of str used for the ticker text bands. ctx.source unused."""
from __future__ import annotations

import numpy as np

from .. import dotkit as dk
from ..registry import effect


def _series(rng, n, smooth=3):
    a = rng.random(n + 12)
    k = np.ones(smooth) / smooth
    return np.convolve(a, k, "same")[6:n + 6]


@effect("dots-led-wall", family="dots", tags=["dots", "led", "data", "all-over", "glow"],
        params={"cols": (64, "LEDs across"), "glow": (0.45, "glow strength 0..2"), "ticker": ("0x3F A1 7C 9E 52 D0 4B", "ticker text (or ctx.data str)"),
                "bands": (9, "approx number of bands"), "shape": ("round", "LED shape: round | square | diamond")})
def dots_led_wall(ctx):
    W, H = ctx.size
    rng = ctx.rng
    P = ctx.palette
    cols = int(ctx.p("cols"))
    pitch = W / cols
    rows = max(8, int(H / pitch))
    m = np.zeros((rows, cols), np.float32)
    data = ctx.data
    series = [np.asarray(s, np.float32) for s in data if np.ndim(s) == 1] if isinstance(data, (list, tuple)) and data and np.ndim(data[0]) == 1 else []
    texts = [d for d in data if isinstance(d, str)] if isinstance(data, (list, tuple)) else ([data] if isinstance(data, str) else [])
    if not texts:
        texts = [str(ctx.p("ticker"))]
    r = 0
    bi = 0
    nb = int(ctx.p("bands"))
    kinds = ["bars", "ticker", "spark", "bits", "bars", "ticker", "bars", "spark", "bits", "bars"]
    target = rows / nb
    while r < rows:
        kind = kinds[bi % len(kinds)] if bi % 3 else str(rng.choice(["bars", "spark", "bits", "ticker"]))
        h = 7 if kind == "ticker" else int(np.clip(round(target * (0.7 + 0.8 * rng.random())), 3, 9))
        h = min(h, rows - r)
        if h < 2:
            break
        band = np.zeros((h, cols), np.float32)
        if kind == "bars":
            s = series[bi % len(series)] if series else _series(rng, cols, 2)
            s = np.interp(np.linspace(0, len(s) - 1, cols), np.arange(len(s)), s)
            s = (s - s.min()) / (np.ptp(s) + 1e-6)
            hh = np.clip(np.round(0.1 * h + s * 0.9 * h).astype(int), 1, h)
            for c in range(cols):
                if c % 4 == 3 and rng.random() < 0.0:
                    continue
                for k in range(hh[c]):
                    band[h - 1 - k, c] = 0.42 + 0.34 * (k + 1) / h
                band[h - hh[c], c] = 0.98 if hh[c] >= h - 1 else 0.8
        elif kind == "spark":
            s = _series(rng, cols, 5) if not series else np.interp(np.linspace(0, len(series[0]) - 1, cols), np.arange(len(series[0])), series[bi % len(series)])
            s = (s - s.min()) / (np.ptp(s) + 1e-6)
            yy = np.clip(np.round((1 - s) * (h - 1)).astype(int), 0, h - 1)
            for c in range(cols):
                band[yy[c], c] = 0.95
                if c and abs(yy[c] - yy[c - 1]) > 1:
                    lo, hi = sorted((yy[c], yy[c - 1]))
                    band[lo:hi + 1, c if rng.random() < .5 else c - 1] = 0.6
                band[yy[c] + 1:, c] = np.maximum(band[yy[c] + 1:, c], 0.0)
            band[band == 0] = 0
        elif kind == "bits":
            runs = rng.random((h, cols)) < 0.5
            for y in range(h):
                runs[y] = np.repeat(rng.random(cols // 3 + 1) < 0.5, 3)[:cols]
            band = np.where(runs, 0.55 + 0.1 * rng.random((h, cols)), 0).astype(np.float32)
        else:
            t = texts[bi % len(texts)]
            band[:] = 0
            tm = dk.tile_text(t, h, cols, row0=max(0, (h - 7) // 2), col0=int(rng.integers(0, 20)), gap=4).astype(np.float32)
            band = tm * (0.78 + 0.2 * rng.random((h, cols)))
            if h > 7:
                band[:] = np.pad(tm[:7], ((0, h - 7), (0, 0)))[:h] * 0.8
        m[r:r + h] = band
        r += h + 1
        bi += 1
    # sprinkle a few accent LEDs evenly
    lit = np.argwhere(m > 0.7)
    if len(lit):
        pick = rng.choice(len(lit), size=min(len(lit), max(3, len(lit) // 14)), replace=False)
        m[lit[pick, 0], lit[pick, 1]] = 1.0
    g = dk.grid(rows, cols, ctx.size, pitch=pitch, origin=(0, (H - rows * pitch) / 2))
    ramp = dk.palette_ramp(P)
    off = P.canvas + (P.structure - P.canvas) * 0.20
    bg = P.canvas
    img = dk.render_dots(m, ctx.size, g, bg=bg, ramp=ramp, shape=str(ctx.p("shape")), size_const=True, r_max=0.40,
                         off=off, off_r=0.15, threshold=0.05, glow_sigma=pitch * 0.55, glow_strength=float(ctx.p("glow")), core=0.5, normalize=False)
    return img
