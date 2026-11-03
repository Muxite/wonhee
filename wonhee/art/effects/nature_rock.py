"""Granular mineral bands and striated rock (sedimentary strata, agate, cliff faces)."""
from __future__ import annotations

import math

import cv2
import numpy as np

from .. import natkit as N
from ..registry import effect


@effect("nature-strata", family="nature",
        tags=["rock", "strata", "bands", "granular", "striated", "all-over", "source", "mood:calm", "mood:earthy", "mood:precise"],
        params=N.with_common({
            "layers": (16, "STRUCTURE: number of beds across the frame (fewer = broader bands, readable at 288 px)"),
            "irregular": (0.8, "STRUCTURE: 0..1 thickness variation between beds (0 = even bands)"),
            "angle": (-0.12, "STRUCTURE: bedding direction in radians (0 = horizontal)"),
            "fold": (0.3, "STRUCTURE: 0..1 how strongly the beds are warped / folded (multiscale)"),
            "faults": (1, "STRUCTURE: number of fault lines that shear the beds"),
            "grain": (0.5, "SURFACE: mineral grain / speckle strength 0..1"),
            "striation": (0.4, "SURFACE: 0..1 fine streaks along the bedding (anisotropic texture)"),
            "erosion": (0.5, "SURFACE: 0..1 ragged bed boundaries (weathering)"),
            "bedtones": (0.8, "COLOUR: 0..1 how far bed to bed tones differ (lightness contrast between beds)"),
            "hue_beds": (0.6, "COLOUR: 0..1 bed to bed hue variety inside the ramp mode (with ramp=dual/spectrum)"),
        }))
def strata(ctx):
    """Sedimentary strata. A bedding coordinate u (perpendicular to the beds) is
    warped by multiscale folds, sheared by faults and optionally bent by a source
    photo's luminance. Beds are cut from u at lognormal thicknesses; each bed has
    its own tone, grain size and internal gradation (darker at its base), with
    ragged weathered contacts, fine lamination, striations along the bedding and
    sparse mineral grains. Colour is a pure scalar tone through the palette
    ramp, so one hue family by default and bed-to-bed hue drift with
    ramp=dual|spectrum. Reads at 288x192 as ~10 broad bands; at full size it is
    granular. Differs from physics-grains (voronoi blobs): here layered,
    anisotropic and photo-steerable."""
    W, H = ctx.size
    rng = ctx.rng
    sw, sh = N.work_size(ctx, 640)
    ang = float(ctx.p("angle"))
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    ca, sa = math.cos(ang), math.sin(ang)
    u = (-sa * (xx - sw / 2) + ca * (yy - sh / 2)) / sw           # across the beds, in frame widths
    v = (ca * (xx - sw / 2) + sa * (yy - sh / 2)) / sw            # along the beds
    fold = float(ctx.p("fold"))
    u = u + fold * 0.10 * N.fbm(rng, sh, sw, sw * 0.5, octaves=4, gain=0.5, aniso=2.5, angle=ang)
    u = u + fold * 0.05 * np.sin(v * rng.uniform(2.0, 4.0) * math.pi + rng.uniform(0, 6)) * (0.6 + 0.4 * N.fbm(rng, sh, sw, sw * 0.4, 2))
    k = float(ctx.p("use_source"))
    src = N.source_fields(ctx, (sw, sh))
    if src is not None and k > 0:
        Ls = N.pnorm(cv2.GaussianBlur(src[1], (0, 0), 0.012 * sw))
        u = u + k * 0.28 * (Ls - 0.5)
    for i in range(int(ctx.p("faults"))):
        a0 = rng.uniform(0.9, 2.2) if i % 2 == 0 else rng.uniform(1.0, 2.1)
        px, py = rng.uniform(0.2, 0.8) * sw, rng.uniform(0.2, 0.8) * sh
        side = ((xx - px) * math.sin(a0) - (yy - py) * math.cos(a0)) / sw
        u = u + (side > 0) * rng.uniform(-0.06, 0.06) * (0.4 + 0.6 * N.smooth(0, 0.15, np.abs(side)) * 0 + 0.6)
    uf = N.up(u, (W, H))
    # erosion: ragged contacts at full resolution
    ero = float(ctx.p("erosion"))
    rf = N.noise(rng, H, W, 3.0 * N.sc(ctx) + 0.7, 7.0 * N.sc(ctx) + 1.0, ang)
    uf = uf + ero * 0.006 * rf * (W / 1500.0 + 0.4)
    n = int(ctx.p("layers"))
    irr = float(ctx.p("irregular"))
    th = np.exp(rng.normal(0, 0.55 * irr, n * 3 + 8)).astype(np.float64)
    th /= th[: n + 2].mean()
    edges = np.cumsum(th) / n                                    # bed boundaries in frame widths
    edges = edges - edges[len(edges) // 3]
    span = float(np.abs(uf).max()) + 0.1
    idx = np.searchsorted(edges, uf).clip(0, len(edges) - 1)
    lo = np.concatenate([[edges[0] - th[0] / n], edges])[idx]
    hi = edges[idx]
    fr = ((uf - lo) / np.maximum(hi - lo, 1e-6)).astype(np.float32)   # 0 base .. 1 top
    rs = np.random.default_rng(ctx.seed + 11)
    nb = len(edges) + 1
    tone = np.zeros(nb)
    for i in range(nb):                                         # AR(1): neighbouring beds correlate a little
        tone[i] = 0.55 * (tone[i - 1] if i else 0) + rs.normal(0, 1)
    tone = (tone - tone.min()) / (np.ptp(tone) + 1e-9)
    tone = 0.5 + (tone - 0.5) * float(ctx.p("bedtones"))
    coarse = rs.uniform(0.3, 1.0, nb)
    hv = rs.uniform(-1, 1, nb) * float(ctx.p("hue_beds"))
    t = tone[idx].astype(np.float32)
    t = t + (fr - 0.5) * 0.22 * coarse[idx].astype(np.float32)   # gradation inside a bed
    s = N.sc(ctx)
    # lamination: thin lines along the bedding whose frequency differs per bed
    lam = np.sin(uf * n * (3 + 9 * coarse[idx]) * 2 * math.pi) * 0.04
    # striations along the bedding + mineral speckle
    st = float(ctx.p("striation"))
    sg = float(ctx.p("grain"))
    stri = N.noise(rng, H, W, 22 * s + 2, 0.8 * s + 0.5, ang)
    gr = N.noise(rng, H, W, 0.9 * s + 0.5, 0.9 * s + 0.5)
    speck = np.clip(N.noise(rng, H, W, 0.8 * s + 0.4) - 1.6, 0, 3) * 0.12
    t = t + st * 0.11 * stri + sg * (0.07 * gr * (0.5 + coarse[idx]) + speck) + lam * st
    # darker thin seams at contacts
    seam = np.exp(-(np.minimum(fr, 1 - fr) / 0.03) ** 2).astype(np.float32)
    t = t - 0.10 * seam * (0.4 + ero)
    t = np.clip(t, 0, 1)
    R = N.ramp(ctx.palette, 768, ctx.p("ramp"), float(ctx.p("tint")), float(ctx.p("contrast")), bool(ctx.p("invert")), hi=0.86 if ctx.palette.polarity == "dark" else None)
    out = N.paint(t, R)
    # bed-to-bed hue variety: a second ramp lookup, slightly shifted tint, blended per bed
    if ctx.p("ramp") != "mono" and float(ctx.p("hue_beds")) > 0:
        R2 = N.ramp(ctx.palette, 768, ctx.p("ramp"), float(ctx.p("tint")) + 55, float(ctx.p("contrast")), bool(ctx.p("invert")), hi=0.86 if ctx.palette.polarity == "dark" else None)
        w = np.clip(hv[idx] * 0.5 + 0.5, 0, 1).astype(np.float32)[..., None]
        out = out * (1 - w * 0.6) + N.paint(t, R2) * w * 0.6
    return N.surface(out, ctx)
