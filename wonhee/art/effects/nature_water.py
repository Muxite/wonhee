"""Broad ribbons on dark water: plankton-bloom / tracer-in-a-flow imagery."""
from __future__ import annotations

import math

import cv2
import numpy as np
from scipy import ndimage as ndi

from .. import natkit as N
from ..registry import effect


def flow_field(ctx, sw, sh, rng, size_frac, drift, angle, boundary, src):
    """Stream function psi (so v = curl psi is divergence free) from anisotropic
    noise + a prevailing drift + optional photo-derived part. Multiplying psi by a
    distance-to-coast factor makes the velocity tangent to the coast."""
    s = sw * size_frac
    shear = 1.0 + 5.0 * drift
    # long along the drift, narrow across: the curl is a sheared current along `angle`
    psi = N.fbm(rng, sh, sw, s * (0.6 + 0.9 * drift), octaves=3, gain=0.5, aniso=shear, angle=angle)
    psi = psi + 0.55 * (1 - drift * 0.6) * N.fbm(rng, sh, sw, s * 1.3, octaves=2, gain=0.5)
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    k = float(ctx.p("use_source", 0.0))
    if src is not None and k > 0:
        _, L, theta, coh = src
        Ls = N.pnorm(cv2.GaussianBlur(L, (0, 0), 0.02 * sw))
        mode = ctx.p("source_mode", "luminance")
        if mode in ("luminance", "both"):
            psi = psi * (1 - 0.5 * k) + k * 2.2 * (Ls - 0.5) / (Ls.std() + 1e-6) * 0.5
        if mode in ("orientation", "both"):
            vx0, vy0 = np.cos(theta) * coh, np.sin(theta) * coh
            psi_or = None
    mask = None
    if boundary > 0:
        land = N.fbm(np.random.default_rng(ctx.seed + 77), sh, sw, sw * 0.22, octaves=3) 
        thr = np.quantile(land, 1 - 0.1 - 0.16 * boundary)
        mask = (land > thr).astype(np.float32)
        d = ndi.distance_transform_edt(1 - mask).astype(np.float32)
        fac = N.smooth(0, sw * 0.06, d)
        psi = psi * fac
        mask = cv2.GaussianBlur(mask, (0, 0), 1.2)
    return psi, mask


@effect("nature-plankton-ribbons", family="nature",
        tags=["water", "ribbons", "flow", "calm", "dreamy", "dark", "source", "mood:dreamy", "mood:calm", "mood:cinematic"],
        params=N.with_common({
            "ribbons": (0.5, "STRUCTURE: bloom coverage 0.2-0.9 (more = more of the water carries ribbons)"),
            "stretch": (0.32, "STRUCTURE: how far the flow drags the tracer, in frame widths (longer = longer ribbons)"),
            "eddy": (0.20, "STRUCTURE: size of the stirring eddies as a fraction of the frame width (bigger = broader ribbons)"),
            "drift": (0.8, "STRUCTURE: prevailing current 0..1 (a coherent flow direction survives at 288 px)"),
            "angle": (-0.25, "STRUCTURE: drift direction in radians"),
            "filament": (0.55, "STRUCTURE: 0..1 fine filament texture inside ribbons (multiscale)"),
            "boundary": (0.0, "STRUCTURE: 0..1 coast / ice shelf the flow must follow (stream function vanishes on it)"),
            "depth": (0.5, "SURFACE: 0..1 slow dark tonal variation of the water"),
            "time": (0.0, "COHERENCE: animation time; pattern slides along the drift (coordinate based, video-safe)"),
        }))
def plankton_ribbons(ctx):
    """Phytoplankton-bloom imagery: a passive tracer is dragged by a smooth,
    divergence-free current (stream-function noise plus a prevailing drift), so
    a few BROAD pale ribbons form on dark water, with finer filaments riding on
    them. The current is a stream function, so a coast (boundary) can be added
    that the ribbons hug; a source photo may steer the current (its luminance
    iso-lines become streamlines) and seed the bloom. Few big shapes, one
    coherent direction: reads at 288x192. Palette: dark water -> structure ->
    pale tinted highlight (light polarity gives ink ribbons on paper).
    Differs from physics-vortex/marbling (all-over high-frequency swirl):
    mostly empty water, low contrast water, bright sparse ribbons."""
    W, H = ctx.size
    sw, sh = N.work_size(ctx, 760)
    rng = ctx.rng
    src = N.source_fields(ctx, (sw, sh))
    angle = float(ctx.p("angle"))
    drift = float(ctx.p("drift"))
    psi, mask = flow_field(ctx, sw, sh, rng, float(ctx.p("eddy")), drift, angle, float(ctx.p("boundary")), src)
    vx, vy = N.curl(psi)
    if src is not None and float(ctx.p("use_source")) > 0 and ctx.p("source_mode") in ("orientation", "both"):
        _, _, theta, coh = src
        k = float(ctx.p("use_source"))
        vx = vx / (np.abs(vx).max() + 1e-6) * (1 - 0.6 * k) + np.cos(theta) * k * 0.7 * coh
        vy = vy / (np.abs(vy).max() + 1e-6) * (1 - 0.6 * k) + np.sin(theta) * k * 0.7 * coh
    vm = np.percentile(np.hypot(vx, vy), 95) + 1e-9
    vx, vy = vx / vm, vy / vm
    steps = 60
    dt = float(ctx.p("stretch")) * sw / steps
    px, py = N.trace_map(vx, vy, steps, dt, mask=None)
    t = float(ctx.p("time"))
    if t:
        px = px - t * drift * sw * 0.12 * math.cos(angle) + 6.0 * N.tnoise(px / sw * 4, py / sw * 4, t * 0.4, ctx.seed, 1.0)
        py = py - t * drift * sw * 0.12 * math.sin(angle) + 6.0 * N.tnoise(px / sw * 4 + 9, py / sw * 4, t * 0.4, ctx.seed + 3, 1.0)
    # tracer seed: broad blobs + finer structure, all sampled at the traced positions
    cov = float(ctx.p("ribbons"))
    base = N.fbm(rng, sh, sw, sw * 0.09, octaves=3, gain=0.55)
    k = float(ctx.p("use_source"))
    if src is not None and k > 0:
        base = base + k * 1.6 * (N.pnorm(src[1]) - 0.5) / 0.3
    thr = np.quantile(base, 1 - cov * 0.42)
    c0 = N.smooth(thr - 0.28, thr + 0.32, base)
    c = N.sample(c0, px, py)
    # filaments: fine tracer pulled along the same map (so it forms streaks, not blotches)
    fil = float(ctx.p("filament"))
    f2 = N.sample(N.fbm(rng, sh, sw, sw * 0.018, octaves=3, gain=0.6), px, py)
    f2 = np.clip(f2 * 0.5, -1, 1)
    c = np.clip(c * (1 + fil * 0.55 * f2) + fil * 0.08 * f2 * c0.mean(), 0, 1)
    c = c ** 1.5
    dep = N.fbm(rng, sh, sw, sw * 0.35, octaves=2)
    dep = np.clip(dep - dep.mean(), -1.5, 1.5)          # very large features on a tiny grid: remove the offset
    water = 0.04 + float(ctx.p("depth")) * 0.05 * (dep * 0.4 + 0.6) + 0.02 * f2
    tt = np.clip(water + c * (0.95 - water), 0, 1)
    if mask is not None:
        tt = tt * (1 - mask) + mask * (0.2 + 0.05 * f2)
    tt = N.up(tt, (W, H))
    # full-res fine filaments follow nothing; keep grain in surface instead
    R = N.ramp(ctx.palette, 512, ctx.p("ramp"), float(ctx.p("tint")), float(ctx.p("contrast")), bool(ctx.p("invert")))
    return N.surface(N.paint(tt, R), ctx)
