"""Cloud layers and aerial cloud fields; broad aurora sheets."""
from __future__ import annotations

import math

import cv2
import numpy as np

from .. import natkit as N
from ..registry import effect


def _shift(a, dx, dy):
    """Sub-pixel translation with wrap (so a layer drifts coherently with time)."""
    h, w = a.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def _layer_density(rng, sh, sw, size, kind, angle, aniso, warp):
    if kind == "cumulus":
        n = np.zeros((sh, sw), np.float32)
        amp, sg = 1.0, size
        for _ in range(4):
            n += amp * np.abs(N.noise(rng, sh, sw, sg, sg * 0.8, angle))     # billows: round domes, creased valleys
            amp *= 0.52
            sg /= 2.0
        n = n + 0.6 * N.fbm(rng, sh, sw, size * 2.0, 2)
    elif kind == "stratus":
        n = N.fbm(rng, sh, sw, size * 1.5, 5, 0.6, aniso=aniso * 2.2, angle=angle)
        n = n + 0.5 * np.abs(N.noise(rng, sh, sw, size * 0.5, size * 0.12, angle))
    elif kind == "cirrus":
        n = N.fbm(rng, sh, sw, size * 1.4, 4, 0.5, aniso=aniso * 5.0, angle=angle)
        n = n + 0.35 * np.abs(N.noise(rng, sh, sw, size * 0.9, size * 0.07, angle))
    else:                            # streets: parallel rolls of cumulus
        n = N.fbm(rng, sh, sw, size, 4, 0.55, aniso=aniso * 1.6, angle=angle)
        yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
        n = n + 0.45 * np.sin((-math.sin(angle) * xx + math.cos(angle) * yy) / (size * 0.9) * math.pi)
        n = n + 0.5 * np.abs(N.noise(rng, sh, sw, size * 0.35, size * 0.35))
    if warp > 0:
        wx = warp * size * 0.5 * N.fbm(rng, sh, sw, size * 2.2, 2)
        wy = warp * size * 0.5 * N.fbm(rng, sh, sw, size * 2.2, 2)
        gx, gy = np.meshgrid(np.arange(sw, dtype=np.float32), np.arange(sh, dtype=np.float32))
        n = cv2.remap(n, gx + wx, gy + wy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return (n - n.mean()) / (n.std() + 1e-6)


@effect("nature-cloudfield", family="nature",
        tags=["clouds", "sky", "aerial", "layers", "calm", "dreamy", "atmosphere", "source", "mood:calm", "mood:dreamy", "mood:cinematic"],
        params=N.with_common({
            "kind": ("cumulus", "STRUCTURE: cumulus (billowy cells) | stratus (long decks) | cirrus (fibres) | streets (cloud rolls)"),
            "layers": (2, "STRUCTURE: stacked cloud decks 1-5, each lit and shadowing the one below"),
            "size": (0.16, "STRUCTURE: biggest cloud feature as a fraction of frame width"),
            "cover": (0.45, "STRUCTURE: 0.15-0.8 cloud coverage (the rest shows the dark ground / sea / sky)"),
            "angle": (-0.2, "STRUCTURE: wind direction in radians (anisotropy and layer drift)"),
            "aniso": (1.6, "STRUCTURE: stretch of the features along the wind"),
            "warp": (0.4, "STRUCTURE: domain warp, how torn / billowed the edges are"),
            "sun": (2.4, "SURFACE: light azimuth in radians; clouds are lit from that side and cast soft shade"),
            "depth": (0.7, "SURFACE: 0..1 shading strength / thickness cue"),
            "time": (0.0, "COHERENCE: animation time; each layer drifts at its own parallax speed (video-safe)"),
        }))
def cloudfield(ctx):
    """Aerial cloud decks over a dark ground. Each layer is a thresholded
    anisotropic density field (billow noise for cumulus, stretched fbm for
    stratus/cirrus, rolls for streets) with domain-warped edges; brightness is
    lit-from-the-sun shading (density gradient along the light azimuth), thin
    edges translucent, thick cores bright; lower layers are shadowed by upper
    ones. Output is a single lightness ramp (ground dark -> clouds pale) so
    colour is independent; a source photo's luminance raises/lowers the cover
    (clouds gather where the photo is bright). Few big shapes: reads at 288 px.
    Differs from physics-grains/swift-hohenberg (no clouds there) and
    aurora-curtains (vertical)."""
    W, H = ctx.size
    rng = ctx.rng
    sw, sh = N.work_size(ctx, 640)
    kind = str(ctx.p("kind"))
    n = int(ctx.p("layers"))
    ang = float(ctx.p("angle"))
    t = float(ctx.p("time"))
    dark = ctx.palette.polarity == "dark"
    sun = float(ctx.p("sun"))
    lx, ly = math.cos(sun), math.sin(sun)
    src = N.source_fields(ctx, (sw, sh))
    k = float(ctx.p("use_source"))
    ground = 0.05 + 0.05 * N.fbm(rng, sh, sw, sw * 0.12, 3)
    bright = ground.copy()
    shadow_from_above = np.zeros((sh, sw), np.float32)
    sh_in = float(ctx.p("depth"))
    for li in range(n):
        frac = li / max(1, n - 1) if n > 1 else 0.5
        size = sw * float(ctx.p("size")) * (1.2 - 0.6 * frac)
        d = _layer_density(rng, sh, sw, size, kind, ang, float(ctx.p("aniso")), float(ctx.p("warp")))
        speed = (0.6 + 0.9 * frac) * sw * 0.05
        if t:
            d = _shift(d, t * speed * math.cos(ang), t * speed * math.sin(ang))
        if src is not None and k > 0:
            d = d + k * 1.6 * (N.pnorm(src[1]) - 0.5) * 2
        thr = np.quantile(d, 1 - float(ctx.p("cover")) * (0.95 + 0.15 * (1 - frac)))
        soft = 2.0 if kind == "cumulus" else 3.2
        T = N.smooth(thr - 0.05, thr + soft, d)                    # cloud thickness, hard-ish edge, domed core
        a = N.smooth(thr - 0.05, thr + 0.45, d)                   # coverage (opacity)
        # rounded lighting: normal from a slightly smoothed thickness, plus a broad term
        Ts = cv2.GaussianBlur(T, (0, 0), 0.010 * sw)
        gy, gx = np.gradient(Ts)
        lit = np.clip(0.5 + (gx * lx + gy * ly) * -sw * (0.22 if kind == "cumulus" else 0.10), 0, 1)
        Tb = cv2.GaussianBlur(T, (0, 0), 0.03 * sw)
        gyb, gxb = np.gradient(Tb)
        big = np.clip(0.5 - (gxb * lx + gyb * ly) * sw * 0.9, 0, 1)
        cloud_b = 0.30 + 0.34 * T + 0.34 * (0.6 * lit + 0.4 * big) * (0.45 + 0.55 * T)
        cloud_b = cloud_b - sh_in * 0.20 * (1 - big) * (0.3 + 0.7 * T) - shadow_from_above * 0.20 * sh_in
        cloud_b = np.clip(cloud_b, 0.2, 0.97)
        # thin edges are translucent: they show the layer below
        bright = bright * (1 - a) + cloud_b * a
        sm = _shift(a, -lx * sw * 0.015, -ly * sw * 0.015)
        shadow_from_above = np.maximum(shadow_from_above, cv2.GaussianBlur(sm, (0, 0), 0.012 * sw))
    tt = bright if dark else 0.64 * (1 - bright) + 0.02
    tt = N.up(tt, (W, H))
    R = N.ramp(ctx.palette, 640, ctx.p("ramp"), float(ctx.p("tint")), float(ctx.p("contrast")), bool(ctx.p("invert")), hi=0.95 if dark else None)
    return N.surface(N.paint(tt, R), ctx)


@effect("nature-aurora-sheets", family="nature",
        tags=["aurora", "sky", "curtains", "dreamy", "cinematic", "atmosphere", "source", "mood:dreamy", "mood:cinematic", "mood:calm"],
        params=N.with_common({
            "sheets": (3, "STRUCTURE: curtains at different distances (parallax: far ones are narrower and fainter)"),
            "fold": (0.45, "STRUCTURE: how far each curtain meanders in depth; folds turn edge-on and flare"),
            "reach": (0.55, "STRUCTURE: curtain height as a fraction of the frame height (taller = fills more sky)"),
            "rays": (0.45, "STRUCTURE: 0..1 vertical ray striation inside a curtain (0 = smooth veil)"),
            "slant": (0.0, "STRUCTURE: shear of the whole system in radians (the sheets lean)"),
            "altitude_hue": (0.7, "COLOUR: 0..1 hue change with altitude (green-ish base to violet / red crown) via a second ramp"),
            "hue_shift": (70.0, "COLOUR: degrees the crown hue sits away from the base hue"),
            "exposure": (1.0, "SURFACE: exposure"),
            "time": (0.0, "COHERENCE: animation time; the curtains ripple sideways (coordinate based, video-safe)"),
        }))
def aurora_sheets(ctx):
    """Aurora as geometry, not as vertical stripes: each curtain is a ribbon
    surface whose ground track meanders in depth, projected in perspective and
    splatted additively. Where a fold turns edge-on the projected density rises
    (real foreshortening flare); distant sheets are smaller and dimmer. The
    ray structure is multiscale noise along the track, brightness falls
    exponentially upward from a bright lower rim, and hue changes with
    altitude through a second ramp. Gives broad draped curtains that cross the
    frame diagonally and overlap, where aurora-curtains is a row of vertical
    bars with black gaps. source: its luminance along x modulates ray strength
    (and the second hue). Reads at 288x192 as 2-4 soft broad curtains."""
    W, H = ctx.size
    rng = ctx.rng
    sc_ = N.sc(ctx)
    n = int(ctx.p("sheets"))
    t = float(ctx.p("time"))
    nsm = int(2600 * max(0.5, W / 1500))
    nh = int(260 * max(0.5, W / 1500))
    I = np.zeros((H, W), np.float32)
    Hc = np.zeros((H, W), np.float32)
    src = N.source_fields(ctx, (W // 4, H // 4))
    ky = float(ctx.p("use_source"))
    slant = float(ctx.p("slant"))
    s = np.linspace(-1.0, 1.0, nsm)
    for i in range(n):
        near = 1.0 - 0.28 * i
        Z0 = 1.0 + 0.9 * i / max(1, n - 1) * 1.6
        # ground track in plan: x = s * span, depth z = Z0 + fold * low-frequency noise
        zr = N.noise(rng, 1, 64, 8, normalise=True)[0]
        zr = np.interp(s, np.linspace(-1, 1, 64), zr)
        ph = rng.uniform(0, 6.28)
        zz = Z0 * (1 + float(ctx.p("fold")) * (0.35 * zr + 0.35 * np.sin(s * rng.uniform(1.6, 3.2) * math.pi + ph + t * 0.35)))
        zz = np.maximum(zz, 0.35 * Z0)
        xx = s * 1.5 * Z0 + 0.16 * Z0 * np.sin(s * 5 + ph + 0.6 * t) * float(ctx.p("fold"))
        sx = W * 0.5 + W * 0.72 * xx / zz
        # rays along the track (also moves with time via a space-time noise)
        r1 = N.tnoise(s * 60, s * 0, t * 0.5, ctx.seed + i, 1.0)
        r2 = N.tnoise(s * 14, s * 0 + 5, t * 0.3, ctx.seed + 31 * i, 1.0)
        rays = np.clip(0.62 + float(ctx.p("rays")) * 1.0 * r1 + 0.7 * r2, 0.05, None) ** 1.6
        if src is not None and ky > 0:
            lx = np.clip(((sx / W) * src[1].shape[1]).astype(int), 0, src[1].shape[1] - 1)
            col = src[1].mean(0)[lx]
            rays = rays * (1 - ky + ky * (0.3 + 1.4 * (col - col.min()) / (np.ptp(col) + 1e-6)))
        hs = np.linspace(0, 1, nh) ** 1.3
        base_y = H * (0.80 - 0.11 * i + rng.uniform(-0.04, 0.04)) + 0.07 * H * np.sin(s * 3 + ph) + 0.05 * H * zr
        reach = float(ctx.p("reach")) * H * (0.55 + 0.45 / zz) * (0.7 + 0.5 * np.clip(rays / (rays.mean() + 1e-6), 0, 2) / 2)
        fall = np.exp(-hs[None, :] * 1.9)
        # foreshortening is implicit in the splat density; add the slope flare explicitly for edge-on folds
        slope = np.abs(np.gradient(sx)) / (np.abs(np.gradient(sx)).mean() + 1e-9)
        flare = 1.0 / np.sqrt(np.clip(slope, 0.5, 3.0))
        px = np.broadcast_to(sx[:, None], (nsm, nh)) + slant * (hs[None, :] * reach[:, None])
        py = base_y[:, None] - hs[None, :] * reach[:, None]
        w = (rays * flare * near * 0.55 * np.clip(slope, 0.2, 1.0) ** 0.9)[:, None] * fall * (0.25 + 1.0 * np.exp(-hs[None, :] * 45))
        wtot = (w * (nsm / 2600)).astype(np.float64)
        lc_splat(I, px.ravel(), py.ravel(), wtot.ravel())
        lc_splat(Hc, px.ravel(), py.ravel(), (wtot * hs[None, :]).ravel())
    I = cv2.GaussianBlur(I, (0, 0), 1.3 * sc_ + 0.5)
    Hc = cv2.GaussianBlur(Hc, (0, 0), 1.3 * sc_ + 0.5)
    alt = np.clip(Hc / (I + 1e-6), 0, 1)
    I = cv2.GaussianBlur(I, (0, 0), 2.5 * sc_) * 0.55 + I * 0.45
    ex = 1 - np.exp(-1.7 * float(ctx.p("exposure")) * I / (np.percentile(I, 99.0) + 1e-9))
    tv = np.clip(ex, 0, 1)
    sky = 0.04 + 0.03 * N.up(N.fbm(rng, max(8, H // 8), max(8, W // 8), 9, 3), (W, H))
    tt = np.clip(sky + tv * 0.92, 0, 1)
    R1 = N.ramp(ctx.palette, 640, ctx.p("ramp"), float(ctx.p("tint")), float(ctx.p("contrast")), bool(ctx.p("invert")))
    R2 = N.ramp(ctx.palette, 640, ctx.p("ramp"), float(ctx.p("tint")) + float(ctx.p("hue_shift")), float(ctx.p("contrast")), bool(ctx.p("invert")))
    h2 = float(ctx.p("altitude_hue"))
    wgt = (N.smooth(0.25, 0.8, alt) * h2 * N.smooth(0.05, 0.4, tv))[..., None]
    out = N.paint(tt, R1) * (1 - wgt) + N.paint(tt, R2) * wgt
    return N.surface(out, ctx)


def lc_splat(buf, x, y, w):
    from . import light_common as lc
    lc.splat(buf, x, y, w)
