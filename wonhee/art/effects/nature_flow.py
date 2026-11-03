"""Photo-derived flow: the photo's own structure is the current that drags it."""
from __future__ import annotations

import math

import numpy as np

from .. import features as F
from .. import natkit as N
from ..registry import effect


@effect("nature-flow-wash", family="nature",
        tags=["photo", "source", "flow", "smear", "painterly", "dreamy", "calm", "anisotropic", "mood:dreamy", "mood:calm", "mood:cinematic"],
        params=N.with_common({
            "smear": (0.22, "STRUCTURE: drag distance in frame widths (broad = long coherent strokes)"),
            "direction": ("orientation", "STRUCTURE: orientation (along the photo's own structure) | contour (along its luminance iso-lines) | drift (one global direction) | curl (stirred current)"),
            "angle": (0.0, "STRUCTURE: global drift direction in radians (also the bias for orientation/contour)"),
            "bias": (0.5, "STRUCTURE: 0..1 weight of the global drift against the photo's own flow"),
            "coherence": (0.6, "STRUCTURE: 0..1 how much the field is smoothed (high = calm broad flow, low = detailed)"),
            "detail": (0.35, "SOURCE: 0..1 how much sharp photo is kept in the strongest-structure areas"),
            "colour_mode": ("photo", "COLOUR: photo (keeps the photo's colour, re-graded by tint/contrast) | ramp (luminance through the palette ramp)"),
            "time": (0.0, "COHERENCE: animation time; drag distance grows smoothly, drift is coordinate based"),
        }, source=False) | {"crop": N.SOURCE_PARAMS["crop"]})
def flow_wash(ctx):
    """The source photo is advected along a flow derived from the photo:
    its structure-tensor orientation (strokes follow edges), its luminance
    iso-contours, a single drift, or a divergence-free curl field, optionally
    biased by a global direction. The result keeps recognisable big forms while
    dissolving detail into long strokes (water-colour wash / wind-blown paint /
    long-exposure drag). colour_mode=ramp discards the photo hue and paints the
    smeared luminance through the palette; 'photo' keeps and re-grades it.
    Needs a source (a synthetic stand-in is used otherwise). Failure modes:
    flat low-contrast photos have no structure, so use direction=drift/curl."""
    W, H = ctx.size
    sw, sh = N.work_size(ctx, 900)
    src = ctx.source if ctx.source is not None else F.synthetic_source((sw, sh), ctx.seed)
    img = F.fit_cover(np.asarray(src, np.float32)[..., :3], (sw, sh), ctx.p("crop", "feature"))
    L = F.luminance(img)
    Le = F.stretch(F.local_equalize(L, 2.0, mix=0.5), 1, 99)
    s0 = F.scale_of((sh, sw))
    coh_s = float(ctx.p("coherence"))
    theta, coh = F.structure_orientation(F.blur(Le, 1.0 * s0), sigma=1.5, rho=(4 + 26 * coh_s), seed=ctx.seed)
    mode = str(ctx.p("direction"))
    ang = float(ctx.p("angle"))
    bias = float(ctx.p("bias"))
    rng = ctx.rng
    if mode == "orientation":
        vx, vy = np.cos(theta), np.sin(theta)
        # make the sign consistent with the global direction so strokes do not cancel
        sg = np.sign(vx * math.cos(ang) + vy * math.sin(ang) + 1e-6)
        vx, vy = vx * sg * (0.35 + 0.65 * coh), vy * sg * (0.35 + 0.65 * coh)
    elif mode == "contour":
        g = F.blur(Le, (3 + 14 * coh_s) * s0)
        gy, gx = np.gradient(g)
        vx, vy = -gy, gx
        m = np.hypot(vx, vy) + 1e-9
        vx, vy = vx / (np.percentile(m, 90) + 1e-9), vy / (np.percentile(m, 90) + 1e-9)
        vx, vy = np.clip(vx, -1.2, 1.2), np.clip(vy, -1.2, 1.2)
    elif mode == "curl":
        psi = N.fbm(rng, sh, sw, sw * (0.10 + 0.15 * coh_s), 3, 0.5)
        vx, vy = N.curl(psi)
        vm = np.percentile(np.hypot(vx, vy), 95) + 1e-9
        vx, vy = vx / vm, vy / vm
    else:
        vx, vy = np.full((sh, sw), math.cos(ang), np.float32), np.full((sh, sw), math.sin(ang), np.float32)
    if mode != "drift":
        vx = vx * (1 - bias * 0.7) + math.cos(ang) * bias * 0.7
        vy = vy * (1 - bias * 0.7) + math.sin(ang) * bias * 0.7
    vx, vy = vx.astype(np.float32), vy.astype(np.float32)
    steps = 56
    tm = float(ctx.p("time"))
    dist = float(ctx.p("smear")) * sw * (1 + 0.5 * math.sin(tm * 0.5))
    out = N.smear_along(img, vx, vy, steps, dist / steps, keep_sign=(mode in ("orientation", "contour")))
    det = float(ctx.p("detail"))
    if det > 0:
        w = (F.smoothstep(0.55, 0.95, coh) * det)[..., None]
        out = out * (1 - w) + img * w
    out = N.up(out, (W, H)) if out.shape[1] != W else out
    if ctx.p("colour_mode") == "ramp":
        lum = F.luminance(np.clip(out, 0, 1))
        lum = F.stretch(lum, 1, 99)
        R = N.ramp(ctx.palette, 640, ctx.p("ramp", "mono"), float(ctx.p("tint", 0.0)), float(ctx.p("contrast", 1.0)), bool(ctx.p("invert", False)))
        rgb = N.paint(lum, R)
    else:
        c = float(ctx.p("contrast", 1.0))
        rgb = np.clip((out - 0.5) * c + 0.5, 0, 1)
        if float(ctx.p("tint", 0.0)):
            lab = F.to_oklab(rgb)
            h = math.radians(float(ctx.p("tint")))
            a2, b2 = lab[..., 1].copy(), lab[..., 2].copy()
            lab[..., 1] = a2 * math.cos(h) - b2 * math.sin(h)
            lab[..., 2] = a2 * math.sin(h) + b2 * math.cos(h)
            rgb = F.from_oklab(lab)
    return N.surface(rgb, ctx)
