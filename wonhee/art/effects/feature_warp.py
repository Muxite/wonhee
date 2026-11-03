"""Physics / displacement edits of a source: the image as a wave medium, seen
through a ripple surface, sliced by its own edge strength."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from .. import features as F
from ..registry import effect
from .feature_common import prep, with_common, mixc, finish, grade


def field_big(f01):
    return f01 - 0.5


def _sources(rng, n, w, h):
    if n <= 0:
        return []
    gx = int(np.ceil(np.sqrt(n * w / h))); gy = int(np.ceil(n / gx))
    cells = [(i, j) for i in range(gx) for j in range(gy)]
    rng.shuffle(cells)
    out = []
    for i, j in cells[:n]:
        out.append(((i + 0.2 + 0.6 * rng.random()) / gx * w, (j + 0.2 + 0.6 * rng.random()) / gy * h,
                    0.6 + 0.8 * rng.random(), 0.8 + 0.5 * rng.random(), rng.random() * 6.28))
    return out


@effect("feature-wave-field", family="feature", tags=["source", "wave", "physics", "simulation", "all-over", "medium"],
        params=with_common({"sources": (5, "point sources on a jittered grid"), "steps": (130, "time steps"), "scale": (3, "sim downscale"),
                            "medium": (0.5, "0..1 how strongly image luminance bends the waves (wave speed)"),
                            "bands": (2.5, "interference band density"),
                            "imprint": (1.0, "0..1 initial condition: the image's high-pass detail is the starting disturbance"),
                            "show": (0.6, "0..1 source tone mixed into the picture so the subject reads")}))
def wave_field(ctx):
    """The picture as a wave medium. Luminance sets the local wave speed
    (bright = fast), the image's own high-pass detail is the starting
    disturbance, and jittered point sources keep ringing: a leapfrog 2D wave
    simulation run to a still, coloured through the palette. Edges of the
    source reflect and refract the rings, so its structure is drawn BY the
    interference. Works on any source with tonal structure; heatmaps and
    plots become rippled terraces; a flat source gives plain ripples."""
    P = prep(ctx)
    rng = ctx.rng
    sc = int(ctx.p("scale"))
    w, h = P.W // sc, P.H // sc
    ts = cv2.resize(P.t, (w, h), interpolation=cv2.INTER_AREA)
    med = float(ctx.p("medium"))
    c2 = (0.08 + 0.30 * (1 - med + med * F.blur(ts, 1.0))).astype(np.float32)
    hp = ts - F.blur(ts, 30)
    u1 = (hp * float(ctx.p("imprint")) * 3).astype(np.float32)
    u0 = u1.copy()
    amp0 = float(np.abs(u1).mean()) + 1e-6
    srcs = _sources(rng, int(ctx.p("sources")), w, h)
    fr = 0.045
    for t in range(int(ctx.p("steps"))):
        lap = np.roll(u1, 1, 0) + np.roll(u1, -1, 0) + np.roll(u1, 1, 1) + np.roll(u1, -1, 1) - 4 * u1
        u2 = 2 * u1 - u0 + c2 * lap
        for x, y, amp, f, ph in srcs:
            u2[int(y) - 1:int(y) + 2, int(x) - 1:int(x) + 2] += 0.3 * amp * (amp0 * 4) * np.sin(2 * np.pi * fr * f * t + ph)
        u2 *= 0.9992
        u0, u1 = u1, u2
    field = u1 / (np.percentile(np.abs(u1), 98) + 1e-6)
    f01 = np.clip(0.5 + 0.5 * field, 0, 1)
    f01 = cv2.resize(f01, (P.W, P.H), interpolation=cv2.INTER_CUBIC)
    sh = float(ctx.p("show"))
    fld = cv2.resize(np.clip(field, -1.5, 1.5), (P.W, P.H), interpolation=cv2.INTER_CUBIC)
    v = F.stretch((1 - sh) * f01 + sh * P.t, 1, 99)
    band = 0.5 + 0.5 * np.cos(2 * np.pi * fld * float(ctx.p("bands")))
    out = P.body(0.06 + 0.94 * v) * (0.72 + 0.28 * band)[..., None]
    crest = F.smoothstep(0.93, 1.0, band) * F.smoothstep(0.35, 0.8, P.t)
    out = mixc(out, P.pal.accent, crest * 0.35)
    return finish(P, out, 0.3, ctx.seed)


@effect("feature-ripple-refraction", family="feature", tags=["source", "refraction", "water", "physics", "all-over", "displacement"],
        params=with_common({"sources": (16, "ripple sources on a jittered grid"), "strength": (14.0, "refraction displacement px at 1500 wide"),
                            "wavelength": (46.0, "ripple wavelength px at 1500 wide"), "caustic": (0.6, "0..1 caustic / specular shading from ripple curvature"),
                            "grade": (True, "OKLCH-grade the source into the palette hue first")}))
def ripple_refraction(ctx):
    """The source seen through a physical ripple surface: a height field from
    sums of damped circular waves displaces the image like refraction (normal
    of the surface bends the view ray), and the curvature of the surface adds
    caustic light bands. Ripples are spread on a jittered grid so distortion
    is everywhere. The subject is still recognisable (displacement is a few
    percent of the frame). Works on any picture, strongest on photos and
    posters with large forms."""
    P = prep(ctx)
    rng = ctx.rng
    s = P.s
    H, W = P.H, P.W
    sw = 2
    w, h = W // sw, H // sw
    yy, xx = np.mgrid[0:h, 0:W // sw].astype(np.float32)
    hf = np.zeros((h, w), np.float32)
    lam = float(ctx.p("wavelength")) * s / sw
    for x, y, amp, f, ph in _sources(rng, int(ctx.p("sources")), w, h):
        r = np.hypot(xx - x, yy - y)
        hf += amp * np.sin(2 * np.pi * r / (lam * f) - ph) * np.exp(-r / (lam * 5.5))
    hf = F.blur(hf, 1.0)
    gy, gx = np.gradient(hf)
    st = float(ctx.p("strength")) * s
    scale = st / (np.percentile(np.hypot(gx, gy), 98) + 1e-6)
    dx = cv2.resize(gx * scale, (W, H), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(gy * scale, (W, H), interpolation=cv2.INTER_CUBIC)
    base = grade(P) if ctx.p("grade") else P.img
    gxm, gym = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    out = cv2.remap(np.ascontiguousarray(base), gxm + dx, gym + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    lap = F.blur(cv2.resize(ndi.laplace(hf), (W, H), interpolation=cv2.INTER_CUBIC), 1.5 * s)
    lap = lap / (np.percentile(np.abs(lap), 98) + 1e-6)
    ca = float(ctx.p("caustic"))
    out = out * (1 - 0.30 * ca * np.clip(-lap, 0, 1)[..., None])           # troughs darker
    out = 1 - (1 - out) * (1 - 0.30 * ca * np.clip(lap, 0, 1)[..., None] * np.asarray(P.pal.structure))
    return finish(P, out, 0.25, ctx.seed)


@effect("feature-edge-glitch", family="feature", tags=["source", "glitch", "slices", "edges", "all-over"],
        params=with_common({"band": (22, "median slice height px at 1500 wide"), "shift": (0.07, "max slice shift as a fraction of width"),
                            "streak": (0.5, "0..1 pixel-streak smear in strong-edge slices"), "blocks": (30, "displaced blocks"),
                            "seed_bands": (1, "reserved")}))
def edge_glitch(ctx):
    """Glitch driven by the picture's own edges: the frame is cut into random
    horizontal slices; each slice shifts in proportion to its edge energy (busy
    rows tear, calm rows hold), the brightest runs in busy slices streak
    sideways like pixel-sorting, displaced blocks copy detail across the frame,
    and an accent-coloured ghost of the edges is misregistered. All slices
    are spread over the whole height so the tearing is everywhere. Works on
    anything with edges (UI, photos, graphs); flat sources barely glitch."""
    P = prep(ctx)
    rng = ctx.rng
    s = P.s
    H, W = P.H, P.W
    e = F.edges_multiscale(P.t, (1.0, 2.0))
    t = P.t
    rows = []
    y = 0
    while y < H:
        hgt = max(2, int(rng.lognormal(np.log(float(ctx.p("band")) * s), 0.7)))
        rows.append((y, min(H, y + hgt)))
        y += hgt
    en = np.array([e[a:b].mean() for a, b in rows]); en = en / (en.mean() + 1e-9)
    maxs = float(ctx.p("shift")) * W
    base = P.body(0.05 + 0.95 * t)
    ghost = np.clip(F.blur(e, 0.6 * s) * 1.5, 0, 1)
    out = base.copy()
    sm = np.zeros_like(t)
    for (a, b), en_b in zip(rows, en):
        if en_b < 0.8 and rng.random() < 0.7:
            continue
        sh = int(rng.choice([-1, 1]) * maxs * min(1.0, en_b / 2.2) * rng.random())
        out[a:b] = np.roll(base[a:b], sh, 1)
        ghost[a:b] = np.roll(ghost[a:b], sh + int(6 * s * rng.choice([-1, 1])), 1)
        if en_b > 1.2 and rng.random() < float(ctx.p("streak")):
            L = int(rng.integers(30, 160) * s) | 1
            tt = cv2.dilate(t[a:b], np.ones((1, L), np.uint8))
            sm[a:b] = np.roll(tt, sh, 1)
    smear = F.smoothstep(0.55, 0.9, sm)
    out = mixc(out, P.body(np.clip(0.55 + 0.45 * sm, 0, 1)), smear * 0.85)
    out = mixc(out, P.pal.accent, ghost * 0.5)
    for _ in range(int(ctx.p("blocks"))):
        bw, bh = int(rng.uniform(0.05, 0.2) * W), int(rng.uniform(0.03, 0.1) * H)
        x0, y0 = int(rng.uniform(0, W - bw)), int(rng.uniform(0, H - bh))
        x1, y1 = int(np.clip(x0 + rng.normal(0, 0.1) * W, 0, W - bw)), int(np.clip(y0 + rng.normal(0, 0.1) * H, 0, H - bh))
        out[y0:y0 + bh, x0:x0 + bw] = out[y1:y1 + bh, x1:x1 + bw]
    return finish(P, out, 0.35, ctx.seed)
