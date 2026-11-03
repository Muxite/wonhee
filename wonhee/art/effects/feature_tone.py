"""Tone / colour edits of a source: thermal, OKLCH hue-graded 'sunset rule',
misregistered multi-layer riso posterise, blueprint annotation."""
from __future__ import annotations

import cv2
import numpy as np

from .. import features as F
from ..registry import effect
from .feature_common import prep, with_common, mixc, flat, finish, grade


@effect("feature-thermal", family="feature", tags=["source", "thermal", "heatmap", "all-over", "isotherms"],
        params=with_common({"isotherms": (10, "number of isotherm lines"), "marks": ("8x5", "measurement marks as COLSxROWS (one per cell, at the local peak; '0x0' off)"),
                            "smooth": (2.0, "tone smoothing px")}))
def thermal(ctx):
    """False-colour thermal camera look in ONE hue family: tone runs through the
    palette ramp (canvas to structure to accent at the hottest few percent),
    with isotherm contour lines and a regular grid of measurement rings each
    placed on the local peak of its cell (so instrument marks spread over the
    whole frame, no single crosshair). Best on anything with bright subjects on
    dark ground (night photos, glows, dashboards, heatmaps)."""
    P = prep(ctx)
    s = P.s
    H, W = P.H, P.W
    t = F.stretch(F.blur(P.t, float(ctx.p("smooth")) * s), 1, 99.5)
    t = 0.4 * t + 0.6 * F.smoothstep(0.03, 0.97, t)
    out = P.lut(0.60 * t + 0.40 * F.smoothstep(0.80, 1.0, t))
    f = t * float(ctx.p("isotherms"))
    gx, gy = np.gradient(F.blur(f, 1.5 * s))
    gm = np.hypot(gx, gy) + 1e-3
    d = np.minimum(f % 1, 1 - f % 1) / gm
    line = np.clip(1.2 * s + 0.4 - d, 0, 1) * (gm > 0.0015)
    out = mixc(out, out * 0.5 + np.asarray(P.pal.canvas) * 0.3, line * 0.7)
    cols, rows = (int(v) for v in str(ctx.p("marks")).lower().split("x"))
    if cols > 0 and rows > 0:
        ss = 2
        big = cv2.resize(np.ascontiguousarray(out), (W * ss, H * ss), interpolation=cv2.INTER_LINEAR)
        lp = F.blur(t, 10 * s)
        cw, ch = W / cols, H / rows
        for r in range(rows):
            for c in range(cols):
                y0, y1, x0, x1 = int(r * ch), int((r + 1) * ch), int(c * cw), int((c + 1) * cw)
                sub = lp[y0:y1, x0:x1]
                iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
                fx, fy = (x0 + ix) * ss, (y0 + iy) * ss
                local = float(sub[iy, ix])
                col = [float(v) for v in (P.pal.canvas if local > 0.7 else P.pal.accent)]
                rad = int(14 * s * ss); arm = int(24 * s * ss)
                th = max(1, int(round(2.2 * s * ss)))
                cv2.circle(big, (fx, fy), rad, col, th, cv2.LINE_AA)
                cv2.line(big, (fx - arm, fy), (fx - rad // 2, fy), col, th, cv2.LINE_AA)
                cv2.line(big, (fx + rad // 2, fy), (fx + arm, fy), col, th, cv2.LINE_AA)
                cv2.line(big, (fx, fy - arm), (fx, fy - rad // 2), col, th, cv2.LINE_AA)
                cv2.line(big, (fx, fy + rad // 2), (fx, fy + arm), col, th, cv2.LINE_AA)
        out = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA)
    return finish(P, out, 0.4, ctx.seed)


@effect("feature-sunset-grade", family="feature", tags=["source", "grade", "photo", "oklch", "hue"],
        params=with_common({"hue": (-1.0, "OKLCH hue degrees; -1 = palette hue, -2 = the source's dominant hue"),
                            "chroma": (0.12, "peak chroma (bounded; rich not neon)"), "source_chroma": (0.5, "0..1 how much the source's own chroma shapes the result"),
                            "bloom": (0.35, "highlight glow 0..1"), "grain": (0.5, "film grain 0..1")}))
def sunset_grade(ctx):
    """The sunset rule applied to any picture: lightness from the (locally
    equalised) source tone, ONE hue family chosen by parameter, bounded chroma,
    calm desaturated darks, a hint of highlight bloom and grain. The subject
    stays fully photographic and recognisable. Use for photos and renders you
    want to match a project's hue; with hue=-2 the grade keeps the picture's
    own dominant hue."""
    P = prep(ctx)
    s = P.s
    h = float(ctx.p("hue"))
    hue = None if h == -1 else (F.dominant_hue(P.img) if h == -2 else h)
    out = grade(P, hue=hue, chroma=float(ctx.p("chroma")), source_chroma=float(ctx.p("source_chroma")))
    b = float(ctx.p("bloom"))
    if b > 0:
        hot = out * F.smoothstep(0.55, 0.9, out.max(-1))[..., None]
        glow = F.blur(hot, 14 * s) * 1.4 * b
        out = 1 - (1 - out) * (1 - np.clip(glow, 0, 1))
    return finish(P, out, float(ctx.p("grain")), ctx.seed)


@effect("feature-riso-layers", family="feature", tags=["source", "risograph", "posterise", "misregistration", "print", "all-over"],
        params=with_common({"layers": (3, "ink layers (2-4)"), "offset": (7.0, "misregistration px at 1500 wide"), "grain": (0.8, "ink grain / speckle 0..1.5"),
                            "texture": (0.6, "0..1 halftone-grain texture in the mid layers")}))
def riso_layers(ctx):
    """Multi-layer posterise with offset misregistration, as risograph: tone is
    split into 2-4 ink layers (palette inks from light to dark) each printed
    with grain, shifted slightly differently, and overprinted (screen on dark
    palettes, multiply on light), so overlaps create the in-between values.
    Equalised tone puts all layers in every region. Strong on photos and
    posters; keeps big shapes of any source; loses fine text."""
    P = prep(ctx)
    s = P.s
    rng = ctx.rng
    H, W = P.H, P.W
    n = int(np.clip(ctx.p("layers"), 2, 4))
    off = float(ctx.p("offset")) * s
    ink = P.ink                                   # 1 = the source is strongly present
    q = np.linspace(0.30, 0.78, n)               # percentile thresholds of ink coverage
    thr = np.quantile(ink, 1 - q)
    gr = float(ctx.p("grain"))
    tex = float(ctx.p("texture"))
    out = flat(P, P.pal.canvas if P.dark else P.pal.canvas)
    cols = [P.body(np.array([0.50 + 0.50 * k / (n - 1)]))[0] for k in range(n)]
    cols[-1] = np.asarray(cols[-1]) * 0.55 + np.asarray(P.pal.accent) * 0.45
    ang = rng.uniform(0, 2 * np.pi, n)
    for k in range(n):
        m = F.smoothstep(thr[k] - 0.015, thr[k] + 0.015, ink)
        # halftone-grain texture in the layer
        g = F.blur(rng.normal(0, 1, (H, W)).astype(np.float32), 0.7 * s)
        m = np.clip(m * (1 - 0.5 * gr * F.smoothstep(0.2, 1.4, g + 0.3 * F.blur(rng.normal(0, 1, (H, W)).astype(np.float32), 5 * s) * tex)), 0, 1)
        dx, dy = off * np.cos(ang[k]) * (1 + k * 0.4), off * np.sin(ang[k]) * (1 + k * 0.4)
        m = np.roll(m, (int(round(dy)), int(round(dx))), (0, 1))
        c = np.asarray(cols[k], np.float32)
        if P.dark:    # light inks overprint additively-ish (screen)
            lay = 1 - (1 - out) * (1 - c * m[..., None] * 0.8)
            out = lay
        else:
            lay = out * (1 - m[..., None] * (1 - c) * 0.9)
            out = lay
    return finish(P, out, 0.3, ctx.seed)


@effect("feature-blueprint-notes", family="feature", tags=["source", "blueprint", "technical", "edges", "annotation", "all-over"],
        params=with_common({"grid": (30, "minor grid pitch px at 1500 wide"), "boxes": (46, "max annotation boxes"), "edge": (0.55, "edge threshold 0..1 (lower = more lines)")}))
def blueprint_notes(ctx):
    """Technical drawing of the source: a drafting grid, the image's multi-scale
    edges as clean linework, and annotation boxes with corner ticks around its
    connected structures and detected text blocks (so a screenshot's panels,
    labels and paragraphs become callouts). Looks best on screenshots,
    dashboards, diagrams, graphs and photos with clear objects; on smooth
    gradients there is little to annotate (grid and faint edges only)."""
    P = prep(ctx, clip=2.0)
    s = P.s
    H, W = P.H, P.W
    ss = 2
    dark = P.dark
    base = flat(P, P.pal.canvas)
    wash = mixc(base, P.body(0.18 + 0.30 * F.blur(P.t, 6 * s)), 0.45)
    big = cv2.resize(np.ascontiguousarray(wash), (W * ss, H * ss), interpolation=cv2.INTER_LINEAR)
    g = float(ctx.p("grid")) * s * ss
    gc = [float(v) for v in P.body(np.array([0.34]))[0]]
    gc2 = [float(v) for v in P.body(np.array([0.5]))[0]]
    k = 0
    x = 0.0
    while x < W * ss:
        cv2.line(big, (int(x), 0), (int(x), H * ss), gc2 if k % 5 == 0 else gc, max(1, int(round((1.6 if k % 5 == 0 else 0.9) * s * ss))), cv2.LINE_AA)
        x += g; k += 1
    k = 0; y = 0.0
    while y < H * ss:
        cv2.line(big, (0, int(y)), (W * ss, int(y)), gc2 if k % 5 == 0 else gc, max(1, int(round((1.6 if k % 5 == 0 else 0.9) * s * ss))), cv2.LINE_AA)
        y += g; k += 1
    out = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA)
    # linework
    t8 = (np.clip(F.blur(P.t, 0.8 * s), 0, 1) * 255).astype(np.uint8)
    lo = int(30 + 80 * (1 - float(ctx.p("edge"))))
    ed = cv2.Canny(t8, lo, lo * 2.2).astype(np.float32) / 255
    ed = F.blur(cv2.dilate(ed, np.ones((2, 2), np.uint8)), 0.6 * s)
    ed = np.clip(ed * 1.6, 0, 1)
    out = mixc(out, P.body(np.full((H, W), 1.0, np.float32)), ed * 0.95)
    # annotation boxes
    comp = cv2.dilate((ed > 0.3).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (int(19 * s) | 1, int(19 * s) | 1)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(comp, 8)
    boxes = [(x0, y0, w, h) for x0, y0, w, h, a in st[1:] if 26 * s < w < W * 0.8 and 16 * s < h < H * 0.7]
    tm = F.text_mask(P.img)
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats((tm > 0.5).astype(np.uint8), 8)
    tboxes = [(x0, y0, w, h) for x0, y0, w, h, a in st2[1:] if w > 30 * s]
    boxes = sorted(boxes, key=lambda b: -b[2] * b[3])[: int(ctx.p("boxes"))] + tboxes[:20]
    big = cv2.resize(np.ascontiguousarray(out), (W * ss, H * ss), interpolation=cv2.INTER_LINEAR)
    acc = [float(v) for v in P.pal.accent]
    th = max(1, int(round(2.0 * s * ss)))
    tick = int(10 * s * ss)
    for (x0, y0, w, h) in boxes:
        x0, y0, x1, y1 = x0 * ss, y0 * ss, (x0 + w) * ss, (y0 + h) * ss
        for (cx, cy, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            cv2.line(big, (cx, cy), (cx + dx * tick, cy), acc, th, cv2.LINE_AA)
            cv2.line(big, (cx, cy), (cx, cy + dy * tick), acc, th, cv2.LINE_AA)
        # dimension tick on the top edge
        cv2.line(big, (x0, y0 - 5 * ss), (x1, y0 - 5 * ss), acc, 1, cv2.LINE_AA)
    out = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA)
    return finish(P, out, 0.25, ctx.seed)
