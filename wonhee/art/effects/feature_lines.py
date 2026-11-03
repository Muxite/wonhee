"""Line-based edits of a source: flow hatching (LIC along the structure
tensor), contour relief, ridge/valley drawing, scan-line relief, circuit traces.
All read ctx.source (a synthetic stand-in is used if none), keep the subject's
structure, and fill the whole frame."""
from __future__ import annotations

import cv2
import numpy as np

from .. import features as F
from ..registry import effect
from .feature_common import prep, with_common, mixc, flat, finish


def lic(noise, vx, vy, length, step=1.0):
    """Line integral convolution of `noise` along the unit field (vx, vy)."""
    H, W = noise.shape
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    acc = noise.copy()
    wsum = 1.0
    for sign in (1.0, -1.0):
        px, py = gx.copy(), gy.copy()
        pvx, pvy = sign * vx, sign * vy
        for k in range(length):
            cx = cv2.remap(vx, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            cy = cv2.remap(vy, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            sg = np.where(cx * pvx + cy * pvy < 0, -1.0, 1.0).astype(np.float32)
            cx, cy = cx * sg, cy * sg
            px = px + cx * step
            py = py + cy * step
            pvx, pvy = cx, cy
            w = 1.0 - k / length
            acc += w * cv2.remap(noise, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            wsum += w
    return acc / wsum


def _uniformize(a):
    q = np.percentile(a, np.linspace(0, 100, 257))
    return np.interp(a, q, np.linspace(0, 1, 257)).astype(np.float32)


@effect("feature-flow-hatch", family="feature", tags=["source", "hatching", "lic", "all-over", "orientation"],
        params=with_common({"length": (22, "stroke half-length in px at 1500 wide (longer = smoother flow)"),
                            "rho": (9.0, "orientation smoothing (px at 1500 wide): bigger = calmer flow"),
                            "gamma": (0.9, "ink density curve (lower = denser strokes)"),
                            "floor": (0.14, "minimum stroke density so calm areas still carry hatching"),
                            "cross": (0.35, "0..1 second perpendicular hatch in the deepest tones")}))
def flow_hatch(ctx):
    """Pen-hatching: noise is smeared along the image's own structure-tensor
    orientation field (line integral convolution), then thresholded by tone, so
    strokes run along edges and contours of the real subject and thicken where
    the source is bright (dark palette) / dark (light palette). Strong on
    photos, plots, heatmaps, radial graphs (strokes follow spokes); weak on
    flat UI screenshots (flow is invented there: the field fills in smoothly)."""
    P = prep(ctx)
    rng = ctx.rng
    s = P.s
    theta, coh = F.structure_orientation(F.blur(P.t, 0.8 * s), sigma=1.5, rho=float(ctx.p("rho")), seed=ctx.seed)
    vx, vy = np.cos(theta).astype(np.float32), np.sin(theta).astype(np.float32)
    n = int(max(6, ctx.p("length") * s))
    noise = F.blur(rng.random((P.H, P.W)).astype(np.float32), 0.5 * s)
    u = _uniformize(lic(noise, vx, vy, n))
    ink = np.maximum(np.clip(P.ink, 0, 1) ** float(ctx.p("gamma")) * 0.92, float(ctx.p("floor")))
    m = np.clip((ink * 1.05 + 0.04 - u) * 9, 0, 1)
    # second hatch across the flow in the deepest ink
    cr = float(ctx.p("cross"))
    if cr > 0:
        u2 = _uniformize(lic(F.blur(rng.random((P.H, P.W)).astype(np.float32), 0.5 * s), -vy, vx, n))
        m2 = np.clip((np.clip(ink - 0.62, 0, 1) * 2.6 - u2) * 9, 0, 1) * cr
        m = np.maximum(m, m2)
    stroke = mixc(P.body(0.35 + 0.65 * P.t), P.pal.accent, F.smoothstep(0.88, 1.0, P.t) * 0.7)
    base = flat(P, P.pal.canvas)
    base = mixc(base, P.body(0.10 + 0.22 * P.t), 0.55)      # faint toned paper so nothing is flat
    out = mixc(base, stroke, m)
    return finish(P, out, 0.3, ctx.seed)


@effect("feature-contour-relief", family="feature", tags=["source", "contours", "topographic", "all-over", "luminance"],
        params=with_common({"levels": (16, "number of contour levels"), "index": (4, "every Nth line is an index line (accent)"),
                            "shade": (0.5, "0..1 hillshade relief strength"), "smooth": (3.5, "tone smoothing (px at 1500 wide)")}))
def contour_relief(ctx):
    """Topographic map of the image: luminance is the height. Banded fill,
    hairline contours with accent index lines every Nth level, and a hillshade
    from the tone gradient. The subject stays recognisable as terrain. Works
    on photos, glows, heatmaps (a heatmap becomes a real height map), radial
    graphs; weak on text screenshots (letters become islands)."""
    P = prep(ctx)
    s = P.s
    tt = F.blur(P.t, float(ctx.p("smooth")) * s)
    n = int(ctx.p("levels"))
    f = tt * n
    fl = np.floor(f)
    fr = f - fl
    gx, gy = np.gradient(F.blur(f, 1.0 * s))
    gm = np.hypot(gx, gy) + 1e-3
    d = np.minimum(fr, 1 - fr) / gm
    lw = 0.8 * s + 0.35
    line = np.clip(lw - d, 0, 1) * (gm > 0.002)
    band = fl / n
    fill = P.body(0.10 + 0.90 * band)
    sh = float(ctx.p("shade"))
    if sh > 0:
        g2x, g2y = np.gradient(F.blur(tt, 2.0 * s))
        shade = np.clip(0.5 + (g2x * -0.7 + g2y * -0.7) * 60 * s, 0, 1) - 0.5
        fill = np.clip(fill * (1 + shade[..., None] * sh * 1.4), 0, 1)
    out = fill
    dark_line = mixc(P.pal.canvas, P.pal.structure, 0.12 if P.dark else 0.55)
    out = mixc(out, np.asarray(dark_line, np.float32), line * 0.8)
    idx = (fl % int(ctx.p("index")) == 0).astype(np.float32)
    # index lines: only the line pixels belonging to an index level (level boundary = nearest integer)
    near = np.round(f)
    idx_line = np.clip(lw * 1.6 - d, 0, 1) * (near % int(ctx.p("index")) == 0) * (gm > 0.002)
    out = mixc(out, P.pal.accent, idx_line * 0.9)
    return finish(P, out, 0.25, ctx.seed)


@effect("feature-ridge-lines", family="feature", tags=["source", "ridges", "line-drawing", "all-over", "veins"],
        params=with_common({"thr": (0.18, "ridge/valley threshold 0..1 (lower = more lines)"), "wash": (0.55, "0..1 tone wash behind the lines"),
                            "sigmas": ("1.6,3.2", "Hessian scales (px at 1500 wide)")}))
def ridge_lines(ctx):
    """Ridge (bright line) and valley (dark line) detection on the source,
    drawn as a two-ink line drawing over a quiet tone wash: veins, cracks,
    text strokes, rows of ducks' feathers. Ridges in the accent, valleys in
    the structure colour, so the pair reads like an engraving. Best on
    textured photos and plots; on flat UI it draws outlines of strokes."""
    P = prep(ctx)
    s = P.s
    sig = tuple(float(x) for x in str(ctx.p("sigmas")).split(","))
    rid, val = F.ridge_valley(P.t, sig)
    thr = float(ctx.p("thr"))

    def thin(a):
        c = a - F.blur(a, 2.2 * s)          # centre-line pick: above its own blur
        return F.smoothstep(thr, thr + 0.2, a) * F.smoothstep(0.0, 0.03, c)

    mr, mv = thin(rid), thin(val)
    wash = P.body(0.12 + 0.55 * F.blur(P.t, 2.5 * s)) * float(ctx.p("wash")) + flat(P, P.pal.canvas) * (1 - float(ctx.p("wash")))
    out = mixc(wash, P.body(0.9 + 0 * P.t), np.clip(mv * 1.4, 0, 1))
    out = mixc(out, P.pal.accent, np.clip(mr * 1.4, 0, 1))
    return finish(P, out, 0.3, ctx.seed)


@effect("feature-line-scan", family="feature", tags=["source", "scanlines", "relief", "all-over"],
        params=with_common({"lines": (56, "number of scan lines"), "amp": (2.2, "relief height in line pitches"),
                            "smooth": (1.5, "tone smoothing (px at 1500 wide)")}))
def line_scan(ctx):
    """Joy-division relief: the image is sliced into horizontal lines whose
    height follows the tone, painter-occluded so the picture reads as a ridged
    landscape. Lines are coloured by tone, so shadows and lights both show.
    Works on any source with tonal structure (photos, glows, heatmaps)."""
    P = prep(ctx)
    s = P.s
    W, H = P.W, P.H
    n = int(ctx.p("lines"))
    pitch = H / n
    tt = F.blur(P.t, float(ctx.p("smooth")) * s)
    amp = float(ctx.p("amp")) * pitch
    ss = 2
    out = np.ascontiguousarray(flat(P, P.pal.canvas))
    big = cv2.resize(out, (W * ss, H * ss), interpolation=cv2.INTER_NEAREST)
    xs = np.arange(0, W, 2)
    fillc = (np.asarray(P.pal.canvas) * 0.9 + np.asarray(P.pal.structure) * 0.06).astype(np.float32)
    for i in range(n + 3):
        y0 = (i + 0.5) * pitch
        row = np.clip(int(min(H - 1, max(0, y0))), 0, H - 1)
        tv = tt[row, xs]
        y = y0 - amp * (tv - 0.2)
        pts = np.stack([xs * ss, y * ss], 1)
        poly = np.concatenate([pts, [[W * ss, (H + 3 * pitch) * ss], [0, (H + 3 * pitch) * ss]]]).astype(np.int32)
        cv2.fillPoly(big, [poly], [float(c) for c in fillc])
        seg_t = tv
        # draw coloured segments (per-sample colour)
        aw = (F.smoothstep(0.85, 1.0, seg_t) * 0.8)[:, None]
        cols = P.body(0.6 + 0.4 * seg_t) * (1 - aw) + np.asarray(P.pal.accent) * aw
        for k in range(0, len(xs) - 1, 1):
            cv2.line(big, (int(pts[k, 0]), int(pts[k, 1])), (int(pts[k + 1, 0]), int(pts[k + 1, 1])), [float(c) for c in cols[k]],
                     max(1, int(round(2.8 * s * ss))), cv2.LINE_AA)
    out = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA)
    return finish(P, out, 0.2, ctx.seed)


@effect("feature-circuit-traces", family="feature", tags=["source", "circuit", "pcb", "contours", "all-over", "technical"],
        params=with_common({"levels": (9, "number of tone levels traced"), "grid": (12, "route grid pitch (px at 1500 wide)"),
                            "min_len": (14, "shortest trace kept (points)"), "pads": (0.5, "0..1 pad / via density")}))
def circuit_traces(ctx):
    """Photo-to-circuit: tone contours of the source are simplified, snapped to
    a routing grid and re-routed as 45-degree chamfered PCB traces with pads
    and vias, over a faint solder-mask wash of the tone. The subject's big
    shapes persist as bundles of parallel traces. Works on photos, heatmaps,
    interference images, graphs; on screenshots it traces the panels."""
    P = prep(ctx)
    s = P.s
    W, H = P.W, P.H
    rng = ctx.rng
    n = int(ctx.p("levels"))
    half = 0.5
    sm = cv2.resize(F.blur(P.t, 2.5 * s), (int(W * half), int(H * half)), interpolation=cv2.INTER_AREA)
    wash = P.body(0.10 + 0.45 * F.blur(P.t, 3 * s))
    ss = 2
    canvas = cv2.resize(np.ascontiguousarray(wash), (W * ss, H * ss), interpolation=cv2.INTER_LINEAR)
    g = max(4, int(ctx.p("grid") * s * ss))
    lw = max(1, int(round(3.4 * s * ss)))
    for li in range(1, n):
        lvl = li / n
        bw = (sm > lvl).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        col = [float(c) for c in (P.pal.accent if li % 4 == 0 else P.body(np.array([0.7 + 0.3 * lvl]))[0])]
        for ci, c in enumerate(cnts):
            if len(c) < int(ctx.p("min_len")):
                continue
            ap = cv2.approxPolyDP(c, 2.5, True)[:, 0, :].astype(np.float64) * (ss / half)
            pts = np.round(ap / g) * g
            keep = [pts[0]]
            for p in pts[1:]:
                if (p != keep[-1]).any():
                    keep.append(p)
            if len(keep) < 3:
                continue
            route = [keep[0]]
            for a, b in zip(keep, keep[1:] + [keep[0]]):
                dx, dy = b[0] - a[0], b[1] - a[1]
                dg = min(abs(dx), abs(dy))
                sx, sy = np.sign(dx), np.sign(dy)
                if ci % 2:                      # diagonal first
                    mid = np.array([a[0] + sx * dg, a[1] + sy * dg])
                elif abs(dx) > abs(dy):         # straight first, then diagonal
                    mid = np.array([b[0] - sx * dg, a[1]])
                else:
                    mid = np.array([a[0], b[1] - sy * dg])
                route.append(mid)
                route.append(b)
            arr = np.round(np.array(route)).astype(np.int32)
            cv2.polylines(canvas, [arr], True, col, lw, cv2.LINE_AA)
            padp = float(ctx.p("pads"))
            if padp > 0:
                for p in keep[:: max(1, int(3 / max(padp, 0.05)))]:
                    cv2.circle(canvas, (int(p[0]), int(p[1])), int(5.5 * s * ss), col, -1, cv2.LINE_AA)
                    cv2.circle(canvas, (int(p[0]), int(p[1])), max(1, int(2.3 * s * ss)), [float(x) for x in P.pal.canvas], -1, cv2.LINE_AA)
    out = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_AREA)
    return finish(P, out, 0.2, ctx.seed)
