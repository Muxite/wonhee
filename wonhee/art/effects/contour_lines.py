"""Line languages: streamlines, engraved hatching, and edge ink."""
from __future__ import annotations

import math

import cv2
import numpy as np

from .. import contourkit as CK
from .. import features as F
from .. import natkit as N
from ..registry import effect


def _vector_field(ctx, f, img, theta, coh, sw, sh):
    """Unit-ish flow field at the working size for the streamlines."""
    mode = str(ctx.p("direction"))
    ang = float(ctx.p("angle"))
    rng = np.random.default_rng(ctx.seed + 17)
    fs = cv2.resize(f, (sw, sh), interpolation=cv2.INTER_AREA)
    if mode == "contour":
        g = F.blur(fs, 5.0 * sw / 1500 * float(ctx.p("scale")) + 1.0)
        gy, gx = np.gradient(g)
        vx, vy = -gy, gx
        m = np.hypot(vx, vy) + 1e-9
        vx, vy = vx / m, vy / m
        # in flat regions fall back to a smooth angle field so lines continue
        flat = N.smooth(0.0, np.percentile(m, 35) + 1e-9, m)
        a0 = ang + 1.2 * N.fbm(rng, sh, sw, sw * 0.25, 2)
        vx, vy = vx * flat + np.cos(a0) * (1 - flat), vy * flat + np.sin(a0) * (1 - flat)
    elif mode == "orientation" and theta is not None:
        th = cv2.resize(theta, (sw, sh), interpolation=cv2.INTER_LINEAR)
        co = cv2.resize(coh, (sw, sh), interpolation=cv2.INTER_LINEAR)
        # doubled-angle smoothing to avoid sign flips
        c2, s2 = F.blur(np.cos(2 * th) * (0.3 + co), 4), F.blur(np.sin(2 * th) * (0.3 + co), 4)
        th = 0.5 * np.arctan2(s2, c2)
        vx, vy = np.cos(th), np.sin(th)
    elif mode == "curl":
        psi = N.fbm(rng, sh, sw, sw * 0.18 * float(ctx.p("scale")), 3, 0.5)
        vx, vy = N.curl(psi)
        m = np.hypot(vx, vy) + 1e-9
        vx, vy = vx / m, vy / m
    elif mode == "radial":
        yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
        cx, cy = rng.uniform(0.1, 0.9) * sw, rng.uniform(0.1, 0.9) * sh
        th = np.arctan2(yy - cy, xx - cx) + math.pi / 2 * (1 + float(ctx.p("angle")) * 0.3)
        th = th + 0.5 * N.fbm(rng, sh, sw, sw * 0.2, 2)
        vx, vy = np.cos(th), np.sin(th)
    else:                                                       # drift: one direction bent by smooth noise
        a0 = ang + 0.9 * N.fbm(rng, sh, sw, sw * 0.3, 3)
        vx, vy = np.cos(a0), np.sin(a0)
    return vx.astype(np.float32), vy.astype(np.float32)


def streamlines(vx, vy, sep_map, rng, max_len=400, h=1.0, min_len=8):
    """Evenly spaced streamlines (Jobard-Lefer). ``sep_map`` (sh, sw) is the
    local desired separation in px (denser in dark tone). Returns a list of
    (N, 2) float arrays."""
    sh, sw = vx.shape
    base = float(sep_map.min())
    cell = max(1.0, base * 0.5)
    gw, gh = int(sw / cell) + 2, int(sh / cell) + 2
    gxp = np.full((gh, gw), -1e9, np.float32)
    gyp = np.full((gh, gw), -1e9, np.float32)
    R = 3

    def free(x, y, d):
        cx, cy = int(x / cell), int(y / cell)
        x0, x1, y0, y1 = max(0, cx - R), min(gw, cx + R + 1), max(0, cy - R), min(gh, cy + R + 1)
        dx = gxp[y0:y1, x0:x1] - x
        dy = gyp[y0:y1, x0:x1] - y
        return float((dx * dx + dy * dy).min()) >= d * d

    def put(x, y):
        cx, cy = int(x / cell), int(y / cell)
        if 0 <= cx < gw and 0 <= cy < gh:
            gxp[cy, cx], gyp[cy, cx] = x, y

    def vel(x, y, px, py):
        ix, iy = min(sw - 1, max(0, int(x))), min(sh - 1, max(0, int(y)))
        a, b = vx[iy, ix], vy[iy, ix]
        if a * px + b * py < 0:
            a, b = -a, -b
        return a, b

    def trace(x, y, sgn):
        pts = []
        px, py = sgn, 0.0
        a, b = vel(x, y, px, py)
        px, py = a * sgn if False else a, b
        px, py = (a, b) if sgn > 0 else (-a, -b)
        for _ in range(max_len):
            a, b = vel(x, y, px, py)
            x2, y2 = x + a * h, y + b * h
            px, py = a, b
            if not (0 <= x2 < sw and 0 <= y2 < sh):
                break
            d = sep_map[int(y2), int(x2)] * 0.5
            if not free(x2, y2, d * 0.999):
                break
            x, y = x2, y2
            pts.append((x, y))
        return pts

    lines = []
    seeds = [(rng.uniform(0, sw), rng.uniform(0, sh)) for _ in range(40)]
    si = 0
    queue = []
    budget = 0
    while True:
        if queue:
            x, y = queue.pop(0)
        elif si < len(seeds):
            x, y = seeds[si]; si += 1
        else:
            break
        if not (0 <= x < sw and 0 <= y < sh):
            continue
        if not free(x, y, sep_map[int(y), int(x)]):
            continue
        fw = trace(x, y, 1)
        bw = trace(x, y, -1)
        pts = bw[::-1] + [(x, y)] + fw
        if len(pts) < min_len:
            continue
        for (px_, py_) in pts[::2]:
            put(px_, py_)
        arr = np.array(pts, np.float32)
        lines.append(arr)
        # seeds on both sides, one separation away from every ~4th point
        for k in range(0, len(arr), 4):
            x0, y0 = arr[k]
            k1 = min(len(arr) - 1, k + 1)
            dx, dy = arr[k1] - arr[max(0, k1 - 1)] if k1 > 0 else (1, 0)
            nrm = math.hypot(dx, dy) + 1e-9
            nx, ny = -dy / nrm, dx / nrm
            d = sep_map[int(min(sh - 1, max(0, y0))), int(min(sw - 1, max(0, x0)))]
            queue.append((x0 + nx * d, y0 + ny * d))
            queue.append((x0 - nx * d, y0 - ny * d))
        budget += len(arr)
        if budget > 400000:
            break
    return lines


@effect("contour-streamlines", family="contour",
        tags=["streamlines", "flow", "lines", "engraving", "source", "data", "hand-drawn", "precise", "scientific", "mood:hand-drawn", "mood:scientific", "mood:precise"],
        params=CK.with_contour(N.with_common({
            "direction": ("contour", "LINES: contour (along the field's iso-lines) | orientation (the photo's own structure) | curl (stirred flow) | drift (one direction, bent) | radial"),
            "angle": (0.35, "LINES: base direction in radians (drift, and the fallback in flat areas)"),
            "spacing": (10.0, "LINES: separation between lines at 1500 px (px); lines in dark/inked tone pack closer"),
            "tone_pack": (0.55, "LINES: 0..1 how much the tone tightens the spacing (0 = even spacing everywhere, 1 = engraving-like)"),
            "weight": (0.34, "LINES: line width relative to spacing (floored by min_px)"),
            "swell": (0.5, "LINES: 0..1 width swelling with the tone (engraved line weight)"),
            "length": (1.0, "LINES: longest line, in frame widths"),
            "ink_vary": (0.6, "COLOUR: 0..1 line colour steps through the ramp with the tone under it"),
        }, source=False)))
def streamlines_fx(ctx):
    """Evenly spaced streamlines (Jobard-Lefer placement) of a direction field,
    drawn as anti-aliased pen lines whose width and spacing follow the tone of
    the field: a photo, a data array, or procedural terrain. With
    direction=contour the lines run along the iso-lines of the tone (form
    following, like a topographic or Gestalt engraving); orientation follows
    the photo's own structure; curl / drift / radial are procedural flows for
    data-free card art. All lines are separated, never crossing and never
    piling up, so at 288x192 the picture is ~40-60 clean flowing lines;
    min_px keeps them visible. Differs from feature-flow-hatch (noise smeared
    by LIC: soft, random density) by its exact spacing control and
    uncrossed, evenly spaced strokes."""
    W, H = ctx.size
    s = W / 1500.0
    f, img, theta, coh = CK.field(ctx, (W, H))
    sw = min(W, 760)
    sh = max(8, int(round(sw * H / W)))
    k = sw / W
    vx, vy = _vector_field(ctx, f, img, theta, coh, sw, sh)
    fs = cv2.resize(f, (sw, sh), interpolation=cv2.INTER_AREA)
    dark = ctx.palette.polarity == "dark"
    ink = fs if dark else 1 - fs
    sp0 = max(float(ctx.p("spacing")) * s, 4.0) * k
    sep = sp0 * (1.0 + float(ctx.p("tone_pack")) * (0.9 - 1.8 * ink))
    sep = np.maximum(sep, float(ctx.p("min_px")) * k * 2.2).astype(np.float32)
    rng = np.random.default_rng(ctx.seed + 23)
    lines = streamlines(vx, vy, sep, rng, max_len=int(520 * float(ctx.p("length"))), h=1.0)
    ss = 2 if W <= 900 else 1
    R = CK.ink_ramp(ctx)
    bg = N.paint(np.zeros((H * ss, W * ss), np.float32) + (0.0 if dark else 0.0), R)
    # faint tone wash so calm areas are not empty
    wash = 0.05 + 0.10 * cv2.resize(ink, (W * ss, H * ss), interpolation=cv2.INTER_LINEAR)
    canvas = np.ascontiguousarray(N.paint(wash, R))
    wob = float(ctx.p("wobble"))
    for arr in lines:
        pts = arr / k * ss
        if wob > 0:
            n_ = len(pts)
            pts = pts + wob * 1.2 * ss * s * np.stack([N.tnoise(pts[:, 0] / W * 4 / ss, pts[:, 1] / W * 4 / ss, 0.0, ctx.seed + 8, 1.0),
                                                       N.tnoise(pts[:, 0] / W * 4 / ss + 5, pts[:, 1] / W * 4 / ss, 0.0, ctx.seed + 9, 1.0)], 1)
        # chunks of ~10 points: width and colour follow the local tone
        for c0 in range(0, len(pts) - 1, 9):
            ch = pts[c0:c0 + 11]
            if len(ch) < 2:
                continue
            mx, my = ch[:, 0].mean() / ss, ch[:, 1].mean() / ss
            t = float(ink[min(sh - 1, int(my * k)), min(sw - 1, int(mx * k))]) if False else float(ink[min(sh - 1, int(my * k)), min(sw - 1, int(mx * k))])
            wpx = max(float(ctx.p("min_px")), sp0 / k * float(ctx.p("weight")) * (1 - float(ctx.p("swell")) + float(ctx.p("swell")) * (0.4 + 1.2 * t)))
            tcol = 0.45 + 0.5 * (t * float(ctx.p("ink_vary")) + (1 - float(ctx.p("ink_vary"))))
            col = N.paint(np.array([tcol], np.float32), R)[0]
            cv2.polylines(canvas, [np.round(ch * 16).astype(np.int32).reshape(-1, 1, 2)], False, tuple(float(c) for c in col),
                          max(1, int(round(wpx * ss))), cv2.LINE_AA, shift=4)
    out = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_AREA) if ss > 1 else canvas
    return N.surface(out, ctx)


@effect("contour-hatch-engrave", family="contour",
        tags=["hatching", "engraving", "lines", "banknote", "source", "data", "hand-drawn", "precise", "mood:hand-drawn", "mood:precise", "mood:cartographic"],
        params=CK.with_contour(N.with_common({
            "spacing": (8.0, "LINES: line pitch at 1500 px (px); never below 3.2 px at the output"),
            "angle": (0.5, "LINES: hatch direction, radians"),
            "bend": (0.6, "LINES: 0..1 how far the lines are displaced by the tone relief (banknote / Joy-Division bend)"),
            "weight": (0.9, "LINES: stroke weight relative to pitch at full ink (0..1.2)"),
            "gamma": (1.0, "LINES: ink curve; lower = more ink in mid-tones"),
            "cross": (0.6, "LINES: 0..1 strength of the second (crossing) hatch in the deepest tones"),
            "cross_angle": (1.1, "LINES: crossing hatch angle offset, radians"),
            "wash": (0.35, "COLOUR: 0..1 faint tone wash under the lines"),
        }, source=False)))
def hatch_engrave(ctx):
    """Engraved hatching: parallel lines whose phase is displaced by the tone
    field (so they bend over forms like banknote engraving), whose width
    swells with the ink density (dark palette: bright areas get more ink;
    light palette: dark areas), with a second crossing hatch in the deepest
    tones and a third in the very deepest. The line phase is an analytic field,
    so spacing is exact and the rendering is antialiased at any size;
    wobble makes the lines meander like a pen. The pitch has a floor in
    output pixels, so a 288x192 render has ~35 lines, not a grey haze.
    Sources: photos (portraits, skies, water), data fields, terrain."""
    W, H = ctx.size
    s = W / 1500.0
    f, img, theta, coh = CK.field(ctx, (W, H))
    mask = CK.mask_field(ctx, f, img)
    dark = ctx.palette.polarity == "dark"
    ink = np.clip(f if dark else 1 - f, 0, 1) ** float(ctx.p("gamma"))
    pitch = max(float(ctx.p("spacing")) * s, 3.2)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    wob = float(ctx.p("wobble"))
    tt = float(ctx.p("time", 0.0))

    def family(ang, lo, amt):
        phi = (xx * math.cos(ang) + yy * math.sin(ang)) / pitch + float(ctx.p("bend")) * 4.0 * (f - 0.5)
        if wob > 0:
            phi = phi + wob * 0.6 * N.tnoise(xx / W, yy / W, tt, ctx.seed + int(ang * 100) + 3, 0.06)
        gy, gx = np.gradient(phi)
        d = np.abs(phi - np.round(phi)) / (np.hypot(gx, gy) + 1e-6)
        wpx = np.maximum(pitch * float(ctx.p("weight")) * np.clip((ink - lo) / (1 - lo + 1e-6), 0, 1) ** 0.8, 0.0)
        a = CK.line_alpha(d, np.maximum(wpx, float(ctx.p("min_px")) * (ink > lo + 0.03)))
        return a * (ink > lo) * amt

    a1 = family(float(ctx.p("angle")), -0.12, 1.0)
    cr = float(ctx.p("cross"))
    a2 = family(float(ctx.p("angle")) + float(ctx.p("cross_angle")), 0.55, cr) if cr > 0 else 0
    a3 = family(float(ctx.p("angle")) - float(ctx.p("cross_angle")) * 0.6, 0.8, cr) if cr > 0 else 0
    a = np.clip(np.maximum(np.maximum(a1, a2), a3) * mask, 0, 1)
    R = CK.ink_ramp(ctx)
    wash = 0.02 + float(ctx.p("wash")) * 0.30 * F.blur(ink, 6 * s)
    out = N.paint(np.clip(wash, 0, 1), R)
    t_line = 0.80 + 0.17 * ink
    out = out * (1 - a[..., None]) + N.paint(np.clip(t_line, 0, 1).astype(np.float32), R) * a[..., None]
    return N.surface(out, ctx)


@effect("contour-edge-ink", family="contour",
        tags=["edges", "ink", "drawing", "line-art", "source", "hand-drawn", "scientific", "mood:hand-drawn", "mood:scientific", "mood:precise"],
        params=CK.with_contour(N.with_common({
            "coarse": (1.0, "LINES: coarse edge scale multiplier (big shapes drawn heavy)"),
            "fine": (0.55, "LINES: 0..1 weight of the fine edge layer"),
            "threshold": (0.5, "LINES: 0..1 edge threshold (higher = fewer, cleaner lines)"),
            "min_area": (22, "FIELD: drop line fragments smaller than this many px at 1500 px wide (speckle removal)"),
            "w_coarse": (3.2, "LINES: coarse line width at 1500 px"),
            "wash": (0.30, "COLOUR: 0..1 tone wash under the drawing"),
        }, source=False)))
def edge_ink(ctx):
    """Pen drawing from edges at two scales: Canny edges of a coarse blur for
    the big shapes (drawn heavy) and of a fine blur for texture
    (drawn thin), each with percentile thresholds, cleaned of speckle by connected-component
    area filtering, optionally wobbled by smooth space-time noise and widened
    so the strokes survive a thumbnail. Dark edges are ink; light edges
    (`light_lines`) can take the accent. Differs from feature-ridge-lines
    (Hessian ridge/valley, no despeckle, no hierarchy) and feature-edge-glitch
    (glitch treatment). Sources: any photo (portraits, objects, architecture),
    data fields; procedural terrain gives a map-like linework."""
    W, H = ctx.size
    s = W / 1500.0
    f, img, theta, coh = CK.field(ctx, (W, H))
    # edges want the sharper source tone, not the heavily smoothed field
    if img is not None:
        L = F.stretch(F.local_equalize(F.luminance(img), 2.0, mix=float(ctx.p("local"))), 1, 99)
        L = cv2.bilateralFilter(L, 5, 0.08, 4 * max(s, 0.3)) if float(ctx.p("denoise")) > 0.3 else L
    else:
        L = f
    wob = CK.wobble_field(ctx, (W, H), float(ctx.p("wobble")))
    if wob is not None:
        L = CK.warp_by(L, wob)
    mask = CK.mask_field(ctx, f, img)
    thr = float(ctx.p("threshold"))

    def canny(sig, pct_lo, pct_hi):
        g = (np.clip(F.blur(L, sig * s), 0, 1) * 255).astype(np.uint8)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.hypot(gx, gy)
        hi = float(np.percentile(mag, pct_hi))
        lo = float(np.percentile(mag, pct_lo))
        return cv2.Canny(g, max(lo, 1.0), max(hi, lo + 1.0), L2gradient=True).astype(np.float32) / 255

    sc_ = float(ctx.p("coarse"))
    ec = canny(3.2 * sc_, 78 + 14 * thr, 90 + 8 * thr)
    ef = canny(1.1, 70 + 20 * thr, 86 + 11 * thr) * float(ctx.p("fine"))
    ec = CK.drop_small(ec, float(ctx.p("min_area")) * s * 1.5)
    ef = CK.drop_small(ef, float(ctx.p("min_area")) * s * 0.8)
    minw = max(float(ctx.p("min_px")), 1.0)
    wc = max(float(ctx.p("w_coarse")) * s, minw)
    rc = max(0.0, (wc - 1.0) * 0.5)
    if rc > 0.4:
        kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(rc * 2) | 1, int(rc * 2) | 1))
        ec = cv2.dilate(ec, kk)
    ec = cv2.GaussianBlur(ec, (0, 0), 0.7)
    rf = max(0.0, (minw - 1.0) * 0.5)
    if rf > 0.4:
        kf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(rf * 2) | 1, int(rf * 2) | 1))
        ef = cv2.dilate(ef, kf)
    ef = cv2.GaussianBlur(ef, (0, 0), 0.6)
    R = CK.ink_ramp(ctx)
    dark = ctx.palette.polarity == "dark"
    ink_t = np.clip(F.blur(f if dark else 1 - f, 5 * s), 0, 1)
    out = N.paint(np.clip(0.02 + float(ctx.p("wash")) * 0.32 * ink_t, 0, 1), R)
    for a, t in ((ef * mask, 0.72), (ec * mask, 0.98)):
        a = np.clip(a, 0, 1)
        out = out * (1 - a[..., None]) + N.paint(np.full(a.shape, t, np.float32), R) * a[..., None]
    return N.surface(out, ctx)
