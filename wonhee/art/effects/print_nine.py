"""print-nine-ways: one subject, a 3x3 sheet of treatments (outline with anchors, pixel blocks, spiral,
scribble, dot rings, banded slices, halftone, hatch, stipple, silhouette, contour)."""
from __future__ import annotations

import math

import cv2
import numpy as np

from .. import features as F
from .. import printkit as K
from ..registry import effect

ALL = ["outline", "pixels", "spiral", "scribble", "rings", "bands", "halftone", "hatch", "stipple", "silhouette", "contour"]
DEFAULT = "outline,pixels,spiral,scribble,rings,bands,halftone,hatch,stipple"
SS = 2


def blob_tone(w, h, rng):
    """Generated subject: a big lobed ink-blot with a hole and a satellite, shaded, filling its cell."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w * 0.5 + rng.uniform(-0.06, 0.06) * w, h * 0.5 + rng.uniform(-0.06, 0.06) * h
    dx, dy = (xx - cx) / (w * 0.5), (yy - cy) / (h * 0.5)
    th = np.arctan2(dy, dx)
    r = np.hypot(dx, dy)
    k = rng.integers(2, 5, 3)
    ph = rng.uniform(0, 6.28, 3)
    edge = 0.62 * (1 + 0.22 * np.sin(k[0] * th + ph[0]) + 0.14 * np.sin((k[1] + 2) * th + ph[1]) + 0.08 * np.sin((k[2] + 5) * th + ph[2]))
    body = F.smoothstep(edge + 0.02, edge - 0.02, r)
    hx, hy = rng.uniform(-0.2, 0.2, 2)
    hole = F.smoothstep(0.17, 0.21, np.hypot(dx - hx, dy - hy))
    sx, sy = rng.choice([-1, 1]) * 0.72, rng.choice([-1, 1]) * 0.62
    sat = F.smoothstep(0.2, 0.17, np.hypot(dx - sx, dy - sy))
    shade = 0.6 + 0.4 * np.clip(0.5 + 0.5 * (dx * 0.6 + dy * 0.8), 0, 1)
    return np.clip(np.maximum(body * hole, sat) * shade, 0, 1).astype(np.float32)


def _thr(T):
    return float(np.clip(np.median(T[T > 0.02]) if (T > 0.02).mean() > 0.6 else 0.42, 0.3, 0.6))


def _cv(w, h):
    return np.zeros((h * SS, w * SS), np.uint8)


def _down(m, w, h):
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255


def _lw(size, v):
    return max(int(round(v * K.U(size) * SS)), SS)


def t_outline(T, size, rng):
    h, w = T.shape
    a, b = _cv(w, h), _cv(w, h)
    d = F.blur(T, 1.5)
    m = (cv2.resize(d, (w * SS, h * SS)) > _thr(T)).astype(np.uint8) * 255
    cs, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for c in cs:
        if len(c) < 40:
            continue
        ap = cv2.approxPolyDP(c, 2.2 * SS * max(K.U(size) * 3, 1), True)
        cv2.polylines(a, [ap], True, 255, _lw(size, 2.0), cv2.LINE_AA)
        for i, p in enumerate(ap[:, 0, :]):
            r = max(SS * 2, _lw(size, 3.5))
            cv2.rectangle(b, (p[0] - r, p[1] - r), (p[0] + r, p[1] + r), 255, -1)
            if i % 3 == 0:     # bezier handles
                q = ap[(i + 1) % len(ap), 0, :]
                hv = ((q - p) * 0.3).astype(int)
                cv2.line(b, tuple(p), tuple(p + hv), 255, max(SS, _lw(size, 1.0)), cv2.LINE_AA)
                cv2.circle(b, tuple(p + hv), max(SS, _lw(size, 2.2)), 255, -1, cv2.LINE_AA)
    return _down(a, w, h), _down(b, w, h)


def t_pixels(T, size, rng):
    h, w = T.shape
    n = 12
    cs = h / n
    nx = int(round(w / cs))
    g = cv2.resize(T, (nx, n), interpolation=cv2.INTER_AREA)
    q = np.floor(np.clip(g, 0, 0.999) * 4) / 3
    out = np.zeros((h, w), np.float32)
    gap = max(1, int(round(0.08 * cs)))
    for j in range(n):
        for i in range(nx):
            if q[j, i] > 0.05:
                y0, x0 = int(j * cs), int(i * cs)
                out[y0:int((j + 1) * cs) - gap, x0:int((i + 1) * cs) - gap] = 0.25 + 0.75 * q[j, i]
    return out, None


def t_spiral(T, size, rng):
    h, w = T.shape
    a = _cv(w, h)
    cx, cy = w / 2 * SS, h / 2 * SS
    pitch = max(h * SS / 11, 5 * SS)
    R = math.hypot(w, h) * SS / 2
    th = 0.0
    pts = []
    while True:
        r = pitch * th / (2 * math.pi)
        if r > R:
            break
        pts.append((cx + r * math.cos(th), cy + r * math.sin(th)))
        th += 0.18 * min(1, 14 * SS / max(r, 14 * SS)) + 0.03
    Tl = cv2.resize(T, (w * SS, h * SS))
    for i in range(1, len(pts)):
        x, y = pts[i]
        xi, yi = int(np.clip(x, 0, w * SS - 1)), int(np.clip(y, 0, h * SS - 1))
        t = Tl[yi, xi]
        wd = max(1, int(round(pitch * (0.08 + 0.78 * t))))
        cv2.line(a, (int(pts[i - 1][0]), int(pts[i - 1][1])), (int(x), int(y)), 255, wd, cv2.LINE_AA)
    return _down(a, w, h), None


def t_scribble(T, size, rng):
    h, w = T.shape
    a = _cv(w, h)
    Tl = cv2.resize(F.blur(T, 1.0), (w * SS, h * SS))
    row = max(h * SS / 16, 5 * SS)
    y = row / 2
    flip = False
    lw = _lw(size, 1.8)
    while y < h * SS + row:
        pts = []
        x = 0.0
        yi = int(np.clip(y, 0, h * SS - 1))
        while x < w * SS:
            t = Tl[yi, int(np.clip(x, 0, w * SS - 1))]
            amp = row * (0.15 + 0.95 * t) * rng.uniform(0.8, 1.1)
            step = row * (0.9 - 0.55 * t) * rng.uniform(0.7, 1.2)
            pts.append((x, y + (amp if flip else -amp)))
            flip = not flip
            x += max(step, 3 * SS)
        cv2.polylines(a, [np.array(pts, np.int32).reshape(-1, 1, 2)], False, 255, lw, cv2.LINE_AA)
        y += row * 0.8
    return _down(a, w, h), None


def t_rings(T, size, rng):
    h, w = T.shape
    a = _cv(w, h)
    cx, cy = w / 2 * SS, h / 2 * SS
    pitch = max(h * SS / 14, 4.5 * SS)
    Tl = cv2.resize(T, (w * SS, h * SS))
    r = pitch
    R = math.hypot(w, h) * SS / 2
    while r < R:
        n = max(8, int(2 * math.pi * r / pitch))
        for k in range(n):
            th = 2 * math.pi * k / n + 0.4 * r / pitch
            x, y = cx + r * math.cos(th), cy + r * math.sin(th)
            if 0 <= x < w * SS and 0 <= y < h * SS:
                t = Tl[int(y), int(x)]
                rad = pitch * (0.06 + 0.42 * t)
                if rad > 0.9:
                    cv2.circle(a, (int(x), int(y)), int(round(rad)), 255, -1, cv2.LINE_AA)
        r += pitch
    return _down(a, w, h), None


def t_bands(T, size, rng):
    h, w = T.shape
    nb = 9
    out = np.zeros((h, w), np.float32)
    bh = h / nb
    Tb = (F.blur(T, 1.2) > _thr(T)).astype(np.float32)
    for j in range(nb):
        y0, y1 = int(j * bh), int((j + 1) * bh) - max(1, int(0.12 * bh))
        sh = int(rng.normal(0, w * 0.06))
        out[y0:y1] = np.roll(Tb, sh, 1)[y0:y1]
    return out, None


def t_halftone(T, size, rng):
    return K.screen(T, size, kind="dot", pitch=26, angle=45, gain=1.1), None


def t_hatch(T, size, rng):
    return K.screen(T, size, kind="line", pitch=24, angle=35, gain=1.0), None


def t_stipple(T, size, rng):
    h, w = T.shape
    a = _cv(w, h)
    n = int(w * h / (max(20 * K.U(size), 5) ** 2) * 2.2)
    cand = rng.uniform(0, 1, (n * 12, 2)) * [w, h]
    t = T[np.clip(cand[:, 1].astype(int), 0, h - 1), np.clip(cand[:, 0].astype(int), 0, w - 1)]
    keep = cand[rng.random(len(cand)) < t ** 1.2]
    rad = max(1.0, 1.15 * max(K.U(size) * 3.2, 1.2)) * SS
    for x, y in keep[: n * 3]:
        cv2.circle(a, (int(x * SS), int(y * SS)), int(round(rad)), 255, -1, cv2.LINE_AA)
    return _down(a, w, h), None


def t_silhouette(T, size, rng):
    t = _thr(T)
    return F.smoothstep(t - 0.06, t + 0.06, F.blur(T, 1.0)), None


def t_contour(T, size, rng):
    h, w = T.shape
    a = _cv(w, h)
    d = F.blur(cv2.resize(T, (w * SS, h * SS)), 2 * SS)
    for lv in (0.15, 0.3, 0.45, 0.6, 0.75):
        m = (d > lv).astype(np.uint8) * 255
        cs, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.polylines(a, cs, True, 255, _lw(size, 1.6), cv2.LINE_AA)
    return _down(a, w, h), None


FN = {"outline": t_outline, "pixels": t_pixels, "spiral": t_spiral, "scribble": t_scribble, "rings": t_rings, "bands": t_bands,
      "halftone": t_halftone, "hatch": t_hatch, "stipple": t_stipple, "silhouette": t_silhouette, "contour": t_contour}


@effect("print-nine-ways", family="print", tags=["editorial", "playful", "structured", "archival", "source", "generated", "sheet", "light", "dark"],
        params=K.params(grid=(3, "cells per side (2-4)"), treatments=(DEFAULT, "comma list from " + ",".join(ALL) + " (cycled to fill the grid)"),
                        frame=(0.8, "0..1 hairline frame/rule strength"), second_ink=(True, "anchors/handles in a second ink"),
                        gutter=(0.025, "gutter as fraction of the width"), smooth=(0.02, "photo subject simplification (blur as fraction of a cell width)")))
def nine_ways(ctx):
    """One subject shown in a 3x3 sheet of treatments, like a printmaker's proof page: outline with
    anchor points and handles, pixel blocks, spiral line, scribble, dot rings, banded slices, halftone,
    hatch, stipple. The subject is the source photo (cropped once, same crop in every cell) or a
    generated ink-blot. Printed in one or two inks on paper; ``clean=true`` for no fibre/grain. Cells
    are big marks, so it reads at 288x192 (cells are 96x64). Cells fill the frame edge to edge."""
    W, H = ctx.size
    rng = ctx.rng
    g = int(np.clip(ctx.p("grid"), 2, 4))
    look = K.resolve_look(ctx, 2)
    dark = look["dark"]
    gut = max(int(float(ctx.p("gutter")) * W), 3)
    cw, ch = (W - gut * (g + 1)) // g, (H - gut * (g + 1)) // g
    names = [t.strip() for t in str(ctx.p("treatments")).split(",") if t.strip() in FN] or DEFAULT.split(",")
    if ctx.source is not None or ctx.data is not None:
        img = K.source_rgb(ctx, zoom=float(ctx.p("zoom")))
        Lt = F.stretch(F.local_equalize(F.luminance(img), 2.5, mix=float(ctx.p("local")) + 0.3), 2, 98)
        D = (Lt if dark else 1 - Lt)
        Dcell = cv2.resize(D, (cw, ch), interpolation=cv2.INTER_AREA)
        Dcell = F.stretch(F.blur(Dcell, float(ctx.p("smooth")) * cw), 4, 96)
    else:
        Dcell = blob_tone(cw, ch, rng)
    A = np.zeros((H, W), np.float32)
    B = np.zeros((H, W), np.float32)
    Ruled = np.zeros((H, W), np.float32)
    for i in range(g * g):
        r, c = divmod(i, g)
        x0, y0 = gut + c * (cw + gut), gut + r * (ch + gut)
        a, b = FN[names[i % len(names)]](Dcell, ctx.size, rng)
        A[y0:y0 + ch, x0:x0 + cw] = np.maximum(A[y0:y0 + ch, x0:x0 + cw], a)
        if b is not None:
            B[y0:y0 + ch, x0:x0 + cw] = np.maximum(B[y0:y0 + ch, x0:x0 + cw], b)
        lw = 1 if W < 600 else 2
        if float(ctx.p("frame")) > 0:
            cv2.rectangle(Ruled, (x0 - 1, y0 - 1), (x0 + cw, y0 + ch), 1.0, lw)
    inks = look["inks"]
    if not ctx.p("inks"):
        h = K.pal.rgb_to_oklch(look["palette"].structure)[2]
        ah = K.pal.rgb_to_oklch(look["palette"].accent)[2]
        inks = [K.pal.oklch_to_rgb(0.34 if not dark else 0.86, 0.15 if not dark else 0.08, h),
                K.pal.oklch_to_rgb(0.58 if not dark else 0.70, 0.20, ah)]
    covs = [A, B if ctx.p("second_ink") else np.zeros_like(B), Ruled * float(ctx.p("frame"))]
    inks3 = [inks[0], inks[1], inks[0]]
    sf = K.surface(ctx)
    return K.press(covs, inks3, ctx.size, rng, paper_rgb=look["paper"], screens=dict(kind="raw"), opacity=[look["opacity"], look["opacity"], 0.55], mode=look["mode"], **sf)
