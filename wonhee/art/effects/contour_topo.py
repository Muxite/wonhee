"""Multiscale contour languages: hierarchical topographic lines and banded hatching."""
from __future__ import annotations

import math

import cv2
import numpy as np

from .. import contourkit as CK
from .. import features as F
from .. import natkit as N
from ..registry import effect


def _w(ctx, w1500, min_px=None):
    W = ctx.size[0]
    mp = float(ctx.p("min_px")) if min_px is None else min_px
    return max(w1500 * W / 1500.0, mp)


@effect("contour-topo-multiscale", family="contour",
        tags=["contours", "topographic", "cartographic", "lines", "multiscale", "source", "data", "precise", "scientific",
              "mood:cartographic", "mood:precise", "mood:scientific", "mood:calm"],
        params=CK.with_contour(N.with_common({
            "levels": (7, "LINES: major (index) levels; the coarse contour set"),
            "minor": (4, "LINES: minor lines per major interval (the intermediate contours; 1 = none)"),
            "fine": (0.5, "LINES: 0..1 hairline layer of fine texture contours riding on the minor ones (from the field's own detail)"),
            "w_major": (3.0, "LINES: major line width at 1500 px (floored by min_px)"),
            "min_gap": (4.5, "LINES: px at the output; minor/fine lines are thinned where they would pack closer than this (noise suppression)"),
            "fill": (0.7, "COLOUR: 0..1 hypsometric band tint under the lines (0 = lines on paper only)"),
            "shade": (0.35, "SURFACE: 0..1 hillshade relief under the lines"),
            "sun": (2.3, "SURFACE: hillshade light azimuth, radians"),
        }, source=False)))
def topo_multiscale(ctx):
    """Cartographic contouring in three weights: heavy index contours, medium
    minor contours and a hairline fine layer driven by the field's own small
    detail, with optional hypsometric tint bands and hillshade. Field: a
    source photo's tone (local-equalised), a 2-D data array in ctx.data, or
    procedural terrain. Where feature-contour-relief draws one uniform set of
    thin lines, this adds the hierarchy, a guaranteed minimum line width in
    output pixels (the lines survive 288x192), packing-density suppression
    (lines thin out instead of turning to mush where the slope is steep),
    despeckled fields, a mask (tone / detail / saliency), wobble for a
    hand-drawn survey look, and dark (glowing lines) / light (ink on paper)
    variants through the palette polarity. Best on smooth sources (skies,
    water, portraits at large scale), data fields, procedural terrain;
    on busy photos raise `scale` and `denoise`."""
    W, H = ctx.size
    f, img, theta, coh = CK.field(ctx, (W, H))
    wob = CK.wobble_field(ctx, (W, H), float(ctx.p("wobble")))
    if wob is not None:
        f = CK.warp_by(f, wob)
    mask = CK.mask_field(ctx, f, img)
    n, mn = int(ctx.p("levels")), max(1, int(ctx.p("minor")))
    s = W / 1500.0
    # field detail for the fine layer
    detail = f - F.blur(f, 6 * s)
    f_fine = f + float(ctx.p("fine")) * 2.2 * detail
    dist_M, gm = CK.level_lines(f, n, 0)
    dist_m, _ = CK.level_lines(f, n * mn, 0)
    dist_f, gf = CK.level_lines(f_fine, n * mn * 3, 0)
    gpx = gm * 1.0                                          # levels per px: spacing = 1 / gpx
    wM = _w(ctx, float(ctx.p("w_major")))
    wm = max(wM * 0.5, float(ctx.p("min_px")))
    wf = max(float(ctx.p("min_px")) * 0.8, 0.7)
    aM = CK.line_alpha(dist_M, wM)
    am = CK.line_alpha(dist_m, wm)
    af = CK.line_alpha(dist_f, wf) * float(ctx.p("fine")) * 0.8
    gap = float(ctx.p("min_gap"))
    sp_m = 1.0 / (gm * mn)                                  # spacing of the minor lines in px
    sp_f = 1.0 / (gf * mn * 3)
    am = am * N.smooth(gap * 0.6, gap * 1.6, sp_m)
    af = af * N.smooth(gap * 0.9, gap * 2.2, sp_f) * N.smooth(gap * 0.6, gap * 1.5, sp_m)
    sp_M = 1.0 / gm
    aM = aM * N.smooth(gap * 0.4, gap * 1.2, sp_M)
    # noise suppression: lines vanish on flat plateaus (almost no slope)
    flat = N.smooth(0.0, 0.4, gm * 12.0) if float(ctx.p("denoise")) > 0 else 1.0
    am, af, aM = am * flat * mask, af * flat * mask, aM * flat * mask
    R = CK.ink_ramp(ctx)
    dark = ctx.palette.polarity == "dark"
    band = np.floor(f * n) / n
    fill = float(ctx.p("fill"))
    t_fill = 0.02 + fill * (0.06 + 0.34 * band) if dark else fill * (0.03 + 0.20 * (1 - band) * 0 + 0.20 * band)
    sh = float(ctx.p("shade"))
    if sh > 0:
        g2y, g2x = np.gradient(F.blur(f, 3.0 * s))
        ca, sa = math.cos(float(ctx.p("sun"))), math.sin(float(ctx.p("sun")))
        shade = np.clip((g2x * ca + g2y * sa) * 55.0 * s, -1, 1)
        t_fill = t_fill + sh * 0.10 * shade
    out = N.paint(np.clip(t_fill, 0, 1), R)
    for a, t in ((af, 0.52), (am, 0.74), (aM, 0.97)):
        out = out * (1 - a[..., None]) + N.paint(np.full(a.shape, t, np.float32), R) * a[..., None]
    return N.surface(out, ctx)


@effect("contour-banded-hatch", family="contour",
        tags=["contours", "bands", "hatching", "lines", "multiscale", "source", "hand-drawn", "precise", "mood:hand-drawn", "mood:cartographic", "mood:calm"],
        params=CK.with_contour(N.with_common({
            "bands": (6, "STRUCTURE: number of coarse tone bands (flat regions)"),
            "spacing": (9.0, "STRUCTURE: hatch spacing at 1500 px (px)"),
            "angle": (0.6, "STRUCTURE: hatch angle of the first band, radians"),
            "angle_step": (0.45, "STRUCTURE: angle change per band (0 = one direction everywhere)"),
            "weight": (0.55, "STRUCTURE: hatch stroke weight relative to spacing (0..1; denser in darker bands)"),
            "edge": (0.7, "LINES: 0..1 strength of the band boundary lines"),
            "fill": (0.55, "COLOUR: 0..1 flat tint per band under the hatching"),
            "bleed": (0.35, "STRUCTURE: 0..1 hatch lines bend with the field inside each band (engraving feel)"),
        }, source=False)))
def banded_hatch(ctx):
    """Coarse contours plus fine texture in one image: the field is cut into a
    few broad tone bands (flat tint each), and every band is hatched at its own
    angle with stroke weight following the band's tone; band boundaries are
    drawn as ink lines. Hatch spacing never drops below what is legible at
    the output (min_px / spacing floor), so the 288x192 thumbnail reads as
    6 tinted bands with texture, the full-size image as a hand-engraved map.
    Hatch lines bend slightly with the field (bleed) and wobble. Sources: any
    photo (best with big tonal masses: sky, sea, portraits), data fields,
    procedural terrain. Failure mode: very busy photos produce confetti
    bands; raise `scale` / `denoise` or lower `bands`."""
    W, H = ctx.size
    f, img, theta, coh = CK.field(ctx, (W, H))
    s = W / 1500.0
    mask = CK.mask_field(ctx, f, img)
    nb = int(ctx.p("bands"))
    # smooth then quantise, drop tiny band fragments (noise suppression)
    fq = np.clip(np.floor(f * nb), 0, nb - 1).astype(np.int32)
    if float(ctx.p("denoise")) > 0:
        k = int(np.clip(int(0.014 * W) | 1, 3, 17))
        fq = cv2.medianBlur(fq.astype(np.uint8), k).astype(np.int32)
    band = fq.astype(np.float32) / max(1, nb - 1)
    dark = ctx.palette.polarity == "dark"
    ink_band = band if dark else 1 - band                      # how much ink a band carries
    sp = max(float(ctx.p("spacing")) * s, 2.4)
    ang = float(ctx.p("angle")) + float(ctx.p("angle_step")) * fq.astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    phi = (xx * np.cos(ang) + yy * np.sin(ang)) / sp
    phi = phi + float(ctx.p("bleed")) * 5.0 * (f - 0.5)
    wob = float(ctx.p("wobble"))
    if wob > 0:
        phi = phi + wob * 0.7 * N.tnoise(xx / W, yy / W, float(ctx.p("time", 0.0)), ctx.seed + 3, 0.08)
    gy, gx = np.gradient(phi)
    gm = np.hypot(gx, gy) + 1e-6
    fr = np.abs(phi - np.round(phi))
    dist = fr / gm
    wpx = np.maximum(sp * float(ctx.p("weight")) * (0.25 + 0.75 * ink_band), float(ctx.p("min_px")))
    a = CK.line_alpha(dist, wpx)
    # the lightest band stays almost bare (paper / ground)
    a = a * (0.08 + 0.92 * N.smooth(0.0, 0.25, ink_band + 0.02 * (nb > 1))) * mask
    # band boundaries
    gyb, gxb = np.gradient(F.blur(band, 0.8 * s))
    edge = np.clip(np.hypot(gxb, gyb) * 8 * max(s, 0.3) * 1.0, 0, 1)
    edge = cv2.dilate(edge, np.ones((3, 3), np.uint8)) if W < 700 else edge
    R = CK.ink_ramp(ctx)
    fill = float(ctx.p("fill"))
    t_fill = 0.02 + fill * (0.05 + 0.30 * band) if dark else fill * (0.04 + 0.24 * band)
    out = N.paint(np.clip(t_fill, 0, 1), R)
    t_line = 0.72 + 0.2 * band if dark else 0.75 + 0.2 * (1 - band) * 0 + 0.1 * band
    out = out * (1 - a[..., None]) + N.paint(np.clip(t_line, 0, 1).astype(np.float32), R) * a[..., None]
    e = np.clip(edge * float(ctx.p("edge")) * 1.6, 0, 1)
    out = out * (1 - e[..., None]) + N.paint(np.full(e.shape, 0.97, np.float32), R) * e[..., None]
    return N.surface(out, ctx)
