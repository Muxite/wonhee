"""Layered / exploded images (family `layers`): split a photo or a scalar field into 3-4 planes and
project them in an oblique / isometric / perspective view with spacing, thickness, tint, edge lines
and soft shadows.

IMPORTANT: depth inferred from an ordinary 2-D image (split=depth) is an approximation: a heuristic
from vertical position, focus, haze and tone, not measured geometry. For real depth give a height map
in ctx.data (2-D array / image path) or a hand mask (split=region).

Control groups: SOURCE (crop, zoom, hue_from_source), STRUCTURE (split, planes, smooth),
VIEW (view, spin, squash, scale, persp, spacing), COLOUR (tint, colour, dim, chroma),
SURFACE (thick, edge, shadow, grain).
"""
from __future__ import annotations

import os

import cv2
import numpy as np
from scipy import ndimage as ndi

from .. import features as F
from .. import layerkit as K
from .. import palette as pal
from .. import regionkit as R
from ..registry import effect

VIEW_PARAMS = {
    "view": ("iso", "VIEW iso | oblique | top"),
    "spin": (45.0, "VIEW rotation of the planes before squashing (iso only)"),
    "squash": (0.5, "VIEW vertical squash (0.5 = classic iso; smaller = flatter angle)"),
    "scale": (0.95, "VIEW size of the planes in the frame (>1 lets them bleed past the frame)"),
    "persp": (0.0, "VIEW 0..0.4 perspective (planes recede upward)"),
    "spacing": (0.17, "VIEW distance between planes as a fraction of the frame height"),
    "shift_x": (0.0, "VIEW horizontal offset of the stack, fraction of width"),
    "shift_y": (0.0, "VIEW vertical offset of the stack, fraction of height"),
}
COLOUR_PARAMS = {
    "tint": (0.45, "COLOUR 0..1 how far each plane is pushed into its own hue family (0 = photo colour)"),
    "colour": ("families", "COLOUR families | palette | source (hue arrangement across planes)"),
    "dim": (0.35, "COLOUR atmospheric darkening of the lower planes"),
    "chroma": (0.9, "COLOUR chroma multiplier"),
}
SURFACE_PARAMS = {
    "thick": (0.018, "SURFACE extruded wall thickness as a fraction of the frame height (0 = paper thin)"),
    "edge": (2.0, "SURFACE px of bright edge line on the plane outlines (0 = none)"),
    "shadow": (0.5, "SURFACE soft shadow cast on the plane below"),
    "shadow_blur": (16.0, "SURFACE shadow softness px"),
    "grain": (0.25, "SURFACE film grain"),
}
STRUCTURE = {
    "split": ("tone", "STRUCTURE tone | colour | depth | region | band | channel (see docstring)"),
    "planes": (4, "STRUCTURE number of planes (3-4 reads best small)"),
    "smooth": (0.55, "STRUCTURE simplification of plane outlines 0..1"),
    "invert": (False, "STRUCTURE swap bottom / top order"),
}


def _bgcol(fr):
    return fr.pal.canvas * (0.9 if fr.dark else 1.0)


def _plane_style(img, fr, hue, tint, chroma, s, dim=0.0):
    st = R.RegionStyle(hue=hue if tint > 0 else None, hue_keep=1.0 - tint, chroma=chroma * (1 - 0.3 * dim), c_abs=0.05 * tint, L_mul=1 - 0.5 * dim)
    one = R.RegionSet(np.zeros(img.shape[:2], np.int32), 1, 0.0)
    return R.restyle(img, one, [st], s_low=30 * s, s_tex=6 * s, dark=fr.dark)


def _hues(fr, n, mode):
    h0 = pal.rgb_to_oklch(fr.pal.structure)[2]
    ha = pal.rgb_to_oklch(fr.pal.accent)[2]
    if mode == "source":
        return [None] * n
    if mode == "palette":
        return [h0 if i % 2 == 0 else ha for i in range(n)]
    seq = [0, 40, 175, -35, 120, 215]
    return [(h0 + seq[i % len(seq)]) % 360 for i in range(n)]


def _dim(img, amt):
    if amt <= 0:
        return img
    lab = F.to_oklab(img)
    lab[..., 0] = lab[..., 0] * (1 - 0.5 * amt) + 0.0
    lab[..., 1:] *= (1 - 0.3 * amt)
    return F.from_oklab(lab)


def _masks(ctx, fr, kind, n):
    """cumulative masks (plane 0 everywhere) for mask-type splits."""
    s = fr.s
    smooth = float(ctx.p("smooth"))
    if kind == "tone":
        L = F.local_equalize(F.luminance(fr.img), 2.0, mix=0.6)
        fld = F.blur(L, 6 * s)
    elif kind == "colour":
        rs = R.segment(fr.img, n=n, by="colour", seed=ctx.seed, smooth=0.6, snap=0.5, merge=0.0)
        fld = rs.labels.astype(np.float32) + 0.01 * F.blur(F.luminance(fr.img), 8 * s)
        cnt = rs.n
        if cnt != n:
            n = cnt
    elif kind == "depth":
        d = ctx.data
        if isinstance(d, (str, os.PathLike)):
            d = R._read_any(d)
        if isinstance(d, dict):
            d = d.get("depth", d.get("field"))
        if isinstance(d, np.ndarray) and d.ndim >= 2:
            dd = d.mean(-1) if d.ndim == 3 else d
            dd = np.asarray(dd, np.float32)
            fld = F.stretch(R._crop_to_frame(dd, fr.box, (fr.W, fr.H), cv2.INTER_LINEAR), 1, 99)
        else:
            fld = K.depth_proxy(fr.img)
    else:   # region
        rs = R.regions_from_ctx(ctx, fr, n=n, by="colour", smooth=smooth, min_area=0.01, snap=0.3)
        fld = rs.labels.astype(np.float32) + 0.01 * F.blur(F.luminance(fr.img), 8 * s)
        n = rs.n
    ms = K.quantile_planes(fld, n, 1.0 * s) if kind not in ("region", "colour") else None
    if ms is None:
        lv = np.unique(np.floor(fld)).astype(int)
        ms = [(fld >= (v - 0.0)).astype(np.float32) for v in lv]
        ms[0] = np.ones_like(ms[0])
    ms = [ms[0]] + [K.simplify_mask(m, smooth, s) for m in ms[1:]]
    return ms


def _build_planes(ctx, fr):
    split = str(ctx.p("split"))
    n = int(np.clip(int(ctx.p("planes")), 2, 6))
    s = fr.s
    tint = float(ctx.p("tint"))
    chroma = float(ctx.p("chroma"))
    dim = float(ctx.p("dim"))
    edge = float(ctx.p("edge")) * s
    planes = []
    if split in ("band", "channel"):
        L = F.luminance(fr.img)
        if split == "band":
            low, mid, high = R.band_split(L, 28 * s, 5 * s)
            fields = [low, mid, high]
            fld_img = [R.lowpass(fr.img, 10 * s), None, None]
        else:
            fields = [fr.img[..., 0], fr.img[..., 1], fr.img[..., 2]]
        hs = _hues(fr, 3, "families" if str(ctx.p("colour")) == "source" else str(ctx.p("colour")))
        for i, f in enumerate(fields[:n]):
            if split == "band" and i == 0:
                base = R.lowpass(fr.img, 6 * s)
                rgb = _plane_style(base, fr, hs[0], tint, chroma, s)
                planes.append(K.Plane(rgb, np.ones_like(L), None, None, "low"))
                continue
            if split == "band":
                a = np.abs(f)
                a = np.clip(a / (np.percentile(a, 97) + 1e-6), 0, 1) ** 0.8
                a = a * (0.95 if i == 1 else 0.85)
                lum = np.clip(0.55 + f * (4.0 if i == 1 else 6.0), 0, 1)
            else:
                a = np.clip(F.stretch(f, 2, 99) ** 1.2, 0, 1) * 0.9
                lum = F.stretch(f, 2, 99)
            col = F.oklch_field(0.35 + 0.5 * lum if fr.dark else 0.45 + 0.5 * lum, 0.12 * chroma, hs[i] if hs[i] is not None else 40)
            planes.append(K.Plane(col.astype(np.float32), a.astype(np.float32), None, None, f"band{i}"))
        return planes
    masks = _masks(ctx, fr, split, n)
    n = len(masks)
    hs = _hues(fr, n, str(ctx.p("colour")))
    for i, m in enumerate(masks):
        rgb = _plane_style(fr.img, fr, hs[i], tint, chroma, s, dim * (n - 1 - i) / max(1, n - 1))
        hard = (m > 0.5).astype(np.float32)
        planes.append(K.Plane(rgb, m, K.plane_edge(hard, edge) if (edge > 0 and i > 0) else None, None, f"plane{i}"))
    return planes


def _render(ctx, fr, planes):
    W, H = fr.W, fr.H
    s = fr.s
    if bool(ctx.p("invert")):
        planes = planes[::-1]
    spin = float(ctx.p("spin"))
    M = K.view_matrix(W, H, str(ctx.p("view")), float(ctx.p("scale")), spin, float(ctx.p("squash")), float(ctx.p("persp")),
                      (float(ctx.p("shift_x")) * W, float(ctx.p("shift_y")) * H))
    p = fr.pal
    edge_col = (p.accent if fr.dark else p.canvas) * 0.9
    out = K.project(planes, (W, H), M, spacing=float(ctx.p("spacing")) * H, thick=float(ctx.p("thick")) * H, shadow=float(ctx.p("shadow")),
                    shadow_blur=float(ctx.p("shadow_blur")), edge_color=edge_col, bg_color=_bgcol(fr), dark=fr.dark, s=s)
    return R.finish(out, ctx.seed, float(ctx.p("grain")), s)


@effect("layers-exploded-planes", family="layers",
        tags=["source", "layers", "exploded", "isometric", "depth-proxy", "mood:technical", "mood:editorial", "mood:playful", "dark-light"],
        params=R.merge(R.SOURCE_PARAMS, STRUCTURE, VIEW_PARAMS, COLOUR_PARAMS, SURFACE_PARAMS))
def layers_exploded_planes(ctx):
    """The photo cut into 3-4 stacked planes and floated apart in an isometric / oblique view, each plane
    tinted into its own hue family, with edge lines, extruded thickness and soft shadows (an exploded
    terrain / map look). split: tone (luminance bands), colour (colour regions), depth (a HEURISTIC
    depth proxy from position, focus and haze; or your own height map in ctx.data), region (hand mask /
    polygons in ctx.data), band (low-frequency colour base + mid structure + fine texture as glass
    sheets), channel (R, G, B as three tinted sheets). Depth from one ordinary image is an
    approximation, so treat `depth` as stylised layering, not geometry. Best on landscapes,
    aerial photos, maps, textured materials; small size: 3-4 planes, thick >= 0.015, edge >= 2.
    Failure: portraits (the face is cut by tone bands; use split=region with a mask)."""
    fr = R.frame_source(ctx, local=0.0)
    return _render(ctx, fr, _build_planes(ctx, fr))


@effect("layers-terrain-slabs", family="layers",
        tags=["layers", "terrain", "isometric", "data", "mood:calm", "mood:technical", "mood:editorial", "dark-light", "all-over"],
        params=R.merge(R.SOURCE_PARAMS, {
            "planes": (6, "STRUCTURE number of slabs (4-8)"), "smooth": (0.6, "STRUCTURE outline simplification 0..1"),
            "invert": (False, "STRUCTURE swap high and low"), "source_mix": (0.0, "COLOUR 0..1 paint slab tops with the source's own colour"),
            "contour": (0.4, "SURFACE faint contour lines on each slab top"), "field_scale": (1.0, "STRUCTURE zoom into the field (>1 = fewer, larger shapes)"),
            "colour": ("ramp", "COLOUR ramp (palette canvas>structure>accent) | families (hue per slab)"),
            "view": ("iso", "VIEW iso | oblique | top"), "spin": (38.0, "VIEW rotation"), "squash": (0.55, "VIEW squash"),
            "scale": (1.55, "VIEW plane size (larger fills the frame)"), "persp": (0.0, "VIEW perspective"),
            "spacing": (0.07, "VIEW slab spacing as a fraction of frame height"), "shift_x": (0.0, "VIEW shift x"), "shift_y": (0.0, "VIEW shift y"),
            "thick": (0.05, "SURFACE slab wall thickness fraction of height"), "edge": (1.5, "SURFACE top edge highlight px"),
            "shadow": (0.55, "SURFACE soft shadow"), "shadow_blur": (18.0, "SURFACE shadow softness"), "grain": (0.2, "SURFACE grain")}))
def layers_terrain_slabs(ctx):
    """A scalar field (a height-map array / image path in ctx.data, else the luminance of the source, else a
    seeded procedural landscape) cut into 4-8 terraced slabs: nested cut-out shapes stacked in an
    isometric view, coloured by the palette ramp (low dark, high light), with extruded walls, lit top
    edges, faint contours and soft shadows. A clean cartographic / paper-cut terrain; works from
    photos (tone as height), simulation fields (use the physics-* output as the source or pass the
    matrix as data) and real elevation grids. Fills the frame (planes bleed), no focal spot.
    Failure: noisy fields give island confetti (raise `smooth`, lower `planes`)."""
    W, H = ctx.size
    d = ctx.data
    if isinstance(d, (str, os.PathLike)):
        d = R._read_any(d)
    if isinstance(d, dict):
        d = d.get("field", d.get("depth", d.get("matrix")))
    fr = R.frame_source(ctx, local=0.0)
    s = fr.s
    if d is not None and not isinstance(d, np.ndarray):
        d = np.asarray(d, np.float32)
    if isinstance(d, np.ndarray) and d.ndim >= 2:
        dd = np.asarray(d.mean(-1) if d.ndim == 3 else d, np.float32)
        fld = cv2.resize(dd, (W, H), interpolation=cv2.INTER_CUBIC)
    else:
        fld = F.blur(F.local_equalize(F.luminance(fr.img), 2.0, mix=0.5), 4 * s)
    fz = float(ctx.p("field_scale"))
    if fz > 1.01:
        h2, w2 = int(H / fz), int(W / fz)
        fld = cv2.resize(fld[(H - h2) // 2:(H - h2) // 2 + h2, (W - w2) // 2:(W - w2) // 2 + w2], (W, H), interpolation=cv2.INTER_CUBIC)
    fld = F.blur(fld, 3 * s)
    if bool(ctx.p("invert")):
        fld = -fld
    n = int(np.clip(int(ctx.p("planes")), 3, 9))
    smooth = float(ctx.p("smooth"))
    ms = K.quantile_planes(fld, n, 0.0)
    ms = [ms[0]] + [K.simplify_mask(m, smooth, s) for m in ms[1:]]
    p = fr.pal
    ramp = p.ramp(256, ("canvas", "structure", "accent"))
    fam = str(ctx.p("colour")) == "families"
    h0 = pal.rgb_to_oklch(p.structure)[2]
    lo, hi = (0.30, 0.84) if fr.dark else (0.40, 0.95)
    mix = float(ctx.p("source_mix"))
    contour = float(ctx.p("contour"))
    planes = []
    for i, m in enumerate(ms):
        t = i / max(1, n - 1)
        if fam:
            col = F.oklch_field(lo + (hi - lo) * t, 0.13, (h0 + 70 * t) % 360).reshape(-1)[:3] if False else pal.oklch_to_rgb(lo + (hi - lo) * t, 0.13, (h0 + 70 * t) % 360)
        else:
            col = ramp[int((0.18 + 0.78 * t) * 255)]
            # lift low slabs so they are visible against the canvas
            col = col * (0.85 + 0.3 * t)
        rgb = np.ones((H, W, 3), np.float32) * col
        if mix > 0:
            rgb = rgb * (1 - mix) + fr.img * mix * (0.7 + 0.5 * t)
        if contour > 0 and i < n - 1:
            nxt = ndi.gaussian_filter((fld >= np.percentile(fld, 100 * (i + 1.5) / n)).astype(np.float32), 1.0)
            ln = np.clip(1 - np.abs(nxt - 0.5) * 6, 0, 1) * contour * 0.5
            rgb = rgb * (1 - ln[..., None]) + np.clip(col * 1.35 + 0.05, 0, 1) * ln[..., None]
        hard = (m > 0.5).astype(np.float32)
        planes.append(K.Plane(rgb, m, K.plane_edge(hard, float(ctx.p("edge")) * s) if i > 0 else None, np.clip(col * 0.45, 0, 1), f"slab{i}"))
    if bool(ctx.p("invert")):
        pass
    M = K.view_matrix(W, H, str(ctx.p("view")), float(ctx.p("scale")), float(ctx.p("spin")), float(ctx.p("squash")), float(ctx.p("persp")),
                      (float(ctx.p("shift_x")) * W, float(ctx.p("shift_y")) * H))
    edge_col = np.clip(ramp[int(0.95 * 255)] * 1.0, 0, 1) if fr.dark else np.clip(p.canvas, 0, 1)
    bg = np.ones((H, W, 3), np.float32) * (p.canvas * 0.9 if fr.dark else p.canvas * 0.97)
    out = K.project(planes, (W, H), M, spacing=float(ctx.p("spacing")) * H, thick=float(ctx.p("thick")) * H, shadow=float(ctx.p("shadow")),
                    shadow_blur=float(ctx.p("shadow_blur")), edge_color=edge_col * 0.9, wall_dark=0.5, bg_color=bg[0, 0], dark=fr.dark, s=s)
    return R.finish(out, ctx.seed, float(ctx.p("grain")), s)
