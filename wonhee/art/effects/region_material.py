"""Region and material interpreter (family `region`): a photo or a real material
behaves like a designed composition while keeping its physical texture.

Four independent control groups on every effect:
  SOURCE    crop, zoom, hue_from_source, local (tone equalisation, default off here)
  STRUCTURE regions, by, smooth, min_area, snap  (or a hand mask / polygons in ctx.data)
  COLOUR    colour (source|palette|families|hues), hues, spread, chroma, contrast, hue_keep
  SURFACE   texture, tex_scale, flat, calm, grain, edge, edge_ink
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .. import features as F
from .. import palette as pal
from .. import regionkit as R
from ..registry import effect

COLOUR_PARAMS = {
    "colour": ("families", "COLOUR source | palette | families | hues: how hue families are arranged over regions"),
    "hues": ("", "COLOUR explicit hues in degrees for colour=hues, e.g. '190,30,260' (dark to light regions)"),
    "spread": (0.35, "COLOUR 0 keep each region's own lightness, 1 spread region lightness evenly (more contrast between regions)"),
    "chroma": (0.9, "COLOUR chroma multiplier (richness)"),
    "contrast": (1.0, "COLOUR lightness contrast inside each region"),
    "hue_keep": (0.35, "COLOUR share of the source's own hue variation kept inside a region"),
}
SURFACE_PARAMS = {
    "texture": (1.0, "SURFACE retention of fine physical texture (0 = smooth poster, 1 = as photographed, 2 = emphasised)"),
    "tex_scale": (6.0, "SURFACE size in px (at 1500 wide) separating fine texture from structure"),
    "flat": (0.5, "SURFACE 0..1 flatten the big shading inside regions (1 = flat shapes with texture on top)"),
    "calm": (0.0, "SURFACE 0..1 edge-aware smoothing of the source first (glassy surfaces, keeps edges)"),
    "grain": (0.3, "SURFACE final film grain"),
    "edge": (0.0, "SURFACE px width of ink boundary lines between regions (0 = none)"),
    "edge_ink": (0.7, "SURFACE 0..1 strength of the boundary lines"),
}
LOCAL = {"local": (0.0, "SOURCE 0..1 local-contrast equalisation of tone (off by default: keeps the real light)")}
ALL = R.merge(R.SOURCE_PARAMS, LOCAL, R.STRUCTURE_PARAMS, COLOUR_PARAMS, SURFACE_PARAMS)


def _setup(ctx):
    fr = R.frame_source(ctx, local=float(ctx.p("local", 0.0)))
    calm = float(ctx.p("calm", 0.0))
    if calm > 0:
        fr.img = R.edge_smooth(fr.img, calm, fr.s)
    rs = R.regions_from_ctx(ctx, fr, n=int(ctx.p("regions")), by=str(ctx.p("by")), smooth=float(ctx.p("smooth")),
                            min_area=float(ctx.p("min_area")), snap=float(ctx.p("snap")), merge=float(ctx.p("merge")))
    base = pal.rgb_to_oklch(fr.pal.structure)[2]
    return fr, rs, base


def _styles(ctx, fr, rs, base, **over):
    st = R.assign_styles(rs, fr.img, fr.pal, colour=str(ctx.p("colour")), hues=str(ctx.p("hues")), spread=float(ctx.p("spread")),
                         chroma=float(ctx.p("chroma")), contrast=float(ctx.p("contrast")), hue_keep=float(ctx.p("hue_keep")), base_hue=base)
    tex = float(ctx.p("texture"))
    for s in st:
        s.flat = float(ctx.p("flat"))
        s.high = tex
        s.mid = max(0.0, 0.5 + 0.5 * tex)
        for k, v in over.items():
            setattr(s, k, v)
    return st


def _lines(ctx, fr, rs, out):
    w = float(ctx.p("edge")) * fr.s
    if w <= 0:
        return out
    ink = fr.pal.canvas if fr.dark else fr.pal.structure * 0.45
    return R.boundary_lines(rs, w, ink, out, float(ctx.p("edge_ink")))


@effect("region-material-poster", family="region",
        tags=["source", "regions", "material", "poster", "mood:bold", "mood:calm", "tactile", "mask", "dark-light"],
        params=R.merge(ALL, {"flat": (0.75, "SURFACE flatten big shading inside regions"),
                              "edge": (0.0, "SURFACE ink boundary px")}))
def region_material_poster(ctx):
    """The photo becomes a few designed colour shapes (turquoise water, warm rock, pale sky) with
    the real surface left on top. Regions come from colour, hue, luma, detail or superpixels (or a hand mask /
    polygons in ctx.data), boundaries are simplified and snapped to real edges. Each region gets
    its own lightness band and hue family; the fine texture and the structure bands are retained by
    scale (`texture`, `tex_scale`), the big shading is flattened (`flat`). Best on materials with
    two or three clearly different surfaces: water against rock, clouds over sea, soil, fabric,
    sunsets. Failure: gradients (sky) split into bands (lower `regions`, raise `smooth`);
    uniform sources give arbitrary regions; photos of people need a hand mask for a clean face."""
    fr, rs, base = _setup(ctx)
    out = R.restyle(fr.img, rs, _styles(ctx, fr, rs, base), s_low=36 * fr.s, s_tex=float(ctx.p("tex_scale")) * fr.s, dark=fr.dark)
    out = _lines(ctx, fr, rs, out)
    return R.finish(out, ctx.seed, float(ctx.p("grain")), fr.s)


@effect("region-tonal-families", family="region",
        tags=["source", "regions", "material", "colour-map", "mood:lush", "mood:dreamy", "mood:cool", "mood:warm", "dark-light"],
        params=R.merge(ALL, {"flat": (0.0, "SURFACE keep the natural shading (0) .. flatten (1)"),
                              "rim": (0.5, "SURFACE light rim along region boundaries (shoreline / foam / sun-lit edge)"),
                              "rim_px": (28.0, "SURFACE rim width px at 1500 wide"),
                              "hue_keep": (0.5, "COLOUR share of the source's hue variation kept inside a region")}))
def region_tonal_families(ctx):
    """Keeps the full light and shading of the photograph, but arranges colour on purpose: every
    region is pushed into its own hue family (cool for the dark regions, the palette hue in the
    middle, warm accents in the lights, or your own `hues`), with chroma shaped by lightness so
    nothing goes neon or muddy. `rim` adds a soft luminous band along the region boundaries
    (shoreline, foam, rim light) in the accent colour. Best on water/land, cloud/sea, night
    cities, fabric; failure: very busy photos give many tiny colour islands (lower `regions`,
    raise `min_area`)."""
    fr, rs, base = _setup(ctx)
    out = R.restyle(fr.img, rs, _styles(ctx, fr, rs, base), s_low=36 * fr.s, s_tex=float(ctx.p("tex_scale")) * fr.s, dark=fr.dark)
    rim = float(ctx.p("rim"))
    if rim > 0:
        e = F.label_edges(rs.labels, 1) > 0.5
        d = ndi.distance_transform_edt(~e).astype(np.float32)
        glow = np.exp(-(d / (float(ctx.p("rim_px")) * fr.s)) ** 1.3)
        glow = ndi.gaussian_filter(glow, 1.5 * fr.s)
        acc = fr.pal.accent if fr.dark else fr.pal.accent * 0.9
        if fr.dark:
            out = 1 - (1 - out) * (1 - (glow * rim * 0.5)[..., None] * acc)
        else:
            out = out * (1 - (glow * rim * 0.3)[..., None]) + acc * (glow * rim * 0.3)[..., None]
    out = _lines(ctx, fr, rs, out)
    return R.finish(out, ctx.seed, float(ctx.p("grain")), fr.s)


@effect("region-paper-map", family="region",
        tags=["source", "regions", "print", "poster", "mood:editorial", "mood:playful", "mood:calm", "tactile", "mask", "dark-light"],
        params=R.merge(ALL, {"regions": (5, "STRUCTURE number of regions"), "flat": (1.0, "SURFACE flat shapes"),
                              "texture": (0.8, "SURFACE fine texture kept as ink tooth"), "steps": (2, "SURFACE tone steps inside a region (0 = none)"),
                              "edge": (2.5, "SURFACE ink boundary px"), "edge_ink": (0.8, "SURFACE ink strength"),
                              "paper": (1.0, "SURFACE paper fibre strength"), "misregister": (2.0, "SURFACE px of print misregistration of the ink lines (0 = off)"),
                              "colour": ("palette", "COLOUR palette | families | hues | source")}))
def region_paper_map(ctx):
    """A screen-print / paper-map reading of the photo: flat region fills in the palette (or hue
    families), a limited number of tone steps per region carrying the real light, the fine
    texture as ink tooth, inked boundaries slightly out of register, and paper fibre over it.
    Reads cleanly at 288x192 because it is shapes first. Best on landscapes, clouds, water,
    aerial and satellite views, terrain, fabric. Failure: portraits become cartoon masks
    (give a hand mask or use region-material-poster)."""
    fr, rs, base = _setup(ctx)
    out = R.restyle(fr.img, rs, _styles(ctx, fr, rs, base, steps=int(ctx.p("steps"))), s_low=30 * fr.s,
                    s_tex=float(ctx.p("tex_scale")) * fr.s, dark=fr.dark)
    mis = float(ctx.p("misregister")) * fr.s
    w = float(ctx.p("edge")) * fr.s
    if w > 0:
        ink = (fr.pal.canvas if fr.dark else fr.pal.structure * 0.4)
        M = np.float32([[1, 0, mis], [0, 1, mis * 0.7]])
        import cv2
        e = rs.edges(w)
        if mis > 0:
            e = cv2.warpAffine(e, M, (fr.W, fr.H), borderMode=cv2.BORDER_REFLECT)
        # ink tooth: line breaks up with fine texture
        tooth = np.clip(0.75 + 3.0 * ndi.gaussian_filter(np.random.default_rng(ctx.seed).normal(0, 1, e.shape), 0.9 * fr.s), 0, 1)
        e = (e * tooth * float(ctx.p("edge_ink")))[..., None]
        out = out * (1 - e) + ink * e
    out = out * (1 + R.paper_grain(out.shape, ctx.seed, fr.s, float(ctx.p("paper")))[..., None] * 2.2)
    return R.finish(out, ctx.seed, float(ctx.p("grain")), fr.s)


@effect("region-scale-retain", family="region",
        tags=["source", "regions", "material", "smoothing", "mood:calm", "mood:dreamy", "mood:serene", "tactile", "dark-light"],
        params=R.merge(ALL, {"keep": ("0.1,0.5,1.0,1.0", "SURFACE texture retention per region, dark to light (e.g. glassy shadows, crisp lights)"),
                              "calm": (0.9, "SURFACE how glassy the calm regions get"),
                              "colour": ("palette", "COLOUR palette | families | hues | source"), "hue_keep": (0.6, "COLOUR source hue variation kept"),
                              "spread": (0.2, "COLOUR lightness spread between regions"), "flat": (0.0, "SURFACE flatten shading")}))
def region_scale_retain(ctx):
    """Texture retention by scale, region by region: each region is blended between an edge-aware
    smoothed version of the surface (glass, mist, still water) and the photographed texture,
    according to `keep` (one number per region, dark to light; cycled). The result keeps real
    material where it matters and calms the rest, with a gentle shared colour family on top.
    Best on water + rock, fog + trees, skin-free product shots, cloud tops; failure: very small
    regions flicker between smooth and rough (raise `min_area`)."""
    fr0 = R.frame_source(ctx, local=float(ctx.p("local", 0.0)))
    calm = float(ctx.p("calm"))
    sm = R.edge_smooth(fr0.img, calm, fr0.s, iters=3)
    ctx.params["calm"] = 0.0
    fr, rs, base = _setup(ctx)
    keep = [float(x) for x in str(ctx.p("keep")).split(",") if x.strip()]
    kk = np.array([keep[i % len(keep)] for i in range(rs.n)], np.float32)
    kmap = (rs.weights * kk[:, None, None]).sum(0)
    mixed = sm * (1 - kmap[..., None]) + fr.img * kmap[..., None]
    fr.img = mixed
    tex = float(ctx.p("texture"))
    st = _styles(ctx, fr, rs, base)
    for i, s in enumerate(st):
        s.high = tex * float(kk[i]) ** 0.7
        s.mid = (0.5 + 0.5 * tex) * (0.3 + 0.7 * float(kk[i]))
    out = R.restyle(mixed, rs, st, s_low=36 * fr.s, s_tex=float(ctx.p("tex_scale")) * fr.s, dark=fr.dark)
    out = _lines(ctx, fr, rs, out)
    return R.finish(out, ctx.seed, float(ctx.p("grain")), fr.s)


@effect("region-mask-overlay", family="region",
        tags=["source", "regions", "mask", "tool", "diagnostic", "mood:technical"],
        params=R.merge(R.SOURCE_PARAMS, R.STRUCTURE_PARAMS, {"alpha": (0.6, "STRUCTURE strength of the region colours over the dimmed source"),
                                                               "edge": (2.0, "SURFACE boundary line px")}))
def region_mask_overlay(ctx):
    """Tool: shows the regions an effect will use (flat region colours over the dimmed photo with
    boundary lines) so you can judge `regions`, `by`, `smooth`, `snap`, or check a hand mask /
    polygon list passed in ctx.data before rendering the real look. Also a legible 'segmentation'
    look on its own."""
    fr = R.frame_source(ctx)
    rs = R.regions_from_ctx(ctx, fr, n=int(ctx.p("regions")), by=str(ctx.p("by")), smooth=float(ctx.p("smooth")),
                            min_area=float(ctx.p("min_area")), snap=float(ctx.p("snap")), merge=float(ctx.p("merge")))
    base = pal.rgb_to_oklch(fr.pal.structure)[2]
    cols = np.stack([F.oklch_field(0.35 + 0.5 * i / max(1, rs.n - 1) if fr.dark else 0.45 + 0.45 * i / max(1, rs.n - 1), 0.12,
                                   base + 360.0 * ((i * 0.381966) % 1.0)) for i in range(rs.n)]).reshape(rs.n, 3)
    col = (rs.weights[..., None] * cols[:, None, None, :]).sum(0)
    a = float(ctx.p("alpha"))
    g = F.luminance(fr.img)[..., None]
    dim = (g * 0.6 + 0.2) if fr.dark else (g * 0.5 + 0.45)
    out = dim * (1 - a) + col * a + (g - 0.5) * 0.25 * a
    out = R.boundary_lines(rs, float(ctx.p("edge")) * fr.s, fr.pal.canvas if fr.dark else fr.pal.structure * 0.4, out, 0.9)
    return R.finish(out)
