"""dither-* effects: quantisation, dither and pixel systems (wonhee/art/ditherkit.py)."""
from __future__ import annotations

import math

import numpy as np

from .. import dotkit as dk
from .. import features as F
from .. import ditherkit as D
from .. import printkit as K
from ..registry import effect

PAL = {
    "colors": ("palette:4", "palette (library palette ramp, 'palette:N') | auto:N (k-means from the image) | preset (gameboy, cga, mac, sepia, riso-cobalt, riso-pink, pico, ocean) | '#hex,#hex,...' | ink names"),
    "hues": (1, "1 = one hue family; 2 = highlights drift to the palette accent hue (library ramps only)"),
    "color_from_source": (-1.0, "0..1 how much source chroma steers the palette choice; -1 = auto (0 for ramps, 1 for auto/presets)"),
    "variant": ("auto", "auto | light | dark: palette polarity for library ramps"),
}
TONE = {
    "local": (0.35, "local tone equalisation of the source"), "contrast": (1.0, "contrast about mid grey (1 = stretched)"),
    "gamma": (1.0, "tone gamma"), "bright": (0.0, "brightness offset"), "crop": ("feature", "feature | center"), "zoom": (0.0, "zoom into the densest region"),
}


def _pal_ctx(ctx):
    var = str(ctx.p("variant", "auto"))
    if var in ("light", "dark") and var != ctx.palette.polarity:
        h = K.pal.rgb_to_oklch(ctx.palette.structure)[2]
        ctx.palette = K.pal.make(h, polarity=var, harmony="analogous")
    return ctx


def _tone_lab(ctx, img, pal_rgb, gw, gh):
    g = D.to_grid(img, gw, gh)
    lab = D.prep_tone(g, local=float(ctx.p("local")), contrast=float(ctx.p("contrast")), gamma=float(ctx.p("gamma")), bright=float(ctx.p("bright")))
    # map the tone range onto the palette's lightness range (nothing left blown or crushed)
    pl = F.to_oklab(pal_rgb[None])[0, :, 0]
    lab[..., 0] = pl.min() + lab[..., 0] * (pl.max() - pl.min())
    cs = float(ctx.p("color_from_source"))
    spec = str(ctx.p("colors")).split(":")[0]
    if cs < 0:
        cs = 0.0 if spec in ("palette", "ramp") else 1.0
    lab[..., 1:] *= cs * 1.0
    if cs == 0.0:
        # ramps live on a hue line: pull a,b to the palette's own chroma at that lightness
        order = np.argsort(F.to_oklab(pal_rgb[None])[0, :, 0])
        pab = F.to_oklab(pal_rgb[None])[0][order]
        lab[..., 1] = np.interp(lab[..., 0], pab[:, 0], pab[:, 1])
        lab[..., 2] = np.interp(lab[..., 0], pab[:, 0], pab[:, 2])
    return lab


@effect("dither-photo", family="dither", tags=["digital", "archival", "playful", "structured", "source", "dither", "pixel", "light", "dark"],
        params={**PAL, **TONE, "algo": ("fs", "none | bayer2/4/8/16 | blue | white | fs | atkinson | sierra | sierra2 | sierra-lite | jjn | stucki | burkes"),
                "cols": (160, "logical pixels across the frame (the pixel scale; same picture at every output size, floor 2.6 px per pixel)"),
                "spread": (1.0, "ordered/noise dither amplitude"), "strength": (1.0, "error diffusion strength (<1 = cleaner)"),
                "gap": (0.0, "0..0.4 gap between logical pixels as a fraction of a pixel (pixel-grid look)")})
def dither_photo(ctx):
    """A photo (or data field, or generated shapes) quantised to ANY palette with a chosen dither:
    ordered Bayer 2-16, blue/white noise or error diffusion (Floyd-Steinberg, Atkinson, Sierra family, JJN,
    Stucki, Burkes), on a logical pixel grid that is a fraction of the frame width so the picture is the
    same at 288x192 and 1500x1000. Palettes: library ramp (one or two hue families), k-means from the
    image, retro presets, custom hex. Best on high-detail photos at cols 90-200 with 4-6 colours;
    failure: faces at cols < 70 turn to blocks; error diffusion on flat gradients shows worms (use bayer)."""
    ctx = _pal_ctx(ctx)
    W, H = ctx.size
    img = K.source_rgb(ctx)
    pal_rgb = D.resolve_palette(ctx.p("colors"), ctx, img, 4, int(ctx.p("hues")))
    gw, gh, b = D.grid_size(ctx.size, int(ctx.p("cols")))
    lab = _tone_lab(ctx, img, pal_rgb, gw, gh)
    idx = D.quantise(lab, pal_rgb, str(ctx.p("algo")), ctx.rng, float(ctx.p("spread")), True, float(ctx.p("strength")))
    return D.to_frame(idx, pal_rgb, ctx.size, float(ctx.p("gap")), pal_rgb[0])


@effect("dither-blocks", family="dither", tags=["digital", "playful", "structured", "source", "pixel", "mosaic", "light", "dark"],
        params={**PAL, **TONE, "colors": ("palette:5", PAL["colors"][1]), "hues": (2, PAL["hues"][1]), "cols": (42, "big pixel blocks across the frame"), "algo": ("bayer4", "none | bayer2/4/8 | blue | fs ... (dither INSIDE the coarse blocks)"),
                "spread": (0.35, "dither amplitude (small = clean flat blocks with only a hint of pattern)"), "gap": (0.06, "gap between blocks as a fraction of a block (0 = none)"),
                "snap": (0.0, "0..1 snap block colours to the palette extremes (more contrast)")})
def dither_blocks(ctx):
    """Large pixel blocks: the picture reduced to a coarse grid of flat palette colours, with a touch of
    ordered dither between neighbours and thin gaps between blocks, like a LED wall or tile mosaic. The
    biggest-feature version of dithering: reads at any card size, hides noise, keeps colour variety
    (default palette = 5 k-means colours from the image). Best on photos and heatmaps."""
    ctx = _pal_ctx(ctx)
    img = K.source_rgb(ctx)
    pal_rgb = D.resolve_palette(ctx.p("colors"), ctx, img, 5, int(ctx.p("hues")))
    gw, gh, b = D.grid_size(ctx.size, int(ctx.p("cols")), min_block=5.0)
    lab = _tone_lab(ctx, img, pal_rgb, gw, gh)
    if float(ctx.p("snap")) > 0:
        L = lab[..., 0]
        lab[..., 0] = L + (np.clip((L - L.mean()) * 2.2 + L.mean(), 0, 1) - L) * float(ctx.p("snap"))
    idx = D.quantise(lab, pal_rgb, str(ctx.p("algo")), ctx.rng, float(ctx.p("spread")), True, 1.0)
    return D.to_frame(idx, pal_rgb, ctx.size, float(ctx.p("gap")), pal_rgb[0] * 0.6)


def _gradient_field(size, rng):
    """Generated, centre-free source: warped sine bands over a diagonal ramp, three hue families blended."""
    W, H = size
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u, v = xx / W, yy / H
    th = rng.uniform(0, math.pi)
    warp = 0.35 * dk.smooth_noise((H, W), W / 2.2, rng, 3) + 0.15 * dk.smooth_noise((H, W), W / 6, rng, 2)
    t = u * math.cos(th) + v * math.sin(th) * (H / W) * 1.5
    bands = 0.5 + 0.5 * np.sin(2 * math.pi * (rng.uniform(1.4, 2.6) * t + warp * 2.0 + rng.uniform(0, 1)))
    L = np.clip(0.12 + 0.76 * (0.55 * bands + 0.45 * (0.5 * t + warp + 0.25)), 0, 1)
    h0 = rng.uniform(0, 360)
    hue = h0 + 120 * dk.smooth_noise((H, W), W / 1.8, rng, 2)
    return F.oklch_field(L, 0.13, hue)


@effect("dither-gradient", family="dither", tags=["digital", "playful", "structured", "generated", "dither", "pixel", "light", "dark"],
        params={**PAL, "colors": ("palette:5", PAL["colors"][1]), "hues": (2, PAL["hues"][1]), "algo": ("bayer8", "dither algorithm (see dither-photo)"), "cols": (200, "logical pixels across"),
                "spread": (1.0, "ordered dither amplitude"), "gap": (0.0, "pixel gap"), "bands": (0.0, "0..1 extra posterise bands (0 = smooth gradient dithered)")})
def dither_gradient(ctx):
    """Procedural digital art with no source: a warped, multi-band colour field dithered to a limited
    palette, the classic gradient-dither look (Bayer lattice or Floyd-Steinberg worms). Spread over the
    whole frame, no centre. Use a library palette ramp (one or two hue families), a preset, or custom
    hex; swap the algorithm for completely different surfaces. ``ctx.data`` (2-D array) replaces the field."""
    ctx = _pal_ctx(ctx)
    W, H = ctx.size
    rng = ctx.rng
    if ctx.data is not None and np.asarray(ctx.data).ndim == 2:
        img = K.source_rgb(ctx)
    else:
        img = _gradient_field(ctx.size, rng)
    pal_rgb = D.resolve_palette(ctx.p("colors"), ctx, img, 5, int(ctx.p("hues")))
    gw, gh, b = D.grid_size(ctx.size, int(ctx.p("cols")))
    g = D.to_grid(img, gw, gh)
    lab = F.to_oklab(g)
    pl = F.to_oklab(pal_rgb[None])[0]
    order = np.argsort(pl[:, 0])
    pl = pl[order]
    L = F.stretch(lab[..., 0], 1, 99)
    if float(ctx.p("bands")) > 0:
        nb = 4
        L = L * (1 - float(ctx.p("bands"))) + np.floor(L * nb) / nb * float(ctx.p("bands"))
    lab = np.stack([pl[0, 0] + L * (pl[-1, 0] - pl[0, 0]), np.interp(L * (pl[-1, 0] - pl[0, 0]) + pl[0, 0], pl[:, 0], pl[:, 1]),
                    np.interp(L * (pl[-1, 0] - pl[0, 0]) + pl[0, 0], pl[:, 0], pl[:, 2])], -1)
    idx = D.quantise(lab, pal_rgb, str(ctx.p("algo")), rng, float(ctx.p("spread")), True, 1.0)
    return D.to_frame(idx, pal_rgb, ctx.size, float(ctx.p("gap")), pal_rgb[0])


@effect("dither-stipple", family="dither", tags=["archival", "editorial", "tactile", "source", "stipple", "light", "dark"],
        params={**PAL, **TONE, "colors": ("palette:3", PAL["colors"][1]), "dots": (9000, "dot count at 1500x1000 (scaled with area, capped for small frames)"),
                "size": (1.0, "dot size scale"), "tint": (0.0, "0..1 colour the dots from the source (via nearest palette colour); 0 = single ink"),
                "iters": (4, "Lloyd relaxation steps")})
def dither_stipple(ctx):
    """Weighted-Voronoi colour stipple: dots denser (and larger) where the source is dark, relaxed so they
    never clump, on the palette's lightest colour. ``tint`` lets each dot take the nearest palette colour of
    the source. Reads at 288x192 by keeping dots >= 2 px and thinning the count. Differs from dots-stipple by
    accepting any palette, colour per dot and a consistent dot density per frame area."""
    ctx = _pal_ctx(ctx)
    W, H = ctx.size
    rng = ctx.rng
    img = K.source_rgb(ctx)
    pal_rgb = D.resolve_palette(ctx.p("colors"), ctx, img, 3, int(ctx.p("hues")))
    dark = ctx.palette.polarity == "dark"
    L = F.stretch(F.local_equalize(F.luminance(img), 2.5, mix=float(ctx.p("local"))), 2, 98)
    dens = (L if dark else 1 - L)
    n = int(float(ctx.p("dots")) * (W * H) / 1.5e6)
    n = int(np.clip(n, 400, 40000)) if W >= 600 else int(np.clip(n * 3.5, 300, 4200))
    pts = dk.stipple(dens ** 1.3, (W, H), n, rng, iters=int(ctx.p("iters")))
    xi = np.clip(pts.x.astype(int), 0, W - 1)
    yi = np.clip(pts.y.astype(int), 0, H - 1)
    t = dens[yi, xi]
    r = np.maximum(pts.cell * 0.5 * (0.35 + 0.75 * t) * float(ctx.p("size")), 1.1)
    bg, ink = (pal_rgb[0], pal_rgb[-1]) if dark else (pal_rgb[-1], pal_rgb[0])
    out = np.ones((H, W, 3), np.float32) * bg[None, None]
    cols = np.tile(ink, (len(xi), 1))
    if float(ctx.p("tint")) > 0:
        lab = F.to_oklab(img[yi, xi][None])[0]
        near = D.Nearest(F.to_oklab(pal_rgb[None])[0])
        k = near.query(lab)
        cols = cols * (1 - float(ctx.p("tint"))) + pal_rgb[k] * float(ctx.p("tint"))
    dk.draw_dots(out, pts.x, pts.y, r, cols, 1.0)
    return np.clip(out, 0, 1)
