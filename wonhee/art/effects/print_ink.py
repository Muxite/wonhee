"""print-* effects: ink and print as a material system (see wonhee/art/printkit.py)."""
from __future__ import annotations

import math

import numpy as np

from .. import features as F
from .. import printkit as K
from ..registry import effect


@effect("print-screenprint", family="print", tags=["tactile", "editorial", "archival", "source", "screenprint", "posterise", "light", "dark"],
        params=K.params(inkn=(2, "number of inks 1-4"), sep=("tone", "tone | levels | hue | cmy"),
                        screen=("none", "none (flat shapes) | dot | line | fm"), pitch=(15.0, "screen pitch, master px"), angle=(45.0, "screen angle for the first ink"),
                        detail=(0.5, "0..1 how many small shapes survive (soft threshold width)")))
def screenprint(ctx):
    """Two-colour screenprint: the picture becomes 1-4 flat ink shapes by tone, printed on paper
    with visible fibres, ink starved in the valleys, slight misregistration and worn edges.
    ``clean=true`` removes the paper/grain/wear (same colours and shapes). Big shapes stay intact
    at 288x192; works on photos, plots and generated shapes. Failure: text screenshots go blobby."""
    n = int(np.clip(ctx.p("inkn"), 1, 4))
    look = K.resolve_look(ctx, n)
    img = K.source_rgb(ctx)
    rng = ctx.rng
    sep = ctx.p("sep")
    if sep in ("hue", "cmy"):
        covs = K.sep_hue(img, max(n, 2), rng) if sep == "hue" else K.sep_cmy(img, n)
        if look["dark"]:
            covs = [c * F.smoothstep(0.1, 0.5, F.luminance(img)) for c in covs]
    else:
        Lt = K.tone_of(img, float(ctx.p("local")), look["dark"])
        w = 0.015 + 0.05 * (1 - float(ctx.p("detail")))
        covs = K.sep_levels(Lt, n, w) if sep == "levels" else K.sep_tone(Lt, n, soft_w=w)
    kind = ctx.p("screen")
    scr = [dict(kind=kind, pitch=float(ctx.p("pitch")), angle=float(ctx.p("angle")) + 30 * k) for k in range(len(covs))]
    return K.press(covs, look["inks"], ctx.size, rng, paper_rgb=look["paper"], screens=scr, opacity=look["opacity"], mode=look["mode"], **K.surface(ctx))


@effect("print-riso-dots", family="print", tags=["tactile", "playful", "editorial", "source", "risograph", "halftone", "overprint", "light", "dark"],
        params=K.params(inkn=(3, "inks 2-4"), sep=("hue", "hue (colour regions + key) | cmy (channels, false colour) | tone"),
                        pitch=(15.0, "dot pitch, master px (floor 4.2 px output)"), angle=(15.0, "base screen angle; inks step by 30 deg"),
                        gain=(1.0, "dot gain (>1 fatter dots)"), kind=("dot", "dot | line | fm | sq"), accent_ink=(True, "first ink takes the palette accent hue (hue families)")))
def riso_dots(ctx):
    """Risograph with real screens: each ink gets its own dot screen (frequency and angle),
    registration drift, grain, overprinted by multiply, so overlaps mix new colours. Separation by
    hue regions or channels (false colour when the inks are arbitrary). Dark variant prints light
    inks on dark stock. Best on photos with colour (ducks, sunset, portraits); poor on flat UI."""
    n = int(np.clip(ctx.p("inkn"), 2, 4))
    look = K.resolve_look(ctx, n, accent_last=bool(ctx.p("accent_ink")))
    if not ctx.p("inks") and not look["dark"]:
        h = K.pal.rgb_to_oklch(look["palette"].structure)[2]
        ah = K.pal.rgb_to_oklch(look["palette"].accent)[2]
        if abs((ah - h + 180) % 360 - 180) < 45:          # mono palette: open up a second hue family
            ah = h + 105
        look["inks"] = [K.pal.oklch_to_rgb(0.64, 0.21, ah), K.pal.oklch_to_rgb(0.48, 0.18, h), K.pal.oklch_to_rgb(0.90, 0.15, ah + 85), K.pal.oklch_to_rgb(0.28, 0.05, h)][:n]
    img = K.source_rgb(ctx)
    rng = ctx.rng
    sep = ctx.p("sep")
    if sep == "tone":
        covs = K.sep_tone(K.tone_of(img, float(ctx.p("local")), look["dark"]), n)
    elif sep == "cmy":
        covs = K.sep_cmy(img, min(n, 3), key=n > 3)
    else:
        covs = K.sep_hue(img, n, rng)
    if look["dark"]:
        covs = [c * F.smoothstep(0.05, 0.5, F.luminance(img)) for c in covs]
    base = float(ctx.p("angle"))
    scr = [dict(kind=ctx.p("kind"), pitch=float(ctx.p("pitch")) * (1 + 0.06 * k), angle=base + 30 * k, gain=float(ctx.p("gain"))) for k in range(len(covs))]
    return K.press(covs, look["inks"], ctx.size, rng, paper_rgb=look["paper"], screens=scr, opacity=look["opacity"], mode=look["mode"], **K.surface(ctx))


@effect("print-newsprint", family="print", tags=["archival", "editorial", "tactile", "source", "halftone", "mono", "light", "dark"],
        params=K.params(pitch=(22.0, "dot pitch, master px"), angle=(45.0, "screen angle"), kind=("dot", "dot | line | sq"),
                        tint=(0.35, "0..1 second flat tint ink behind the screen (0 = single ink)"), gain=(1.1, "dot gain")))
def newsprint(ctx):
    """Single-ink halftone on newsprint: one coarse dot (or line) screen carries the photo, an optional
    flat pale tint block sits behind, paper has fibre and mottle. The most readable small-size print
    look because the dots are big. Clean variant is a crisp magazine screen."""
    n = 2 if float(ctx.p("tint")) > 0 else 1
    look = K.resolve_look(ctx, 2, light_ls=(0.6, 0.3))
    img = K.source_rgb(ctx)
    rng = ctx.rng
    Lt = K.tone_of(img, float(ctx.p("local")) + 0.2, look["dark"])
    ink = look["inks"][-1] if not ctx.p("inks") else look["inks"][0]
    if not ctx.p("inks"):
        h = K.pal.rgb_to_oklch(look["palette"].structure)[2]
        ink = K.pal.oklch_to_rgb(0.26 if not look["dark"] else 0.82, 0.07, h)
    covs = [1 - Lt]
    inks = [ink]
    scr = [dict(kind=ctx.p("kind"), pitch=float(ctx.p("pitch")), angle=float(ctx.p("angle")), gain=float(ctx.p("gain")))]
    if n == 2:
        tint = look["inks"][0] if not ctx.p("inks") else look["inks"][1]
        covs = [F.smoothstep(0.2, 0.7, F.blur(1 - Lt, 5)) * float(ctx.p("tint")) * 2.4] + covs
        inks = [tint] + inks
        scr = [dict(kind="none")] + scr
    return K.press(covs, inks, ctx.size, rng, paper_rgb=look["paper"], screens=scr, opacity=look["opacity"], mode=look["mode"], **K.surface(ctx))


def _shape_masks(size, rng, n_inks: int, cols: int, rows: int, form: str, data=None):
    """Large overlapping shapes on a jittered grid, one mask per ink. All forms bleed off the frame."""
    W, H = size
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    masks = [np.zeros((H, W), np.float32) for _ in range(n_inks)]
    cw, ch = W / cols, H / rows
    vals = None
    if data is not None:
        vals = np.asarray(data, np.float32).ravel()
        vals = (vals - vals.min()) / (np.ptp(vals) + 1e-9)
    forms = ["circle", "ring", "half", "bar", "quarter", "diamond"] if form == "mix" else [form]
    for r in range(-1, rows + 1):
        for c in range(-1, cols + 1):
            if rng.random() < 0.18:
                continue
            k = int(rng.integers(0, n_inks))
            cx = (c + 0.5 + rng.uniform(-0.28, 0.28)) * cw
            cy = (r + 0.5 + rng.uniform(-0.28, 0.28)) * ch
            v = vals[(r % rows) * cols + (c % cols) if len(vals) >= rows * cols else int(rng.integers(0, len(vals)))] if vals is not None else rng.uniform(0.35, 1.0)
            rad = (0.42 + 0.5 * v) * min(cw, ch) * 1.15
            f = forms[int(rng.integers(0, len(forms)))]
            d = np.hypot(xx - cx, yy - cy)
            if f == "circle":
                m = d < rad
            elif f == "ring":
                m = (d < rad) & (d > rad * 0.55)
            elif f == "half":
                m = (d < rad) & ((yy < cy) if rng.random() < 0.5 else (xx < cx))
            elif f == "bar":
                m = (np.abs(xx - cx) < rad * 1.3) & (np.abs(yy - cy) < rad * 0.32)
            elif f == "quarter":
                m = (d < rad * 1.2) & (xx > cx) & (yy > cy)
            else:
                m = (np.abs(xx - cx) + np.abs(yy - cy)) < rad * 1.1
            masks[k] = np.maximum(masks[k], m.astype(np.float32))
    # anti-alias
    return [F.blur(m, 0.7) for m in masks]


@effect("print-overprint-shapes", family="print", tags=["playful", "tactile", "structured", "editorial", "generated", "overprint", "light", "dark"],
        params=K.params(inkn=(3, "inks 2-4"), cols=(4, "shape grid columns"), form=("mix", "mix | circle | ring | half | bar | quarter | diamond"),
                        screen=("none", "none | dot | line: texture inside the shapes"), pitch=(14.0, "screen pitch (master px)"),
                        tint=(0.0, "0..1 gradient tint across the shapes (needs screen dot/line)")))
def overprint_shapes(ctx):
    """Generated geometry, printed: large circles, rings, bars and quarter discs spread edge to
    edge, one ink each, overprinted (multiply on paper, opaque on dark stock) so every overlap
    is a third colour. No central motif: shapes bleed off all edges. ``ctx.data`` (any array) sets
    shape sizes. With ``screen=dot`` the shapes carry a halftone gradient. Strong at 288x192."""
    n = int(np.clip(ctx.p("inkn"), 2, 4))
    look = K.resolve_look(ctx, n, accent_last=True)
    if not ctx.p("inks"):
        h = K.pal.rgb_to_oklch(look["palette"].structure)[2]
        ah = K.pal.rgb_to_oklch(look["palette"].accent)[2]
        if look["dark"]:
            look["inks"] = [K.pal.oklch_to_rgb(l, 0.15, hh) for l, hh in zip((0.74, 0.55, 0.82, 0.4), (ah, h, h + 160, h))][:n]
        else:
            look["inks"] = [K.pal.oklch_to_rgb(l, c, hh) for l, c, hh in zip((0.70, 0.46, 0.86, 0.30), (0.18, 0.19, 0.12, 0.05), (ah, h, h + 160, h))][:n]
    W, H = ctx.size
    rng = ctx.rng
    cols = int(ctx.p("cols"))
    rows = max(2, int(round(cols * H / W)))
    masks = _shape_masks(ctx.size, rng, n, cols, rows, ctx.p("form"), ctx.data if ctx.source is None and ctx.data is not None else None)
    kind = ctx.p("screen")
    covs = []
    yy = np.linspace(0, 1, H)[:, None] * np.ones((1, W), np.float32)
    xx = np.linspace(0, 1, W)[None, :] * np.ones((H, 1), np.float32)
    for k, m in enumerate(masks):
        if kind != "none":
            g = 0.55 + 0.45 * np.sin((xx * math.cos(k) + yy * math.sin(k)) * math.pi * 2 * (0.6 + 0.3 * k))
            tint = float(ctx.p("tint"))
            covs.append(m * (1 - tint + tint * g))
        else:
            covs.append(m)
    scr = [dict(kind=kind, pitch=float(ctx.p("pitch")), angle=15 + 30 * k) if kind != "none" else dict(kind="none") for k in range(n)]
    out = K.press(covs, look["inks"], ctx.size, rng, paper_rgb=look["paper"], screens=scr, opacity=look["opacity"], mode=look["mode"], **K.surface(ctx))
    return out
