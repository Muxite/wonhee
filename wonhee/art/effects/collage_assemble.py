"""Collage and image assembly (family `collage`): photos, cut-out masks, generated surfaces from other
library effects and flat colour pieces, assembled with overlaps, scale, alignment and colour
unification. Two surface modes: `paper` (torn edges, fibre, soft shadows) and `clean` (precise digital:
hard edges, crisp offset shadows, aligned grids).

Control groups: SOURCE (crop, zoom, hue_from_source), STRUCTURE (pieces, by, smooth, shuffle, rot, overlap),
COLOUR (unify, colour, chroma), SURFACE (mode, torn, paper, shadow, rim, gutter, grain).
Small-card rule: <= 7 pieces, big pieces, gutters >= 0.012 of the height.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from .. import collagekit as C
from .. import features as F
from .. import palette as pal
from .. import regionkit as R
from ..registry import effect

SURF = {
    "mode": ("paper", "SURFACE paper (torn edges, fibre, soft shadows) | clean (precise digital, crisp offset shadows)"),
    "torn": (0.7, "SURFACE torn-edge roughness (paper mode)"),
    "paper": (1.0, "SURFACE paper fibre strength (paper mode)"),
    "shadow": (0.5, "SURFACE shadow strength"),
    "rim": (1.5, "SURFACE px of light rim along piece edges"),
    "grain": (0.2, "SURFACE grain"),
}
COL = {
    "unify": (0.45, "COLOUR 0..1 how hard pieces are pulled into hue families (0 = natural, 1 = fully re-hued)"),
    "colour": ("families", "COLOUR families | palette | source: hue arrangement across pieces"),
    "chroma": (0.9, "COLOUR chroma multiplier"),
}


def _hues(fr, n, mode):
    h0 = pal.rgb_to_oklch(fr.pal.structure)[2]
    ha = pal.rgb_to_oklch(fr.pal.accent)[2]
    if mode == "source":
        return [None] * n
    if mode == "palette":
        return [h0 if i % 3 else ha for i in range(n)]
    seq = [0, 175, 35, -40, 140, 70, 215, -75]
    return [(h0 + seq[i % len(seq)]) % 360 for i in range(n)]


def _paper_base(fr, seed, paper, mode):
    p = fr.pal
    base = np.ones((fr.H, fr.W, 3), np.float32) * (p.canvas * (1.15 if fr.dark else 0.99))
    if mode == "paper" and paper > 0:
        base = base * (1 + R.paper_grain(base.shape, seed, fr.s, paper)[..., None] * 2.5)
    return base


@effect("collage-paper-cutouts", family="collage",
        tags=["source", "collage", "cutout", "paper", "mood:playful", "mood:editorial", "mood:tactile", "mask", "dark-light"],
        params=R.merge(R.SOURCE_PARAMS, {
            "regions": (7, "STRUCTURE number of pieces (5-8 reads best small)"), "by": ("colour", "STRUCTURE how pieces are found: colour | hue | luma | detail | superpixel"),
            "smooth": (0.85, "STRUCTURE piece outline simplification"), "merge": (0.0, "STRUCTURE merge similar neighbours (0 = keep all pieces)"),
            "shuffle": (0.28, "STRUCTURE 0..1 how far piece content is taken from elsewhere in the photo (0 = each piece shows its own part)"),
            "rot": (7.0, "STRUCTURE max rotation of piece content, degrees"), "zoomvar": (1.5, "STRUCTURE max extra magnification of piece content"),
            "overlap": (6.0, "STRUCTURE px each piece grows over its neighbours"),
            "calm": (0.3, "SURFACE edge-aware smoothing of the content (flatter, more 'printed paper')")}, COL, SURF))
def collage_paper_cutouts(ctx):
    """The photo recut into 5-8 shapes (regions of the picture, or your hand mask / polygons in ctx.data) and
    reassembled as overlapping pieces: each piece shows content moved, rotated and enlarged from other
    parts of the same photo, pulled toward its own hue family, with a soft shadow and a light rim, on a
    paper ground. mode=clean swaps torn edges and fibre for crisp edges and hard offset shadows (a digital
    cut-and-paste). Best on photos with texture everywhere (ducks, water, fabric, city); shuffle=0 gives a
    cut-paper rendition of the picture itself. Failure: faces get sliced (give a mask, or shuffle=0).
    Layer count = `regions`; at 288x192 keep it <= 7."""
    fr = R.frame_source(ctx)
    s = fr.s
    img = R.edge_smooth(fr.img, float(ctx.p("calm")), s) if float(ctx.p("calm")) > 0 else fr.img
    rs = R.regions_from_ctx(ctx, fr, n=int(ctx.p("regions")), by=str(ctx.p("by")), smooth=float(ctx.p("smooth")), min_area=0.015, snap=0.3,
                            merge=float(ctx.p("merge")))
    rng = np.random.default_rng(ctx.seed)
    mode = str(ctx.p("mode"))
    paper = mode == "paper"
    out = _paper_base(fr, ctx.seed, float(ctx.p("paper")), mode)
    hs = _hues(fr, rs.n, str(ctx.p("colour")))
    order = np.argsort(-rs.area())          # big pieces underneath
    W, H = fr.W, fr.H
    pad = int((float(ctx.p("overlap")) + 40) * s) + 8
    fibre = R.paper_grain((H, W), ctx.seed + 5, s, float(ctx.p("paper")) * 0.7) if (paper and float(ctx.p("paper")) > 0) else None
    for rank, j in enumerate(order):
        a_full = rs.labels == int(j)
        if a_full.sum() < 20:
            continue
        ys, xs = np.nonzero(a_full)
        by0, by1 = max(0, ys.min() - pad), min(H, ys.max() + 1 + pad)
        bx0, bx1 = max(0, xs.min() - pad), min(W, xs.max() + 1 + pad)
        sl = (slice(by0, by1), slice(bx0, bx1))
        a0 = rs.mask(int(j), 0.0)[sl]
        a = C.grow(a0, float(ctx.p("overlap")) * s)
        a = C.torn_alpha(a, float(ctx.p("torn")), s, ctx.seed + 31 * int(j)) if paper else ndi.gaussian_filter(a, 0.7 * s)
        cx, cy = float(xs.mean()), float(ys.mean())
        sh = float(ctx.p("shuffle"))
        ang = rng.uniform(-1, 1) * float(ctx.p("rot")) * (0.3 + sh)
        zm = 1.0 + rng.uniform(0, 1) * (float(ctx.p("zoomvar")) - 1.0)
        dx, dy = rng.uniform(-1, 1, 2) * sh * 0.5 * np.array([W, H])
        M = cv2.getRotationMatrix2D((cx, cy), ang, zm)
        M[:, 2] += (dx - bx0, dy - by0)
        content = cv2.warpAffine(img, M, (bx1 - bx0, by1 - by0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        content = C.unify(content, hs[int(j)], float(ctx.p("unify")), fr.dark, s, float(ctx.p("chroma")))
        # lift each piece a little in lightness by rank so overlaps separate
        content = np.clip(content + (0.035 if fr.dark else -0.02) * ((rank % 3) - 1), 0, 1)
        if fibre is not None:
            content = content * (1 + np.roll(fibre, 97 * int(j), axis=1)[sl][..., None] * 2.0)
        o = out[sl]
        if float(ctx.p("shadow")) > 0:
            if paper:
                shd = R.drop_shadow(a, (3.5 * s, 5.5 * s), 6 * s)
                o = o * (1 - float(ctx.p("shadow")) * 0.55 * shd[..., None])
            else:
                shd = R.drop_shadow(a, (7 * s, 7 * s), 0.8)
                o = o * (1 - float(ctx.p("shadow")) * 0.45 * shd[..., None])
        o = C.over(o, content, a)
        rl = C.rim_light(a, float(ctx.p("rim")) * s)
        if rl.any():
            hi = np.clip(content + (0.12 if fr.dark else 0.05), 0, 1) if paper else np.ones_like(content) * np.asarray(fr.pal.canvas, np.float32)
            o = C.over(o, hi, rl * (0.7 if paper else 0.9))
        out[sl] = o
    return R.finish(out, ctx.seed, float(ctx.p("grain")), s)


# ---------------------------------------------------------------- digital panels

def _bsp(W, H, n, rng, gutter):
    rects = [(gutter, gutter, W - gutter, H - gutter)]
    ratios = (0.382, 0.5, 0.618)
    while len(rects) < n:
        i = int(np.argmax([(r[2] - r[0]) * (r[3] - r[1]) for r in rects]))
        x0, y0, x1, y1 = rects.pop(i)
        r = ratios[int(rng.integers(3))]
        if (x1 - x0) >= (y1 - y0) * 1.1:
            m = x0 + (x1 - x0) * r
            rects += [(x0, y0, m - gutter / 2, y1), (m + gutter / 2, y0, x1, y1)]
        else:
            m = y0 + (y1 - y0) * r
            rects += [(x0, y0, x1, m - gutter / 2), (x0, m + gutter / 2, x1, y1)]
    return [tuple(int(round(v)) for v in r) for r in rects]


@effect("collage-digital-panels", family="collage",
        tags=["source", "collage", "grid", "panels", "surfaces", "mood:technical", "mood:editorial", "mood:cool", "dark-light"],
        params=R.merge(R.SOURCE_PARAMS, {
            "pieces": (6, "STRUCTURE number of panels (5-7 reads best small)"),
            "fills": ("crop,physics-interference,crop,dots-hex-field,flat,crop", "STRUCTURE per-panel content, cycled: crop (a detailed crop of the photo) | flat | any library effect name (a generated surface)"),
            "gutter": (0.018, "STRUCTURE gap between panels, fraction of the height"),
            "overlap": (0.25, "STRUCTURE 0..1 chance that a panel is enlarged to overlap neighbours (with a crisp offset shadow)"),
            "crop_frac": (0.38, "STRUCTURE share of the photo a crop panel shows (smaller = closer in)"),
            "hairline": (1.5, "SURFACE px accent hairline around panels (0 = none)"), "surface_params": ("{}", "STRUCTURE JSON {effect: params} for generated panels")}, COL,
            {"unify": (0.5, "COLOUR pull of crops toward panel hue families"), "mode": ("clean", "SURFACE clean | paper"),
             "shadow": (0.5, "SURFACE offset shadow strength under overlapping panels"), "grain": (0.15, "SURFACE grain")}))
def collage_digital_panels(ctx):
    """A precise modular layout (golden-ratio splits with even gutters) filled with detailed crops of the photo,
    surfaces generated by other library effects (physics-interference, dots-hex-field, ...) rendered in
    per-panel hue-shifted palettes, and flat colour panels; some panels overlap their neighbours with a crisp
    offset shadow. Colour unification via a shared lightness structure and hue families; hairlines align
    everything. The generated surfaces are not AI images: they are the library's own deterministic art.
    Best for hero / blog covers that mix a photo with project visuals. Layer count = `pieces`; keep <= 7 at
    288x192. Failure: a fill effect that needs `ctx.data` (graph-*) renders flat canvas."""
    import json
    fr = R.frame_source(ctx)
    s = fr.s
    W, H = fr.W, fr.H
    rng = np.random.default_rng(ctx.seed)
    g = max(2, int(float(ctx.p("gutter")) * H))
    rects = _bsp(W, H, int(ctx.p("pieces")), rng, g)
    rects.sort(key=lambda r: (r[1] // max(1, H // 3), r[0]))
    fills = [x.strip() for x in str(ctx.p("fills")).split(",") if x.strip()]
    over = json.loads(str(ctx.p("surface_params")) or "{}")
    p = fr.pal
    out = np.ones((H, W, 3), np.float32) * (p.canvas * (1.0 if fr.dark else 0.98))
    hs = _hues(fr, len(rects), str(ctx.p("colour")))
    img = fr.img
    crops = F.detail_crops(img, max(len(rects), 3), aspect=W / H, frac=float(ctx.p("crop_frac")) + 0.2, seed=ctx.seed)
    ci = 0
    pol = p.polarity
    clean = str(ctx.p("mode")) == "clean"
    enlarge = [bool(rng.random() < float(ctx.p("overlap"))) for _ in rects]
    for k, (x0, y0, x1, y1) in enumerate(rects):
        if enlarge[k]:
            d = int(g * 2.2 + 10 * s)
            x0, y0, x1, y1 = max(0, x0 - d), max(0, y0 - d), min(W, x1 + d), min(H, y1 + d)
        pw, ph = x1 - x0, y1 - y0
        if pw < 8 or ph < 8:
            continue
        kind = fills[k % len(fills)]
        hue = hs[k] if hs[k] is not None else pal.rgb_to_oklch(p.structure)[2]
        if kind == "crop":
            bx0, by0, bx1, by1 = crops[ci % len(crops)]
            ci += 1
            cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
            ch = (by1 - by0)
            cw = int(ch * pw / ph)
            if cw > img.shape[1]:
                cw = img.shape[1]; ch = int(cw * ph / pw)
            xa = int(np.clip(cx - cw / 2, 0, img.shape[1] - cw)); ya = int(np.clip(cy - ch / 2, 0, img.shape[0] - ch))
            tile = cv2.resize(img[ya:ya + ch, xa:xa + cw], (pw, ph), interpolation=cv2.INTER_AREA)
            tile = C.unify(tile, hs[k], float(ctx.p("unify")), fr.dark, s, float(ctx.p("chroma")))
        elif kind == "flat":
            # calm panel: a soft two-stop gradient in the panel hue with fine grain (never a dead flat block)
            yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
            t = np.clip((xx / pw * 0.7 + yy / ph * 0.5) / 1.2, 0, 1)
            L0, L1 = (0.30, 0.46) if fr.dark else (0.80, 0.92)
            tile = F.oklch_field(L0 + (L1 - L0) * t, 0.10, hue + 25 * t)
            tile = tile * (1 + R.paper_grain(tile.shape, ctx.seed + k, s, 1.0)[..., None] * 1.5)
        else:
            pk = pal.make(hue, polarity=pol, harmony="analogous", three=p.structure2 is not None)
            tile = C.surface(kind, (pw, ph), ctx.seed + 17 * k, pk, over.get(kind))
        a = np.zeros((H, W), np.float32)
        a[y0:y1, x0:x1] = 1.0
        if enlarge[k] and float(ctx.p("shadow")) > 0:
            shd = R.drop_shadow(a, (8 * s, 8 * s), 0.8 if clean else 6 * s)
            out = out * (1 - float(ctx.p("shadow")) * 0.5 * shd[..., None])
        out[y0:y1, x0:x1] = tile
        hl = float(ctx.p("hairline")) * s
        if hl > 0:
            col = p.accent * (0.75 if fr.dark else 0.9)
            t = max(1, int(round(hl)))
            for (a0, a1, b0, b1) in ((y0, y0 + t, x0, x1), (y1 - t, y1, x0, x1), (y0, y1, x0, x0 + t), (y0, y1, x1 - t, x1)):
                out[a0:a1, b0:b1] = out[a0:a1, b0:b1] * 0.45 + col * 0.55
    return R.finish(out, ctx.seed, float(ctx.p("grain")), s)


# ---------------------------------------------------------------- subject on a generated surface

@effect("collage-subject-sticker", family="collage",
        tags=["source", "collage", "cutout", "subject", "sticker", "surfaces", "mood:playful", "mood:bold", "mask", "dark-light"],
        params=R.merge(R.SOURCE_PARAMS, {
            "surface": ("physics-caustics", "STRUCTURE library effect used as the generated ground (any effect name; 'source' = a blurred version of the photo)"),
            "surface_params": ("{}", "STRUCTURE JSON params for the surface effect"),
            "regions": (5, "STRUCTURE regions searched for the subject when no mask is given"),
            "subject": (0, "STRUCTURE index of the nth most salient region to use (0 = best); ctx.data mask (bright = subject) overrides"),
            "scale": (0.78, "STRUCTURE subject height as a fraction of the frame height"), "place": ("auto", "STRUCTURE left | right | auto (rule of thirds)"),
            "outline": (9.0, "SURFACE px of sticker outline"), "echo": (14.0, "SURFACE px offset of the accent silhouette behind (0 = none)"),
            "smooth": (0.7, "STRUCTURE subject outline simplification"), "surface_hue": (0.0, "COLOUR hue offset of the surface palette vs the palette (deg; 0 = same family, 180 = complement)"),
            "unify": (0.25, "COLOUR pull of the subject toward the palette family"), "shadow": (0.55, "SURFACE soft shadow"), "grain": (0.2, "SURFACE grain"),
            "chroma": (0.9, "COLOUR chroma multiplier")}))
def collage_subject_sticker(ctx):
    """A cut-out subject (the most salient region, or a hand mask / polygon in ctx.data) lifted off the photo
    as a thick-outlined sticker with an offset accent echo and soft shadow, placed on the rule of thirds over
    a generated ground from another library effect (a physics field, dots, anything) in a related or
    complementary hue. Reads as a clear silhouette at 288x192: two layers plus ground. Best for photos with
    one clear subject (a person, an object); the auto subject is a guess, so for people pass a mask
    (grey-level image, polygon list) for a clean cut. Failure: busy photos give an arbitrary blob."""
    fr = R.frame_source(ctx)
    s = fr.s
    W, H = fr.W, fr.H
    p = fr.pal
    sal = F.blur(F.saliency(fr.img), 6 * s)
    data = ctx.data
    rs = R.regions_from_ctx(ctx, fr, n=int(ctx.p("regions")), by="colour", smooth=float(ctx.p("smooth")), min_area=0.015, snap=0.7, merge=0.25)
    has_mask = data is not None
    if has_mask:
        i = rs.n - 1          # hand masks: the brightest region is the subject
        m0 = rs.mask(i, 0.0)
    else:
        sc = []
        for i in range(rs.n):
            m = rs.labels == i
            border = np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]]).mean()
            area = m.mean()
            sc.append(float(sal[m].mean() * (1 - 0.9 * border) * (1 if 0.04 < area < 0.6 else 0.3)))
        order = np.argsort(sc)[::-1]
        i = int(order[min(int(ctx.p("subject")), rs.n - 1)])
        m0 = rs.mask(i, 0.0)
        # keep the largest connected piece
        cc, k = ndi.label(m0 > 0.5)
        if k > 1:
            big = np.argmax(np.bincount(cc.ravel())[1:]) + 1
            m0 = (cc == big).astype(np.float32)
    ys, xs = np.nonzero(m0 > 0.5)
    if len(xs) < 50:
        m0 = np.ones((H, W), np.float32) * 0.0
        m0[H // 4:3 * H // 4, W // 4:3 * W // 4] = 1
        ys, xs = np.nonzero(m0 > 0.5)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    subj = fr.img[y0:y1, x0:x1]
    am = ndi.gaussian_filter(m0, 1.0 * s)[y0:y1, x0:x1]
    sc_ = float(ctx.p("scale")) * H / (y1 - y0)
    sc_ = min(sc_, 0.6 * W / (x1 - x0), 2.2)
    nw, nh = max(4, int((x1 - x0) * sc_)), max(4, int((y1 - y0) * sc_))
    subj = cv2.resize(subj, (nw, nh), interpolation=cv2.INTER_CUBIC if sc_ > 1 else cv2.INTER_AREA)
    am = cv2.resize(am.astype(np.float32), (nw, nh), interpolation=cv2.INTER_LINEAR)
    place = str(ctx.p("place"))
    if place == "auto":
        place = "left" if (xs.mean() / W) < 0.5 else "right"
    cx = W * (0.35 if place == "left" else 0.65)
    cy = H * 0.52
    ox, oy = int(cx - nw / 2), int(cy - nh / 2)
    # canvas-size layers
    A = np.zeros((H, W), np.float32)
    S = np.zeros((H, W, 3), np.float32)
    sx0, sy0 = max(0, ox), max(0, oy)
    sx1, sy1 = min(W, ox + nw), min(H, oy + nh)
    A[sy0:sy1, sx0:sx1] = am[sy0 - oy:sy1 - oy, sx0 - ox:sx1 - ox]
    S[sy0:sy1, sx0:sx1] = subj[sy0 - oy:sy1 - oy, sx0 - ox:sx1 - ox]
    # ground
    sp = str(ctx.p("surface"))
    h0 = pal.rgb_to_oklch(p.structure)[2]
    gp = pal.make((h0 + float(ctx.p("surface_hue"))) % 360, polarity=p.polarity, harmony="analogous", three=p.structure2 is not None)
    if sp == "source":
        ground = F.blur(fr.img, 18 * s)
    else:
        import json
        ground = C.surface(sp, (W, H), ctx.seed, gp, json.loads(str(ctx.p("surface_params")) or "{}"))
    out = ground.copy()
    hue = h0
    S = C.unify(S, hue, float(ctx.p("unify")), fr.dark, s, float(ctx.p("chroma")))
    ow = float(ctx.p("outline")) * s
    ring = C.grow((A > 0.5).astype(np.float32), ow)
    ring = ndi.gaussian_filter(ring, 0.8 * s)
    echo = float(ctx.p("echo")) * s
    if echo > 0:
        M = np.float32([[1, 0, echo], [0, 1, echo * 0.6]])
        er = cv2.warpAffine(ring, M, (W, H))
        acc = p.accent * (0.95 if fr.dark else 0.9)
        shd = R.drop_shadow(er, (3 * s, 5 * s), 7 * s)
        out = out * (1 - float(ctx.p("shadow")) * 0.5 * shd[..., None])
        out = C.over(out, np.ones_like(out) * acc, er)
    shd = R.drop_shadow(ring, (4 * s, 7 * s), 9 * s)
    out = out * (1 - float(ctx.p("shadow")) * 0.55 * shd[..., None])
    paper = np.asarray(p.canvas * (1.0 if fr.dark else 0.99) + (0.62 if fr.dark else 0.0), np.float32).clip(0, 0.97)
    out = C.over(out, np.ones_like(out) * paper, ring)
    out = C.over(out, S, A)
    return R.finish(out, ctx.seed, float(ctx.p("grain")), s)
