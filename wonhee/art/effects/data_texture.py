"""data-pixel-field, data-stripe-matrix, data-terminal-panels: information textures built from data cells.

Same data contract as data_matrix.py (ctx.data = 2D array / {"matrix": ..., "rows": ..., "cols": ...}; no data =
seeded SYNTHETIC values, flagged by provenance(...)['synthetic']; ``mode=full`` stamps the frame). data-terminal-panels
also accepts a dict of named 1D series."""
from __future__ import annotations

import numpy as np

from .. import dotkit as dkit
from .. import datakit as dk
from ..registry import effect
from .data_matrix import COMMON, _p, numbers_and_legend, resolve, scaled


# ----------------------------------------------------------------------------------------------------------------
def _blocks(Q: np.ndarray, max_b: int = 4) -> np.ndarray:
    """Block size (1, 2, 4) of the largest aligned square of equal quantised value each cell belongs to."""
    R, C = Q.shape
    bs = np.ones((R, C), np.int8)
    b = 2
    while b <= max_b:
        for r in range(0, R - b + 1, b):
            for c in range(0, C - b + 1, b):
                blk = Q[r:r + b, c:c + b]
                if (blk == blk[0, 0]).all() and (bs[r:r + b, c:c + b] == b // 2).all():
                    bs[r:r + b, c:c + b] = b
        b *= 2
    return bs


@effect("data-pixel-field", family="data", tags=["data", "pixels", "texture", "heatmap", "all-over", "analytical", "synthetic-default", "dark-light", "print"],
        params={**{k: v for k, v in COMMON.items() if k not in ("order",)},
                "scale": ("cream-red", COMMON["scale"][1]),
                "cols": (64, "pixel columns across the frame"),
                "levels": (6, "quantisation levels (4..9); fewer = more merged blocks"),
                "merge": (True, "merge equal 2x2 / 4x4 blocks into larger pixels (mixed pixel sizes)"),
                "accent": (0.02, "share of blocks drawn in the accent end of an alternative scale"),
                "accent_scale": ("ink", "scale for accent blocks (its high end)"),
                "streaks": (0.5, "row / column streak strength in synthetic fields (0..1)"),
                "gap": (0.0, "gap px at 1500 wide (0 = solid pixel field)")})
def pixel_field(ctx):
    """Cream/red pixel field: a coarse, quantised value field drawn as hard pixels, equal neighbours merged into
    bigger squares so the pixel size varies across the frame; a few accent blocks in a second hue. Real data is
    upsampled with nearest-neighbour (so a 38x9 matrix becomes big flat blocks). Best with the light palettes
    (red-on-cream / ink-on-paper) and scale cream-red; works dark too."""
    W, H = ctx.size; rng = ctx.rng; s = dk.scale_px(W)
    cols = int(_p(ctx, "cols")); pitch = W / cols
    rows = int(np.ceil(H / pitch)) + 1
    d = ctx.data
    if d is None:
        P = dk.synth_field(rng, rows, cols)
        # streaks: horizontal runs and vertical bands (print-fault / telemetry look)
        st = float(_p(ctx, "streaks"))
        rs = dk.ndimage.gaussian_filter(rng.normal(0, 1, rows), 0.9)[:, None]
        cs = dk.ndimage.gaussian_filter(rng.normal(0, 1, cols), 1.6)[None, :]
        M = P.M + st * (0.16 * rs + 0.12 * cs)
        run = dk.ndimage.gaussian_filter(rng.normal(0, 1, (rows, cols)), (0.4, 4.0))
        M = M + st * 0.22 * run
        P.M = M.astype(np.float32)
    else:
        P = dk.prepare_matrix(ctx, "field")
        M0 = P.M
        ri = (np.arange(rows) * M0.shape[0] / rows).astype(int); ci = (np.arange(cols) * M0.shape[1] / cols).astype(int)
        P.M = M0[ri][:, ci]
    V, used = scaled(ctx, P.M)
    L = int(np.clip(_p(ctx, "levels"), 3, 12))
    Q = np.clip(np.floor(V * L), 0, L - 1).astype(np.int16)
    bs = _blocks(Q) if _p(ctx, "merge") else np.ones_like(Q, np.int8)
    tab = dk.lut(_p(ctx, "scale"), ctx.palette, polarity=ctx.palette.polarity)
    if _p(ctx, "invert"):
        tab = tab[::-1].copy()
    cols_q = tab[(np.arange(L) * (len(tab) - 1) / (L - 1)).astype(int)]
    atab = dk.lut(_p(ctx, "accent_scale"), ctx.palette, polarity=ctx.palette.polarity)
    ys, xs = np.arange(H) + 0.5, np.arange(W) + 0.5
    ri = np.minimum((ys / pitch).astype(int), rows - 1); ci = np.minimum((xs / pitch).astype(int), cols - 1)
    b_img = bs[ri][:, ci].astype(np.int32)
    r0 = (ri[:, None] // b_img) * b_img; c0 = (ci[None, :] // b_img) * b_img
    q_img = Q[r0, c0]
    img = cols_q[q_img].astype(np.float32)
    # accent blocks (decided per block origin, seeded)
    acc = float(_p(ctx, "accent"))
    if acc > 0:
        hit = rng.random((rows, cols)) < acc
        pick = hit[r0, c0] & (bs[r0, c0] >= 1)
        lvl = rng.random((rows, cols))
        a_img = dk.apply_lut(0.55 + 0.45 * lvl[r0, c0], atab)
        img = np.where(pick[..., None], a_img, img)
    gap = float(_p(ctx, "gap")) * s
    if gap > 0:
        by0, by1 = r0 * pitch, (r0 + b_img) * pitch
        bx0, bx1 = c0 * pitch, (c0 + b_img) * pitch
        dist = np.minimum(np.minimum(ys[:, None] - by0, by1 - ys[:, None]), np.minimum(xs[None, :] - bx0, bx1 - xs[None, :]))
        cov = np.clip(dist - gap / 2 + 0.5, 0, 1)[..., None]
        img = img * cov + ctx.palette.canvas * (1 - cov)
    if str(_p(ctx, "surface")) == "paper":
        g = rng.normal(0, 1, (H, W)).astype(np.float32)
        g = dk.cv2.GaussianBlur(g, (0, 0), 0.9)
        img = img * (1 + float(_p(ctx, "surf")) * 0.05 * g)[..., None]
    img = np.clip(img, 0, 1).astype(np.float32)
    if str(_p(ctx, "mode")) == "full":
        ov = dk.Overlay(img)
        pad = int(18 * s + 8); px = max(10, int(12 * s))
        ink = np.array([0.95, 0.94, 0.92]) if ctx.palette.polarity == "dark" else np.array([0.1, 0.1, 0.12])
        bw = int(W * 0.26); x1 = W - pad; y1 = H - pad - px - 8
        ov.rect(x1 - bw - 14, y1 - 14 * s - px - 12, x1 + 14, H - pad + 8, fill=ctx.palette.canvas, a=0.9, radius=int(6 * s + 2))
        ov.colorbar(x1 - bw, y1 - max(8, int(14 * s)), x1, y1, cols_q if False else tab, used[0], used[1], ink, "{:.2f}", px, 3, label="value (synthetic)" if P.synthetic else "value")
        if P.synthetic:
            ov.text((pad, pad), "synthetic data - decorative, not results", px, ink, 0.85)
        img = ov.done()
    return img


# ----------------------------------------------------------------------------------------------------------------
@effect("data-stripe-matrix", family="data", tags=["data", "matrix", "stripes", "heatmap", "benchmark", "analytical", "all-over", "synthetic-default", "dark-light"],
        params={**COMMON, "order": ("mean", COMMON["order"][1]),
                "stripes": (9, "horizontal stripes (matrix columns after transposing)"),
                "segs": (38, "segments per stripe (matrix rows)"),
                "runs": (3, "each segment split into this many strips (repeat runs); 1 = plain"),
                "stripe_var": (0.35, "stripe height variation"),
                "seg_var": (0.5, "segment width variation (ragged stripes)"),
                "stagger": (0.0, "per-stripe horizontal offset, fraction of a segment (0..1)"),
                "sidebar": ("bars", "bars | none: per-stripe mean bars on the right"),
                "gap": (3.0, COMMON["gap"][1])})
def stripe_matrix(ctx):
    """Horizontal stripe matrix: every configuration is a stripe, every test a segment (segment widths and stripe
    heights vary, optionally split into repeat-run strips), coloured by score; stripes ordered by mean; an optional
    sidebar of per-stripe means. Real data: dict {"scores": [test][config][run]} or any 2D [test][config]. The frame
    is filled edge to edge; stripes bleed off the left edge."""
    W, H = ctx.size; rng = ctx.rng; s = dk.scale_px(W)
    n_st = int(_p(ctx, "stripes")); n_sg = int(_p(ctx, "segs")); runs = int(max(1, _p(ctx, "runs")))
    d = ctx.data
    if d is None:
        P = dk.synth_bench(rng, n_sg, n_st)
        # synthetic repeat runs: each segment has `runs` noisy copies, one pass-fraction per run
        S3 = np.clip(P.M[..., None] + rng.normal(0, 0.18, P.M.shape + (runs,)) * (P.M[..., None] > 0.05) * (P.M[..., None] < 0.95), 0, 1)
    else:
        if isinstance(d, dict) and "scores" in d and np.asarray(d["scores"]).ndim == 3:
            S3 = np.asarray(d["scores"], np.float32)
            P = dk.Prep(S3.mean(-1), None, None, False, "scores", "data")
        else:
            P = dk.prepare_matrix(ctx)
            S3 = P.M[..., None]
        runs = S3.shape[2]
    # transpose: stripes = columns of P.M
    order = np.argsort(-np.nanmean(P.M, 0)) if str(_p(ctx, "order")) == "mean" else (dk.cluster_order(P.M, 1)[0] if str(_p(ctx, "order")) in ("cluster", "cols") else np.arange(P.M.shape[1]))
    S3 = S3[:, order, :]
    n_sg, n_st = S3.shape[0], S3.shape[1]
    lo, hi = _p(ctx, "lo"), _p(ctx, "hi")
    V, used = dk.normalize(S3, lo, hi, "minmax" if S3.max() <= 1.0 + 1e-6 and lo is None and hi is None else "robust")
    tab = dk.lut(_p(ctx, "scale"), ctx.palette, polarity=ctx.palette.polarity)
    if _p(ctx, "invert"):
        tab = tab[::-1].copy()
    canvas = ctx.palette.canvas
    img = np.empty((H, W, 3), np.float32); img[:] = canvas
    side = 0.14 * W if str(_p(ctx, "sidebar")) == "bars" else 0.0
    row_e = dk.edges(n_st, H, rng, float(_p(ctx, "stripe_var")), 1, 0, bleed=False)
    gap = float(_p(ctx, "gap")) * s
    means = np.nanmean(S3, (0, 2))
    for k in range(n_st):
        y0, y1 = int(round(row_e[k])), int(round(row_e[k + 1]))
        if y1 <= y0:
            continue
        # sub-strips: segments x runs on a ragged column grid
        wts = np.repeat(np.exp(rng.normal(0, float(_p(ctx, "seg_var")), n_sg)), runs)
        wts = np.where(np.arange(len(wts)) % runs == 0, wts, wts * 0 + wts.mean() * 0 + wts) if False else wts
        e = dk.edges(n_sg * runs, W - side, rng, 0.0, 1, 0, weights=wts, bleed=False)
        off = float(_p(ctx, "stagger")) * (W - side) / (n_sg * runs) * (rng.random() * 2 * runs)
        e = e - off
        # cut so the stripe covers [0, W-side]; wrap by extending
        e = np.concatenate([e, e[1:] + e[-1] - e[0]]) if off > 0 else e
        Vk = np.repeat(V[:, k, :].reshape(1, -1), 1, 0)
        Vk = np.concatenate([Vk, Vk[:, 1:]], 1) if off > 0 else Vk
        sub, _ = dk.render_cells(Vk[:, : len(e) - 1], np.array([0.0, y1 - y0]), e, (W, y1 - y0), tab, canvas, gap=gap * 0.8, surface=str(_p(ctx, "surface")),
                                 surf_amt=float(_p(ctx, "surf")), rng=rng, cell_jitter=float(_p(ctx, "jitter")))
        # horizontal gap (stripe separation) is the render_cells gap at top / bottom; segment borders inside get a darker line
        img[y0:y1] = sub
        if side:
            img[y0:y1, int(W - side):] = canvas
            frac = (means[k] - used[0]) / (used[1] - used[0] + 1e-9)
            bw = max(2, int(frac * (side - 8 * s)))
            c = dk.apply_lut(np.array([[np.clip(frac, 0, 1)]]), tab)[0, 0]
            by0, by1 = y0 + int(gap), y1 - int(gap)
            img[by0:by1, int(W - side + 6 * s): int(W - side + 6 * s) + bw] = c
    img = np.clip(img, 0, 1).astype(np.float32)
    if str(_p(ctx, "mode")) == "full":
        img = numbers_and_legend(ctx, img, P, V[..., 0], used, tab, np.array([0.0, 1]), np.array([0.0, 1]), 0, 0, np.full((1, 1), np.nan), cells=False)
    return img


# ----------------------------------------------------------------------------------------------------------------
def _series(rng, n, kind):
    t = np.linspace(0, 1, n)
    if kind == "walk":
        return np.cumsum(rng.normal(0, 1, n)) * 0.2 + 0.3 * np.sin(t * rng.uniform(2, 7))
    if kind == "burst":
        base = np.abs(rng.normal(0, 0.15, n))
        for c in rng.random(3):
            base += np.exp(-((t - c) / 0.03) ** 2) * rng.uniform(0.6, 1.2)
        return base
    return np.sin(t * rng.uniform(3, 9) + rng.random() * 6) * rng.uniform(0.5, 1) + rng.normal(0, 0.15, n)


@effect("data-terminal-panels", family="data", tags=["data", "panels", "terminal", "dashboard", "analytical", "scientific", "all-over", "synthetic-default", "dark-light", "small-multiples"],
        params={**{k: v for k, v in COMMON.items() if k in ("scale", "lo", "hi", "invert", "mode", "surf")},
                "panels": (8, "number of panels (binary space partition of the frame)"),
                "gutter": (7, "gutter px at 1500 wide"),
                "kinds": ("auto", "auto | heat | bars | spark | table | log (comma list cycles through panels)"),
                "chrome": (True, "title bars with window dots"),
                "density": (1.0, "row density multiplier")})
def terminal_panels(ctx):
    """Terminal / dashboard panels tiled over the whole frame: each panel is a small multiple (mini heatmap, bar
    rows, sparkline, coloured table, log lines). In card mode text is drawn as 'greeked' bars whose LENGTH and COLOUR
    carry the data; in full mode they are real monospace numbers from the data. ctx.data: dict name -> 1D series (a
    panel each), or a 2D matrix (rows feed bars/tables, the whole matrix feeds the heat panel). Synthetic otherwise."""
    W, H = ctx.size; rng = ctx.rng; s = dk.scale_px(W); P = ctx.palette
    full = str(_p(ctx, "mode")) == "full"
    gut = int(round(float(_p(ctx, "gutter")) * s))
    panels = [(x, y, x + w, y + h) for x, y, w, h in dkit.bsp_panels(ctx.size, rng, int(_p(ctx, "panels")), 0.18, 0)]
    tab = dk.lut(_p(ctx, "scale"), ctx.palette, polarity=P.polarity)
    if _p(ctx, "invert"):
        tab = tab[::-1].copy()
    dark = P.polarity == "dark"
    bg = P.canvas
    panel_bg = P.canvas + (P.structure - P.canvas) * (0.07 if dark else 0.05)
    border = P.canvas + (P.structure - P.canvas) * 0.45
    ink = np.array([0.93, 0.92, 0.9]) if dark else np.array([0.1, 0.1, 0.12])
    dim = P.canvas + (ink - P.canvas) * 0.38
    img = np.empty((H, W, 3), np.float32); img[:] = bg
    # data source
    d = ctx.data
    series = None
    if isinstance(d, dict) and not any(k in d for k in ("matrix", "scores", "values")):
        series = {k: np.asarray(v, np.float32) for k, v in d.items()}
        Pm = dk.Prep(np.array([v[: min(map(len, series.values()))] for v in series.values()]), list(series), None, False, "scores", "data")
    elif d is not None:
        Pm = dk.prepare_matrix(ctx)
    else:
        Pm = dk.synth_bench(rng, 24, 12)
    Mn, used = dk.normalize(Pm.M, _p(ctx, "lo"), _p(ctx, "hi"))
    kinds_spec = str(_p(ctx, "kinds"))
    kinds = ["heat", "bars", "spark", "table", "log"] if kinds_spec == "auto" else kinds_spec.split(",")
    big_first = sorted(range(len(panels)), key=lambda i: -(panels[i][2] - panels[i][0]) * (panels[i][3] - panels[i][1]))
    kind_of = {}
    for r, i in enumerate(big_first):
        kind_of[i] = kinds[r % len(kinds)] if kinds_spec != "auto" else ["heat", "spark", "bars", "heat", "table", "log", "spark", "bars", "heat"][r % 9]
    ov_items = []
    for pi, (x0, y0, x1, y1) in enumerate(panels):
        x0 += gut // 2; y0 += gut // 2; x1 -= gut // 2; y1 -= gut // 2
        img[y0:y1, x0:x1] = panel_bg
        ov_items.append((pi, x0, y0, x1, y1))
    ov = dk.Overlay(img)
    px = max(9, int(11 * s))
    for pi, x0, y0, x1, y1 in ov_items:
        w, h = x1 - x0, y1 - y0
        ov.rect(x0, y0, x1 - 1, y1 - 1, outline=border, w=max(1, int(round(1.5 * s))), a=1.0, radius=int(4 * s))
        ty = y0
        if _p(ctx, "chrome"):
            bar = max(int(16 * s), 8)
            ov.rect(x0, y0, x1 - 1, y0 + bar, fill=border, a=0.35, radius=int(4 * s))
            for k in range(3):
                ov.rect(x0 + 6 * s + k * 9 * s, y0 + bar / 2 - 2.2 * s, x0 + 6 * s + k * 9 * s + 4.4 * s, y0 + bar / 2 + 2.2 * s, fill=tab[int(len(tab) * (0.35 + 0.3 * k))], a=0.95, radius=int(2 * s))
            if full:
                nm = (Pm.rows[pi % len(Pm.rows)] if Pm.rows else f"panel {pi + 1}")
                ov.text((x0 + 36 * s, y0 + bar / 2), str(nm), px, dim, 1.0, anchor="lm")
            else:
                ov.rect(x0 + 40 * s, y0 + bar / 2 - 2 * s, x0 + 40 * s + min(w * 0.3, 60 * s + (pi % 3) * 24 * s), y0 + bar / 2 + 2 * s, fill=dim, a=0.8, radius=int(2 * s))
            ty = y0 + bar + 4 * s
        ix0, ix1, iy0, iy1 = x0 + 6 * s, x1 - 6 * s, ty + 4 * s, y1 - 6 * s
        iw, ih = ix1 - ix0, iy1 - iy0
        if iw < 20 or ih < 14:
            continue
        kind = kind_of[pi]
        rp = np.random.default_rng(ctx.seed * 131 + pi)
        n_series = Mn.shape[0]
        if kind == "heat":
            R, C = Mn.shape
            cs = max(6 * s, min(iw / C, ih / R)) if False else None
            cw, chh = iw / C, ih / R
            if cw < 3 or chh < 3:
                f = int(max(np.ceil(3 / cw), np.ceil(3 / chh)))
                sub = dk.aggregate(Mn, max(1, R // f), max(1, C // f))
            else:
                sub = Mn
            r0 = int(rp.integers(0, max(1, Mn.shape[0] - sub.shape[0] + 1))); sub = sub[: sub.shape[0]]
            RR, CC = sub.shape
            ce = ix0 + np.linspace(0, iw, CC + 1); re = iy0 + np.linspace(0, ih, RR + 1)
            patch = np.empty((int(ih), int(iw), 3), np.float32)
            blk, _ = dk.render_cells(sub, re - iy0, ce - ix0, (int(iw), int(ih)), tab, panel_bg, gap=max(0.0, 1.2 * s), gap_color=panel_bg)
            ov.im.paste(__import__("PIL.Image", fromlist=["Image"]).fromarray((blk * 255).astype(np.uint8)), (int(ix0), int(iy0)))
        elif kind in ("bars", "table", "log"):
            nrow = max(2, int(ih / (16 * s) * float(_p(ctx, "density"))))
            rh = ih / nrow
            for r in range(nrow):
                yy = iy0 + r * rh
                v = Mn[(pi * 3 + r) % Mn.shape[0]]
                if kind == "bars":
                    lab_w = min(iw * 0.22, 44 * s)
                    ov.rect(ix0, yy + rh * 0.3, ix0 + lab_w * (0.5 + 0.5 * rp.random()), yy + rh * 0.7, fill=dim, a=0.7, radius=int(1.5 * s))
                    val = float(np.nanmean(v))
                    c = dk.apply_lut(np.array([[val]]), tab)[0, 0]
                    ov.rect(ix0 + lab_w + 6 * s, yy + rh * 0.2, ix0 + lab_w + 6 * s + (iw - lab_w - 8 * s) * max(0.04, val), yy + rh * 0.8, fill=c, a=1.0, radius=int(2 * s))
                    if full:
                        ov.text((ix0 + iw - 2, yy + rh / 2), f"{val * (used[1] - used[0]) + used[0]:.2f}", px, ink, 0.9, anchor="rm")
                elif kind == "table":
                    ncell = min(len(v), max(3, int(iw / (28 * s))))
                    cw2 = iw / ncell
                    for c in range(ncell):
                        col = dk.apply_lut(np.array([[float(v[c % len(v)]) if np.isfinite(v[c % len(v)]) else 0]]), tab)[0, 0]
                        ov.rect(ix0 + c * cw2 + 1, yy + 1, ix0 + (c + 1) * cw2 - 1, yy + rh - 1, fill=col, a=0.92 if r % 2 == 0 else 0.7)
                        if full and cw2 > 30 and rh > 11:
                            ov.text((ix0 + (c + .5) * cw2, yy + rh / 2), f"{v[c % len(v)] * (used[1] - used[0]) + used[0]:.2f}", max(8, int(rh * 0.55)), dk.ink_for(col), 0.95, anchor="mm")
                else:   # log lines
                    x = ix0
                    lvl = float(np.nanmean(v))
                    ov.rect(x, yy + rh * 0.3, x + 6 * s, yy + rh * 0.7, fill=dk.apply_lut(np.array([[lvl]]), tab)[0, 0], a=1.0)
                    x += 12 * s
                    for _ in range(int(rp.integers(2, 5))):
                        wseg = rp.uniform(0.06, 0.25) * iw
                        if x + wseg > ix1:
                            break
                        ov.rect(x, yy + rh * 0.36, x + wseg, yy + rh * 0.64, fill=dim if rp.random() > 0.2 else ink, a=0.65, radius=int(1.5 * s))
                        x += wseg + 6 * s
        else:   # spark
            kinds_s = ["walk", "burst", "wave"]
            n = max(30, int(iw / (3 * s)))
            if series is not None and len(series):
                key = list(series)[pi % len(series)]
                y = np.interp(np.linspace(0, len(series[key]) - 1, n), np.arange(len(series[key])), series[key])
            else:
                y = _series(rp, n, kinds_s[pi % 3])
            y = (y - y.min()) / (np.ptp(y) + 1e-9)
            xs_ = ix0 + np.linspace(0, iw, n)
            ys_ = iy1 - (0.08 + 0.84 * y) * ih
            col = tab[int(len(tab) * 0.72)]
            for gx in range(1, 4):
                ov.line([(ix0, iy0 + ih * gx / 4), (ix1, iy0 + ih * gx / 4)], border, 1, 0.5)
            poly = [(ix0, iy1)] + list(zip(xs_, ys_)) + [(ix1, iy1)]
            ov.d.polygon(poly, fill=ov._c(col, 0.22))
            ov.line(list(zip(xs_, ys_)), col, max(2, int(round(2 * s))), 1.0)
            if full:
                ov.text((ix0 + 2, iy0 + 2), f"max {used[1]:.2f}", px, dim, 1.0)
    img = ov.done()
    if full and Pm.synthetic:
        o2 = dk.Overlay(img)
        o2.rect(int(8 * s), H - int(10 * s) - px - 8, int(8 * s) + int(px * 24), H - int(10 * s) + 2, fill=bg, a=0.9, radius=3)
        o2.text((int(12 * s), H - int(10 * s) - 2), "synthetic data - decorative, not results", px, ink, 0.95, anchor="lb")
        img = o2.done()
    return img
