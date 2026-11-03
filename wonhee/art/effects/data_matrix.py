"""data-heat-bleed, data-annotated-grow, data-cluster-matrix: matrix heatmaps as information texture.

Shared by all three (see wonhee/art/datakit.py): ctx.data is a 2D array or {"matrix"|"scores": 2D/3D, "rows": [..],
"cols": [..], "mask": bool}; with no data a SEEDED SYNTHETIC matrix is used and the image is decorative only
(provenance(...)['synthetic'] is True; ``mode=full`` stamps 'synthetic data' on the frame).

Controls are independent: colour (scale, lo, hi, center, invert, palette), structure (cols, rows, col_var, row_var,
growth, order, fit), source (data / synth), surface (gap, surface, surf, jitter). ``mode=card`` (default) draws no
text, keeps bands and value differences; ``mode=full`` adds numbers where they are readable plus a real legend."""
from __future__ import annotations

import math

import numpy as np

from .. import datakit as dk
from ..registry import effect

COMMON = {
    "scale": ("auto", "colour scale: auto (keyed on the palette hue) | palette | magma | ember | ocean | kelp | ice | rose | cream-red | ink | mono | blue-red | teal-rose | violet-gold | green-plum"),
    "lo": (None, "explicit range low (data units); default robust 1st percentile"),
    "hi": (None, "explicit range high (data units); default robust 99th percentile"),
    "invert": (False, "flip the scale"),
    "order": ("data", "row/col order: data | mean | cluster | rows | cols"),
    "gap": (2.0, "gap between cells in px at 1500 wide (0 = flush)"),
    "surface": ("flat", "flat | glass | bevel | paper"),
    "surf": (0.5, "surface strength"),
    "jitter": (0.0, "per-cell value jitter (0..0.1), hand-made texture"),
    "mode": ("card", "card (no text; aggregates; always legible at 288x192) | full (numbers, legend, synthetic stamp)"),
}


def _p(ctx, k):
    return ctx.params.get(k)


def window(n_data: int, n_need: int, rng) -> tuple[int, int]:
    n = min(n_data, n_need)
    s = int(rng.integers(0, n_data - n + 1)) if n_data > n else 0
    return s, n


def axis_edges(nvis: float, n_data: int, total: float, rng, var: float, growth: float = 1.0, off: float = 0.5):
    """-> (start, n, edges): a window of n cells of a data axis covering ``total`` px, cut by the frame edges."""
    n_need = int(math.ceil(nvis)) + 1
    s, n = window(n_data, n_need, rng)
    nominal = total / max(nvis, 1e-6)
    e = dk.edges(n, nominal * n, rng, var, growth, off if n >= n_need else off * 0.0)
    if e[-1] < total + 1:                      # not enough cells (small matrix): stretch to cover
        e = (e - e[0]) / (e[-1] - e[0]) * (total + 2) if n < n_need else e * (total + 1) / e[-1]
    return s, n, e


def resolve(ctx, kind="bench", shape=(38, 9), diverging=False):
    P = dk.prepare_matrix(ctx, kind, shape)
    P = dk.order_matrix(P, str(_p(ctx, "order")))
    pol = ctx.palette.polarity
    spec = _p(ctx, "scale")
    tab = dk.lut(spec, ctx.palette, polarity=pol)
    if _p(ctx, "invert"):
        tab = tab[::-1].copy()
    return P, tab


def scaled(ctx, M, center=None):
    return dk.normalize(M, _p(ctx, "lo"), _p(ctx, "hi"), "robust", center)


def numbers_and_legend(ctx, img, P, V, rng_used, tab, row_e, col_e, r0, c0, raw, info=None, force_legend=True, cells=True):
    """Full mode: numbers inside cells that are big enough, a legend, the synthetic stamp."""
    W, H = ctx.size
    s = dk.scale_px(W)
    pal = ctx.palette
    ink_d, ink_l = np.array([0.09, 0.09, 0.1]), np.array([0.96, 0.95, 0.93])
    ov = dk.Overlay(img)
    R, C = V.shape if cells else (0, 0)
    for i in range(R):
        ya, yb = row_e[i], row_e[i + 1]
        if yb < 0 or ya > H:
            continue
        for j in range(C):
            xa, xb = col_e[j], col_e[j + 1]
            if xb < 0 or xa > W or not np.isfinite(raw[i, j]):
                continue
            px = int(min((yb - ya) * 0.34, (xb - xa) * 0.2))
            if px < max(9, 10 * s) or xa < 2 or xb > W - 2 or ya < 2 or yb > H - 2:
                continue
            col = dk.apply_lut(np.array([V[i, j]]), tab)[0]
            ov.text(((xa + xb) / 2, (ya + yb) / 2), f"{raw[i, j]:.2f}", px, dk.ink_for(col, ink_d, ink_l), 0.92, anchor="mm")
    pad = int(18 * s + 8)
    ink = pal.structure * 0.0 + (ink_l if pal.polarity == "dark" else ink_d)
    bw, bh = int(W * 0.26), max(8, int(14 * s))
    px = max(10, int(12 * s))
    x1 = W - pad; x0 = x1 - bw; y1 = H - pad - px - 8; y0 = y1 - bh
    ov.rect(x0 - 14, y0 - px - 12, x1 + 14, H - pad + 8, fill=pal.canvas, a=0.88, radius=int(6 * s + 2))
    ov.colorbar(x0, y0, x1, y1, tab, rng_used[0], rng_used[1], ink, "{:.2f}", px, 3, label=("value" if not P.synthetic else "value (synthetic)"))
    if P.synthetic:
        ov.text((pad, pad), "synthetic data - decorative, not results", px, ink, 0.85)
    return ov.done()


@effect("data-heat-bleed", family="data", tags=["data", "matrix", "heatmap", "scientific", "analytical", "all-over", "synthetic-default", "dark-light"],
        params={**COMMON,
                "cols": (6.5, "columns visible across the frame (fractional: the edge column is cut)"),
                "rows": (10.4, "rows visible down the frame"),
                "col_var": (0.0, "column width variation (lognormal sigma 0..0.6)"),
                "row_var": (0.0, "row height variation"),
                "fit": ("crop", "crop (window into the matrix, bleeds) | all (whole matrix at its own cell size) | aggregate (area-average to cols x rows)")})
def heat_bleed(ctx):
    """The gold-standard heatmap: a flat-cell matrix magnified so cells bleed off every edge (38 tests x 9 model
    pairs in magma). Any real matrix; synthetic bench-like scores otherwise (flagged). Light and dark follow the
    palette polarity (light: cream low values)."""
    W, H = ctx.size; rng = ctx.rng; s = dk.scale_px(W)
    P, tab = resolve(ctx)
    M = P.M
    cols_v, rows_v = float(_p(ctx, "cols")), float(_p(ctx, "rows"))
    fit = str(_p(ctx, "fit"))
    if fit == "aggregate":
        M = dk.aggregate(M, int(round(rows_v)), int(round(cols_v)))
        ar = ac = 0
        r0, c0, nr, nc = 0, 0, M.shape[0], M.shape[1]
        row_e = dk.edges(nr, H, rng, float(_p(ctx, "row_var")), 1, 0, bleed=False); col_e = dk.edges(nc, W, rng, float(_p(ctx, "col_var")), 1, 0, bleed=False)
    elif fit == "all":
        r0 = c0 = 0; nr, nc = M.shape
        row_e = dk.edges(nr, H, rng, float(_p(ctx, "row_var")), 1, 0, bleed=False); col_e = dk.edges(nc, W, rng, float(_p(ctx, "col_var")), 1, 0, bleed=False)
    else:
        r0, nr, row_e = axis_edges(rows_v, M.shape[0], H, rng, float(_p(ctx, "row_var")), 1.0, float(rng.uniform(0.15, 0.85)))
        c0, nc, col_e = axis_edges(cols_v, M.shape[1], W, rng, float(_p(ctx, "col_var")), 1.0, float(rng.uniform(0.15, 0.85)))
    sub = M[r0:r0 + nr, c0:c0 + nc]
    V, used = scaled(ctx, sub)
    gc = ctx.palette.canvas
    img, info = dk.render_cells(V, row_e, col_e, ctx.size, tab, gc, gap=float(_p(ctx, "gap")) * s, surface=str(_p(ctx, "surface")),
                                surf_amt=float(_p(ctx, "surf")), rng=rng, cell_jitter=float(_p(ctx, "jitter")))
    if str(_p(ctx, "mode")) == "full":
        img = numbers_and_legend(ctx, img, P, V, used, tab, row_e, col_e, r0, c0, sub)
    return img


# ----------------------------------------------------------------------------------------------------------------
def grow_edges(total: float, w0: float, g: float, nmax: int, flip: bool):
    """Cell boundaries that start at size w0 and grow by ratio g until they pass ``total`` (cut by the frame).
    If the data has too few cells, g is raised until they cover."""
    for _ in range(40):
        n = 1; acc = w0
        while acc < total * 1.02 and n < nmax:
            acc += w0 * g ** n; n += 1
        if acc >= total * 1.02 or n >= nmax and acc >= total * 1.02:
            break
        g *= 1.06
    sizes = w0 * g ** np.arange(n)
    e = np.concatenate([[0], np.cumsum(sizes)])
    if e[-1] < total * 1.02:
        e = e * total * 1.02 / e[-1]
    if flip:
        e = total - e[::-1]
    return e, n


@effect("data-annotated-grow", family="data", tags=["data", "matrix", "heatmap", "annotated", "analytical", "scientific", "synthetic-default", "dark-light"],
        params={**COMMON,
                "w0": (0.045, "size of the smallest cell, fraction of frame width"),
                "growth": (1.2, "ratio between neighbouring cell sizes (1.0 = uniform)"),
                "corner": ("random", "tl | tr | bl | br | random: where the small annotated block sits"),
                "callouts": (4, "outlined cells with leader ticks (the 'annotated' part)"),
                "aspect_growth": (0.0, "extra growth bias for rows vs cols (-0.1..0.1)")})
def annotated_grow(ctx):
    """A small, crisp executive heatmap in one corner (thin gaps, outlined call-out cells, tick marks) that keeps
    growing: neighbouring cells swell geometrically until the last ones are huge flat colour fields bleeding off the
    opposite edges. Reads as 'detail -> overview' with no centre. Card mode draws no text; full mode adds numbers
    and the legend."""
    W, H = ctx.size; rng = ctx.rng; s = dk.scale_px(W)
    P, tab = resolve(ctx, shape=(30, 30))
    corner = str(_p(ctx, "corner"))
    if corner == "random":
        corner = ["tl", "tr", "bl", "br"][int(rng.integers(0, 4))]
    fx, fy = corner[1] == "r", corner[0] == "b"
    M = P.M
    g = float(_p(ctx, "growth")); ag = float(_p(ctx, "aspect_growth"))
    w0 = float(_p(ctx, "w0")) * W
    h0 = w0 * 0.72
    col_e, nc = grow_edges(W, w0, g + ag, M.shape[1], fx)
    row_e, nr = grow_edges(H, h0, g - ag, M.shape[0], fy)
    sub = M[:nr, :nc]
    if fy: sub = sub[::-1]
    if fx: sub = sub[:, ::-1]
    V, used = scaled(ctx, sub)
    # small cells keep thin gaps, big cells get proportionally the same: gap fixed in px (reads as 'fine at the corner, flat at the far end')
    img, _ = dk.render_cells(V, row_e, col_e, ctx.size, tab, ctx.palette.canvas, gap=float(_p(ctx, "gap")) * s, surface=str(_p(ctx, "surface")),
                             surf_amt=float(_p(ctx, "surf")), rng=rng, cell_jitter=float(_p(ctx, "jitter")))
    ov = dk.Overlay(img)
    ink = np.array([0.95, 0.94, 0.92]) if ctx.palette.polarity == "dark" else np.array([0.1, 0.1, 0.12])
    # call-outs: outline the strongest cells among the visible small ones (those narrower than 0.12 W)
    ys, xs = np.nonzero(np.isfinite(V))
    small = [(V[i, j], i, j) for i, j in zip(ys, xs) if (col_e[j + 1] - col_e[j]) < 0.13 * W and (row_e[i + 1] - row_e[i]) < 0.19 * H]
    small.sort(key=lambda t: -t[0])
    picked = []
    for v, i, j in small:
        if all(abs(i - a) + abs(j - b) > 2 for _, a, b in picked):
            picked.append((v, i, j))
        if len(picked) >= int(_p(ctx, "callouts")):
            break
    lw = max(2, int(round(3 * s + 0.5)))
    halo = ctx.palette.canvas
    for v, i, j in picked:
        x0, x1, y0, y1 = col_e[j], col_e[j + 1], row_e[i], row_e[i + 1]
        ov.rect(x0 + 1, y0 + 1, x1 - 1, y1 - 1, outline=halo, w=lw + 2, a=0.9)
        ov.rect(x0 + 1 + 1, y0 + 2, x1 - 2, y1 - 2, outline=ink, w=lw, a=0.98)
    # tick marks along the small block's edges (axis ticks)
    tl = int(8 * s + 3)
    for j in range(min(nc, 9)):
        xc = (col_e[j] + col_e[j + 1]) / 2
        if 0 < xc < W:
            ov.line([(xc, H if fy else 0), (xc, (H - tl) if fy else tl)], ink, max(1, lw - 1), 0.85)
    for i in range(min(nr, 9)):
        yc = (row_e[i] + row_e[i + 1]) / 2
        if 0 < yc < H:
            ov.line([(W if fx else 0, yc), ((W - tl) if fx else tl, yc)], ink, max(1, lw - 1), 0.85)
    img = ov.done()
    if str(_p(ctx, "mode")) == "full":
        img = numbers_and_legend(ctx, img, P, V, used, tab, row_e, col_e, 0, 0, sub)
    return img


# ----------------------------------------------------------------------------------------------------------------
def dendrogram_segments(Z, order, pos):
    """Segments (pos_a, h_a, pos_b, h_b) for a scipy linkage, leaves placed at pos[leaf index in display order]."""
    n = len(order)
    where = {leaf: k for k, leaf in enumerate(order)}
    node_pos = {leaf: pos[where[leaf]] for leaf in range(n)}
    node_h = {leaf: 0.0 for leaf in range(n)}
    hmax = Z[:, 2].max() + 1e-9
    segs = []
    for k, (a, b, d, _) in enumerate(Z):
        a, b = int(a), int(b)
        h = d / hmax
        segs.append((node_pos[a], node_h[a], node_pos[a], h))
        segs.append((node_pos[b], node_h[b], node_pos[b], h))
        segs.append((node_pos[a], h, node_pos[b], h))
        node_pos[n + k] = (node_pos[a] + node_pos[b]) / 2
        node_h[n + k] = h
    return segs


@effect("data-cluster-matrix", family="data", tags=["data", "matrix", "heatmap", "clustered", "correlation", "scientific", "analytical", "synthetic-default", "dark-light"],
        params={**COMMON, "order": ("cluster", "cluster | data | mean | rows | cols"), "scale": ("blue-red", COMMON["scale"][1]),
                "n": (26, "synthetic matrix size (correlation of latent factors)"),
                "margin": (0.15, "dendrogram margin, fraction of frame size"),
                "dendro": (True, "draw dendrograms (real linkage of the displayed order)"),
                "strip": (True, "categorical cluster strips next to the dendrograms"),
                "k": (4, "number of cluster strips / colours"),
                "center": (0.0, "diverging centre value (None for a sequential scale)")})
def cluster_matrix(ctx):
    """Clustered correlation/score matrix: rows and columns ordered by hierarchical clustering with optimal leaf
    ordering (blocks appear on the diagonal), diverging colour about ``center``, dendrograms and categorical cluster
    strips as side overlays. Real data: any matrix (square symmetric matrices cluster rows and columns together).
    The dendrogram margin and the cell block both fill the frame; no centre."""
    W, H = ctx.size; rng = ctx.rng; s = dk.scale_px(W)
    n = int(_p(ctx, "n"))
    d = ctx.data
    P = dk.prepare_matrix(ctx, "corr", (n, n))
    sym = P.M.shape[0] == P.M.shape[1] and P.kind == "corr" or (d is not None and P.M.shape[0] == P.M.shape[1] and np.allclose(np.nan_to_num(P.M), np.nan_to_num(P.M.T), atol=1e-4))
    if sym:
        P.kind = "corr"
    how = str(_p(ctx, "order"))
    Zr = Zc = None
    ro, Zr = dk.cluster_order(P.M, 0) if how in ("cluster", "rows") else (np.arange(P.M.shape[0]), None)
    if sym:
        co, Zc = ro, Zr
    else:
        co, Zc = dk.cluster_order(P.M, 1) if how in ("cluster", "cols") else (np.arange(P.M.shape[1]), None)
    if how == "mean":
        ro = np.argsort(-np.nanmean(P.M, 1)); co = np.argsort(-np.nanmean(P.M, 0))
    M = P.M[np.ix_(ro, co)]
    center = _p(ctx, "center")
    if center is None and str(_p(ctx, "scale")) in dk.DIV:
        center = 0.0
    tab = dk.lut(_p(ctx, "scale"), ctx.palette)
    if _p(ctx, "invert"):
        tab = tab[::-1].copy()
    V, used = dk.normalize(M, _p(ctx, "lo"), _p(ctx, "hi"), "robust", center if str(_p(ctx, "scale")) in dk.DIV else None)
    dend = bool(_p(ctx, "dendro"))
    mg = float(_p(ctx, "margin")) if dend else 0.0
    mx, my = mg * W * 0.8, mg * H * 1.0
    strip = bool(_p(ctx, "strip")) and dend
    sw = (0.016 * W if strip else 0.0)
    x0f, y0f = mx + sw, my + sw * 1.3
    R, C = M.shape
    # bleeding: matrix cells run past the right/bottom edge by a fraction of a cell
    cw = (W - x0f) / (C - 0.35); rh = (H - y0f) / (R - 0.35)
    col_e = x0f + np.arange(C + 1) * cw; row_e = y0f + np.arange(R + 1) * rh
    img, _ = dk.render_cells(V, row_e, col_e, ctx.size, tab, ctx.palette.canvas, gap=float(_p(ctx, "gap")) * s * 0.8, surface=str(_p(ctx, "surface")),
                             surf_amt=float(_p(ctx, "surf")), rng=rng, cell_jitter=float(_p(ctx, "jitter")))
    ov = dk.Overlay(img)
    ink = np.array([0.9, 0.9, 0.9]) if ctx.palette.polarity == "dark" else np.array([0.25, 0.25, 0.3])
    k = int(_p(ctx, "k"))
    cats = dk.categorical(k, ctx.palette.polarity, 25.0)
    lw = max(1, int(round(2 * s)))
    if dend:
        for Z, order, e, vert in ((Zr, ro, row_e, True), (Zc, co, col_e, False)):
            if Z is None:
                continue
            mid = (e[:-1] + e[1:]) / 2
            segs = dendrogram_segments(Z, np.arange(len(order)), mid)
            # leaves are indexed by the ORIGINAL id; order maps display slot -> original id
            slot = np.empty(len(order), int); slot[order] = np.arange(len(order))
            segs = dendrogram_segments(Z, order, mid)
            ext = (mx if vert else my) * 0.92
            base = (mx - 4 * s) if vert else (my - 4 * s)
            for pa, ha, pb, hb in segs:
                if vert:
                    ov.line([(base - ha * ext, pa), (base - hb * ext, pb)], ink, lw, 0.75)
                else:
                    ov.line([(pa, base - ha * ext), (pb, base - hb * ext)], ink, lw, 0.75)
        if strip:
            from scipy.cluster.hierarchy import fcluster
            for Z, order, e, vert in ((Zr, ro, row_e, True), (Zc, co, col_e, False)):
                if Z is None:
                    continue
                lab = fcluster(Z, k, "maxclust")[order] - 1
                for i in range(len(order)):
                    c = cats[lab[i] % k]
                    if vert:
                        ov.rect(mx + 1, row_e[i] + 1, mx + sw - 2, row_e[i + 1] - 1, fill=c)
                    else:
                        ov.rect(col_e[i] + 1, my + 1, col_e[i + 1] - 1, my + sw * 1.3 - 2, fill=c)
    img = ov.done()
    if str(_p(ctx, "mode")) == "full":
        img = numbers_and_legend(ctx, img, P, V, used, tab, row_e, col_e, 0, 0, M)
    return img
