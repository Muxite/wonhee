"""Fractured microstructure: large jagged plates filled with coherent, oriented,
hierarchical rectilinear traces over a dark (or light) substrate.

The frame is cut into a few broad plates (Voronoi / strata / macro-block
boundaries with multi-scale roughness). Every plate is filled by its own trace
field in plate coordinates: orientation, lane pitch, bundles with gaps, long
segments with pads, a phase offset against its neighbours, optional crossing
layers and sub-blocks that turn 90 degrees, plus a coarse trunk layer with
haloes and vias. Plates get their own tilt shading and substrate grain; cracks
and bevels separate them. Boundaries can come from a real image (`ctx.source`,
k-means regions of colour + position, trace orientation read from the image's
own structure), from real data (`ctx.data`: a 2D field is banded into plates, an
integer label image is used as is, an (n, 2) point set in 0..1 is a set of
Voronoi seeds) or purely from the seed.

Why it is not feature-circuit-traces / dots-circuit / graph-*: those draw
thin wires or dot pads on a calm field; here the picture is the large fractured
areas, each one a coherent directional fine grid with its own phase.

Moods: engineered, precise, tactile (substrate grain), nocturnal (dark), monumental (big plates)."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from .. import microkit as mk
from .. import palette as pal
from ..registry import effect

BASE = {
    "boundary": ("voronoi", "plate boundaries: voronoi | strata | blocks | source | data"),
    "plates": (9, "number of broad plates (3-6 for a small card, 8-16 for a hero)"),
    "rough": (0.8, "structure: boundary roughness / jaggedness 0..1.5"),
    "angle": (-1.0, "structure: global trace angle in degrees (-1 = seeded)"),
    "orient": (0.5, "structure: orientation variety between plates 0 (all alike) .. 1 (diagonals, tilts)"),
    "pitch": (0.0042, "structure: fine lane pitch as a fraction of the width"),
    "hierarchy": (0.6, "structure: coarse trunk layer, bundles and vias 0..1"),
    "layers": (2, "structure: crossing trace layers per plate (1-3)"),
    "blocks": (0.18, "structure: share of sub-blocks inside a plate whose traces turn 90 degrees"),
    "offset": (0.8, "structure: phase offset between plates 0..1 (0 = one continuous grid)"),
    "density": (0.85, "structure: share of lane segments present 0..1"),
    "crack": (1.0, "surface: crack width multiplier between plates"),
    "bevel": (0.5, "surface: plate edge highlight 0..1"),
    "tilt": (0.5, "surface: per-plate tilt shading 0..1"),
}


def _warp(W, H, rough, rng):
    """Smooth + jagged displacement fields (dx, dy) in px at full resolution."""
    q = 4
    h, w = max(8, H // q), max(8, W // q)
    d = []
    for _ in range(2):
        big = mk.fbm((h, w), w * 0.28, rng, 3) - 0.5
        mid = mk.fbm((h, w), w * 0.06, rng, 2) - 0.5
        fine = np.abs(mk.fbm((h, w), w * 0.015, rng, 2) * 2 - 1) - 0.5      # ridged -> jagged
        f = big * 0.09 * W + mid * 0.03 * W + fine * 0.022 * W
        d.append(cv2.resize((f * rough).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC))
    return d[0], d[1]


def _seeds(n, rng, W, H, kind):
    """Jittered best-candidate seeds. strata: seeds along a line (long shards)."""
    pts = []
    for _ in range(n):
        best, bd = None, -1
        for _ in range(14):
            c = rng.random(2) * [W, H]
            dmin = min((np.hypot(*(c - p)) for p in pts), default=1e9)
            if dmin > bd:
                best, bd = c, dmin
        pts.append(best)
    return np.array(pts, np.float32)


def _merge_small(lab, minfrac=0.004):
    n = int(lab.max()) + 1
    cnt = np.bincount(lab.ravel(), minlength=n)
    small = cnt < lab.size * minfrac
    if not small.any() or small.all():
        return lab
    bad = small[lab]
    _, (iy, ix) = ndi.distance_transform_edt(bad, return_indices=True)
    out = lab[iy, ix]
    # compact ids
    u, inv = np.unique(out, return_inverse=True)
    return inv.reshape(out.shape).astype(np.int32)


def _lowres_labels_source(src, n, rng):
    """k-means regions of an image: OKLab colour + position, at low resolution."""
    from .. import features as F
    h, w = 120, 180
    small = cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)
    lab = F.to_oklab(small)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    feat = np.stack([lab[..., 0] * 3.0, lab[..., 1] * 5.0, lab[..., 2] * 5.0, xx / w * 2.4, yy / h * 2.4], -1).reshape(-1, 5)
    cv2.setRNGSeed(int(rng.integers(0, 2 ** 31 - 1)))
    _, labels, _ = cv2.kmeans(feat.astype(np.float32), int(n), None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1), 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(h, w).astype(np.uint8)
    labels = cv2.medianBlur(labels, 15)
    return _merge_small(labels.astype(np.int32), 0.03), small


def _lowres_labels_data(data, n, rng):
    d = np.asarray(data)
    if d.ndim == 2 and d.dtype.kind in "iu" and len(np.unique(d)) <= 96:
        u, inv = np.unique(d, return_inverse=True)
        return _merge_small(inv.reshape(d.shape).astype(np.int32), 0.004)
    f = np.asarray(d, np.float32)
    f = cv2.resize(f, (180, 120), interpolation=cv2.INTER_AREA if f.shape[0] > 120 else cv2.INTER_LINEAR)
    f = cv2.GaussianBlur(f, (0, 0), 1.6)
    lo, hi = np.percentile(f, [3, 97])
    f = np.clip((f - lo) / max(1e-6, hi - lo), 0, 0.9999)
    bands = np.floor(f * max(3, int(n * 0.6))).astype(np.int32)
    out = np.zeros(bands.shape, np.int32)
    k = 0
    for b in np.unique(bands):
        cc, m = ndi.label(bands == b)
        out[cc > 0] = cc[cc > 0] + k
        k += m
    return _merge_small(out, 0.006)


def _plate_labels(ctx, W, H, rng, src):
    """-> labels (H, W) int32, plate stats dict (optional per-plate source info)."""
    kind = str(ctx.p("boundary"))
    n = int(ctx.p("plates"))
    rough = float(ctx.p("rough"))
    dx, dy = _warp(W, H, rough, rng)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    X, Y = xx + dx, yy + dy
    stats = {}
    low = None
    if kind == "source" and src is not None:
        low, small = _lowres_labels_source(src, n, rng)
        sx, sy = low.shape[1] / W, low.shape[0] / H
        lab = cv2.remap(low.astype(np.float32), (X * sx).astype(np.float32), (Y * sy).astype(np.float32), cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE).astype(np.int32)
        stats["lowres"], stats["small"] = low, small
    elif kind == "data" and ctx.data is not None and np.ndim(ctx.data) == 2:
        low = _lowres_labels_data(ctx.data, n, rng)
        sx, sy = low.shape[1] / W, low.shape[0] / H
        lab = cv2.remap(low.astype(np.float32), (X * sx).astype(np.float32), (Y * sy).astype(np.float32), cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE).astype(np.int32)
        stats["lowres"] = low
    else:
        th = np.deg2rad(rng.uniform(0, 180) if kind != "blocks" else 0)
        aniso, p = 1.0, 1.4
        if kind == "strata":
            aniso, p = 0.16, 1.2
        if kind == "blocks":
            aniso, p = 1.0, 1.0       # L1 metric after warp -> rectilinear-ish diamonds; combined with snapping below
        pts = None
        if kind == "data" and ctx.data is not None and np.ndim(ctx.data) == 2 and np.shape(ctx.data)[1] == 2:
            pts = np.asarray(ctx.data, np.float32) * [W, H]
        elif kind == "data":
            kind = "voronoi"
        if pts is None:
            pts = _seeds(n, rng, W, H, kind)
        if kind == "blocks":
            # macro blocks: axis-aligned BSP rectangles; roughness then fractures their edges
            from ..dotkit import bsp_panels
            bl = bsp_panels((W, H), rng, n, 0.12)
            lab = np.zeros((H, W), np.int32)
            ix = np.clip(X, 0, W - 1).astype(np.int32)
            iy = np.clip(Y, 0, H - 1).astype(np.int32)
            raw = np.zeros((H, W), np.int32)
            for i, (x, y, w, h) in enumerate(bl):
                raw[y:y + h, x:x + w] = i
            lab = raw[iy, ix]
        else:
            c, s = np.cos(th), np.sin(th)

            def tf(a, b):
                return np.stack([(a * c + b * s), (-a * s + b * c) * aniso], -1)
            tree = cKDTree(tf(pts[:, 0], pts[:, 1]))
            q = tf(X.ravel(), Y.ravel())
            _, idx = tree.query(q, p=p, workers=-1)
            lab = idx.reshape(H, W).astype(np.int32)
        u, inv = np.unique(lab, return_inverse=True)
        lab = inv.reshape(lab.shape).astype(np.int32)
    return lab, stats


def _lanes(u, v, pitch, wf, G, dens, seglen, phu, phv, lab, seed, padlen, padw):
    """Trace coverage + pad coverage of an oriented lane field in plate coordinates."""
    uu = (u + phu) / pitch
    i = np.floor(uu)
    across = np.abs(uu - i - 0.5) * pitch
    m = np.mod(i, G + 1) != G
    h1 = mk.hash01(i, lab, seed)
    Ls = pitch * seglen * (0.5 + h1)
    vv = (v + phv) / Ls + h1 * 7
    k = np.floor(vv)
    fv = vv - k
    act = mk.hash01(i, k, lab, seed + 1) < dens
    on = m & act
    w = pitch * wf
    cov = np.clip(w * 0.5 - across + 0.5, 0, 1) * on
    dend = np.minimum(fv, 1 - fv) * Ls
    padm = mk.hash01(i, k, lab, seed + 2) < 0.55
    pad = np.clip(padw * 0.5 - across + 0.5, 0, 1) * np.clip(padlen - dend + 0.5, 0, 1) * (on & padm)
    return cov, pad, h1


def render_plates(ctx):
    W, H = ctx.size
    rng = ctx.rng
    P = ctx.palette
    dark = P.polarity == "dark"
    s = W / 1500.0
    src = mk.fit_source(ctx)
    smix = float(ctx.p("source_mix", 1.0))
    tone = None
    if src is not None and smix > 0:
        from .. import features as F
        L = F.luminance(src)
        tone = cv2.GaussianBlur(F.stretch(F.local_equalize(L, 2.5, mix=0.7), 1, 99), (0, 0), 3 * max(s, 0.3))
    lab, stats = _plate_labels(ctx, W, H, rng, src)
    N = int(lab.max()) + 1

    # ---- per-plate attributes
    pitch0 = max(1.6, float(ctx.p("pitch")) * W)
    th0 = float(ctx.p("angle"))
    th0 = rng.uniform(0, 180) if th0 < 0 else th0
    orient = float(ctx.p("orient"))
    ang_set = [0, 90, 0, 90, 45, 135, 18, 72]
    ang = np.zeros(N, np.float32)
    for i in range(N):
        pick = rng.integers(0, 2 + int(6 * orient))
        ang[i] = th0 + ang_set[pick] + rng.normal(0, 4 * orient)
    # source: orientation read from the image's own structure, snapped mod 90
    if "small" in stats and smix > 0:
        small = stats["small"]
        g = cv2.GaussianBlur(small.mean(-1), (0, 0), 1.2)
        gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
        mag = np.hypot(gx, gy)
        th = np.arctan2(gy, gx)
        low = stats["lowres"]
        for i in range(N):
            mm = low == i
            if mm.sum() < 20:
                continue
            z = (mag[mm] * np.exp(4j * th[mm])).sum()
            if abs(z) > 1e-6:
                ang[i] = (np.degrees(np.angle(z)) / 4 + 90) * smix + ang[i] * (1 - smix) + (rng.integers(0, 2) * 0)
    pitch = (pitch0 * rng.uniform(0.8, 1.7, N)).astype(np.float32)
    G = rng.integers(2, 8, N).astype(np.int32)
    dens = (float(ctx.p("density")) * rng.uniform(0.65, 1.0, N)).astype(np.float32)
    seglen = rng.uniform(4, 22, N).astype(np.float32)
    off = float(ctx.p("offset"))
    phu = (rng.random(N) * 1e4 * off).astype(np.float32)
    phv = (rng.random(N) * 1e4 * off).astype(np.float32)
    blkU = (pitch * rng.uniform(14, 40, N)).astype(np.float32)
    blkV = (pitch * rng.uniform(14, 40, N)).astype(np.float32)
    hier = float(ctx.p("hierarchy"))
    has_coarse = (rng.random(N) < 0.35 + 0.65 * hier).astype(np.float32)
    cpitch = (pitch * rng.uniform(6.5, 11, N) * (1.0 + 0.5 * (1 - hier))).astype(np.float32)
    cx = np.zeros(N, np.float32); cy = np.zeros(N, np.float32)
    for i in range(N):
        yy_, xx_ = np.nonzero(lab[::8, ::8] == i)
        cx[i], cy[i] = (xx_.mean() * 8, yy_.mean() * 8) if len(xx_) else (W / 2, H / 2)
    tilt = float(ctx.p("tilt"))
    tdir = rng.uniform(0, 2 * np.pi, N)
    tmag = rng.uniform(0.3, 1.0, N) * tilt

    # ---- colour families per plate
    nfam = max(1, int(ctx.p("families")))
    sh = mk.source_hues(src, 3) if (src is not None and ctx.p("hue_from_source")) else None
    hues = mk.hue_list(P, nfam, float(ctx.p("hue_spread")), sh)
    roles = [mk.role_colours(P, h, float(ctx.p("chroma")), float(ctx.p("contrast")), v) for h in hues for v in [0.0]]
    # family weights: first family dominates, others are patches
    w = np.array([0.62] + [0.38 / max(1, len(hues) - 1)] * (len(hues) - 1))[:len(hues)]
    fam = rng.choice(len(hues), N, p=w / w.sum())
    if "small" in stats and sh and smix > 0:   # plates take the source region's hue family
        from .. import features as F
        lab_s = F.to_oklab(stats["small"])
        hh = (np.degrees(np.arctan2(lab_s[..., 2], lab_s[..., 1])) % 360)
        for i in range(N):
            mm = stats["lowres"] == i
            if mm.sum() > 10:
                hv = float(np.degrees(np.angle(np.exp(1j * np.deg2rad(hh[mm])).mean())) % 360)
                fam[i] = int(np.argmin([min(abs(hv - h), 360 - abs(hv - h)) for h in hues]))
    vary = rng.random(N).astype(np.float32)
    C = {k: np.stack([r[k] for r in roles]) for k in ("body", "body_hi", "trace", "bright")}
    body = C["body"][fam]; body_hi = C["body_hi"][fam]
    body = body + (body_hi - body) * vary[:, None] * 0.8
    trace = C["trace"][fam] ; bright = C["bright"][fam]
    hot_col = np.stack([mk.role_colours(P, hues[-1], float(ctx.p("chroma")), float(ctx.p("contrast")))["bright"]])[0]
    if "small" in stats and smix > 0:
        # plates take their lightness from the source region (calm tonal map)
        small = stats["small"]
        Ls = small.mean(-1)
        for i in range(N):
            mm = stats["lowres"] == i
            if mm.sum() > 10:
                vary[i] = 0.2 + 0.8 * float(np.clip((Ls[mm].mean() - 0.1) / 0.6, 0, 1))
        body = C["body"][fam] + (C["body_hi"][fam] - C["body"][fam]) * vary[:, None]

    # ---- per-pixel plate coordinates
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    a = np.deg2rad(ang)[lab]
    ca, sa = np.cos(a), np.sin(a)
    u = xx * ca + yy * sa
    v = -xx * sa + yy * ca
    pit = pitch[lab]
    # sub-blocks that turn 90 degrees
    bp = float(ctx.p("blocks"))
    if bp > 0:
        bi = np.floor(u / blkU[lab]); bj = np.floor(v / blkV[lab])
        flip = mk.hash01(bi, bj, lab, 77) < bp
        u, v = np.where(flip, v, u), np.where(flip, u, v)
    d = dens[lab]
    if tone is not None:
        d = np.clip(d * (1 - 0.8 * smix + smix * (0.25 + 1.5 * tone)), 0, 1)
    PU, PV, GG, SL = phu[lab], phv[lab], G[lab], seglen[lab]
    cov1, pad1, h1 = _lanes(u, v, pit, 0.40, GG, d, SL, PU, PV, lab, 1, pit * 2.0, pit * 1.35)
    nl = int(ctx.p("layers"))
    cov2 = pad2 = None
    if nl >= 2:   # crossing layer, coarser, dimmer, in the other direction
        cov2, pad2, _ = _lanes(v, u, pit * 1.7, 0.34, GG + 1, d * 0.8, SL * 1.4, PV, PU, lab, 9, pit * 2.0, pit * 1.2)
    cov3 = None
    if nl >= 3:
        cov3, _, _ = _lanes(u, v, pit * 3.1, 0.30, 1, d * 0.6, SL * 2.0, PU + 40, PV, lab, 21, 0, 0)

    # ---- substrate: plate body with tilt shading, grain
    shade = 1 + 0.35 * tmag[lab] * ((xx - cx[lab]) * np.cos(tdir)[lab] + (yy - cy[lab]) * np.sin(tdir)[lab]) / (0.35 * W)
    tex = mk.fbm((H, W), 40 * max(s, 0.4), rng, 3)
    out = body[lab] * (shade * (0.92 + 0.16 * tex))[..., None]
    cv = lambda c: np.asarray(c, np.float32)

    def paint(base, colour, alpha):
        return base + (colour - base) * np.clip(alpha, 0, 1)[..., None]
    tcol = trace[lab] * (0.80 + 0.45 * h1)[..., None]
    if cov2 is not None:
        out = paint(out, body_hi[lab] * 0.5 + trace[lab] * 0.5, np.maximum(cov2, pad2) * 0.55)
    if cov3 is not None:
        out = paint(out, trace[lab] * 0.8, cov3 * 0.5)
    out = paint(out, tcol, np.maximum(cov1, pad1))
    # ---- coarse trunk layer with halo, vias, a few hot pads
    if hier > 0.02:
        cp = cpitch[lab]
        cc, cpad, ch = _lanes(v, u, cp, 0.20, 1 + (GG % 3), np.clip(d * 0.9, 0, 1), SL * 1.6, PV * 0.7, PU * 0.7, lab, 33, cp * 0.55, cp * 0.42)
        cc = cc * has_coarse[lab]; cpad = cpad * has_coarse[lab]
        solid = np.maximum(cc, cpad)
        halo = cv2.GaussianBlur(solid.astype(np.float32), (0, 0), max(0.8, 1.6 * s * 2.2))
        dk = body[lab] * (0.55 if dark else 1.0) if dark else body[lab] * 0.80
        out = paint(out, dk, np.clip(halo * 1.5, 0, 0.85) * (1 - solid))
        out = paint(out, bright[lab] * (0.85 + 0.2 * ch)[..., None], solid)
        hot = (cpad > 0.5) & (mk.hash01(np.floor(xx / 3), np.floor(yy / 3), 5, 4) < 0.12 * hier)
        out = paint(out, np.asarray(hot_col, np.float32), hot.astype(np.float32))
    # ---- cracks and bevels
    e = np.zeros((H, W), np.uint8)
    e[:, 1:] |= (lab[:, 1:] != lab[:, :-1]); e[1:, :] |= (lab[1:, :] != lab[:-1, :])
    cw = max(1, int(round(2.4 * s * float(ctx.p("crack")) + 0.4)))
    ker = np.ones((cw, cw), np.uint8)
    crack = cv2.dilate(e, ker)
    crack_f = cv2.GaussianBlur(crack.astype(np.float32), (0, 0), 0.7)
    crk = P.canvas * 0.6 if dark else np.asarray(body_hi.mean(0), np.float32) * 0.55
    bev = float(ctx.p("bevel"))
    if bev > 0:
        hi = np.roll(np.roll(crack, cw + 1, 0), cw + 1, 1).astype(np.float32) * (1 - crack)
        hi = cv2.GaussianBlur(hi, (0, 0), 0.8)
        lift = (mk.role_colours(P, hues[0], 1.0, 1.0)["bright"] if dark else np.array([1, 1, 1], np.float32))
        out = paint(out, lift * (0.8 if dark else 1.0), hi * 0.30 * bev)
    out = paint(out, cv(crk), crack_f)
    return mk.surface(out, rng, grain=float(ctx.p("grain")), wear=float(ctx.p("wear")), glow=float(ctx.p("glow")), dark=dark)


def _params(**over):
    p = mk.with_groups(BASE)
    for k, v in over.items():
        p[k] = (v, p[k][1])
    return p


@effect("micro-plates", family="micro", tags=["micro", "plates", "fractured", "traces", "engineered", "precise", "tactile", "nocturnal", "all-over", "light-dark"],
        params=_params())
def micro_plates(ctx):
    """Fractured microstructure (Voronoi plates, oriented trace fills, coarse trunk layer)."""
    return render_plates(ctx)


@effect("micro-plates-card", family="micro", tags=["micro", "plates", "fractured", "traces", "engineered", "precise", "monumental", "small-card", "all-over", "light-dark"],
        params=_params(plates=5, rough=0.9, pitch=0.0075, hierarchy=0.9, layers=2, blocks=0.10))
def micro_plates_card(ctx):
    """Small-card preset: 3-6 broad plates, coarser fine grid, strong visible trunk layer."""
    return render_plates(ctx)


@effect("micro-strata", family="micro", tags=["micro", "plates", "strata", "fractured", "traces", "engineered", "tactile", "all-over", "light-dark"],
        params=_params(boundary="strata", plates=13, rough=0.55, orient=0.2, families=2))
def micro_strata(ctx):
    """Long fractured strips (anisotropic shards) with aligned trace fills."""
    return render_plates(ctx)


@effect("micro-die", family="micro", tags=["micro", "plates", "die", "blocks", "traces", "engineered", "precise", "all-over", "light-dark"],
        params=_params(boundary="blocks", plates=10, rough=0.45, orient=0.0, blocks=0.25, crack=1.4))
def micro_die(ctx):
    """Macro-block die map: rectangular regions with slightly fractured edges, axis-aligned fills."""
    return render_plates(ctx)


@effect("micro-plates-source", family="micro", tags=["micro", "plates", "source", "photo", "traces", "engineered", "urban", "tactile", "light-dark"],
        params=_params(boundary="source", plates=11, rough=0.6, orient=0.3, hue_from_source=True, families=3))
def micro_plates_source(ctx):
    """Plates from a real image: k-means regions, trace orientation from the image's structure,
    trace density and substrate lightness from its tone, hues from its dominant colours."""
    return render_plates(ctx)


@effect("micro-plates-data", family="micro", tags=["micro", "plates", "data", "traces", "engineered", "precise", "light-dark"],
        params=_params(boundary="data", plates=10, rough=0.6))
def micro_plates_data(ctx):
    """Plates from real data: a 2D field banded into regions, an integer label image, or (n,2) points as seeds."""
    return render_plates(ctx)
