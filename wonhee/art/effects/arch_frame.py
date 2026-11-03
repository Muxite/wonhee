"""Space-frame perspective: a vector renderer for a lattice of columns, beams,
girders and braces seen from inside, drawn far to near with depth fog.

The lattice has a hierarchy: primary members (thick, every `primary` nodes),
secondary members (thin) and X-braces in random bays. Some bays on the side
planes are glazed: panels in groups, lit or dark, so the frame also reads as a
building under construction or a night hall. The camera is rolled and offset,
never centred on a vanishing point. Drawn at 2x and area-downsampled for
clean anti-aliased linework; thick members stay readable at 288x192.

Moods: engineered, precise, monumental, nocturnal, urban."""
from __future__ import annotations

import cv2
import numpy as np

from .. import microkit as mk
from .. import palette as pal
from ..registry import effect

BASE = {
    "fov": (84.0, "structure: field of view in degrees"),
    "roll": (-1.0, "structure: camera roll in degrees (-1 = seeded, 6..22)"),
    "primary": (3, "structure: every n-th lattice line is a thick primary member"),
    "thick": (1.0, "structure: member thickness multiplier"),
    "brace": (0.28, "structure: share of bays with X-bracing"),
    "glaze": (0.42, "structure: share of side bays glazed with window panels"),
    "lit": (0.35, "source: share of glazed panels that are lit"),
    "depth": (18, "structure: lattice depth in bays"),
    "haze": (0.7, "surface: depth fog 0..1"),
}


def _rot(yaw, pitch, roll):
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float32)
    Rp = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], np.float32)
    Rr = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], np.float32)
    return Rr @ Rp @ Ry


def render_frame(ctx):
    W, H = ctx.size
    rng = ctx.rng
    P = ctx.palette
    dark = P.polarity == "dark"
    SS = 2
    w2, h2 = W * SS, H * SS
    fov = np.deg2rad(float(ctx.p("fov")))
    f = (h2 / 2) / np.tan(fov / 2)
    roll = float(ctx.p("roll"))
    roll = rng.uniform(6, 22) * rng.choice([-1, 1]) if roll < 0 else roll
    R = _rot(np.deg2rad(rng.uniform(-14, 14)), np.deg2rad(rng.uniform(-10, 12)), np.deg2rad(roll))
    cam = np.array([rng.uniform(-1.4, 1.4), rng.uniform(0.6, 2.2), 0.0], np.float32)
    nx, ny, nz = 9, 5, int(ctx.p("depth"))
    prim = max(2, int(ctx.p("primary")))
    brace = float(ctx.p("brace")); glaze = float(ctx.p("glaze")); litp = float(ctx.p("lit"))
    th = float(ctx.p("thick"))
    hs = mk.hue_list(P, 3, float(ctx.p("hue_spread")))
    ch, ct = float(ctx.p("chroma")), float(ctx.p("contrast"))
    if dark:
        bg_top, bg_bot = pal.oklch_to_rgb(0.12, 0.03 * ch, hs[0]), pal.oklch_to_rgb(0.22, 0.05 * ch, hs[0])
        c_prim = pal.oklch_to_rgb(0.60 + 0.08 * (ct - 1), 0.07 * ch, hs[0])
        c_sec = pal.oklch_to_rgb(0.42, 0.07 * ch, hs[0])
        c_brace = pal.oklch_to_rgb(0.72, 0.09 * ch, hs[0] - 25)
        glass = pal.oklch_to_rgb(0.18, 0.04 * ch, hs[0]); glass2 = pal.oklch_to_rgb(0.25, 0.06 * ch, hs[0])
        lit_a = pal.oklch_to_rgb(0.88, 0.075 * ch, hs[-1]); lit_b = pal.oklch_to_rgb(0.80, 0.08 * ch, hs[1 % len(hs)])
        fog = bg_bot
    else:
        bg_top, bg_bot = pal.oklch_to_rgb(0.97, 0.01 * ch, hs[0]), pal.oklch_to_rgb(0.88, 0.04 * ch, hs[0])
        c_prim = pal.oklch_to_rgb(0.30 - 0.06 * (ct - 1), 0.07 * ch, hs[0])
        c_sec = pal.oklch_to_rgb(0.52, 0.07 * ch, hs[0])
        c_brace = pal.oklch_to_rgb(0.40, 0.11 * ch, hs[0] - 25)
        glass = pal.oklch_to_rgb(0.74, 0.05 * ch, hs[0]); glass2 = pal.oklch_to_rgb(0.84, 0.05 * ch, hs[0])
        lit_a = pal.oklch_to_rgb(0.92, 0.03 * ch, hs[-1]); lit_b = pal.oklch_to_rgb(0.62, 0.11 * ch, hs[1 % len(hs)])
        fog = bg_bot
    # background gradient
    t = np.linspace(0, 1, h2, dtype=np.float32)[:, None, None]
    img = (bg_top + (bg_bot - bg_top) * t) * np.ones((1, w2, 1), np.float32)
    img = img.astype(np.float32)

    def proj(p):
        q = (p - cam) @ R.T
        return q

    def to_px(q):
        z = np.maximum(q[..., 2], 0.05)
        return np.stack([w2 / 2 + f * q[..., 0] / z, h2 / 2 - f * q[..., 1] / z], -1), q[..., 2]

    prims = []   # (depth, kind, pts, colour, thickness_world, extra)
    X = np.arange(-nx // 2, nx // 2 + 1)
    Yv = np.arange(0, ny + 1)
    Z = np.arange(1, nz + 1)
    sp = 1.0
    def node(i, j, k):
        return np.array([i * sp * 1.15, j * sp * 0.95, k * sp * 1.1], np.float32)

    for k in Z:
        for i in X:
            primary_x = (i % prim == 0)
            primary_z = (k % prim == 0)
            for j in Yv:
                a = node(i, j, k)
                if j < ny:   # columns
                    prims.append(("l", a, node(i, j + 1, k), 0.095 * th if (primary_x and primary_z) else 0.030 * th, c_prim if (primary_x and primary_z) else c_sec))
                if i < X[-1]:   # beams along x
                    pr = primary_z and (j % 2 == 0)
                    prims.append(("l", a, node(i + 1, j, k), 0.075 * th if pr else 0.024 * th, c_prim if pr else c_sec))
                if k < Z[-1]:   # girders along z
                    pr = primary_x and (j % 2 == 0)
                    prims.append(("l", a, node(i, j, k + 1), 0.075 * th if pr else 0.024 * th, c_prim if pr else c_sec))
    # bays: braces and glazing panels
    for k in Z[:-1]:
        for i in X[:-1]:
            for j in range(ny):
                r = rng.random()
                if r < brace:
                    a, b, c, d = node(i, j, k), node(i + 1, j + 1, k), node(i + 1, j, k), node(i, j + 1, k)
                    prims.append(("l", a, b, 0.032 * th, c_brace)); prims.append(("l", c, d, 0.032 * th, c_brace))
    glaze_planes = []
    for k in Z[:-1]:
        for j in range(ny):
            for side, i in ((0, X[0]), (1, X[-1])):
                if rng.random() < glaze:
                    glaze_planes.append(("x", i, j, k))
    for k in Z[:-1]:
        for i in X[:-1]:
            if rng.random() < glaze * 0.5:
                glaze_planes.append(("y", i, ny, k))      # roof glazing / ceiling
            if rng.random() < glaze * 0.4:
                glaze_planes.append(("y", i, 0, k))       # floor panels
    for kind, i, j, k in glaze_planes:
        if kind == "x":
            pts = np.stack([node(i, j, k), node(i, j + 1, k), node(i, j + 1, k + 1), node(i, j, k + 1)])
        else:
            pts = np.stack([node(i, j, k), node(i + 1, j, k), node(i + 1, j, k + 1), node(i, j, k + 1)])
        gi, gj = (k // 2, j // 2) if kind == "x" else (k // 2, i // 2)
        lit = mk.hash01(gi, gj, i if kind == "x" else j, 11) * 0.6 + mk.hash01(i, j, k, 12) * 0.4 < litp
        prims.append(("q", pts, lit, kind))

    # ---- painter: far to near
    def depth_of(pr):
        pts = pr[1:3] if pr[0] == "l" else pr[1]
        q = proj(np.asarray(pts, np.float32))
        return -q[..., 2].mean()
    order = sorted(prims, key=depth_of)
    fogk = float(ctx.p("haze"))
    near = 0.25
    for pr in order:
        if pr[0] == "q":
            q = proj(pr[1])
            if (q[:, 2] < near).any():
                continue
            xy, z = to_px(q)
            zm = float(z.mean())
            fg = 1 - np.exp(-zm / 14.0 * fogk * 1.3)
            wallc = c_sec * (0.55 if dark else 1.0) + (fog - c_sec * 0.55) * fg * 0.8
            cv2.fillConvexPoly(img, xy.astype(np.int32), tuple(float(c) for c in wallc), cv2.LINE_AA)
            p0, p1, p3 = pr[1][0], pr[1][1], pr[1][3]
            nxw, nyw = 3, 2
            for wi in range(nxw):
                for wj in range(nyw):
                    m = 0.14
                    s0, s1 = (wi + m) / nxw, (wi + 1 - m) / nxw
                    t0, t1 = (wj + m) / nyw, (wj + 1 - m) / nyw
                    cs = np.stack([p0 + (p3 - p0) * s0 + (p1 - p0) * t0, p0 + (p3 - p0) * s1 + (p1 - p0) * t0,
                                   p0 + (p3 - p0) * s1 + (p1 - p0) * t1, p0 + (p3 - p0) * s0 + (p1 - p0) * t1])
                    xyw, _ = to_px(proj(cs))
                    litw = pr[2] and (mk.hash01(wi, wj, int(pr[1][0][0] * 7 + pr[1][0][2] * 13), 5) < 0.8)
                    base = (lit_a if mk.hash01(wi, wj, int(pr[1][0][2] * 5), 6) < 0.7 else lit_b) if litw else (glass if pr[3] == "x" else glass2)
                    col = base + (fog - base) * fg * (0.5 if litw else 0.85)
                    cv2.fillConvexPoly(img, xyw.astype(np.int32), tuple(float(c) for c in col), cv2.LINE_AA)
            continue
        _, a, b, wthick, colr = pr
        qa, qb = proj(a), proj(b)
        if qa[2] < near and qb[2] < near:
            continue
        if qa[2] < near or qb[2] < near:   # clip at the near plane
            if qa[2] < near:
                qa, qb = qb, qa
            tt = (near - qa[2]) / (qb[2] - qa[2])
            qb = qa + (qb - qa) * tt
        (pa, pb), (za, zb) = to_px(np.stack([qa, qb]))
        zm = (za + zb) / 2
        width = max(1.0, wthick * f / max(zm, 0.3))
        fg = 1 - np.exp(-zm / 14.0 * fogk * 1.3)
        col = colr + (fog - colr) * fg * 0.9
        cv2.line(img, tuple(int(round(v)) for v in pa), tuple(int(round(v)) for v in pb), tuple(float(c) for c in col), int(round(width)), cv2.LINE_AA)
    out = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    hot = None
    return mk.surface(out, rng, grain=float(ctx.p("grain")), wear=float(ctx.p("wear")), glow=float(ctx.p("glow")), dark=dark, hot=hot)


def _params(**over):
    p = mk.with_groups(BASE, source=False)
    for k, v in over.items():
        p[k] = (v, p[k][1])
    return p


@effect("arch-space-frame", family="arch", tags=["arch", "truss", "lattice", "perspective", "engineered", "precise", "monumental", "nocturnal", "all-over", "light-dark"],
        params=_params())
def arch_space_frame(ctx):
    """Rolled-camera space frame: primary/secondary members, X-braces, glazed bays lit in groups."""
    return render_frame(ctx)


@effect("arch-truss-hall", family="arch", tags=["arch", "truss", "lattice", "perspective", "engineered", "monumental", "tactile", "all-over", "light-dark"],
        params=_params(brace=0.55, glaze=0.12, thick=1.35, primary=2, fov=92.0))
def arch_truss_hall(ctx):
    """Heavier, braced, mostly unglazed frame: truss hall / scaffold."""
    return render_frame(ctx)
