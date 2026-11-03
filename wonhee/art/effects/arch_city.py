"""Architectural perspective systems: a tiny vectorised ray caster over a field
of towers, with a parametric curtain-wall facade on every face.

Scenes (`scene`): `canyon` (worm's-eye, towers converging overhead: monumental),
`aerial` (oblique high view over a block grid with roofs), `street` (corridor of
facades receding to a vanishing point near a frame corner: two-point feel).
Every face carries a facade: floor height and bay width per building, thick
structural piers every K bays (so windows come in groups), service floors /
belt courses every M floors, mullions, glass reflecting a sky gradient and a lit
state per window. The lit-window distribution and its mask shape are separate
parameters (`lit`, `lit_shape`); with `ctx.source` (a photo) the lit pattern
follows the photo's tone at each window's own projected position, and window
hue follows the photo's local colour (`source_mix`).

Windows smaller than ~3 px fade to their mean tone (no moire); depth haze keeps
near/far readable; both dark (night: glass dark, lit windows warm) and light
(day: pale concrete, glass in the structure hue) polarities.

Moods: urban, monumental, nocturnal, precise, engineered."""
from __future__ import annotations

import cv2
import numpy as np

from .. import features as F
from .. import microkit as mk
from .. import palette as pal
from ..registry import effect

BASE = {
    "scene": ("canyon", "scene: canyon | aerial | street | wall"),
    "tilt": (-1.0, "structure: camera pitch in degrees (-1 = scene default)"),
    "yaw": (-1.0, "structure: camera yaw in degrees (-1 = scene default); off-axis = two-point"),
    "fov": (74.0, "structure: vertical field of view in degrees"),
    "floors": (1.0, "structure: floor density multiplier (higher = more, smaller floors)"),
    "bay": (1.0, "structure: bay width multiplier"),
    "group": (1.0, "structure: window group size multiplier (piers every K bays, K 3..6)"),
    "heights": (1.0, "structure: tower height multiplier"),
    "pier": (0.5, "structure: thickness of structural piers and belt courses 0..1"),
    "lit": (0.42, "source: share of windows lit 0..1 (night) / sunlit glass share (day)"),
    "lit_shape": ("mixed", "source: lit mask shape per building: mixed | scatter | floors | columns | stair | blocks"),
    "haze": (0.55, "surface: depth haze 0..1"),
}


def _boxes(scene, rng, hmul):
    """-> list of (x0, x1, z0, z1, h)."""
    B = []
    if scene == "canyon":
        S, g, st = 3.6, 0.55, 2.4
        for side in (-1, 1):
            for k in range(3):
                for j in range(9):
                    x0 = side * (st / 2 + k * (S + g)); x1 = x0 + side * S
                    z0 = 1.0 + j * (S + g)
                    h = rng.uniform(9, 26) * hmul * (1.0 + 0.12 * k)
                    B.append((min(x0, x1), max(x0, x1), z0, z0 + S, h))
    elif scene == "aerial":
        S, g = 2.6, 0.7
        for i in range(-6, 7):
            for j in range(0, 22):
                if abs(i * (S + g)) < 0.9:
                    continue
                if rng.random() < 0.08:
                    continue
                x0, z0 = i * (S + g) - S / 2, 3.0 + j * (S + g)
                h = (rng.uniform(1.5, 9) ** 1.0) * hmul * (1.3 if rng.random() < 0.15 else 1.0)
                B.append((x0, x0 + S, z0, z0 + S, h))
    elif scene == "wall":
        x = -40.0
        while x < 40:
            w = rng.uniform(6, 13)
            z0 = 9.0 + rng.uniform(-1.6, 1.6)
            B.append((x, x + w, z0, z0 + 3.0, rng.uniform(26, 40) * hmul))
            x += w
    else:   # street: continuous wall on the right, towers and gaps on the left, corridor
        z = 1.0
        while z < 48:
            d = rng.uniform(2.6, 5.0)
            gap = 0.0 if rng.random() < 0.7 else 0.5
            B.append((1.8, 5.8, z, z + d, rng.uniform(8, 22) * hmul))
            B.append((-5.8, -1.8, z + rng.uniform(-1, 1), z + d + rng.uniform(-1, 1), rng.uniform(5, 18) * hmul))
            B.append((5.8, 9.8, z, z + d, rng.uniform(12, 30) * hmul))
            z += d + gap
    return B


def render_city(ctx, night_default=True):
    W, H = ctx.size
    rng = ctx.rng
    P = ctx.palette
    dark = P.polarity == "dark"
    scene = str(ctx.p("scene"))
    s = W / 1500.0
    hmul = float(ctx.p("heights"))
    B = _boxes(scene, rng, hmul)
    nb = len(B)
    # ---- camera
    pitch = float(ctx.p("tilt")); yaw = float(ctx.p("yaw"))
    default = {"canyon": (24.0, 16.0, 0.5), "aerial": (-34.0, 0.0, 15.0), "street": (6.0, 20.0, 1.7), "wall": (2.0, 16.0, 9.0)}[scene]
    pitch = default[0] if pitch < -0.5 and pitch == -1.0 else pitch
    yaw = default[1] if yaw == -1.0 else yaw
    if scene == "street":
        yaw = abs(yaw) * (1 if rng.random() < 0.5 else -1)
    cam_y = default[2]
    fov = np.deg2rad(float(ctx.p("fov")))
    th, ph = np.deg2rad(pitch), np.deg2rad(yaw)
    fwd = np.array([np.sin(ph) * np.cos(th), np.sin(th), np.cos(ph) * np.cos(th)], np.float32)
    right = np.array([np.cos(ph), 0, -np.sin(ph)], np.float32)
    up = np.cross(fwd, right).astype(np.float32)
    up = -up if up[1] < 0 else up
    tan = np.tan(fov / 2)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    nx = ((xx + 0.5) / W * 2 - 1) * tan * W / H
    ny = (1 - (yy + 0.5) / H * 2) * tan
    d = fwd[None, None, :] + nx[..., None] * right + ny[..., None] * up
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    dx, dy, dz = d[..., 0].copy(), d[..., 1].copy(), d[..., 2].copy()
    ox, oy, oz = 0.0, cam_y, 0.0
    eps = 1e-6
    idx = lambda a: np.where(np.abs(a) < eps, np.where(a < 0, -eps, eps), a)
    ix, iy, iz = 1 / idx(dx), 1 / idx(dy), 1 / idx(dz)
    best = np.full((H, W), 1e9, np.float32)
    bid = np.full((H, W), -1, np.int16)
    face = np.zeros((H, W), np.int8)       # 0 = x face, 1 = z face, 2 = roof
    for b, (x0, x1, z0, z1, h) in enumerate(B):
        # screen ROI of the box (full frame when a corner is behind the camera)
        cs = np.array([[x, y, z] for x in (x0, x1) for y in (0.0, h) for z in (z0, z1)], np.float32) - [ox, oy, oz]
        zr = cs @ fwd
        if (zr > 0.05).all():
            sx = ((cs @ right) / zr / (tan * W / H) * 0.5 + 0.5) * W
            sy = (0.5 - (cs @ up) / zr / tan * 0.5) * H
            xa, xb = int(max(0, np.floor(sx.min()) - 2)), int(min(W, np.ceil(sx.max()) + 3))
            ya, yb = int(max(0, np.floor(sy.min()) - 2)), int(min(H, np.ceil(sy.max()) + 3))
            if xb <= xa or yb <= ya:
                continue
        else:
            xa, xb, ya, yb = 0, W, 0, H
        sl = (slice(ya, yb), slice(xa, xb))
        tx0, tx1 = (x0 - ox) * ix[sl], (x1 - ox) * ix[sl]
        tz0, tz1 = (z0 - oz) * iz[sl], (z1 - oz) * iz[sl]
        ty0, ty1 = (0 - oy) * iy[sl], (h - oy) * iy[sl]
        txn, txf = np.minimum(tx0, tx1), np.maximum(tx0, tx1)
        tzn, tzf = np.minimum(tz0, tz1), np.maximum(tz0, tz1)
        tyn, tyf = np.minimum(ty0, ty1), np.maximum(ty0, ty1)
        tn = np.maximum(np.maximum(txn, tzn), tyn)
        tf = np.minimum(np.minimum(txf, tzf), tyf)
        hit = (tn < tf) & (tn > 0.05) & (tn < best[sl])
        if not hit.any():
            continue
        best[sl] = np.where(hit, tn, best[sl])
        bid[sl] = np.where(hit, b, bid[sl])
        fc = np.where((txn >= tzn) & (txn >= tyn), 0, np.where(tzn >= tyn, 1, 2)).astype(np.int8)
        face[sl] = np.where(hit, fc, face[sl])
    # ground
    tg = np.where(dy < -1e-4, -oy / np.minimum(dy, -1e-4), 1e9).astype(np.float32)
    gmask = (tg < best)
    hitany = bid >= 0
    t = np.where(gmask, tg, best)
    px_, py_, pz_ = ox + t * dx, oy + t * dy, oz + t * dz

    # ---- per building attributes
    Bn = np.array(B, np.float32)
    flo = float(ctx.p("floors")); bay = float(ctx.p("bay")); grp = float(ctx.p("group"))
    hf = (rng.uniform(0.26, 0.44, nb) / flo).astype(np.float32)
    wb = (rng.uniform(0.30, 0.55, nb) / flo * bay).astype(np.float32)
    K = np.clip(np.round(rng.integers(3, 7, nb) * grp), 2, 9).astype(np.int32)
    M = rng.integers(4, 9, nb).astype(np.int32)
    gw = rng.integers(1, 4, nb).astype(np.int32)
    gh = rng.integers(1, 4, nb).astype(np.int32)
    wfrac = rng.uniform(0.62, 0.84, nb).astype(np.float32)
    litp = np.clip(float(ctx.p("lit")) * rng.uniform(0.5, 1.5, nb), 0.02, 0.98).astype(np.float32)
    shapes = ["scatter", "floors", "columns", "stair", "blocks"]
    ls = str(ctx.p("lit_shape"))
    shp = rng.integers(0, 5, nb) if ls == "mixed" else np.full(nb, shapes.index(ls) if ls in shapes else 0)
    sphase = rng.random(nb).astype(np.float32) * 100
    tint = rng.random(nb).astype(np.float32)
    pw = float(ctx.p("pier"))
    BI = np.clip(bid, 0, nb - 1).astype(np.int32)

    # ---- face coordinates
    fx = (face == 0)
    u = np.where(fx, pz_, px_).astype(np.float32)
    v = np.where(face == 2, pz_, py_).astype(np.float32)
    hfp, wbp, Kp, Mp, gwp, ghp, wfp = hf[BI], wb[BI], K[BI], M[BI], gw[BI], gh[BI], wfrac[BI]
    cu = u / wbp + sphase[BI]
    cvv = v / hfp
    i = np.floor(cu); j = np.floor(cvv)
    fu = cu - i; fvv = cvv - j
    # footprint in cell units (pixel size / cell size)
    gux = np.abs(np.gradient(cu, axis=1)) + np.abs(np.gradient(cu, axis=0))
    gvx = np.abs(np.gradient(cvv, axis=1)) + np.abs(np.gradient(cvv, axis=0))
    gux = np.clip(gux, 1e-4, 1.5); gvx = np.clip(gvx, 1e-4, 1.5)
    px_cell = 1.0 / np.maximum(gux, gvx)
    # structural piers every K bays: solid cell
    ib = np.mod(i, Kp)
    is_pier = (ib == 0)
    # service floors every M floors: solid belt
    is_belt = (np.mod(j, Mp) == 0)
    mx = (1 - wfp) / 2 * (1 + 0.9 * pw)
    my0, my1 = 0.16 + 0.12 * pw, 0.86 - 0.08 * pw
    cov = np.clip((fu - mx) / gux + 0.5, 0, 1) * np.clip((1 - mx - fu) / gux + 0.5, 0, 1) \
        * np.clip((fvv - my0) / gvx + 0.5, 0, 1) * np.clip((my1 - fvv) / gvx + 0.5, 0, 1)
    cov = cov * (~is_pier) * (~is_belt)
    # mullion: thin centre bar in wide windows
    mull = np.clip(1 - np.abs(fu - 0.5) / np.maximum(gux, 0.04) * 3.0, 0, 1) * (wbp > 0.75) * cov * 0.0
    # ---- lit state
    gi = np.floor(i / gwp); gj = np.floor(j / ghp)
    bk = BI
    hg = mk.hash01(gi, gj, bk, 3)
    hw = mk.hash01(i, j, bk, 4)
    sp = shp[BI]
    shape_m = np.ones((H, W), np.float32)
    shape_m = np.where(sp == 1, (mk.hash01(j, bk, 5) < 0.5).astype(np.float32) * 1.7 + 0.15, shape_m)
    shape_m = np.where(sp == 2, (mk.hash01(i // 2, bk, 6) < 0.5).astype(np.float32) * 1.7 + 0.15, shape_m)
    shape_m = np.where(sp == 3, np.clip(0.2 + 1.8 * (((j + i * 0.6) % 14) < 7), 0, 2), shape_m)
    shape_m = np.where(sp == 4, (mk.hash01(i // 4, j // 3, bk, 7) < 0.45).astype(np.float32) * 1.8 + 0.1, shape_m)
    lit_p = np.clip(litp[BI] * shape_m, 0, 1)
    src = mk.fit_source(ctx)
    smix = float(ctx.p("source_mix", 1.0))
    local_hue = None
    if src is not None and smix > 0:
        # project each window's own centre into the frame and read the photo there
        uc = (i + 0.5 - sphase[BI]) * wbp
        vc = (j + 0.5) * hfp
        cx_ = np.where(fx, px_, uc); cz_ = np.where(fx, uc, pz_); cy_ = np.where(face == 2, py_, vc)
        rx, ry, rz = cx_ - ox, cy_ - oy, cz_ - oz
        xr = rx * right[0] + ry * right[1] + rz * right[2]
        yr = rx * up[0] + ry * up[1] + rz * up[2]
        zr = np.maximum(rx * fwd[0] + ry * fwd[1] + rz * fwd[2], 0.05)
        sx = (xr / zr / (tan * W / H) * 0.5 + 0.5) * W
        sy = (0.5 - yr / zr / tan * 0.5) * H
        Ls = F.luminance(src)
        tone = cv2.GaussianBlur(F.stretch(F.local_equalize(Ls, 2.5, mix=0.6), 1, 99), (0, 0), 3 * max(s, 0.3))
        tw = cv2.remap(tone.astype(np.float32), sx.astype(np.float32), sy.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        lit_p = np.clip(lit_p * (1 - smix) + smix * mk.smoothstep(0.30, 0.70, tw) * (0.55 + 0.45 * min(1.0, float(ctx.p("lit")))), 0, 1)
        lab_s = F.to_oklab(cv2.GaussianBlur(src, (0, 0), 6 * max(s, 0.3)))
        local_hue = np.degrees(np.arctan2(lab_s[..., 2], lab_s[..., 1])) % 360
        local_hue = cv2.remap(local_hue.astype(np.float32), sx.astype(np.float32), sy.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    lit = ((0.55 * hg + 0.45 * hw) < lit_p).astype(np.float32)

    # ---- colours
    rr = mk.role_colours
    hs = mk.hue_list(P, 3, float(ctx.p("hue_spread")))
    ch, ct = float(ctx.p("chroma")), float(ctx.p("contrast"))
    cA = rr(P, hs[0], ch, ct); cB = rr(P, hs[min(1, len(hs) - 1)], ch, ct); cC = rr(P, hs[-1], ch, ct)
    if dark:
        facex = pal.oklch_to_rgb(0.30, 0.045 * ch, hs[0]); facez = pal.oklch_to_rgb(0.24, 0.04 * ch, hs[0])
        pier_c = pal.oklch_to_rgb(0.36, 0.05 * ch, hs[0]); belt_c = pal.oklch_to_rgb(0.40, 0.05 * ch, hs[0])
        glass = pal.oklch_to_rgb(0.15, 0.03 * ch, hs[0]); glass_hi = pal.oklch_to_rgb(0.22, 0.05 * ch, hs[0])
        lit_a = pal.oklch_to_rgb(0.90, 0.075 * ch, hs[-1]); lit_b = pal.oklch_to_rgb(0.80 + 0.05 * (ct - 1), 0.075 * ch, hs[1 % len(hs)] if len(hs) > 1 else hs[0] + 30)
        sky_lo = pal.oklch_to_rgb(0.30, 0.06 * ch, hs[0]); sky_hi = pal.oklch_to_rgb(0.13, 0.03 * ch, hs[0])
        haze_c = pal.oklch_to_rgb(0.30, 0.06 * ch, hs[0])
        roof_c = pal.oklch_to_rgb(0.27, 0.04 * ch, hs[0])
    else:
        facex = pal.oklch_to_rgb(0.90, 0.02 * ch, hs[0]); facez = pal.oklch_to_rgb(0.82, 0.025 * ch, hs[0])
        pier_c = pal.oklch_to_rgb(0.96, 0.012 * ch, hs[0]); belt_c = pal.oklch_to_rgb(0.97, 0.01 * ch, hs[0])
        glass = pal.oklch_to_rgb(0.45, 0.07 * ch, hs[0]); glass_hi = pal.oklch_to_rgb(0.62, 0.08 * ch, hs[0])
        lit_a = pal.oklch_to_rgb(0.80 + 0.04 * (ct - 1), 0.06 * ch, hs[0] + 20); lit_b = pal.oklch_to_rgb(0.72, 0.08 * ch, hs[1 % len(hs)] if len(hs) > 1 else hs[0] + 30)
        sky_lo = pal.oklch_to_rgb(0.92, 0.025 * ch, hs[0]); sky_hi = pal.oklch_to_rgb(0.84, 0.05 * ch, hs[0])
        haze_c = pal.oklch_to_rgb(0.92, 0.025 * ch, hs[0])
        roof_c = pal.oklch_to_rgb(0.74, 0.02 * ch, hs[0])
    face_c = np.where(fx[..., None], facex, facez).astype(np.float32)
    # per-building tone drift
    face_c = face_c * (0.88 + 0.24 * tint[BI])[..., None]
    # vertical light falloff: brighter on upper floors in dark mode (city glow from below is dimmer)
    # glass: reflect a gradient by height within the building + per-window variation
    gmix = np.clip(fvv, 0, 1) * 0.7 + 0.3 * hw
    glass_c = glass + (glass_hi - glass) * gmix[..., None]
    # lit colour: two hue variants per window, hue follows the photo where available
    pick = (mk.hash01(i, j, bk, 8) < 0.12 + 0.25 * tint[BI])
    lc = np.where(pick[..., None], lit_b, lit_a).astype(np.float32)
    if local_hue is not None and smix > 0:
        tc = pal.oklch_to_rgb  # per-pixel hue: approximate with oklch_field
        lcs = F.oklch_field(np.full((H, W), 0.80 if dark else 0.74, np.float32), np.full((H, W), 0.085 * ch, np.float32), local_hue)
        lc = lc * (1 - 0.45 * smix) + lcs * 0.45 * smix
    lc = lc * (0.82 + 0.3 * mk.hash01(i, j, bk, 9))[..., None]
    win_c = glass_c + (lc - glass_c) * lit[..., None]
    wall = face_c.copy()
    wall = np.where(is_pier[..., None], pier_c * (0.9 + 0.1 * face_c / np.maximum(face_c.max(), 1e-3)), wall)
    wall = np.where(is_belt[..., None] & (~is_pier[..., None]), belt_c, wall)
    # sill / shadow lines under belt
    fac = wall * (1 - cov[..., None]) + win_c * cov[..., None]
    # average tone for sub-3px detail (no moire)
    avg = wall * (1 - 0.62 * np.clip(lit_p * 0 + 0.55, 0, 1)[..., None]) + win_c * 0 + (glass_c + (lc - glass_c) * lit_p[..., None]) * 0.62
    fade = mk.smoothstep(1.6, 4.5, px_cell)[..., None]
    fac = avg + (fac - avg) * fade
    # roofs
    ra = np.mod(np.floor(u / 0.9) + np.floor(v / 0.9) * 7, 5)
    roof = roof_c * (0.85 + 0.15 * (ra / 4))[..., None]
    surf = np.where((face == 2)[..., None], roof, fac)
    # ground / street
    gx_ = np.abs(np.mod(px_, 3.2) - 1.6); gz_ = np.abs(np.mod(pz_, 3.2) - 1.6)
    gl = np.clip(1 - np.minimum(gx_, gz_) / 0.12, 0, 1)
    ground = (sky_lo * 0.8 + (haze_c - sky_lo) * 0.4)[None, None, :] * np.ones((H, W, 1), np.float32)
    ground = ground * (1 - 0.35 * gl[..., None]) + (lit_a if dark else pier_c)[None, None, :] * (0.12 * gl[..., None])
    surf = np.where(gmask[..., None] & ~hitany[..., None], ground, surf)
    # depth haze
    hz = 1 - np.exp(-t / (14.0 + 40 * (1 - float(ctx.p("haze")))) * float(ctx.p("haze")) * 1.4)
    surf = surf + (haze_c - surf) * (hz * 0.8)[..., None]
    # sky
    sky_t = np.clip(1 - 2.2 * np.maximum(dy, 0), 0, 1)[..., None]
    sky = sky_hi + (sky_lo - sky_hi) * (sky_t ** 1.6)
    cl = mk.fbm((H, W), 120 * max(s, 0.4), rng, 3)
    sky = sky * (0.94 + 0.12 * cl)[..., None]
    out = np.where((hitany | gmask)[..., None], surf, sky).astype(np.float32)
    hot = (lit * cov * (hitany) * fade[..., 0]).astype(np.float32) if dark else None
    return mk.surface(out, rng, grain=float(ctx.p("grain")), wear=float(ctx.p("wear")), glow=float(ctx.p("glow")), dark=dark, hot=hot)


def _params(**over):
    p = mk.with_groups(BASE, source=True)
    for k, v in over.items():
        p[k] = (v, p[k][1])
    return p


@effect("arch-canyon", family="arch", tags=["arch", "city", "towers", "perspective", "urban", "monumental", "nocturnal", "precise", "all-over", "light-dark"],
        params=_params(scene="canyon"))
def arch_canyon(ctx):
    """Worm's-eye tower canyon: converging curtain-wall facades, grouped windows, piers."""
    return render_city(ctx)


@effect("arch-aerial", family="arch", tags=["arch", "city", "towers", "perspective", "urban", "engineered", "nocturnal", "all-over", "light-dark"],
        params=_params(scene="aerial", tilt=-24.0, fov=66.0, lit=0.5, floors=0.9, heights=1.5))
def arch_aerial(ctx):
    """Oblique aerial block grid: roofs, facades, dense and even."""
    return render_city(ctx)


@effect("arch-street", family="arch", tags=["arch", "city", "facade", "perspective", "urban", "monumental", "precise", "all-over", "light-dark"],
        params=_params(scene="street", tilt=6.0, yaw=24.0, fov=80.0))
def arch_street(ctx):
    """Two-point street corridor of facades receding off-centre."""
    return render_city(ctx)


@effect("arch-wall", family="arch", tags=["arch", "facade", "curtain-wall", "perspective", "urban", "monumental", "precise", "engineered", "all-over", "light-dark"],
        params=_params(scene="wall", tilt=2.0, yaw=16.0, fov=62.0, floors=1.0, lit=0.5))
def arch_wall(ctx):
    """A frame-filling curtain wall seen off-axis: facades with different grids and setbacks, piers, belts, grouped windows."""
    return render_city(ctx)


@effect("arch-night-photo", family="arch", tags=["arch", "city", "photo", "source", "perspective", "urban", "nocturnal", "tactile", "light-dark"],
        params=_params(scene="wall", tilt=2.0, yaw=14.0, fov=62.0, lit=0.9, floors=1.0, lit_shape="scatter", source_mix=1.0))
def arch_night_photo(ctx):
    """A photo re-lit as a tower field: each window is lit by the photo's tone at its own
    projected position, hued by the photo's local colour. Needs ctx.source (a procedural fallback otherwise)."""
    return render_city(ctx)
