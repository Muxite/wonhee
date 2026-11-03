"""proj-trainnet-review: TrainNet v2's review-repair pass as a mosaic.

ctx.data = list of {"state","x","y"} (x,y in 0..1 of the frame): the real 94
markers of one review pass, states confirmed/recovered/reshaped/dropped/deferred.
Each marker owns its Voronoi cell, filled with a hex dot field that swells
towards the marker; its review outcome is the ring drawn around it:
confirmed = thin ring, recovered = bright disc + heavy ring, reshaped = original
and shifted ring joined by a tick, dropped = crossed out over a thinned cell,
deferred = dashed ring. Without data a seeded stand-in scatter is used."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from .. import dotkit as dk
from ..registry import effect

STATES = ("confirmed", "recovered", "reshaped", "dropped", "deferred")
# ramp position of the dot field and of the ring per state
CELL_HI = {"confirmed": 0.50, "recovered": 0.85, "reshaped": 0.62, "dropped": 0.20, "deferred": 0.40}
CELL_LO = {"confirmed": 0.02, "recovered": 0.12, "reshaped": 0.06, "dropped": 0.0, "deferred": 0.04}


def _standin(rng):
    cnt = {"confirmed": 78, "recovered": 6, "reshaped": 7, "dropped": 2, "deferred": 1}
    out = []
    for s, n in cnt.items():
        for _ in range(n):
            out.append({"state": s, "x": rng.random(), "y": rng.random()})
    return out


@effect("proj-trainnet-review", family="dots", tags=["dots", "voronoi", "rings", "data", "all-over", "project"],
        params={"pitch": (0.0105, "dot pitch, fraction of width"), "ring": (1.0, "ring size multiplier"),
                "gap": (3.0, "gap between cells, px at 1500 wide"),
                "glyph": ("ring", "ring | box (detection boxes)"), "lattice": ("hex", "hex | square"), "shape": ("round", "round | square")})
def proj_trainnet_review(ctx):
    W, H = ctx.size
    rng = ctx.rng
    P = ctx.palette
    sc = W / 1500.0
    data = ctx.data if isinstance(ctx.data, list) and ctx.data and isinstance(ctx.data[0], dict) else _standin(rng)
    X = np.array([[d["x"] * W, d["y"] * H] for d in data], np.float32)
    st = [d["state"] for d in data]
    n = len(X)
    tree = cKDTree(X)
    # Dot lattice, each dot tagged with its nearest marker and the Voronoi seam distance.
    lat = (dk.square if str(ctx.p("lattice")) == "square" else dk.hexagonal)(ctx.size, float(ctx.p("pitch")) * W, angle=0.0)
    pts = np.stack([lat.x, lat.y], 1)
    dd, nn = tree.query(pts, k=2)  # nn: (N,2) nearest and 2nd-nearest marker
    d1, d2 = dd[:, 0], dd[:, 1]
    sep = np.linalg.norm(X[nn[:, 0]] - X[nn[:, 1]], axis=1)
    seam = (d2 ** 2 - d1 ** 2) / (2 * np.maximum(sep, 1e-3))
    keep = seam > float(ctx.p("gap")) * sc + 0.5 * lat.cell
    # cell radius per marker (distance to its farthest own dot)
    R = np.full(n, 1.0, np.float32)
    np.maximum.at(R, nn[:, 0], d1.astype(np.float32))
    R = np.maximum(R, 20)
    t = np.clip(d1 / R[nn[:, 0]], 0, 1)
    hi = np.array([CELL_HI[s] for s in st], np.float32)[nn[:, 0]]
    lo = np.array([CELL_LO[s] for s in st], np.float32)[nn[:, 0]]
    ripple = 0.5 + 0.5 * np.cos(2 * math.pi * d1 / (lat.cell * 5.0))
    val = lo + (hi - lo) * (1 - t) ** 1.2
    size_v = np.clip(0.10 + 0.70 * (1 - t) ** 0.9 + 0.20 * ripple * (1 - t), 0, 1)
    ramp = dk.palette_ramp(P)
    cv = dk.new_canvas(ctx.size, P.canvas)
    sub = lat.subset(keep)
    rad = sub.cell * (0.14 + 0.36 * size_v[keep])
    cols = dk.ramp_lookup(ramp, val[keep])
    dk.draw_dots(cv, sub.x, sub.y, rad, cols, 1.0, str(ctx.p("shape")))
    # Marker glyphs, supersampled on an RGBA layer.
    SS = 3
    lay = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    g = ImageDraw.Draw(lay)

    def rgba(c, a=255):
        return tuple(int(255 * float(v)) for v in np.clip(c, 0, 1)) + (a,)

    c_st = rgba(ramp[int(0.70 * 255)])
    c_ac = rgba(P.accent)
    c_dim = rgba(ramp[int(0.45 * 255)])
    c_ink = rgba(P.canvas)
    nnd = tree.query(X, k=2)[0][:, 1]
    for i in range(n):
        x, y = X[i] * SS
        r = min(0.40 * nnd[i], 0.30 * R[i], 64 * sc) * float(ctx.p("ring")) * SS
        r = max(r, 17 * sc * SS)
        s = st[i]
        lw = max(2, int(4.0 * sc * SS))
        def ring(cx, cy, rr, col, w=lw):
            g.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=col, width=w)
        def disc(cx, cy, rr, col):
            g.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=col)
        if str(ctx.p("glyph")) == "box":
            def box(cx, cy, rr, col, w=lw, dash=False):
                if dash:
                    seg = rr * 0.5
                    for a0 in (-1, 1):
                        for t0 in np.arange(-rr, rr, seg * 2):
                            g.line([cx + t0, cy + a0 * rr, cx + min(t0 + seg, rr), cy + a0 * rr], fill=col, width=w)
                            g.line([cx + a0 * rr, cy + t0, cx + a0 * rr, cy + min(t0 + seg, rr)], fill=col, width=w)
                else:
                    g.rectangle([cx - rr, cy - rr, cx + rr, cy + rr], outline=col, width=w)
            bw = r * 1.15
            if s == "confirmed":
                box(x, y, bw, c_st, lw)
            elif s == "recovered":
                g.rectangle([x - bw, y - bw, x + bw, y + bw], fill=c_ac)
                box(x, y, bw * 1.35, c_ac, lw * 2)
            elif s == "reshaped":
                a = rng.random() * 6.28
                ox, oy = x + math.cos(a) * bw * 0.6, y + math.sin(a) * bw * 0.6
                box(x, y, bw, c_dim, lw)
                box(ox, oy, bw * 0.85, c_st, lw * 2)
            elif s == "dropped":
                g.rectangle([x - bw, y - bw, x + bw, y + bw], fill=c_ink)
                box(x, y, bw, c_dim, lw)
                g.line([x - bw, y - bw, x + bw, y + bw], fill=c_dim, width=lw * 2)
                g.line([x - bw, y + bw, x + bw, y - bw], fill=c_dim, width=lw * 2)
            else:
                box(x, y, bw, c_ac, lw * 2, dash=True)
            continue
        if s == "confirmed":
            disc(x, y, r * 0.34, c_st)
            ring(x, y, r, c_st, lw)
        elif s == "recovered":
            disc(x, y, r * 0.62, c_ac)
            ring(x, y, r * 1.0, c_ac, lw * 2)
            ring(x, y, r * 1.35, c_st, max(2, lw - 1))
        elif s == "reshaped":
            a = rng.random() * 6.28
            ox, oy = x + math.cos(a) * r * 0.55, y + math.sin(a) * r * 0.55
            ring(x, y, r, c_dim, max(2, lw - 1))
            disc(ox, oy, r * 0.30, c_st)
            ring(ox, oy, r * 0.8, c_st, lw)
            g.line([x, y, ox, oy], fill=c_st, width=lw)
        elif s == "dropped":
            disc(x, y, r * 1.05, c_ink)
            ring(x, y, r, c_dim, lw)
            k = r * 0.55
            g.line([x - k, y - k, x + k, y + k], fill=c_dim, width=lw * 2)
            g.line([x - k, y + k, x + k, y - k], fill=c_dim, width=lw * 2)
        else:
            disc(x, y, r * 0.30, c_ac)
            for j in range(12):
                a0 = j * 360 / 12
                g.arc([x - r, y - r, x + r, y + r], a0, a0 + 17, fill=c_ac, width=lw * 2)
    lay = lay.resize((W, H), Image.LANCZOS)
    arr = np.asarray(lay, np.float32) / 255.0
    a = arr[..., 3:4]
    rgb = arr[..., :3] / np.maximum(a, 1e-4)
    cv = cv * (1 - a) + rgb * a
    return np.clip(cv, 0, 1).astype(np.float32)
