"""proj-euglena-bench: the real euglena benchmark (38 tests x 9 model/variant
pairs x 3 repeat runs, 1,026 runs) drawn straight from the data, no labels.

ctx.data: {"scores": [test][col][run], "costs": [test][col][run]} (the
muksite benchData runs; also accepted: {"matrix": [test][col]}).
Modes
  heat     the score matrix as big flat cells that bleed off every edge
           (the owner's gold-standard heatmap), magma / ember / cyan ramp
  runs     transposed: one horizontal stripe per configuration, one column
           per test, each cell split into its 3 repeat runs
  skyline  per configuration a band; each run is a bar whose height is its score
  scatter  all 1026 runs as glowing dots, x = log cost, y = score
Ramps never reach pure black or white."""
from __future__ import annotations

import numpy as np

from ..registry import effect

RAMPS = {
    "magma": [(0.0, (0.11, 0.06, 0.20)), (0.22, (0.32, 0.08, 0.46)), (0.48, (0.70, 0.20, 0.47)),
              (0.74, (0.97, 0.52, 0.38)), (0.90, (0.99, 0.80, 0.55)), (1.0, (0.97, 0.93, 0.73))],
    "ember": [(0.0, (0.12, 0.07, 0.05)), (0.3, (0.38, 0.14, 0.06)), (0.6, (0.80, 0.38, 0.08)),
              (0.85, (0.96, 0.66, 0.20)), (1.0, (0.98, 0.88, 0.58))],
    "cyan": [(0.0, (0.05, 0.12, 0.19)), (0.3, (0.07, 0.30, 0.45)), (0.6, (0.12, 0.55, 0.74)),
             (0.85, (0.42, 0.80, 0.92)), (1.0, (0.80, 0.95, 0.98))],
}


def ramp(name, v):
    st = RAMPS[name]
    xs = np.array([s[0] for s in st]); cs = np.array([s[1] for s in st], np.float32)
    v = np.clip(v, 0, 1)
    return np.stack([np.interp(v, xs, cs[:, i]) for i in range(3)], -1).astype(np.float32)


def _data(ctx):
    d = ctx.data
    if d is None:
        raise ValueError("proj-euglena-bench requires ctx.data: a score matrix (38 x 9) or {'scores','costs'}")
    if isinstance(d, dict) and "scores" in d:
        return np.asarray(d["scores"], np.float32), np.asarray(d["costs"], np.float32)
    m = np.asarray(d["matrix"] if isinstance(d, dict) else d, np.float32)
    return np.repeat(m[..., None], 3, 2), np.full(m.shape + (3,), 0.01, np.float32)


@effect("proj-euglena-bench", family="dots", tags=["euglena", "benchmark", "heatmap", "data", "all-over", "project"],
        params={"mode": ("heat", "heat | runs | skyline | scatter"), "ramp": ("magma", "magma | ember | cyan"),
                "col_w": (0.19, "heat: cell width, fraction of frame width"), "row_h": (0.068, "heat: cell height, fraction of frame height"),
                "x0": (0.6, "heat: first column offset in cells"), "y0": (3.3, "heat: first row offset in cells"),
                "gap": (2.0, "gap px"), "order": ("data", "data | mean (rows sorted by mean score)"),
                "grain": (0.03, "per-cell texture")})
def euglena_bench(ctx):
    W, H = ctx.size
    rng = ctx.rng
    S, C = _data(ctx)           # (38, 9, 3)
    nt, nc, nr = S.shape
    rn = str(ctx.p("ramp")); mode = str(ctx.p("mode")); gap = float(ctx.p("gap")) * W / 1500
    if str(ctx.p("order")) == "mean":
        o = np.argsort(-S.mean((1, 2))); S = S[o]; C = C[o]
    if mode == "heat":
        m = S.mean(2)
        cw, rh = ctx.p("col_w") * W, ctx.p("row_h") * H
        X, Y = np.meshgrid(np.arange(W), np.arange(H))
        u = X / cw + float(ctx.p("x0")); v = Y / rh + float(ctx.p("y0"))
        ci = np.floor(u).astype(int) % nc; ri = np.floor(v).astype(int) % nt
        val = m[ri, ci]
        img = ramp(rn, 0.04 + 0.96 * val)
        fu = (u - np.floor(u)) * cw; fv = (v - np.floor(v)) * rh
        edge = (fu < gap) | (fv < gap)
        img[edge] = img[edge] * 0.78
        # slow shading across each cell (real cells are flat, this only gives paper/ink feel)
        sh = 1 + 0.06 * (0.5 - fv / rh)[..., None]
        img = img * sh + (rng.random((H, W, 1)).astype(np.float32) - 0.5) * float(ctx.p("grain"))
        return np.clip(img, 0.04, 0.97).astype(np.float32)
    if mode == "runs":
        # stripes = configurations (9), columns = tests (38), each split into 3 run strips
        order = np.argsort(-S.mean((0, 2)))
        X, Y = np.meshgrid(np.arange(W), np.arange(H))
        u = X / W * nt * nr; v = Y / H * nc
        ti = np.minimum((u // nr).astype(int), nt - 1); ri = np.minimum((u % nr).astype(int), nr - 1)
        ci = order[np.minimum(v.astype(int), nc - 1)]
        val = S[ti, ci, ri]
        img = ramp(rn, 0.03 + 0.97 * val)
        fu = (u - np.floor(u)) * W / (nt * nr); fv = (v - np.floor(v)) * H / nc
        img[fu < gap * 0.6] *= 0.80
        img[(u // nr).astype(int) != ((u - gap / (W / (nt * nr))) // nr).astype(int)] *= 0.55
        img[fv < gap * 1.6] *= 0.45
        return np.clip(img + (rng.random((H, W, 1)).astype(np.float32) - 0.5) * 0.02, 0.04, 0.97).astype(np.float32)
    if mode == "skyline":
        order = np.argsort(-S.mean((0, 2)))
        img = np.zeros((H, W, 3), np.float32) + ramp(rn, 0.0)
        bandh = H / nc
        nbar = nt * nr
        bw = W / nbar
        for k, c in enumerate(order):
            y1 = (k + 1) * bandh
            for t in range(nt):
                for r in range(nr):
                    s = S[t, c, r]
                    x0 = int((t * nr + r) * bw); x1 = int((t * nr + r + 1) * bw - 1)
                    hh = (0.06 + 0.9 * s) * (bandh - gap * 1.5)
                    ya, yb = int(y1 - gap * 1.5 - hh), int(y1 - gap * 1.5)
                    col = ramp(rn, 0.18 + 0.8 * s)
                    img[ya:yb, x0:x1] = col
            img[int(y1 - gap * 1.5):int(y1)] = ramp(rn, 0.0) * 0.7
        return np.clip(img, 0.04, 0.97)
    # scatter
    img = np.zeros((H, W, 3), np.float32) + ramp(rn, 0.02)
    lc = np.log10(np.clip(C, 1e-5, None))
    lo, hi = lc.min(), lc.max()
    xs = (lc - lo) / (hi - lo)
    ys = S + (rng.random(S.shape) - 0.5) * 0.02
    glow = np.zeros((H, W), np.float32)
    col = np.zeros((H, W, 3), np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    pad = 0.04
    px = (pad + (1 - 2 * pad) * xs) * W; py = (1 - (pad + (1 - 2 * pad) * ys)) * H
    tier = (S.mean((0, 2)))
    for t in range(nt):
        for c in range(nc):
            for r in range(nr):
                x, y = px[t, c, r], py[t, c, r]
                rad = 7 + 6 * rng.random()
                x0, x1, y0, y1 = int(x - 3 * rad), int(x + 3 * rad), int(y - 3 * rad), int(y + 3 * rad)
                x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, W), min(y1, H)
                if x1 <= x0 or y1 <= y0:
                    continue
                d2 = (xx[y0:y1, x0:x1] - x) ** 2 + (yy[y0:y1, x0:x1] - y) ** 2
                g = np.exp(-d2 / (2 * (rad * 0.55) ** 2)) * 0.5 + np.exp(-d2 / (2 * (rad * 1.6) ** 2)) * 0.12
                glow[y0:y1, x0:x1] += g
                cc = ramp(rn, 0.25 + 0.75 * S[t, c, r])
                col[y0:y1, x0:x1] += g[..., None] * cc
    w = np.maximum(glow, 1e-6)[..., None]
    mix = 1 - np.exp(-glow[..., None] * 1.3)
    out = img * (1 - mix) + (col / w) * mix * 1.05
    return np.clip(out, 0.04, 0.97).astype(np.float32)
