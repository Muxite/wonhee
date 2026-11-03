"""dots-circuit: pad arrays and traces. The frame is split into blocks;
each is a BGA-style ring-pad array, a bus of parallel dotted traces with 45
degree jogs ending in pads, or a chip (dense square dots with pins). Everything
sits on a dim substrate dot grid. Blocks are spread by BSP, so detail is even.

ctx.data (optional): 2D array in 0..1 read at the pad sites (higher = more
'powered', accent pads), e.g. a score matrix. ctx.source unused."""
from __future__ import annotations

import numpy as np

from .. import dotkit as dk
from ..registry import effect


@effect("dots-circuit", family="dots", tags=["dots", "circuit", "pcb", "pads", "traces", "all-over"],
        params={"cols": (66, "grid cells across"), "blocks": (13, "number of blocks"), "hot": (0.10, "share of pads in accent")})
def dots_circuit(ctx):
    W, H = ctx.size
    rng = ctx.rng
    P = ctx.palette
    cols = int(ctx.p("cols"))
    pitch = W / cols
    rows = int(H / pitch)
    sub = np.ones((rows, cols), np.float32)
    pads = np.zeros((rows, cols), np.float32)
    trace = np.zeros((rows, cols), np.float32)
    chip = np.zeros((rows, cols), np.float32)
    blocks = dk.bsp_panels((cols, rows), rng, int(ctx.p("blocks")), 0.14)
    data = np.asarray(ctx.data, np.float32) if ctx.data is not None and np.ndim(ctx.data) == 2 else None
    hot_p = float(ctx.p("hot"))
    for bi, (x, y, w, h) in enumerate(blocks):
        x0, y0, x1, y1 = x + 1, y + 1, x + w - 1, y + h - 1
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        kind = ["bga", "bus", "chip", "bga", "bus"][bi % 5] if rng.random() < 0.8 else str(rng.choice(["bga", "bus", "chip"]))
        if kind == "bga":
            st = int(rng.choice([1, 2, 2]))
            pads[y0:y1:st, x0:x1:st] = 1
        elif kind == "chip":
            chip[y0 + 1:y1 - 1, x0 + 1:x1 - 1] = 1
            pads[y0:y1:2, x0] = 1; pads[y0:y1:2, x1 - 1] = 1
            pads[y0, x0:x1:2] = 1; pads[y1 - 1, x0:x1:2] = 1
            chip[y0 + 1:y1 - 1:3, x0 + 1:x1 - 1:3] = 0
        else:
            horiz = (x1 - x0) >= (y1 - y0)
            n_t = (y1 - y0) if horiz else (x1 - x0)
            L = (x1 - x0) if horiz else (y1 - y0)
            for t in range(0, n_t, 2):
                pos = t
                jog = int(rng.integers(L // 4, 3 * L // 4 + 1))
                jd = int(rng.choice([-1, 1]))
                for s in range(L):
                    if s == jog and 0 <= pos + jd * 2 < n_t:
                        for q in range(2):
                            pos += jd
                            cy, cx = (y0 + pos, x0 + s) if horiz else (y0 + s, x0 + pos)
                            if 0 <= cy < rows and 0 <= cx < cols:
                                trace[cy, cx] = 1
                    cy, cx = (y0 + pos, x0 + s) if horiz else (y0 + s, x0 + pos)
                    if 0 <= cy < rows and 0 <= cx < cols:
                        trace[cy, cx] = 1
                for end in (0, L - 1):
                    cy, cx = (y0 + pos if end else y0 + t, x0 + end) if horiz else (y0 + end, x0 + pos if end else x0 + t)
                    if 0 <= cy < rows and 0 <= cx < cols:
                        pads[cy, cx] = 1
    g = dk.grid(rows, cols, ctx.size, pitch=pitch, origin=(0, (H - rows * pitch) / 2))
    ramp = dk.palette_ramp(P)
    off = P.canvas + (P.structure - P.canvas) * 0.20
    cv = dk.new_canvas(ctx.size, P.canvas)
    dk.draw_dots(cv, g.x, g.y, g.cell * 0.09, off, 1.0, "round")
    mid = ramp[int(0.5 * 255)]
    lo = ramp[int(0.34 * 255)]
    hi = ramp[255]
    L = g
    tr = trace.ravel() > 0
    dk.draw_dots(cv, L.x[tr], L.y[tr], L.cell[tr] * 0.2, mid, 1.0, "round")
    ch = chip.ravel() > 0
    dk.draw_dots(cv, L.x[ch], L.y[ch], L.cell[ch] * 0.40, lo, 1.0, "square")
    pd = pads.ravel() > 0
    pv = (data[(L.row[pd] % data.shape[0]), (L.col[pd] % data.shape[1])] if data is not None else rng.random(int(pd.sum())))
    hotm = pv > (1 - hot_p)
    px, py, pc = L.x[pd], L.y[pd], L.cell[pd]
    dk.draw_dots(cv, px[~hotm], py[~hotm], pc[~hotm] * 0.42, mid, 1.0, "ring")
    dk.draw_dots(cv, px[hotm], py[hotm], pc[hotm] * 0.42, hi, 1.0, "round")
    return cv
