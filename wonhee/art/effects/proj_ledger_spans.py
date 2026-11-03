"""proj-ledger-spans: Ledger's real source windows as an all-over field of word bars.
Each word of a recorded source window becomes a rounded bar whose length is the
word's character count; paragraphs wrap edge to edge and bleed off every side.
Words inside the span that Ledger verified (evidence.start..end) are lit in the
accent with a soft glow; everything else is quiet structure. Nothing is labelled.

ctx.data: {"lines": [{"words": [[len, verified 0/1], ...]}, ...]} (the project's
public/ledger/*.json evidence contexts). Without data, a seeded demo is drawn."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from ..registry import effect


@effect("proj-ledger-spans", family="dots", tags=["project", "text", "bars", "all-over"],
        params={"pitch": (0.036, "row pitch, fraction of height"), "cw": (0.0085, "bar width per character, fraction of width"),
                "para_gap": (0.35, "extra blank fraction of a row between paragraphs"), "glow": (7, "accent glow sigma px")})
def proj_ledger_spans(ctx):
    W, H = ctx.size
    rng = ctx.rng
    P = ctx.palette
    d = ctx.data
    paras = [p["words"] for p in d["lines"]] if isinstance(d, dict) and d.get("lines") else \
        [[[int(rng.integers(2, 9)), int(rng.random() < 0.25)] for _ in range(40)] for _ in range(12)]
    S = 2
    pitch = float(ctx.p("pitch")) * H
    bh = pitch * 0.5
    cw = float(ctx.p("cw")) * W
    gapw = cw * 1.6
    ramp = P.ramp(256, ("edge", "structure", "accent"))
    base = Image.new("RGB", (W * S, H * S), tuple(int(c * 255) for c in P.canvas))
    glow_l = Image.new("L", (W * S, H * S), 0)
    bd, gd = ImageDraw.Draw(base), ImageDraw.Draw(glow_l)
    order = rng.permutation(len(paras))
    y = -float(rng.uniform(0, pitch))
    k = 0
    while y < H + pitch:
        words = paras[order[k % len(paras)]]
        shade = 0.20 + 0.28 * rng.random()
        x = -float(rng.uniform(0, cw * 4))
        # long paragraphs wrap on the row; every row of a paragraph keeps its words
        for w, v in words:
            wl = max(w, 1) * cw
            if x + wl > W + cw * 3:
                x = -float(rng.uniform(0, cw * 3)); y += pitch
                if y > H + pitch: break
            col = ramp[int((0.97 if v else shade + 0.1 * rng.random()) * 255)]
            c = tuple(int(t * 255) for t in col)
            box = [x * S, (y - bh / 2) * S, (x + wl) * S, (y + bh / 2) * S]
            bd.rounded_rectangle(box, radius=bh * S * 0.5, fill=c)
            if v:
                gd.rounded_rectangle(box, radius=bh * S * 0.5, fill=255)
            x += wl + gapw
        y += pitch * (1.0 + float(ctx.p("para_gap")) * 0.0)
        k += 1
    img = np.asarray(base.resize((W, H), Image.LANCZOS), np.float32) / 255
    gl = np.asarray(glow_l.resize((W, H), Image.BILINEAR), np.float32) / 255
    halo = ndimage.gaussian_filter(gl, float(ctx.p("glow")))[..., None]
    acc = np.asarray(P.accent, np.float32)
    img = img + halo * (acc - img) * 0.45
    return np.clip(img, 0, 1)
