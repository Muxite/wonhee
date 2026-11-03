"""dots-word: dot-matrix lettering mosaic. The text is rasterised with the
5x7 bitmap font (dotkit.text_matrix) into bands that run edge to edge as a
marquee; bands are staggered so letters are everywhere, over a dim field of
off dots whose size breathes with a wave field. Reads at 288x192: the letters
are ~1/6 of the frame height.

ctx.data (optional): a str (or list of str, one per band, cycled), e.g. the
project name, a number, a score. Param ``text`` is the default. ctx.source unused."""
from __future__ import annotations

import numpy as np

from .. import dotkit as dk
from ..registry import effect


@effect("dots-word", family="dots", tags=["dots", "text", "led", "lettering", "all-over"],
        params={"text": ("WONHEE", "word to set (or ctx.data str / list of str)"), "cols": (62, "dots across the frame"), "gap": (4, "blank columns between repeats"),
                "shape": ("round", "round | square | diamond"), "bands": (0, "text bands (0 = as many as fit)")})
def dots_word(ctx):
    W, H = ctx.size
    rng = ctx.rng
    P = ctx.palette
    d = ctx.data
    texts = [d] if isinstance(d, str) else ([str(x) for x in d] if isinstance(d, (list, tuple)) and d else [str(ctx.p("text"))])
    cols = int(ctx.p("cols"))
    pitch = W / cols
    rows = int(H / pitch)
    pitch_y = pitch
    band_h = 7
    gapr = 2
    nb = int(ctx.p("bands")) or max(1, (rows + gapr) // (band_h + gapr))
    used = nb * band_h + (nb - 1) * gapr
    top = (rows - used) // 2
    m = np.zeros((rows, cols), np.float32)
    wv = dk.resize_field(dk.wave_field((max(8, H // 8), max(8, W // 8)), rng, 6, 0.18 * W / 8), (cols, rows))
    for b in range(nb):
        t = texts[b % len(texts)]
        unit = dk.text_matrix(t).shape[1] + int(ctx.p("gap"))
        tm = dk.tile_text(t, band_h, cols, gap=int(ctx.p("gap")), col0=int((b * 0.5 * unit + b * 2) % unit))
        r0 = top + b * (band_h + gapr)
        m[r0:r0 + band_h] = tm.astype(np.float32)
    g = dk.grid(rows, cols, ctx.size, pitch=pitch, origin=(0, (H - rows * pitch_y) / 2))
    ramp = dk.palette_ramp(P)
    off = P.canvas + (P.structure - P.canvas) * 0.20
    # background dots breathe with the wave, lit letters sit brighter
    bgv = dk.norm01(wv)
    cv = dk.new_canvas(ctx.size, P.canvas)
    dk.draw_dots(cv, g.x, g.y, g.cell * (0.10 + 0.13 * bgv.ravel()[0:0].sum() if False else 0.10), off, 1.0, "round")
    sv = (0.55 + 0.35 * bgv)
    lit = m > 0
    mv = np.where(lit, sv, 0).astype(np.float32)
    mv[lit & (rng.random(m.shape) < 0.07)] = 1.0
    return dk.render_dots(mv, ctx.size, g, canvas=cv, bg=P.canvas, ramp=ramp, shape=str(ctx.p("shape")), size_const=True, r_max=0.42, threshold=0.05,
                          off=off, off_r=0.12, glow_sigma=pitch * 0.5, glow_strength=0.55, core=0.35, normalize=False)
