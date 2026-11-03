"""swiss-* effects: a Swiss-inspired modular layout engine (wonhee/art/swisskit.py)."""
from __future__ import annotations

import numpy as np

from .. import printkit as K
from .. import swisskit as S
from ..registry import effect


def _params(**extra):
    d = K.params(**extra)
    d["texture"] = (0.6, "0..1.5 overall surface strength (paper fibre, grain, wear); clean=true for flat colour")
    d["reg"] = (2.0, "registration drift between inks (px at 1500 wide)")
    d["wear"] = (0.15, "edge wear")
    d["grain"] = (0.35, "ink grain")
    d["fibre"] = (0.5, "paper fibre visibility")
    for k in ("inks",):
        d[k] = ("", "up to 4 colours in this order: A (block colour), B (small accent), tint, ink; '' = from the palette hue")
    return d


GRID = dict(cols=(6, "grid columns (rows follow so modules are near square)"), gutter=(0.008, "gutter, fraction of width (floor 2 px)"),
            margin=(0.0, "frame margin, fraction of width (0 = blocks bleed off all edges)"), grid_lines=(0.55, "hairline module grid strength"))
TYPE = dict(type=(True, "draw the type layer when ctx.data holds text (str | [title, subtitle, caption...] | {title,subtitle,caption})"),
            type_scale=(1.0, "0.3-1 shrink the fitted title"), font=("", "path to a TTF; default Liberation Sans (Helvetica metrics)"))


@effect("swiss-grid-poster", family="swiss", tags=["structured", "editorial", "tactile", "archival", "source", "type", "light", "dark", "generated"],
        params=_params(density=(0.68, "share of modules filled; the rest is negative space"), big=(0.3, "area share of the anchor block"),
                       photo=(0.0, "0..1 chance/amount of photo blocks (needs a source; 0 = purely geometric)"),
                       photo_style=("duo", "duo (tint+A+ink duotone) | dot (halftone) | solid2 | mix"),
                       **GRID, **TYPE))
def grid_poster(ctx):
    """Swiss-inspired poster: a modular grid (merged modules), one big anchor block, large geometric
    forms (circle, quarter, half, rings, bars, diagonal, dots, steps, cross) on flat colour blocks that
    overprint, optional photo crops as duotone or halftone, asymmetric negative space, hairline grid.
    Typography is a separate optional layer from ctx.data (real text only; Liberation Sans) fitted into
    the largest free rectangle. Light = cream paper + cobalt; dark = night paper + lifted colours.
    One block in every quadrant: no focal point, no central emblem. Strong at 288x192 (6x4 modules)."""
    comp = S.compose(ctx, cols=int(ctx.p("cols")), density=float(ctx.p("density")), margin=float(ctx.p("margin")),
                     gutter=float(ctx.p("gutter")), big=float(ctx.p("big")), photo=float(ctx.p("photo")),
                     photo_style=str(ctx.p("photo_style")), type_on=bool(ctx.p("type")), grid_lines=float(ctx.p("grid_lines")))
    return S.print_swiss(ctx, comp)


@effect("swiss-form-tiling", family="swiss", tags=["structured", "playful", "tactile", "generated", "pattern", "light", "dark"],
        params=_params(cols=(8, "grid columns"), gutter=(0.0, "gutter fraction"), margin=(0.0, "margin fraction"), grid_lines=(0.0, "hairline grid"),
                       forms=("circle,quarter,half,rings,bars,diag,dots,steps,cross", "comma list of forms"),
                       mix=(0.5, "0 = one form family, 1 = every module different")))
def form_tiling(ctx):
    """Every module carries a large geometric form (quarter discs, half discs, rings, bars, diagonals,
    dot grids, steps), on coloured and tinted grounds, merged into bigger modules where the packing
    allows. A true all-over pattern: it fills the frame edge to edge with no centre. The strongest
    Swiss-inspired look for small cards (forms are module-sized, never tiny)."""
    forms = [f for f in str(ctx.p("forms")).split(",") if f in S.FORMS]
    mix = float(ctx.p("mix"))
    if mix < 1.0 and len(forms) > 2:
        rng = np.random.default_rng(ctx.seed + 5)
        k = max(2, int(round(2 + mix * (len(forms) - 2))))
        forms = list(rng.choice(forms, k, replace=False))
    comp = S.compose(ctx, cols=int(ctx.p("cols")), density=1.0, margin=float(ctx.p("margin")), gutter=float(ctx.p("gutter")), big=0.12,
                     kinds={"solid": 0.12, "form": 0.8, "rules": 0.04, "outline": 0.04}, forms=forms, type_on=False,
                     grid_lines=float(ctx.p("grid_lines")), full=True)
    return S.print_swiss(ctx, comp)


@effect("swiss-photo-grid", family="swiss", tags=["structured", "editorial", "tactile", "source", "light", "dark"],
        params=_params(cols=(5, "grid columns"), photo_style=("mix", "duo | dot | solid2 | mix"), density=(0.85, "fill"), big=(0.34, "anchor block share"), **TYPE))
def photo_grid(ctx):
    """The photo cut into merged grid modules: every block is a different crop of the source (chosen
    by detail), printed as duotone, halftone or two-colour tones in the Swiss-inspired inks, with solid
    colour blocks and a form or two between them. The photo is never whole: this reads as a layout, not
    a filter. Needs a source (a generated shape field stands in); works best on busy photos."""
    comp = S.compose(ctx, cols=int(ctx.p("cols")), density=float(ctx.p("density")), margin=0.0, gutter=0.008, big=float(ctx.p("big")), photo=1.0,
                     photo_style=str(ctx.p("photo_style")), type_on=bool(ctx.p("type")), grid_lines=0.45,
                     kinds={"photo": 0.55, "solid": 0.2, "form": 0.2, "outline": 0.05})
    return S.print_swiss(ctx, comp)
