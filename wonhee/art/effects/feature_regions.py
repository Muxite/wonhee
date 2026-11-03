"""Region masks (C): apply different feature-edits to different regions of
one source, with soft or hard seams."""
from __future__ import annotations

import json

import numpy as np

from .. import features as F
from ..registry import effect
from .feature_common import prep, with_common


def render_regions(source, size, assignments, *, palette="ember", mode="detail", soft=0.03, seed=0, order=None):
    """Helper: `assignments` = [(effect_name, params_dict), ...]; one region per
    entry. Regions come from `F.region_weights(mode)` computed on the source as
    it is framed. Returns float32 RGB. Equivalent to the 'feature-regions' effect."""
    from ..compose import render_effect
    W, H = size
    img = F.fit_cover(np.asarray(source, np.float32)[..., :3], (W, H), "feature")
    wts = F.region_weights(img, len(assignments), mode, soft, seed)
    imgs = [render_effect(n, size=size, seed=seed + i, palette=palette, params=p, source=source) for i, (n, p) in enumerate(assignments)]
    return F.composite_regions(imgs, wts)


@effect("feature-regions", family="feature", tags=["source", "regions", "composite", "all-over", "mask"],
        params=with_common({"effects": ("feature-halftone-screen,feature-contour-relief", "comma-separated feature effects, one per region (ordered shadows-to-lights / calm-to-busy)"),
                            "mode": ("detail", "detail | tone | cells | superpixel | diagonal | rows | columns"),
                            "soft": (0.03, "seam softness as a fraction of the width (0 = hard seam)"),
                            "params": ("{}", "JSON {effect_name: {param: value}} overrides")}))
def feature_regions(ctx):
    """One source, several effects: the frame is partitioned by a feature
    (detail energy, tone, jittered cells, superpixel colour groups, or a
    geometric split) and each region gets its own feature-edit, joined with a
    soft or hard seam. 'detail' puts the dense effect on busy areas and the
    quiet one on calm areas; 'cells' scatters effects evenly. Pick effects
    from the feature-* family that suit the same source."""
    P = prep(ctx)
    names = [n.strip() for n in str(ctx.p("effects")).split(",") if n.strip()]
    over = json.loads(ctx.p("params") or "{}")
    from ..compose import render_effect
    wts = F.region_weights(P.img, len(names), str(ctx.p("mode")), float(ctx.p("soft")), ctx.seed)
    imgs = []
    for i, n in enumerate(names):
        pr = dict(over.get(n, {}))
        for k in ("crop", "local", "hue_from_source"):
            pr.setdefault(k, ctx.p(k))
        imgs.append(render_effect(n, size=ctx.size, seed=ctx.seed + i, palette=ctx.palette, params=pr, source=ctx.source))
    return F.composite_regions(imgs, wts)
