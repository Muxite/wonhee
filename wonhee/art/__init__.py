"""wonhee.art: a growing library of deterministic image effects ("art") and the
tools to combine them. No generative AI: every effect is numpy/scipy math with
a seed (physics sims, light, dot matrices, graphs, filters on real images).

Contract (see docs/ART.md):

    from wonhee.art import effect

    @effect("wave-ripple", family="physics", tags=["wave", "all-over"],
            params={"sources": (3, "number of wave sources")})
    def wave_ripple(ctx):            # ctx: size, rng, palette, params, source
        ...
        return rgb                   # float32 H x W x 3 in 0..1 (sRGB)

An effect is registered by importing its module under ``wonhee/art/effects``
(auto-discovered). It must be deterministic in (params, seed, size, palette),
fill the whole frame with evenly spread detail (no central focal point), and
stay readable at 288x192.
"""
from __future__ import annotations

from .compose import render_effect, render_recipe
from .palette import Palette, make, named
from .registry import REGISTRY, Ctx, discover, effect, get, names

__all__ = ["REGISTRY", "Ctx", "Palette", "discover", "effect", "get", "make", "named", "names",
           "render_effect", "render_recipe"]
