"""Atmosphere effects for real photos: bloom, halation, haze, fog, streaks, pools, wet ground.

Each is the shared pipeline in ``atmos_common.atmosphere`` with one family of stages
on; ``atmos-nocturne`` exposes all stages with presets. Input: ``ctx.source`` (a photo;
without one a generated night street is used). Optional ``ctx.data``: hand-authored
masks ({"polygons": [[(x, y), ...]]} in 0..1 or a 2D array) for depth (``depth=mask``,
1 = far) and for the wet ground.
"""
from __future__ import annotations

from ..registry import effect
from .atmos_common import (P, SOURCE, SURFACE, COMMON, BLOOM, HALATION, STREAK, POOLS, HAZE, FOG, WET,
                           atmosphere, prep_source)


def _run(ctx, off=()):
    img = prep_source(ctx)
    p = dict(ctx.params)
    for k in off:
        p[k] = 0.0
    return atmosphere(img, p, ctx.seed, ctx.data)


@effect("atmos-glow", family="atmos", tags=["source", "photo", "glow", "bloom", "halation", "nocturnal", "luminous", "nostalgic"],
        params=P(COMMON, SOURCE, BLOOM, HALATION, SURFACE))
def glow(ctx):
    """Bloom and halation as two separate dials on the real lights of a photo. Emitters are
    extracted (luminance, colour or small local peaks, skin excluded), bloomed over four scales
    in linear light, and given a warm halation fringe that lives OUTSIDE the highlight. Cores
    stay crisp, shadows stay dark. Best on night streets, signs, stage lights; fails on flat
    daylight (nothing emits: raise `bloom` or lower `threshold`) and on photos whose brightest
    area is a face (skin is protected, so little happens)."""
    return _run(ctx, ("haze", "fog", "wet", "streak", "pools"))


@effect("atmos-haze", family="atmos", tags=["source", "photo", "haze", "depth", "mysterious", "nocturnal", "luminous"],
        params=P(COMMON, SOURCE, HAZE, SURFACE, haze=0.8))
def haze(ctx):
    """Depth-dependent veil: distant parts of the frame (gradient from the horizon, top-down,
    dark-channel estimate, or a hand-authored mask) sink into airlight of the scene's own
    ambient colour (or a chosen hue) while the foreground, skin and lamp cores stay clear;
    lamps light the haze around them. Not a colour grade: contrast and saturation fall with
    distance. Fails when `depth=gradient` does not match the picture's geometry (set
    `horizon`) and when `haze` is high on a frame with no distance cue."""
    return _run(ctx, ("fog", "wet", "streak", "pools", "bloom", "halation"))


@effect("atmos-fog", family="atmos", tags=["source", "photo", "fog", "mysterious", "intimate", "nocturnal"],
        params=P(COMMON, SOURCE, FOG, SURFACE, {"haze_level": HAZE["haze_level"], "haze_hue": HAZE["haze_hue"], "haze_chroma": HAZE["haze_chroma"],
                                                 "haze_scatter": HAZE["haze_scatter"], "ground": WET["ground"]}, fog=0.8))
def fog(ctx):
    """Patchy fog banks (stretched fractal noise, a pure function of coordinates and seed, so it
    does not crawl in video) laid over the picture; fog near the ground when `fog_height` is
    high, lit by nearby lamps. Complements `atmos-haze` (smooth with depth). Fails when coverage
    is high and the scene has no lamps: a flat grey veil."""
    return _run(ctx, ("haze", "wet", "streak", "pools", "bloom", "halation"))


@effect("atmos-streaks", family="atmos", tags=["source", "photo", "scatter", "anamorphic", "nocturnal", "luminous", "mediated"],
        params=P(COMMON, SOURCE, STREAK, SURFACE, streak=0.8))
def streaks(ctx):
    """Directional scattering of lights: each emitter is smeared along an axis (horizontal =
    anamorphic lens, vertical = rain on glass, diagonal), energy conserving. Pair `streak_cross`
    for star shapes. Fails with many small lights (turns into stripes): raise `threshold`."""
    return _run(ctx, ("haze", "fog", "wet", "pools", "bloom", "halation"))


@effect("atmos-light-pools", family="atmos", tags=["source", "photo", "pools", "luminous", "intimate", "nocturnal"],
        params=P(COMMON, SOURCE, POOLS, SURFACE, pools=1.0))
def light_pools_fx(ctx):
    """A few large soft pools of coloured light under the strongest lamps (the five biggest, well
    separated), instead of a haze of tiny glows: reads at thumbnail size. Fails if the scene has
    one dominant light only (a single pool)."""
    return _run(ctx, ("haze", "fog", "wet", "streak", "bloom", "halation"))


@effect("atmos-wet-ground", family="atmos", tags=["source", "photo", "wet", "reflection", "nocturnal", "mysterious", "luminous"],
        params=P(COMMON, SOURCE, WET, SURFACE, wet=1.0))
def wet_ground(ctx):
    """Wet-street reflections: lights above the horizon are mirrored into the ground mask,
    stretched down the vertical, blurred sideways by `roughness`, rippled by seeded noise and
    weighted toward the horizon; the surface darkens like wet asphalt. Pass the ground as
    `ground` (row fraction) or ctx.data polygons. Fails when the horizon is wrong (reflections
    float) and on photos whose ground already carries a mirror image (set `wet` lower)."""
    return _run(ctx, ("haze", "fog", "streak", "pools", "bloom", "halation"))


PRESETS = {
    "none": {},
    "night-street": dict(bloom=0.45, halation=0.4, pools=0.3, wet=0.5, haze=0.2, fog=0.0, streak=0.0),
    "mist": dict(bloom=0.3, halation=0.2, pools=0.2, wet=0.0, haze=0.6, fog=0.5, streak=0.0, depth="top", haze_level=0.36, ground=0.7),
    "neon-rain": dict(bloom=0.6, halation=0.35, pools=0.25, wet=0.9, haze=0.15, streak=0.3, streak_angle=90.0, mode="both", roughness=0.6),
    "stage": dict(bloom=0.6, halation=0.35, pools=0.5, wet=0.0, haze=0.3, streak=0.35, depth="top", mode="peak", haze_level=0.26),
    "dream": dict(bloom=0.9, halation=0.6, pools=0.0, wet=0.0, haze=0.25, streak=0.0, threshold=0.5, local=0.2, bloom_radius=1.5, haze_level=0.4),
}


@effect("atmos-nocturne", family="atmos", tags=["source", "photo", "stack", "nocturnal", "mysterious", "intimate", "luminous", "candid"],
        params=P({"preset": ("night-street", "night-street | mist | neon-rain | stage | dream | none; sets the stage amounts, any param you pass explicitly overrides it, including 0 (use none to set every stage yourself)")},
                 COMMON, SOURCE, BLOOM, HALATION, STREAK, POOLS, HAZE, FOG, WET, SURFACE))
def nocturne(ctx):
    """The full stack with independent dials: bloom, halation, streaks, pools, haze, fog and wet
    ground, over one shared emitter extraction. `preset` picks a starting point; any param you
    pass explicitly wins over the preset (even an explicit 0 or a value equal to the default). Best on night streets, signs,
    stage and window light; with no source it renders a generated night street."""
    ps = PRESETS.get(str(ctx.p("preset", "night-street")), PRESETS["night-street"])
    p = dict(ctx.params)
    for k, v in ps.items():
        if k not in ctx.explicit:              # a param the caller passed (even 0) beats the preset
            p[k] = v
    img = prep_source(ctx)
    from .atmos_common import atmosphere as run
    return run(img, p, ctx.seed, ctx.data)
