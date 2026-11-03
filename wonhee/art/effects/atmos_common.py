"""Shared prelude + the one atmosphere pipeline for ``atmos_*`` effects (not an effect itself).

Every ``atmos-*`` effect is the same pipeline with different stages switched on, so
stages combine cleanly and each control is independent:

  colour    haze_hue / haze_chroma / halation_hue / glow_tint
  structure radii, horizon, depth mode, streak angle, pool count
  source    mode / threshold / softness / hue / local / skin (which pixels are lights)
  surface   grain, anchor (dark end), core protection

Light transport is in linear light (gamma 2.2).
"""
from __future__ import annotations


import cv2
import numpy as np

from .. import atmoskit as K
from .. import features as F

# ---- parameter groups (name -> (default, description)) ----------------------

SOURCE = {
    "mode": ("luminance", "which pixels emit: luminance | colour | peak | both"),
    "threshold": (0.62, "display luminance where emission starts (0..1)"),
    "softness": (0.3, "knee width of the threshold"),
    "hue": (-1.0, "restrict emitters to this OKLCH hue in degrees (-1 = any); with mode=colour picks one sign colour"),
    "hue_width": (40.0, "half width of the hue range, degrees"),
    "local": (0.6, "0..1 suppress big bright areas (walls, sky); only lamp-sized peaks emit"),
    "skin": (0.9, "0..1 skin protection: people do not glow, are not hazed, keep natural tones"),
}
SURFACE = {
    "anchor": (0.8, "0..1 keep the dark end where the photo had it (no grey lift)"),
    "core": (0.6, "0..1 protect blown cores: glow is not stacked on near-white pixels"),
    "grain": (0.15, "0..1 fine film grain (coordinate-based, same on every frame)"),
}
COMMON = {
    "crop": ("center", "how a source of another aspect is cropped: center | feature"),
}
BLOOM = {
    "bloom": (0.8, "0..1.5 bright-region glow amount (soft, white-ish, energy-conserving)"),
    "bloom_radius": (1.0, "size multiplier of the glow scales (base radii 5/16/48/140 px at 1500 wide)"),
    "bloom_falloff": (0.5, "0..1 how much tight glow dominates wide glow"),
    "glow_tint": (-1.0, "tint the bloom toward this OKLCH hue in degrees (-1 = the lights' own colour)"),
}
HALATION = {
    "halation": (0.6, "0..1.5 warm fringe around highlights (film base reflection)"),
    "halation_hue": (32.0, "fringe hue in OKLCH degrees (32 = red-orange)"),
    "halation_radius": (9.0, "fringe radius px at 1500 wide"),
}
STREAK = {
    "streak": (0.0, "0..1.5 directional scattering: lights smeared along an axis"),
    "streak_length": (160.0, "mean smear length px at 1500 wide"),
    "streak_angle": (0.0, "degrees; 0 = horizontal (anamorphic), 90 = vertical (rain, wet glass)"),
    "streak_cross": (0.0, "0..1 add a second perpendicular streak (star-like)"),
}
POOLS = {
    "pools": (0.0, "0..1.5 a few large soft pools of light under the strongest lamps"),
    "pool_count": (5, "number of pools (fewer, larger is better at thumbnail size)"),
    "pool_radius": (0.08, "pool radius as a fraction of the frame diagonal"),
    "pool_squash": (1.0, "vertical squash of pools (0.5 = ground-hugging ellipses)"),
}
HAZE = {
    "haze": (0.0, "0..1.5 depth-dependent veiling haze (airlight)"),
    "depth": ("gradient", "depth proxy: gradient | top | dark-channel | mask (ctx.data polygons / array) | flat"),
    "horizon": (0.45, "horizon row as a fraction of height (gradient depth, ground)"),
    "haze_level": (0.28, "luminance of the airlight 0..1 (night: low, morning: high)"),
    "haze_hue": (-1.0, "airlight OKLCH hue in degrees (-1 = the scene's own ambient light)"),
    "haze_chroma": (0.05, "airlight chroma when haze_hue is set"),
    "haze_scatter": (0.8, "0..2 lamps light up the haze around them (in-scattering)"),
    "haze_depth_power": (1.0, "gamma on depth: >1 keeps the foreground clear for longer"),
}
FOG = {
    "fog": (0.0, "0..1.5 patchy fog banks (seeded fractal noise)"),
    "fog_coverage": (0.5, "0..1 how much of the frame carries fog"),
    "fog_scale": (260.0, "feature size of the banks px at 1500 wide"),
    "fog_height": (0.5, "0..1 pull the fog toward the ground"),
    "fog_drift": (0.0, "0..1 slide the noise sideways (animation hook, pure function)"),
}
WET = {
    "wet": (0.0, "0..1.5 wet-ground reflections of the lights (vertical stretch inside the ground mask)"),
    "ground": (0.62, "ground starts at this fraction of the height (or ctx.data polygons / mask)"),
    "roughness": (0.4, "0 = mirror, 1 = rough wet asphalt (long, wide smear)"),
    "ripple": (0.3, "seeded ripple distortion of the reflection"),
    "reflect_length": (0.16, "reach of the reflection as a fraction of the height"),
    "wet_darken": (0.25, "0..1 wet surfaces darken and gain contrast"),
    "ground_dark": (0.6, "0..1 only reflect into darker ground pixels (not onto legs and walls)"),
}


def P(*groups, **over):
    out = {}
    for g in groups:
        out.update(g)
    for k, v in over.items():
        d, doc = out[k]
        out[k] = (v, doc)
    return out


# ---- procedural night field ---------------------------------------------------

def synth_night(size, seed: int = 0) -> np.ndarray:
    """A generated night street: dark sky, skyline with lit windows across the whole width, lamps, two neon signs,
    a wet dark road. No focal point; used when no source is given and as a stand-in test field."""
    W, H = size
    rng = np.random.default_rng(seed)
    ss = 2
    w, h = W * ss, H * ss
    img = np.zeros((h, w, 3), np.float32)
    yy = np.linspace(0, 1, h)[:, None]
    sky = np.array([0.03, 0.04, 0.09]) * (1 - yy) + np.array([0.08, 0.05, 0.10]) * yy
    img[:] = sky[:, None, :]
    horizon = int(0.62 * h)
    x = 0
    while x < w:
        bw = int(rng.uniform(0.05, 0.14) * w)
        bh = int(rng.uniform(0.18, 0.55) * h)
        top = horizon - bh
        shade = rng.uniform(0.015, 0.05)
        cv2.rectangle(img, (x, top), (x + bw, horizon), (shade, shade * 1.1, shade * 1.6), -1)
        cell = max(4, int(w * 0.012))
        for cy in range(top + cell, horizon - cell, cell * 2):
            for cx in range(x + cell, x + bw - cell, cell * 2):
                if rng.random() < 0.34:
                    warm = rng.random() < 0.75
                    col = (1.0, 0.78, 0.45) if warm else (0.6, 0.85, 1.0)
                    k = rng.uniform(0.5, 1.0)
                    cv2.rectangle(img, (cx, cy), (cx + cell // 2, cy + cell // 2), tuple(float(c * k) for c in col), -1)
        x += bw + int(rng.uniform(0, 0.012) * w)
    cv2.rectangle(img, (0, horizon), (w, h), (0.02, 0.022, 0.03), -1)
    # signs
    for col, cx in (((1.0, 0.15, 0.35), 0.22), ((0.1, 0.9, 1.0), 0.7)):
        sx, sy = int(cx * w), int(horizon - 0.14 * h)
        cv2.rectangle(img, (sx, sy), (sx + int(0.13 * w), sy + int(0.03 * h)), col, -1)
    # lamps along the road, in perspective
    for side in (-1, 1):
        for i in range(7):
            t = i / 6
            px = int(w * (0.5 + side * (0.08 + 0.52 * t ** 1.4)))
            py = int(horizon + (h - horizon) * (0.04 + 0.7 * t ** 1.4) - 0.1 * h * (0.3 + t))
            r = max(2, int(w * 0.004 * (1 + 2.5 * t)))
            cv2.circle(img, (px, py), r, (1.0, 0.85, 0.55), -1)
    out = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    n = K.hash_noise((H, W), seed) * 0.006
    return np.clip(out + n[..., None], 0, 1).astype(np.float32)


def prep_source(ctx):
    W, H = ctx.size
    src = ctx.source
    if src is None:
        return synth_night((W, H), ctx.seed)
    return F.fit_cover(np.asarray(src, np.float32)[..., :3], (W, H), ctx.p("crop", "feature"))


def _unit_tint(hue_deg, chroma=0.15, L=0.78):
    c = K.lin(K.hue_rgb(hue_deg, chroma, L))
    return c / (float(c @ K.LUM) + 1e-4)


# ---- the pipeline -------------------------------------------------------------

def atmosphere(img: np.ndarray, p: dict, seed: int = 0, data=None) -> np.ndarray:
    """Apply the stages selected by the params in ``p`` to display-referred ``img``. All stage
    amounts default to 0 when missing, so an effect enables only what it documents."""
    g = lambda k, d=0.0: float(p.get(k, d))
    H, W = img.shape[:2]
    s = K.scale_of(img.shape)
    base = img
    w, col = K.emission(img, str(p.get("mode", "luminance")), g("threshold", 0.62), g("softness", 0.3),
                        g("hue", -1.0), g("hue_width", 40.0), g("local", 0.6), g("skin", 0.9),
                        mask=K.mask_from_data(data, img.shape) if p.get("mode") == "mask" else None)
    core = K.core_mask(img)
    skin = K.skin_mask(img) if g("skin", 0.9) > 0 and (g("haze") > 0 or g("fog") > 0) else None
    protect = None
    if g("haze") > 0 or g("fog") > 0:
        protect = np.clip(g("core", 0.6) * 0.5 * core + (g("skin", 0.9) * skin if skin is not None else 0), 0, 1)

    x = img
    # 1. wet ground: darken the surface, then add reflections of the lights
    if g("wet") > 0:
        hz = g("ground", 0.62)
        gm = K.ground_mask(img.shape, hz, 0.07, data, img, g("ground_dark", 0.6))
        xl = K.lin(x)
        wd = g("wet_darken", 0.25) * min(1.0, g("wet")) * gm
        xl = xl * (1 - 0.45 * wd[..., None])
        # lights above the horizon only
        yy = np.arange(H, dtype=np.float32)[:, None] / H
        above = (1 - K.smoothstep(hz - 0.02, hz + 0.02, yy))[..., None]
        wcol = K.lin(img) * K.emission(img, str(p.get("mode", "luminance")), max(0.35, g("threshold", 0.62) - 0.2), 0.3,
                                       g("hue", -1.0), g("hue_width", 40.0), 0.3, g("skin", 0.9))[0][..., None] * above
        wr = K.wet_reflect(wcol, gm, hz * H, g("reflect_length", 0.16), g("roughness", 0.4), g("ripple", 0.3), seed)
        xl = xl + wr * g("wet") * 2.6
        x = K.srgb(np.clip(xl, 0, 1))

    # 2. airlight haze and fog banks (in front of the scene, behind the glow)
    if g("haze") > 0 or g("fog") > 0:
        depth = K.depth_field(img, str(p.get("depth", "gradient")), g("horizon", 0.45), 0.0,
                              K.mask_from_data(data, img.shape), g("haze_depth_power", 1.0))
        colr = K.ambient_colour(img)
        m_ = float(colr.mean()); colr = np.clip(m_ + (colr - m_) * 1.4, 0.002, None)
        if g("haze_hue", -1.0) >= 0:
            colr = _unit_tint(g("haze_hue"), g("haze_chroma", 0.05) + 0.02, 0.8)
        scat = K.blur_big(col, 70 * s) * g("haze_scatter", 0.8) * 2.0 if g("haze_scatter", 0.8) > 0 else None
        if g("haze") > 0:
            x = K.veil(x, depth, 1.2 * g("haze"), g("haze_level", 0.34), colr, scatter=scat, protect=protect)
        if g("fog") > 0:
            dens = K.fog_banks(img.shape, seed, g("fog_scale", 260.0), g("fog_coverage", 0.5), g("fog_height", 0.5),
                               g("ground", 0.62) if g("wet") > 0 else 0.6, 3, g("fog_drift", 0.0))
            x = K.veil(x, dens, 2.4 * g("fog"), g("haze_level", 0.34) + 0.08, colr, scatter=scat, protect=protect)

    # 3. light: bloom, streaks, halation, pools (additive, linear)
    add = np.zeros_like(img)
    if g("bloom") > 0:
        r = g("bloom_radius", 1.0)
        gl = K.glow_stack(col, tuple(v * r for v in (5, 16, 48, 140)), falloff=g("bloom_falloff", 0.5))
        if g("glow_tint", -1.0) >= 0:
            t = _unit_tint(g("glow_tint"))
            gl = (gl @ K.LUM)[..., None] * t * 0.7 + gl * 0.3
        add += gl * g("bloom") * 2.2
    if g("streak") > 0:
        st = K.streak_blur(col, g("streak_length", 160.0), g("streak_angle", 0.0))
        if g("streak_cross") > 0:
            st = st + g("streak_cross") * K.streak_blur(col, g("streak_length", 160.0), g("streak_angle", 0.0) + 90)
        add += st * g("streak") * 2.2
    if g("halation") > 0:
        add += K.halation_field(w, col, _unit_tint(g("halation_hue", 32.0), 0.18), g("halation_radius", 9.0)) * g("halation") * 1.8
    if g("pools") > 0:
        add += K.light_pools(w, col, int(g("pool_count", 5)), g("pool_radius", 0.12), g("pool_squash", 1.0)) * g("pools") * 0.45
    if np.any(add):
        add = add * (1 - g("core", 0.6) * core)[..., None]
        if g("skin", 0.9) > 0:
            add = add * (1 - 0.85 * g("skin", 0.9) * (skin if skin is not None else K.skin_mask(img)))[..., None]
        out = K.srgb(np.clip(1 - (1 - K.lin(x)) * np.exp(-add), 0, 1))
    else:
        out = x

    out = K.anchor_blacks(out, base, g("anchor", 0.8))
    gr = g("grain", 0.15)
    if gr > 0:
        n = K.hash_noise(img.shape, seed + 11)
        n = K.blur(n, 0.5 * max(1.0, s * 1.2))
        L = K.luma(out)
        out = out + (n * gr * 0.05 * (0.3 + np.sqrt(np.clip(4 * L * (1 - L), 0, 1))) * min(1.0, 0.3 + s))[..., None]
    return np.clip(out, 0, 1).astype(np.float32)
