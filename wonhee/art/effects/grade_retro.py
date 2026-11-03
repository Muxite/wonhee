"""Digital-era and screen looks: Y2K digicam, VHS, CRT, thermal, infrared false colour."""
from ..grading import Grade, Finish, Look, register_look, Curve, s_curve, ink

register_look(Look(
    "y2k-digicam", "2000s compact camera, flash, early digital",
    "Hard on-camera flash feel: blown highlights, cyan-blue cast in shadows, crisp over-sharpened edges, chroma noise, tiny lateral aberration, saturated colour.",
    "friends, parties, cars, street snaps, anything candid; reads great at small size.",
    "calm landscapes (looks wrong on purpose); skin gets a magenta-pink push (protected 0.45).",
    lambda p: Grade(exposure=0.2, temp=-0.08, tint=0.06, curve=Curve([(0, 0.0), (0.2, 0.17), (0.55, 0.62), (0.85, 0.96), (1, 1)]),
                    sat=0.22, vib=0.1, split=dict(sh=(235, 0.025), hi=(95, 0.015)), skin=0.45),
    lambda p: Finish(grain=0.4, grain_size=0.9, grain_chroma=0.7, chroma_noise=0.8, sharpen=0.55, ca=1.3, flash=(0.5, 0.7), vignette=(0.12, 2.0, (0, 0, 0))),
    tags=["digital", "flash", "y2k", "snapshot"]))

register_look(Look(
    "vhs", "tape, home video, 80s-90s",
    "Chroma bleeding sideways, soft luma with tape edge halos, ghosting, row jitter and a tracking band, cool-green shadows and magenta highlights, low saturation but pushed colour.",
    "video stills, people, signage, anything you want to feel like a recording.",
    "small text and fine detail disappear; tracking band is seed-dependent.",
    lambda p: Grade(sat=-0.12, curve=Curve([(0, 0.04), (0.5, 0.5), (0.85, 0.88), (1, 0.97)]), temp=-0.03, tint=0.05,
                    split=dict(sh=(185, 0.03), hi=(340, 0.03)), skin=0.3),
    lambda p: Finish(grain=0.4, grain_size=1.1, vhs=dict(bleed=7, shift=4, luma_blur=1.6, jitter=2.0, ghost=(9, 0.5), noise=0.8, tracking=0.5), vignette=(0.15, 2.2, (0, 0, 0))),
    tags=["retro", "video", "tape", "glitch"]))

register_look(Look(
    "crt", "phosphor monitor, scanlines, glow",
    "Horizontal scanlines, RGB phosphor mask, bloom around bright pixels, slightly curved glass and corner falloff; colour a bit saturated.",
    "pixel art, UI screenshots, graphics, hero backgrounds, games.",
    "photos of people get the 'monitor photo' feel; fine mask moire at some downsizes (period is clamped to 2 px).",
    lambda p: Grade(curve=s_curve(0.5), sat=0.12, skin=0.0, split=dict(sh=(190, 0.012))),
    lambda p: Finish(grain=0.12, crt=dict(scan=0.9, period=4.0, mask=0.55, barrel=0.8), bloom=(0.3, 12, 0.5, (1, 1, 1)), vignette=(0.4, 2.2, (0, 0, 0)), ca=1.0),
    tags=["retro", "screen", "scanlines", "pixel"]))

_HEAT = [(0.0, ink(0.1, 0.05, 280)), (0.2, ink(0.28, 0.16, 300)), (0.4, ink(0.45, 0.2, 350)), (0.6, ink(0.62, 0.2, 30)),
         (0.8, ink(0.82, 0.17, 80)), (1.0, ink(0.98, 0.04, 100))]
register_look(Look(
    "thermal-map", "heat camera, false colour",
    "Lightness mapped through a thermal ramp (black, violet, magenta, red, amber, near-white). Colour information is replaced; structure stays.",
    "night photos, glowing lights, faces (reads as a heat portrait), textures.",
    "low-contrast flat images give a one-colour wash.",
    lambda p: Grade(contrast=1.05, curve=s_curve(0.5), gmap=_HEAT, gmap_mix=1.0),
    lambda p: Finish(grain=0.18, bloom=(0.18, 14, 0.7, (1, 0.9, 0.9))),
    tags=["false-colour", "thermal", "gradient-map"]))

register_look(Look(
    "infrared-false-colour", "Aerochrome, pink foliage, dreamy",
    "Infrared colour film: foliage and green water turn hot pink-red and bright, sky and blue deepen to violet, soft glow. Only greens and cyans are remapped, so skin and neutrals survive.",
    "trees, grass, parks, birds in grass, water edges.",
    "scenes without green barely change (a cool tint only).",
    lambda p: Grade(curve=s_curve(0.35), vib=0.2, sat=0.05, shoulder=0.3,
                    ranges=[(142, 38, -120, 0.35, 0.16), (110, 25, -95, 0.3, 0.12), (195, 35, 70, 0.1, -0.02), (264, 40, 40, 0.05, -0.05), (100, 22, -75, 0.0, 0.1)],
                    split=dict(sh=(300, 0.02)), skin=0.7),
    lambda p: Finish(grain=0.3, soften=0.3, bloom=(0.2, 20, 0.6, (1, 0.9, 0.95))),
    tags=["false-colour", "infrared", "nature", "pink"]))
