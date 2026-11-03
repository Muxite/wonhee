"""Cinema and night looks: teal-orange, bleach bypass, Cinestill, night city, cold blue flash ..."""
from ..grading import Grade, Finish, Look, register_look, Curve, s_curve, ink

W = (1, 1, 1)
RED = (1.0, 0.25, 0.1)

register_look(Look(
    "teal-orange", "cinematic, punchy, blockbuster",
    "Shadows pushed to teal, highlights and warm midtones to amber, S-curve contrast, selective cyan/blue shift. Skin stays orange (protected).",
    "landscapes with sky or water, street scenes, people (skin protection keeps faces natural).",
    "already orange-heavy scenes (sunsets turn flat orange), pure foliage.",
    lambda p: Grade(contrast=1.06, curve=s_curve(0.55), sat=0.06, vib=0.15,
                    ranges=[(250, 40, -55, 0.2, -0.02), (200, 35, -8, 0.15, 0.0), (140, 30, 22, -0.25, 0.0), (30, 30, 8, 0.1, 0.0)],
                    split=dict(sh=(210, 0.055), mid=(190, 0.008), hi=(60, 0.045), bal=0.0), skin=0.55, shoulder=0.35),
    lambda p: Finish(grain=0.25, grain_size=1.0, vignette=(0.18, 2.2, (0, 0, 0)), bloom=(0.10, 14, 0.8, (1, .95, .85))),
    tags=["cinema", "teal", "orange", "contrast"]))

register_look(Look(
    "bleach-bypass", "gritty, desaturated, high contrast",
    "Silver-retention film process: colour overlaid with its own luminance, half the saturation, hard contrast, crunchy grain.",
    "faces, architecture, war/doc moods, rain, metal and concrete.",
    "pastel subjects; heavy shadows crush on dark photos.",
    lambda p: Grade(bypass=0.6, sat=-0.28, curve=s_curve(0.55), contrast=1.04, lift=0.012, skin=0.2, shoulder=0.2),
    lambda p: Finish(grain=0.55, grain_size=1.2, grain_chroma=0.1, vignette=(0.22, 2.4, (0, 0, 0)), clarity=0.25),
    tags=["gritty", "desaturated", "contrast"]))

register_look(Look(
    "cinestill-800t", "neon night, tungsten film, red halation",
    "Tungsten-balanced stock shot under street light: cyan-blue shadows, warm highlights, strong red halation round every light, soft bloom, visible grain.",
    "night photos, signage, car lights, city streets, people lit by lamps.",
    "daylight (turns everything cyan); flat low-light scenes with no bright lights have no halation to show.",
    lambda p: Grade(temp=-0.12, curve=Curve([(0, 0.02), (0.12, 0.13), (0.5, 0.52), (0.85, 0.9), (1, 0.98)]),
                    curve_r=Curve([(0, 0), (0.5, 0.5), (1, 1.0)]), curve_b=Curve([(0, 0.03), (0.3, 0.33), (1, 0.97)]),
                    split=dict(sh=(215, 0.035), hi=(55, 0.03)), sat=0.08, vib=0.2, skin=0.4, shoulder=0.4, lift=0.01),
    lambda p: Finish(grain=0.6, grain_size=1.4, grain_chroma=0.3, halation=(0.9, 22, RED, 0.5), bloom=(0.22, 26, 0.65, (1, .9, .85)), vignette=(0.12, 2.2, (0, 0, 0))),
    tags=["night", "halation", "film", "tungsten"]))

register_look(Look(
    "night-city", "electric, blue hour, urban",
    "Lifted navy blacks, magenta-orange highlights, teal mids, a bit of bloom: city lights glow without crushing the dark.",
    "skylines, plazas, signage, rain, people under lamps.",
    "bright daytime scenes (look turns murky).",
    lambda p: Grade(exposure=0.1, curve=Curve([(0, 0.035), (0.1, 0.13), (0.4, 0.43), (0.75, 0.80), (1, 0.98)]), sat=0.12, vib=0.25,
                    ranges=[(260, 35, -8, 0.1, 0.02), (330, 35, 0, 0.1, 0.0), (55, 30, 0, 0.0, 0.0)],
                    split=dict(sh=(262, 0.045), mid=(200, 0.012), hi=(25, 0.03)), skin=0.35, shoulder=0.5),
    lambda p: Finish(grain=0.3, bloom=(0.28, 20, 0.6, (1, .92, 1.0)), halation=(0.25, 16, (1.0, 0.35, 0.5), 0.55), vignette=(0.2, 2.2, (0, 0, 0.1))),
    tags=["night", "blue", "city", "neon"]))

register_look(Look(
    "cold-blue-flash", "cold, flash-lit, candid, film",
    "Direct-flash party/portrait look: whole frame cast cold blue (red channel pulled down hard through the mids), deep navy blacks, bright near-white highlights, flash falloff (bright centre, dark corners), soft bloom, visible grain and a little chromatic aberration. Calibrated on the statistics of a film-flash portrait (blue 255 deg, black point L .16, white L .94); the photo itself is not used.",
    "people, interiors, night street portraits, anything with a close subject.",
    "daylight landscapes (become blue monochrome); skin intentionally turns cool, so skin=0 by default.",
    lambda p: Grade(exposure=-0.05, temp=-0.6, tint=-0.04, curve=s_curve(0.6), contrast=1.04,
                    curve_r=Curve([(0, 0.0), (0.3, 0.15), (0.6, 0.45), (1, 0.93)]), curve_g=Curve([(0, 0), (0.5, 0.47), (1, 0.97)]),
                    curve_b=Curve([(0, 0.07), (0.3, 0.38), (0.7, 0.8), (1, 1)]), lift=0.015,
                    sat=-0.05, ranges=[(55, 40, 0, -0.3, 0.0)],
                    split=dict(sh=(262, 0.04), mid=(255, 0.02), hi=(235, 0.012)), shoulder=0.45, vib=0.1),
    lambda p: Finish(grain=0.5, grain_size=1.4, grain_chroma=0.2, bloom=(0.28, 18, 0.55, (0.85, 0.92, 1.0)), flash=(0.6, 0.62),
                     vignette=(0.2, 2.0, (0, 0.02, 0.08)), ca=1.6, soften=0.12, halation=(0.12, 14, (0.5, 0.7, 1.0), 0.7)),
    tags=["cold", "blue", "flash", "portrait", "grain"], web_note="the flash falloff and grain are finishing only"))

register_look(Look(
    "overcast-cool", "calm, cool, quiet",
    "Slightly cool white balance, desaturated, soft contrast with blue shadows: grey-day Nordic mood.",
    "landscapes, buildings, forests, calm portraits.",
    "already blue/grey images (little change); night.",
    lambda p: Grade(temp=-0.22, sat=-0.2, curve=s_curve(0.2), lift=0.012, split=dict(sh=(245, 0.025), hi=(215, 0.012)), skin=0.5, shoulder=0.25),
    lambda p: Finish(grain=0.22, vignette=(0.1, 2.2, (0, 0, 0))),
    tags=["cool", "calm", "muted"]))

register_look(Look(
    "moody-forest", "dark, green, damp, quiet",
    "Greens shifted toward deep teal-moss and darkened, yellows muted, blacks lifted green, low saturation elsewhere, hard vignette.",
    "foliage, parks, hiking photos, wildlife, birds in grass.",
    "no green to work with (becomes dull grey); skin keeps its colour but sits in a green-grey world.",
    lambda p: Grade(exposure=-0.15, sat=-0.12, curve=s_curve(0.5), lift=0.01,
                    ranges=[(142, 40, 14, 0.0, -0.06), (100, 30, 22, -0.45, -0.04), (55, 30, -4, -0.25, 0.0), (264, 40, 0, -0.2, -0.04)],
                    split=dict(sh=(175, 0.035), hi=(95, 0.02)), skin=0.6, shoulder=0.3),
    lambda p: Finish(grain=0.35, vignette=(0.32, 2.0, (0, 0.02, 0)), bloom=(0.10, 16, 0.75, (0.95, 1, 0.9))),
    tags=["green", "dark", "nature", "moody"]))

register_look(Look(
    "desaturated-documentary", "sober, honest, news-like",
    "Muted colour (-40%), mild contrast, neutral-cool shadows, slight clarity and light grain. Looks 'unstyled' on purpose.",
    "reportage, people, screenshots-with-photos, product shots that must not shout.",
    "subjects that rely on colour (sunsets, neon): they go grey.",
    lambda p: Grade(sat=-0.38, curve=s_curve(0.35), split=dict(sh=(235, 0.015)), skin=0.5, shoulder=0.2),
    lambda p: Finish(grain=0.3, clarity=0.25, vignette=(0.08, 2.0, (0, 0, 0))),
    tags=["desaturated", "documentary", "neutral"]))

register_look(Look(
    "high-key-editorial", "bright, clean, magazine",
    "Exposure up, white roll-off kept, lifted shadows, soft low contrast, gentle warm skin and a clean cool-white cast; subtle diffusion.",
    "portraits, product, light interiors, fashion, daytime street.",
    "dark or night images (washes out), already overexposed skies.",
    lambda p: Grade(exposure=0.45, shoulder=0.75, knee=0.6, curve=Curve([(0, 0.04), (0.2, 0.27), (0.5, 0.58), (0.85, 0.91), (1, 0.97)]),
                    sat=-0.05, vib=0.1, split=dict(sh=(30, 0.012), hi=(230, 0.008)), skin=0.7),
    lambda p: Finish(grain=0.12, soften=0.25, bloom=(0.12, 22, 0.7, (1, 0.97, 0.95))),
    tags=["bright", "clean", "portrait", "high-key"]))

register_look(Look(
    "golden-hour", "warm, glowing, late sun",
    "Warm balance, amber highlights with a soft glow, lifted warm shadows, slightly richer oranges and greens-to-gold.",
    "landscapes, faces, water, buildings, anything that should feel like 6pm.",
    "scenes with important blues (skies turn teal-grey); night.",
    lambda p: Grade(exposure=0.1, temp=0.4, curve=s_curve(0.3), sat=0.05, vib=0.2, lift=0.01,
                    ranges=[(142, 40, -25, 0.0, 0.03), (264, 40, 12, -0.15, 0.0)],
                    split=dict(sh=(40, 0.02), hi=(75, 0.035)), skin=0.5, shoulder=0.5),
    lambda p: Finish(grain=0.2, bloom=(0.30, 24, 0.55, (1, 0.85, 0.6)), soften=0.12, vignette=(0.16, 2.0, (0.05, 0.02, 0))),
    tags=["warm", "gold", "glow"]))

register_look(Look(
    "cyber-neon", "synthetic, nightclub, cyberpunk",
    "Shadows to deep violet-blue, highlights pushed pink/cyan, saturation high, strong contrast, heavy bloom. Colour, not realism.",
    "night photos, signage, lit people, glass and metal.",
    "skin tones go magenta or cyan (skin=0.15 only); washed daylight scenes just look tinted.",
    lambda p: Grade(curve=s_curve(0.75), contrast=1.08, sat=0.28, vib=0.3,
                    ranges=[(330, 40, 0, 0.2, 0.0), (195, 35, 0, 0.2, 0.0), (100, 40, 25, 0.0, 0.0)],
                    split=dict(sh=(285, 0.06), mid=(330, 0.02), hi=(195, 0.04)), skin=0.15, shoulder=0.4),
    lambda p: Finish(grain=0.3, bloom=(0.45, 20, 0.5, (1, 0.7, 1.0)), halation=(0.3, 14, (1, 0.2, 0.8), 0.55), ca=1.2, vignette=(0.25, 2.0, (0.05, 0, 0.1))),
    tags=["neon", "night", "synthetic", "saturated"]))

register_look(Look(
    "sunset-hue", "dusk, one-hue family, parametric",
    "Pulls every hue toward a target (default warm 40 deg OKLCH; try 20 red, 330 magenta, 195 cyan, 265 blue), then splits shadows cool-dark, highlights warm. The photographic sibling of feature-sunset-grade.",
    "any photo you need to match a project's hue, skies, water, plazas.",
    "stays recognisable but loses real colour differences: wants a strong subject; heavy pull on people makes skin flat (lower `pull`).",
    lambda p: Grade(pull=(float(p["hue"]), float(p["pull"]), 0.0), sat=0.25, vib=0.3, curve=s_curve(0.45), shoulder=0.4,
                    split=dict(sh=(float(p["hue"]) + 200, 0.04), mid=(float(p["hue"]), 0.02), hi=(float(p["hue"]) + 10, 0.05)), skin=0.15),
    lambda p: Finish(grain=0.25, bloom=(0.22, 20, 0.6, (1, 0.9, 0.8)), vignette=(0.18, 2.2, (0, 0, 0))),
    params={"hue": (40.0, "target OKLCH hue in degrees"), "pull": (0.92, "0..1 how far every hue is pulled toward it")},
    tags=["hue", "parametric", "dusk"]))
