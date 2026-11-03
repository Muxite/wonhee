"""Film-stock and analogue-photo looks."""
from ..grading import Grade, Finish, Look, register_look, Curve, s_curve

register_look(Look(
    "portra-warm", "soft, warm, natural, wedding",
    "Portra-like: warm-neutral balance, gentle contrast, lifted shadows, creamy pastel skin, slightly yellow-green foliage, smooth highlight shoulder, fine grain.",
    "portraits, weddings, everyday people photos, landscapes in soft light, food.",
    "scenes that need punch (looks polite); very cool/blue-hour scenes lose their mood.",
    lambda p: Grade(exposure=0.12, temp=0.14, shoulder=0.6, knee=0.62, curve=Curve([(0, 0.02), (0.15, 0.17), (0.5, 0.53), (0.85, 0.88), (1, 0.98)]),
                    sat=-0.04, vib=0.12, ranges=[(142, 35, -12, -0.15, 0.02), (264, 40, 4, -0.12, 0.02)],
                    split=dict(sh=(215, 0.012), hi=(70, 0.02)), skin=0.6),
    lambda p: Finish(grain=0.3, grain_size=1.0, grain_chroma=0.35, soften=0.08, bloom=(0.08, 18, 0.75, (1, 0.95, 0.9))),
    tags=["film", "warm", "portrait", "soft"]))

register_look(Look(
    "kodachrome", "saturated, dense, 1970s slide",
    "Rich saturated reds and deep blues, warm cast, inky blacks, firm contrast, creamy yellows. Dense and graphic.",
    "sunny landscapes, signs, cars, buildings, people outdoors, ducks and birds.",
    "dim indoor photos (blacks go dead); skin gets ruddy in strong sun (skin protection moderate).",
    lambda p: Grade(temp=0.12, curve=s_curve(0.75), contrast=1.05, sat=0.2, vib=0.1,
                    ranges=[(29, 30, -4, 0.2, -0.02), (264, 40, -2, 0.18, -0.05), (142, 40, 6, 0.0, -0.03), (100, 25, 0, 0.1, 0.02)],
                    split=dict(sh=(20, 0.012), hi=(85, 0.015)), skin=0.5, shoulder=0.3),
    lambda p: Finish(grain=0.32, grain_size=1.0, vignette=(0.12, 2.2, (0, 0, 0))),
    tags=["film", "saturated", "slide", "warm"]))

register_look(Look(
    "faded-film", "worn, nostalgic, washed",
    "Lifted blacks, capped whites, 20% less colour, green-cyan shadows and warm highlights, visible grain: a print left on a windowsill.",
    "everything casual: street, friends, landscapes, interiors.",
    "needs no contrast: poorly lit already-flat photos go milky.",
    lambda p: Grade(lift=0.07, cap=0.94, sat=-0.2, curve=s_curve(0.25), split=dict(sh=(185, 0.03), hi=(75, 0.025)), skin=0.5, shoulder=0.3),
    lambda p: Finish(grain=0.55, grain_size=1.2, grain_chroma=0.3, vignette=(0.15, 2.2, (0.08, 0.06, 0.03))),
    tags=["film", "faded", "nostalgic", "matte"]))

register_look(Look(
    "polaroid", "instant print, sun-bleached, cream",
    "Cream-yellow cast with cyan-teal shadows, low contrast, soft focus and edge falloff, slightly blown highlights.",
    "friends, food, summer, small snapshots, window light.",
    "dark scenes (muddy); no frame is drawn, add one in layout if wanted.",
    lambda p: Grade(exposure=0.1, temp=0.16, lift=0.05, cap=0.96, sat=-0.1, curve=Curve([(0, 0.03), (0.25, 0.27), (0.6, 0.6), (0.9, 0.94), (1, 0.97)]),
                    curve_b=Curve([(0, 0.06), (0.5, 0.47), (1, 0.92)]), split=dict(sh=(190, 0.03), hi=(95, 0.03)), skin=0.5, shoulder=0.4),
    lambda p: Finish(grain=0.35, grain_size=1.3, soften=0.30, vignette=(0.3, 2.0, (0.05, 0.04, 0.0)), bloom=(0.12, 16, 0.7, (1, 0.97, 0.85))),
    tags=["instant", "faded", "cream", "soft"]))

register_look(Look(
    "cross-process", "chemical accident, punk, poster",
    "E6 slide in C41 chemistry: steeper red and green curves, blue shadows lifted and highlights capped, yellow-green highlights, cyan-blue darks. Tamed so highlights keep detail.",
    "skies, fashion, street, anything you want hotter and weirder.",
    "skin goes yellow-green (protection 0.35); text-heavy images.",
    lambda p: Grade(curve_r=Curve([(0, 0), (0.25, 0.18), (0.6, 0.66), (1, 0.98)]), curve_g=Curve([(0, 0), (0.3, 0.27), (0.7, 0.77), (1, 0.98)]),
                    curve_b=Curve([(0, 0.1), (0.45, 0.45), (0.8, 0.68), (1, 0.84)]), sat=0.1, split=dict(sh=(235, 0.02), hi=(105, 0.025)), skin=0.35, shoulder=0.4),
    lambda p: Finish(grain=0.4, vignette=(0.2, 2.0, (0, 0, 0)), ca=1.0),
    tags=["film", "cross-process", "saturated", "weird"]))

register_look(Look(
    "seventies-warm", "brown-orange, sun-faded, retro",
    "Warm orange cast, greens turned olive-mustard, soft red/orange, lifted brown shadows, low contrast, soft halation.",
    "family-album feel, cars, interiors, parks, people.",
    "blue skies turn grey-green; very dark pictures turn muddy brown.",
    lambda p: Grade(temp=0.3, lift=0.04, cap=0.95, sat=-0.02, curve=s_curve(0.25),
                    ranges=[(142, 40, -42, -0.1, -0.02), (264, 40, 45, -0.35, 0.0), (29, 30, 10, 0.1, 0.0)],
                    split=dict(sh=(45, 0.025), hi=(80, 0.03)), skin=0.4, shoulder=0.4),
    lambda p: Finish(grain=0.5, grain_size=1.2, grain_chroma=0.35, halation=(0.35, 14, (1, 0.5, 0.2), 0.7), soften=0.12, vignette=(0.2, 2.2, (0.1, 0.05, 0.0))),
    tags=["film", "retro", "warm", "olive"]))

register_look(Look(
    "ektachrome-slide", "clean, cool, crisp slide",
    "Slide film: bluish-clean whites, strong blues and greens, high contrast and sharpness, little warmth. Fresh and commercial.",
    "daytime landscapes, water, architecture, cars, outdoors.",
    "warm indoor light (turns cold); skin slightly cool, protected 0.5.",
    lambda p: Grade(temp=-0.1, curve=s_curve(0.6), sat=0.16, vib=0.1, ranges=[(264, 35, 0, 0.18, -0.03), (142, 40, 8, 0.1, 0.0)],
                    split=dict(sh=(250, 0.015), hi=(220, 0.01)), skin=0.5, shoulder=0.3),
    lambda p: Finish(grain=0.18, clarity=0.15, sharpen=0.15, vignette=(0.08, 2.0, (0, 0, 0))),
    tags=["film", "slide", "cool", "clean"]))

register_look(Look(
    "lomo", "toy camera, vignette, saturated",
    "Cross-processed colour, high saturation and contrast, a hard vignette and a touch of chromatic aberration.",
    "street, friends, parties, food, anything cheerful.",
    "subtle subjects; the vignette hides corners.",
    lambda p: Grade(curve=s_curve(0.8), curve_b=Curve([(0, 0.05), (0.5, 0.46), (1, 0.93)]), curve_r=Curve([(0, 0), (0.5, 0.54), (1, 1)]),
                    sat=0.3, vib=0.1, skin=0.35, shoulder=0.3),
    lambda p: Finish(grain=0.35, vignette=(0.62, 1.9, (0, 0, 0)), ca=2.4, soften=0.1),
    tags=["film", "toy", "vignette", "saturated"]))

register_look(Look(
    "tropical", "lush, bright, holiday",
    "Teal-turquoise water, vivid leafy greens leaning yellow, warm skin, lifted shadows, high vibrance and light bloom.",
    "beaches, plants, birds, summer, food, bright daylight.",
    "dull grey scenes (nothing to saturate); night.",
    lambda p: Grade(exposure=0.15, temp=0.08, vib=0.5, sat=0.12, curve=s_curve(0.35),
                    ranges=[(195, 35, 0, 0.35, 0.03), (142, 35, -14, 0.25, 0.02), (264, 35, -25, 0.2, 0.0), (55, 30, 0, 0.1, 0.0)],
                    split=dict(sh=(185, 0.02), hi=(85, 0.02)), skin=0.6, shoulder=0.4),
    lambda p: Finish(grain=0.15, bloom=(0.18, 18, 0.7, (1, 1, 0.9)), clarity=0.12),
    tags=["bright", "saturated", "nature", "summer"]))
