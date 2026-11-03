"""Black and white, toned monochrome and duotones."""
from ..grading import Grade, Finish, Look, register_look, Curve, s_curve, ink

register_look(Look(
    "noir-bw", "black and white, hard, graphic, grainy",
    "Red-filter monochrome (dark skies, bright skin), strong S-curve, deep blacks and bright whites, heavy grain and a vignette.",
    "faces, architecture, night scenes, textures, birds.",
    "very dark photos crush; low-contrast subjects turn grey.",
    lambda p: Grade(mono=1.0, mono_w=(0.5, 0.42, 0.08), curve=s_curve(0.9), contrast=1.05, shoulder=0.2),
    lambda p: Finish(grain=0.6, grain_size=1.2, grain_chroma=0.0, vignette=(0.4, 2.0, (0, 0, 0)), clarity=0.2),
    tags=["bw", "hard", "grain"]))

register_look(Look(
    "bw-airy", "black and white, soft, light, quiet",
    "High-key monochrome: lifted blacks, bright mids, soft contrast and a slightly cool silver tone, fine grain.",
    "portraits, flowers, snow, minimal architecture, quiet street.",
    "needs enough light; dark photos go grey and flat.",
    lambda p: Grade(mono=1.0, mono_w=(0.34, 0.5, 0.16), exposure=0.35, shoulder=0.6, lift=0.04, curve=Curve([(0, 0.03), (0.25, 0.3), (0.6, 0.66), (1, 0.97)]),
                    split=dict(sh=(250, 0.01), hi=(70, 0.006))),
    lambda p: Finish(grain=0.3, grain_chroma=0.0, soften=0.18),
    tags=["bw", "soft", "high-key"]))

def _gm(dark, light, mid=None):
    g = [(0.0, dark), (1.0, light)]
    if mid is not None:
        g.insert(1, (0.5, mid))
    return g

register_look(Look(
    "cyanotype", "Prussian blue print, botanical, sun print",
    "Cyanotype: tone mapped from deep Prussian blue to a pale blue-white paper, matte, grainy, slightly stained.",
    "plants, birds, architecture, portraits as botanical-plate cards.",
    "colour contrast is lost: red and green subjects merge in tone.",
    lambda p: Grade(mono=1.0, mono_w=(0.3, 0.55, 0.15), curve=s_curve(0.5), gmap=_gm(ink(0.2, 0.08, 255), ink(0.95, 0.025, 215), ink(0.5, 0.115, 245)), gmap_mix=1.0),
    lambda p: Finish(grain=0.4, grain_chroma=0.0, vignette=(0.2, 2.0, (0, 0.03, 0.1)), soften=0.1),
    tags=["mono", "blue", "print", "cyan"]))

register_look(Look(
    "sepia-albumen", "antique print, brown, quiet",
    "Albumen-print sepia: warm brown shadows to cream highlights, lowered contrast and capped whites, soft vignette.",
    "portraits, old-town architecture, landscapes in haze, heritage pages.",
    "colour-driven subjects; modern signage looks anachronistic.",
    lambda p: Grade(mono=1.0, mono_w=(0.32, 0.55, 0.13), curve=s_curve(0.35), gmap=_gm(ink(0.22, 0.045, 55), ink(0.94, 0.04, 85), ink(0.6, 0.08, 62)), gmap_mix=1.0),
    lambda p: Finish(grain=0.4, grain_chroma=0.0, vignette=(0.3, 2.0, (0.1, 0.06, 0.0)), soften=0.12),
    tags=["mono", "sepia", "antique", "brown"]))

_DUO = {
    "indigo": ("indigo ink on pale paper, calm and graphic", ink(0.2, 0.12, 275), ink(0.96, 0.02, 270), None),
    "crimson": ("crimson ink on warm paper, loud print", ink(0.2, 0.1, 15), ink(0.95, 0.04, 75), ink(0.5, 0.16, 28)),
    "forest": ("deep green ink on cream paper, botanical", ink(0.22, 0.07, 160), ink(0.95, 0.04, 105), None),
    "amber": ("dark brown to amber glow, warm screenprint", ink(0.16, 0.05, 50), ink(0.9, 0.15, 75), ink(0.6, 0.15, 55)),
}
for key, (mood, dk, lt, md) in _DUO.items():
    register_look(Look(
        f"duotone-{key}", f"duotone, {mood}",
        "Tone is mapped between two inks (dark ink to paper) in OKLab, with firm contrast so detail stays; every source collapses to the same palette, which makes mixed photos consistent.",
        "card art that must match a project's colour; photos of people (reads as a print); textures.",
        "colour contrast is lost; very flat images give a one-tone wash.",
        (lambda dk=dk, lt=lt, md=md: (lambda p: Grade(mono=1.0, mono_w=(0.3, 0.57, 0.13), curve=s_curve(0.85), contrast=1.04, gmap=_gm(dk, lt, md), gmap_mix=1.0)))(),
        (lambda key=key: (lambda p: Finish(grain=0.3, grain_chroma=0.0, grain_size=1.1, vignette=(0.1, 2.0, (0, 0, 0)))))(),
        tags=["mono", "duotone", "ink", key]))
