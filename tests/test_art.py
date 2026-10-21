"""Project art: the colour rules from DESIGN.md › Project art › Colour, checked for every domain, polarity and harmony."""
import itertools

import pytest

from wonhee.edit.viz import palette as pal

CASES = list(itertools.product(sorted(pal.DOMAIN_HUE), ("light", "dark"), ("mono", "analogous", "complementary", "split")))


def _spread(hex_color: str) -> int:
    ch = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]
    return max(ch) - min(ch)


@pytest.mark.parametrize("domain,polarity,harmony", CASES)
def test_roles_keep_the_lightness_rules(domain, polarity, harmony):
    r = pal.for_domain(domain, polarity, harmony)
    L = {k: pal.lightness(getattr(r, k)) for k in ("canvas", "structure", "edge", "accent", "ink")}
    if polarity == "light":
        assert .94 <= L["canvas"] <= .98
    else:
        assert .12 <= L["canvas"] <= .18
    assert abs(L["structure"] - L["canvas"]) >= .25
    assert abs(L["accent"] - L["canvas"]) >= .30
    # edges sit closer to the canvas than the vertices they join, by 25-35% (±rounding)
    ratio = abs(L["edge"] - L["canvas"]) / abs(L["structure"] - L["canvas"])
    assert .62 <= ratio <= .78
    assert r.canvas not in ("#000000", "#ffffff") and r.ink not in ("#000000", "#ffffff")


@pytest.mark.parametrize("polarity", ("light", "dark"))
def test_canvas_is_a_tinted_neutral_and_accent_is_not(polarity):
    r = pal.for_domain("compute", polarity)
    assert 0 < _spread(r.canvas) <= 14  # near-neutral, but not pure grey: tinted toward the key hue
    assert _spread(r.accent) >= 60


def test_editorial_stays_near_neutral():
    assert _spread(pal.for_domain("editorial", "light").canvas) < _spread(pal.for_domain("compute", "light").canvas) + 1
    assert _spread(pal.for_domain("editorial", "dark").structure) <= _spread(pal.for_domain("compute", "dark").structure)


def test_gamut_clip_keeps_lightness():
    # a very saturated blue at L .72 is outside sRGB; chroma shrinks, lightness must not move
    assert abs(pal.lightness(pal.to_hex(.72, .35, 260)) - .72) < .02
    assert abs(pal.lightness(pal.to_hex(.50, .05, 100)) - .50) < .01


def test_split_tones_the_neutrals_cool_whatever_the_accent():
    warm = pal.roles(60, "dark", "split")
    cool = pal.roles(240, "dark", "split")
    assert warm.canvas == cool.canvas and warm.structure == cool.structure


def test_unknown_domain_is_refused():
    with pytest.raises(ValueError, match="unknown art domain"):
        pal.for_domain("vibes", "light")
