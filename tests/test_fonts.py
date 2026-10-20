"""Bundled fonts (docs/specs/2026-10-20-reel-styles-design.md §4, §9): every file has a licence next to it and loads,
and every font role resolves to a bundled file."""
from pathlib import Path

import pytest
from PIL import ImageFont

from wonhee.edit import text as T

FONT_FILES = sorted(p for p in T.FONTS.iterdir() if p.suffix == ".ttf")
SPEC_FAMILIES = ["BarlowCondensed", "Tinos", "InstrumentSerif", "Inter", "InterTight", "Figtree", "Poppins",
                 "Fraunces", "CaveatBrush", "GochiHand", "Mansalva", "Arimo", "PressStart2P"]


def _family(p: Path) -> str:
    fam = p.stem.split("-")[0]
    return "NotoSans" if fam == "NotoSansMono" else fam  # the mono face shares the Noto Sans OFL


@pytest.mark.parametrize("font", FONT_FILES, ids=lambda p: p.name)
def test_bundled_font_loads_and_has_a_licence_next_to_it(font):
    ImageFont.truetype(str(font), 32).getlength("Hamburgefonstiv")
    lic = [p for p in T.FONTS.iterdir() if p.name in (f"OFL-{_family(font)}.txt", f"LICENSE-{_family(font)}.txt")]
    assert lic, f"no OFL-/LICENSE- file for {font.name}"
    text = lic[0].read_text()
    assert "Open Font License" in text or "Apache License" in text


def test_every_spec_family_is_bundled():
    have = {_family(p) for p in FONT_FILES}
    assert set(SPEC_FAMILIES) <= have, set(SPEC_FAMILIES) - have


def test_every_role_resolves_to_a_bundled_file():
    assert {"headline", "transcript", "editorial", "editorial_italic", "grotesk", "ui", "caption", "caption_round",
            "soft_serif", "marker", "marker_alt", "hand", "autocaption", "pixel", "sans", "mono", "script"} <= set(T.ROLES)
    for role, r in T.ROLES.items():
        assert (T.FONTS / r.file).exists(), role
        T.font_for(role, 24)
    assert T.available()["bundled fonts"]


def test_no_font_is_left_without_a_role():
    used = {r.file for r in T.ROLES.values()}
    assert {p.name for p in FONT_FILES} - used <= {"NotoSans-Regular.ttf"}  # the spec-sheet poster's body face


def _copyright(path: Path) -> str:
    """The font's own name ID 0 string, read straight from its name table (no fontTools dependency)."""
    import struct
    d = path.read_bytes()
    n = struct.unpack(">H", d[4:6])[0]
    off = next(struct.unpack(">4sIII", d[12 + 16 * i:28 + 16 * i])[2] for i in range(n) if d[12 + 16 * i:16 + 16 * i] == b"name")
    count, str_off = struct.unpack(">HH", d[off + 2:off + 6])
    for i in range(count):
        plat, _enc, _lang, nid, length, o = struct.unpack(">HHHHHH", d[off + 6 + 12 * i:off + 18 + 12 * i])
        if nid == 0:
            raw = d[off + str_off + o:off + str_off + o + length]
            return (raw.decode("utf-16-be") if plat in (0, 3) else raw.decode("latin-1")).strip()
    raise AssertionError(f"{path.name} has no copyright string")


def test_licence_files_open_with_the_fonts_own_copyright_line():
    for font in FONT_FILES:
        lic = next(p for p in T.FONTS.iterdir() if p.name in (f"OFL-{_family(font)}.txt", f"LICENSE-{_family(font)}.txt"))
        # the Noto Sans files share one licence file, so any of their strings may open it
        owns = {_copyright(f) for f in FONT_FILES if _family(f) == _family(font)}
        assert lic.read_text().splitlines()[0].strip() in owns, (font.name, lic.name)
