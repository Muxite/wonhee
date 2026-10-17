"""Text layers for reels and slides: Pillow-rendered, colour emoji and CJK included (libass can do neither well).

Adapted from slopcut/text.py, plus per-run font fallback: emoji runs are drawn with Noto Color Emoji
(embedded colour, no stroke), CJK/Hangul runs with Noto Sans CJK, everything else with the preset's font.
`render_layer` returns a full-frame RGBA layer, so callers can overlay it with ffmpeg in the same pass as a look.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFont

FONTS = Path(__file__).resolve().parents[1] / "assets" / "fonts"
SANS_BOLD = FONTS / "NotoSans-Bold.ttf"
SCRIPT = FONTS / "DancingScript.ttf"
EMOJI = Path("/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf")
CJK = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
CJK_FACE = {"kr": 1, "sc": 2}  # .ttc face index: 0 JP, 1 KR, 2 SC, 3 TC
EMOJI_PX = 109  # the only size the bitmap emoji font loads at; glyphs are scaled afterwards

Preset = Literal["outline", "band", "script"]
Position = Literal["top", "upper", "center", "lower", "bottom"]
# Instagram reel UI covers about the top 12% and the bottom 20%; text blocks stay inside.
SAFE_TOP, SAFE_BOTTOM = 0.12, 0.80
ANCHOR_Y = {"upper": 0.30, "center": 0.50, "lower": 0.66}


@dataclass(frozen=True)
class TextSpec:
    content: str
    preset: Preset = "outline"
    position: Position = "upper"
    size_frac: float | None = None  # glyph size as a fraction of frame height; None = preset default
    width_frac: float = 0.84  # wrap width as a fraction of frame width
    band: Literal["dark", "light"] = "dark"  # band preset only: white-on-black or black-on-white


DEFAULT_SIZE = {"outline": 0.040, "band": 0.034, "script": 0.052}


def available() -> dict[str, bool]:
    return {"sans": SANS_BOLD.exists(), "script": SCRIPT.exists(), "emoji": EMOJI.exists(), "cjk": CJK.exists()}


# ---- classification ------------------------------------------------------------------------------------------

def _is_emoji(ch: str) -> bool:
    cp = ord(ch)
    return (0x1F000 <= cp <= 0x1FAFF or 0x2600 <= cp <= 0x27BF or 0x2B00 <= cp <= 0x2BFF
            or cp in (0x00A9, 0x00AE, 0x203C, 0x2049, 0x2122, 0x2139, 0x3030, 0x303D))


def _is_emoji_joiner(ch: str) -> bool:  # ZWJ, variation selectors, skin tones, keycap
    cp = ord(ch)
    return cp in (0x200D, 0xFE0F, 0xFE0E, 0x20E3) or 0x1F3FB <= cp <= 0x1F3FF or 0xE0020 <= cp <= 0xE007F


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return (0x1100 <= cp <= 0x11FF or 0x2E80 <= cp <= 0x9FFF or 0xA960 <= cp <= 0xA97F or 0xAC00 <= cp <= 0xD7FF
            or 0xF900 <= cp <= 0xFAFF or 0xFF00 <= cp <= 0xFFEF or 0x20000 <= cp <= 0x2FA1F)


def runs(text: str) -> list[tuple[str, str]]:
    """Split into (kind, text) runs: kind is 'emoji', 'cjk' or 'base'. Joiners stay with their emoji."""
    out: list[list[str]] = []
    for ch in text:
        if out and out[-1][0] == "emoji" and _is_emoji_joiner(ch):
            kind = "emoji"
        elif _is_emoji(ch):
            kind = "emoji"
        elif _is_cjk(ch):
            kind = "cjk"
        else:
            kind = "base"
        if out and out[-1][0] == kind:
            out[-1][1] += ch
        else:
            out.append([kind, ch])
    return [(k, s) for k, s in out]


# ---- drawing ---------------------------------------------------------------------------------------------------

class _Fonts:
    def __init__(self, preset: str, px: int, text: str):
        if preset == "script":
            self.base = ImageFont.truetype(str(SCRIPT), px)
            self.base.set_variation_by_name("Bold")
        else:
            self.base = ImageFont.truetype(str(SANS_BOLD), px)
        hangul = any(0xAC00 <= ord(c) <= 0xD7AF or 0x1100 <= ord(c) <= 0x11FF for c in text)
        self.cjk = ImageFont.truetype(str(CJK), px, index=CJK_FACE["kr" if hangul else "sc"]) if CJK.exists() else self.base
        self.emoji = ImageFont.truetype(str(EMOJI), EMOJI_PX) if EMOJI.exists() else None
        self.px = px

    def font(self, kind: str):
        return self.cjk if kind == "cjk" else self.base


@dataclass
class _Style:
    fill: str
    stroke: int
    stroke_fill: str


def _style(spec: TextSpec, px: int) -> _Style:
    if spec.preset == "band":
        return _Style("white" if spec.band == "dark" else "black", 0, "black")
    if spec.preset == "script":
        return _Style("white", max(2, round(px * 0.06)), "black")
    return _Style("white", max(2, round(px * 0.09)), "black")


def _emoji_img(fonts: _Fonts, s: str, height: int) -> Image.Image | None:
    if fonts.emoji is None:
        return None
    w = max(1, round(fonts.emoji.getlength(s)))
    im = Image.new("RGBA", (w + 8, EMOJI_PX + 32), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((4, 4), s, font=fonts.emoji, embedded_color=True)
    box = im.getbbox()
    if box is None:
        return None
    im = im.crop(box)
    return im.resize((max(1, round(im.width * height / im.height)), height), Image.LANCZOS)


def _piece(fonts: _Fonts, st: _Style, kind: str, s: str, line_h: int, base_y: int) -> Image.Image:
    """One run as an RGBA strip of the line's height, drawn on the shared baseline."""
    if kind == "emoji":
        h = round(fonts.px * 1.05)
        em = _emoji_img(fonts, s, h)
        if em is not None:
            strip = Image.new("RGBA", (em.width + max(2, fonts.px // 12), line_h), (0, 0, 0, 0))
            strip.alpha_composite(em, (0, max(0, base_y - h + round(fonts.px * 0.12))))
            return strip
        kind = "base"  # no emoji font: fall back to the base font (likely tofu, but never crash)
    f = fonts.font(kind)
    w = round(f.getlength(s)) + 2 * st.stroke + 2
    strip = Image.new("RGBA", (max(1, w), line_h), (0, 0, 0, 0))
    ImageDraw.Draw(strip).text((st.stroke + 1, base_y), s, font=f, fill=st.fill, anchor="ls",
                               stroke_width=st.stroke, stroke_fill=st.stroke_fill)
    return strip


def _tokens(text: str) -> list[tuple[str, bool]]:
    """Wrap units: (text, space_before). Words stay whole; each CJK character is its own unit; '\\n' forces a break."""
    out: list[tuple[str, bool]] = []
    for li, line in enumerate(text.split("\n")):
        if li:
            out.append(("\n", False))
        for wi, word in enumerate(line.split(" ")):
            if not word:
                continue
            first = True
            buf = ""
            for kind, s in runs(word):
                if kind == "cjk":
                    if buf:
                        out.append((buf, first and wi > 0))
                        first, buf = False, ""
                    for ch in s:
                        out.append((ch, first and wi > 0))
                        first = False
                else:
                    buf += s
            if buf:
                out.append((buf, first and wi > 0))
    return out


def _render_lines(spec: TextSpec, size: tuple[int, int]) -> tuple[list[Image.Image], int]:
    W, H = size
    px = max(10, round(H * (spec.size_frac or DEFAULT_SIZE[spec.preset])))
    fonts = _Fonts(spec.preset, px, spec.content)
    st = _style(spec, px)
    line_h = round(px * 1.32) + 2 * st.stroke
    base_y = round(px * 1.02) + st.stroke
    space = round(fonts.base.getlength(" "))
    max_w = W * spec.width_frac

    pad = 2 * st.stroke + 2  # each text strip carries its stroke on both sides; neighbours overlap by it

    def unit(s: str) -> list[tuple[Image.Image, int]]:
        return [(im, 0 if k == "emoji" else pad)
                for k, t in runs(s) for im in [_piece(fonts, st, k, t, line_h, base_y)]]

    def advance(parts) -> int:
        return sum(im.width - p for im, p in parts)

    lines: list[list[tuple[Image.Image, int]]] = [[]]
    widths = [0]
    for tok, space_before in _tokens(spec.content.strip()):
        if tok == "\n":
            lines.append([])
            widths.append(0)
            continue
        parts = unit(tok)
        gap = space if space_before and lines[-1] else 0
        if lines[-1] and widths[-1] + gap + advance(parts) > max_w:
            lines.append([])
            widths.append(0)
            gap = 0
        if gap:
            lines[-1].append((Image.new("RGBA", (gap, line_h), (0, 0, 0, 0)), 0))
        lines[-1].extend(parts)
        widths[-1] += gap + advance(parts)
    out = []
    for pieces in lines:
        if not pieces:
            continue
        img = Image.new("RGBA", (advance(pieces) + pieces[-1][1], line_h), (0, 0, 0, 0))
        x = 0
        for im, p in pieces:
            img.alpha_composite(im, (x, 0))
            x += im.width - p
        out.append(img)
    return out, px


def render_layer(spec: TextSpec, size: tuple[int, int]) -> Image.Image:
    """A full-frame transparent RGBA layer with the text block placed per preset and position."""
    W, H = size
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    if not spec.content.strip():
        return layer
    lines, px = _render_lines(spec, size)
    gap = round(px * 0.08)
    block_h = sum(l.height for l in lines) + gap * (len(lines) - 1)
    if spec.position == "top":
        y0 = round(H * SAFE_TOP)
    elif spec.position == "bottom":
        y0 = round(H * SAFE_BOTTOM) - block_h
    else:
        y0 = round(H * ANCHOR_Y[spec.position] - block_h / 2)
    y0 = max(round(H * 0.02), min(y0, H - block_h - round(H * 0.02)))
    if spec.preset == "band":
        pad = round(px * 0.55)
        colour = (0, 0, 0, 235) if spec.band == "dark" else (255, 255, 255, 240)
        ImageDraw.Draw(layer).rectangle((0, y0 - pad, W, y0 + block_h + pad), fill=colour)
    y = y0
    for l in lines:
        layer.alpha_composite(l, (max(0, (W - l.width) // 2), y))
        y += l.height + gap
    return layer


def draw_on(base: Image.Image, spec: TextSpec) -> Image.Image:
    """Composite a text block onto an image (RGB in, RGB out)."""
    rgba = base.convert("RGBA")
    rgba.alpha_composite(render_layer(spec, rgba.size))
    return rgba.convert("RGB")
