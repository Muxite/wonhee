"""Drawing for the GitHub slop machine (docs/specs/2026-10-20-github-slop-design.md): the SlopScript a model writes,
and the 4 card slides (1080x1350, 4:5) a deterministic renderer makes from a dossier plus that script.

Facts on a card (stars, languages, dates, licence, owner/name) are read from the dossier by code. The script holds
only voice lines, so a model cannot put a number on a card. Real pixels only: text, gradients, bars, and the repo's
own dependency graph.
"""
from __future__ import annotations

import colorsys
import re
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from wonhee.edit.text import SANS_BOLD, TextSpec, draw_on
from wonhee.edit.viz import depgraph, depgraph_html
from wonhee.edit.viz.colors import hash_hue
from wonhee.edit.viz.export import ExportUnavailable, export_png
from wonhee.edit.viz.ignore import globs_spec
from wonhee.edit.viz.source import SourceError, snapshot

W, H = 1080, 1350
FOOTER_Y = 1262  # the credit band: owner/name is drawn here on every slide
GRAPH_X, GRAPH_Y, GRAPH_W = 48, 560, 984  # where the depgraph still goes on slide 2 (it may run down to the footer)
EDGE_RGB = (0x9A, 0xA6, 0xD0)  # edges are redrawn heavier and brighter than the viz default so they read at phone size
MIN_EDGE_PIXELS = 400  # fewer bright edge pixels than this and the graph counts as unreadable


def _words(label: str, limit: int):
    def check(v: str) -> str:
        v = v.strip()
        if not v or len(v.split()) > limit:
            raise ValueError(f"{label} must be 1 to {limit} words, got {len(v.split())}")
        return v
    return check


class SlopScript(BaseModel):
    """Voice lines only. Facts come from the dossier, never from here."""
    model_config = ConfigDict(extra="forbid")
    hook: str = Field(description="big opening line, at most 9 words")
    what: str = Field(description="what the repo is, at most 24 words")
    why: str = Field(description="why anyone would care, at most 24 words")
    roast: str = Field(description="affectionate joke about the code, never the author, at most 16 words")
    caption: str = Field(max_length=300, description="Instagram caption, at most 300 characters")
    hashtags: list[str] = Field(max_length=6, description="at most 6, without the # sign")

    _hook = field_validator("hook")(_words("hook", 9))
    _what = field_validator("what")(_words("what", 24))
    _why = field_validator("why")(_words("why", 24))
    _roast = field_validator("roast")(_words("roast", 16))

    @field_validator("caption")
    @classmethod
    def _caption(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("caption is empty")
        return v.strip()

    @field_validator("hashtags")
    @classmethod
    def _tags(cls, v: list[str]) -> list[str]:
        return [t.strip().lstrip("#").replace(" ", "") for t in v if t.strip().lstrip("#")]


# ---- facts (code only) -----------------------------------------------------------------------------------------

def language_shares(dossier: dict, top: int = 5) -> list[tuple[str, float]]:
    langs = {k: v for k, v in (dossier.get("languages") or {}).items() if v > 0}
    total = sum(langs.values())
    if not total:
        return []
    return [(k, v / total) for k, v in sorted(langs.items(), key=lambda kv: -kv[1])[:top]]


def pct(share: float) -> str:
    return "<1%" if share * 100 < 1 else f"{share * 100:.0f}%"


_NUM = re.compile(r"\d[\d.,]*\d|\d")


def check_numbers(script: "SlopScript", dossier_text: str) -> None:
    """design: github-slop §Rules (numbers are checked in code). Every digit token in a voice line must appear in
    the dossier text; a cheap guard against invented figures, not a fact-checker."""
    plain = dossier_text.replace(",", "")
    for field in ("hook", "what", "why", "roast", "caption"):
        for tok in _NUM.findall(getattr(script, field)):
            if tok not in dossier_text and tok.replace(",", "") not in plain:
                raise ValueError(f"{field} has the number {tok!r}, which is not in the dossier; "
                                 "leave numbers out or use only ones from the dossier")


def checked_script_class(dossier_text: str) -> type[SlopScript]:
    """SlopScript plus the digit guard for one dossier, so a model's validation retry sees the refusal."""
    class CheckedSlopScript(SlopScript):
        @model_validator(mode="after")
        def _numbers(self):
            check_numbers(self, dossier_text)
            return self
    CheckedSlopScript.__name__ = "SlopScript"
    return CheckedSlopScript


def stat_lines(dossier: dict) -> list[tuple[str, str]]:
    """(label, value) rows for slide 3, all from the dossier."""
    return [("stars", f"{dossier.get('stars', 0):,}"), ("forks", f"{dossier.get('forks', 0):,}"),
            ("created", str(dossier.get("created_at", ""))[:10] or "?"),
            ("last push", str(dossier.get("pushed_at", ""))[:10] or "?"),
            ("licence", dossier.get("license") or "none stated")]


# ---- drawing ---------------------------------------------------------------------------------------------------

def _font(px: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(SANS_BOLD), px)


def gradient(seed: str) -> Image.Image:
    """Dark vertical gradient whose hue comes from the repo name, so each set has its own colour."""
    h = hash_hue(seed) / 360
    top = np.array(colorsys.hsv_to_rgb(h, 0.55, 0.38)) * 255
    bot = np.array(colorsys.hsv_to_rgb((h + 0.12) % 1, 0.6, 0.07)) * 255
    t = np.linspace(0, 1, H)[:, None, None]
    arr = (top * (1 - t) + bot * t).repeat(W, axis=1).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _footer(im: Image.Image, dossier: dict, n: int) -> None:
    d = ImageDraw.Draw(im)
    f = _font(34)
    d.text((48, FOOTER_Y + 36), dossier["full_name"], font=f, fill="white", anchor="lm")
    d.text((W - 48, FOOTER_Y + 36), f"{n}/4", font=f, fill=(200, 200, 200), anchor="rm")


def _text(im: Image.Image, content: str, position: str, size: float) -> Image.Image:
    return draw_on(im, TextSpec(content, "outline", position, size_frac=size))


def _slide1(d: dict, s: SlopScript) -> Image.Image:
    im = _text(gradient(d["full_name"]), s.hook, "center", 0.062)
    ImageDraw.Draw(im).text((W // 2, 1180), f"{d['owner']} / {d['name']}", font=_font(40), fill=(230, 230, 230),
                            anchor="mm")
    return im


def _slide2(d: dict, s: SlopScript, graph: Image.Image | None) -> Image.Image:
    base = gradient(d["full_name"])
    if graph is None:  # no picture: the line gets the room instead
        return _text(base, s.what, "center", 0.046)
    im = _text(base, s.what, "top", 0.032)
    g = graph.convert("RGB")
    g = g.resize((GRAPH_W, round(g.height * GRAPH_W / g.width)), Image.LANCZOS)
    im.paste(g.crop((0, 0, GRAPH_W, min(g.height, FOOTER_Y - GRAPH_Y - 16))), (GRAPH_X, GRAPH_Y))
    return im


def _slide3(d: dict) -> Image.Image:
    im = gradient(d["full_name"])
    dr = ImageDraw.Draw(im)
    y = 150
    for label, value in stat_lines(d):
        big = label == "stars"
        dr.text((72, y), label.upper(), font=_font(30), fill=(170, 170, 170))
        dr.text((72, y + 36), value, font=_font(120 if big else 64), fill="white")
        y += 220 if big else 130
    shares = language_shares(d)
    if shares:
        x, bar_y, bw = 72, y + 20, W - 144
        for name, share in shares:
            seg = max(2, round(bw * share))
            r, g, b = colorsys.hsv_to_rgb(hash_hue(name) / 360, 0.65, 0.9)
            dr.rectangle((x, bar_y, x + seg, bar_y + 40), fill=(int(r * 255), int(g * 255), int(b * 255)))
            x += seg
        dr.text((72, bar_y + 60), "  ".join(f"{n} {pct(s)}" for n, s in shares[:4]), font=_font(32),
                fill=(225, 225, 225))
    return im


def _slide4(d: dict, s: SlopScript) -> Image.Image:
    im = _text(gradient(d["full_name"]), s.roast, "upper", 0.05)
    im = _text(im, s.why, "lower", 0.032)
    ImageDraw.Draw(im).text((W // 2, 1140), f"github.com/{d['full_name']}", font=_font(40), fill="white", anchor="mm")
    return im


def render_slides(dossier: dict, script: SlopScript, graph: Image.Image | None = None) -> list[Image.Image]:
    slides = [_slide1(dossier, script), _slide2(dossier, script, graph), _slide3(dossier), _slide4(dossier, script)]
    for i, im in enumerate(slides, 1):
        _footer(im, dossier, i)
    return slides


def graph_still(clone: Path, title: str, size: tuple[int, int] = (1080, 700)) -> Image.Image | None:
    """The dependency graph of a cloned repo as an image, or None when there is no browser or nothing resolves."""
    try:
        with snapshot(clone, None, True) as src:
            graph = depgraph.build_graph(src.root, globs_spec([]), globs_spec([]), src.only)
    except SourceError:
        return None
    if not graph.nodes or not graph.edges:
        return None
    html = depgraph_html.render_html(graph, title=title, theme="dark", show_hint=False)
    # heavier, brighter edges (the viz default is a 1 px dim line that vanishes at 1080 px wide)
    html = (html.replace('.join("line").attr("stroke-width", 1)', '.join("line").attr("stroke-width", 2.5)')
            .replace('"stroke", "#3a4157"', '"stroke", "#9aa6d0"').replace('"stroke-opacity", 0.5', '"stroke-opacity", 0.9'))
    with tempfile.TemporaryDirectory(prefix="wonhee-slop-") as tmp:
        page, png = Path(tmp) / "page.html", Path(tmp) / "out.png"
        page.write_text(html, encoding="utf-8")
        try:
            export_png(page, png, *size)
        except ExportUnavailable:
            return None
        im = Image.open(png).convert("RGB")
    return _dense_crop(im)


def _dense_crop(im: Image.Image) -> Image.Image | None:
    """Crop to where the picture is, or None when too few edge pixels are visible to read as a graph."""
    a = np.asarray(im).astype(int)
    edge = (np.abs(a - np.array(EDGE_RGB)).sum(axis=2) < 90)
    if edge.sum() < MIN_EDGE_PIXELS:
        return None
    bg = a[0, 0]
    ys, xs = np.nonzero(np.abs(a - bg).sum(axis=2) > 30)
    pad = 24
    box = (max(0, xs.min() - pad), max(0, ys.min() - pad), min(im.width, xs.max() + pad), min(im.height, ys.max() + pad))
    return im.crop(box)
