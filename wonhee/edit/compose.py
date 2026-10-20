"""Layouts: a canvas plus an ordered list of layers, rendered the same way every time.

Design: docs/specs/2026-10-20-meme-and-popart-formats-design.md, "Generalized design (approved form)". Every
format (meme frame, sticker frame, brag collage, pop-art cutout, slop card) is a JSON template of this one engine,
and agents can write one-off layouts directly. No DB here: `render(layout, inputs, seed)` takes PIL images and returns
a PIL image; the same layout, inputs and seed give identical pixels (all randomness comes from a numpy generator
seeded with seed + layer index, and nothing reads the clock).

Reuses edit/text.py for every glyph (emoji and CJK included; its `sticker` preset is the two-stroke slanted one) and
the mask engine (edit/mask.py) for `cutout`.

Layer kinds, painted in order: photo, pattern, shape, sticker, text, band. The layer type is the field `type`
(the spec's `kind` is the subkind: pattern kind, shape kind, sticker kind), a naming choice the spec left open.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import cv2
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont, ImageOps
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from wonhee.edit import text as T

TEMPLATES = Path(__file__).resolve().parent / "compose_templates"
SS = 3  # supersampling for shape masks
Shape = Literal["rect", "rounded", "circle", "heart", "star", "burst", "bubble"]

PALETTES: dict[str, list[str]] = {  # [background, accent, ink, highlight]; generic, chosen by the user, not lifted
    "popart-yellow": ["#ffe600", "#ff3fa4", "#1b1030", "#ffffff"],
    "popart-pink": ["#ff5fb0", "#ffe600", "#2a0a3a", "#ffffff"],
    "lilac": ["#c8a2e8", "#5b21d6", "#15101f", "#fff3b0"],
    "mint": ["#7be0b8", "#ff6b4a", "#10201a", "#ffffff"],
    "tangerine": ["#ff8a1f", "#2a6cff", "#1d1208", "#fff6d6"],
    "ink": ["#101114", "#ffe600", "#f5f5f5", "#ff3fa4"],
}


class Box(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: float = 0.0
    y: float = 0.0
    w: float = Field(default=1.0, gt=0)
    h: float = Field(default=1.0, gt=0)


class Point(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: float
    y: float


class Outline(BaseModel):
    model_config = ConfigDict(extra="forbid")
    width_frac: float = Field(gt=0, le=0.2)  # of the canvas's short side
    colour: str


class PatternSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["flat", "halftone", "stars", "glyph", "stripes"] = "flat"
    colours: list[str] | None = None  # [ground, ink, ...]; default palette[0], palette[1]
    density: float = Field(default=0.5, ge=0, le=1)
    angle: float = 45.0
    glyph: str = "+"


class Canvas(BaseModel):
    model_config = ConfigDict(extra="forbid")
    w: int = Field(default=1080, ge=64, le=4096)
    h: int = Field(default=1350, ge=64, le=4096)
    bg: str | PatternSpec = "#ffffff"


class Layer(BaseModel):
    """One layer. Only the fields of its `type` are read; the rest must stay at their defaults' meaning."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    type: Literal["photo", "pattern", "shape", "sticker", "text", "band"]
    box: Box = Box()
    rotate: float = Field(default=0.0, ge=-180, le=180)
    opacity: float = Field(default=1.0, ge=0, le=1)
    kind: str | None = None  # pattern: flat|halftone|stars|glyph|stripes; shape: Shape; sticker: bubble|badge|...
    # photo
    src: int | str | None = None  # input index, or "clip" (video formats: a transparent window for the clip)
    fit: Literal["cover", "contain"] = "cover"
    shape: Shape = "rect"
    outlines: list[Outline] = []  # offset strokes around a photo/shape/sticker, innermost first, outermost last
    cutout: bool = False  # subject only, through the mask engine
    focus: Literal["center", "subject"] = "center"  # subject: keep the mask's box inside a non-rect shape
    # pattern / shape / sticker colours
    colours: list[str] | None = None
    density: float = Field(default=0.5, ge=0, le=1)
    angle: float = 45.0
    glyph: str = "+"
    fill: str | PatternSpec | None = None
    tail: Point | None = None  # bubble: where the tail points, as fractions of the box
    # text / sticker text
    content: str | None = Field(default=None, validation_alias=AliasChoices("content", "text"))
    items: list[str] | None = None  # checklist rows
    preset: Literal["outline", "band", "script", "sticker"] | None = None
    size_frac: float | None = Field(default=None, gt=0.005, le=0.4)  # of the canvas height
    align: Literal["left", "center", "right"] = "center"
    colour: str | None = None  # text fill (or the band colour)
    stroke: list[str] | None = None  # sticker preset: inner and outer stroke colours
    slant: float = Field(default=0.0, ge=-12, le=12)
    # band
    side: Literal["top", "bottom"] = "top"
    h_frac: float = Field(default=0.15, gt=0, le=1)

    @field_validator("kind")
    @classmethod
    def _kind(cls, v):
        return v


STICKER_KINDS = ("bubble", "badge", "checklist", "burst", "note", "label")


class Layout(BaseModel):
    model_config = ConfigDict(extra="forbid")
    canvas: Canvas = Canvas()
    palette: list[str] | str = "popart-yellow"
    seed: int = 0
    layers: list[Layer] = Field(default_factory=list, max_length=200)

    @model_validator(mode="after")
    def _check(self):
        if isinstance(self.palette, str) and self.palette not in PALETTES:
            raise ValueError(f"unknown palette '{self.palette}'; known: {', '.join(PALETTES)}")
        for i, ly in enumerate(self.layers):
            if ly.type == "sticker" and ly.kind not in STICKER_KINDS:
                raise ValueError(f"layer {i}: sticker kind must be one of {', '.join(STICKER_KINDS)}")
            if ly.type == "shape" and ly.kind not in (None, "rect", "rounded", "circle", "heart", "star", "burst",
                                                      "bubble"):
                raise ValueError(f"layer {i}: unknown shape kind '{ly.kind}'")
            if ly.type == "pattern" and ly.kind not in (None, "flat", "halftone", "stars", "glyph", "stripes"):
                raise ValueError(f"layer {i}: unknown pattern kind '{ly.kind}'")
        return self


def palette_of(layout: Layout) -> list[str]:
    return list(PALETTES[layout.palette]) if isinstance(layout.palette, str) else list(layout.palette)


def colour(c: str, pal: list[str], alpha: int = 255) -> tuple[int, int, int, int]:
    """'pal:N' indexes the palette (wrapping); anything else is a PIL colour (#rrggbb, names)."""
    if c.startswith("pal:"):
        c = pal[int(c[4:]) % len(pal)]
    r, g, b, a = ImageColor.getcolor(c, "RGBA")
    return (r, g, b, a if a != 255 else alpha)


# ---- shapes ------------------------------------------------------------------------------------------------------

def _heart_points(w: int, h: int) -> list[tuple[float, float]]:
    t = np.linspace(0, 2 * math.pi, 240)
    x = 16 * np.sin(t) ** 3
    y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    x = (x - x.min()) / (x.max() - x.min()) * (w - 1)
    y = (y - y.min()) / (y.max() - y.min()) * (h - 1)
    return list(zip(x.tolist(), y.tolist()))


def _spiky(w: int, h: int, n: int, inner: float, rot: float = -math.pi / 2) -> list[tuple[float, float]]:
    pts = []
    for i in range(2 * n):
        r = 1.0 if i % 2 == 0 else inner
        a = rot + i * math.pi / n
        pts.append((w / 2 + math.cos(a) * r * w / 2, h / 2 + math.sin(a) * r * h / 2))
    return pts


def shape_mask(kind: str, w: int, h: int, tail: Point | None = None) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """An anti-aliased L mask of the shape filling w x h, plus the inner box (x0, y0, x1, y1) where text fits."""
    W, H = w * SS, h * SS
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    inner = (0.04, 0.04, 0.96, 0.96)
    if kind == "rect":
        d.rectangle((0, 0, W, H), fill=255)
    elif kind == "rounded":
        d.rounded_rectangle((0, 0, W - 1, H - 1), radius=min(W, H) * 0.14, fill=255)
        inner = (0.06, 0.06, 0.94, 0.94)
    elif kind == "circle":
        d.ellipse((0, 0, W - 1, H - 1), fill=255)
        inner = (0.15, 0.15, 0.85, 0.85)
    elif kind == "heart":
        d.polygon(_heart_points(W, H), fill=255)
        inner = (0.22, 0.2, 0.78, 0.66)
    elif kind == "star":
        d.polygon(_spiky(W, H, 5, 0.42), fill=255)
        inner = (0.3, 0.3, 0.7, 0.7)
    elif kind == "burst":
        d.polygon(_spiky(W, H, 14, 0.78), fill=255)
        inner = (0.2, 0.2, 0.8, 0.8)
    elif kind == "bubble":
        tail = tail or Point(x=0.25, y=1.0)
        by0, by1, bx0, bx1 = 0.0, 1.0, 0.0, 1.0
        if tail.y > 0.8:
            by1 = 0.78
        elif tail.y < 0.2:
            by0 = 0.22
        body = (bx0 * W, by0 * H, bx1 * W - 1, by1 * H - 1)
        d.ellipse(body, fill=255)
        cx, cy = (body[0] + body[2]) / 2, (body[1] + body[3]) / 2
        rx, ry = (body[2] - body[0]) / 2, (body[3] - body[1]) / 2
        tx, ty = tail.x * W, tail.y * H
        ang = math.atan2((ty - cy) / ry, (tx - cx) / rx)
        base = [(cx + math.cos(ang + s) * rx, cy + math.sin(ang + s) * ry) for s in (-0.28, 0.28)]
        d.polygon([base[0], (tx, ty), base[1], (cx, cy)], fill=255)
        inner = (0.16, by0 + 0.16 * (by1 - by0), 0.84, by1 - 0.16 * (by1 - by0))
    else:
        raise ValueError(f"unknown shape '{kind}'")
    mask = im.resize((w, h), Image.LANCZOS)
    return mask, (round(inner[0] * w), round(inner[1] * h), round(inner[2] * w), round(inner[3] * h))


def _regions(mask: Image.Image, radii: list[float]) -> list[Image.Image]:
    """The shape grown by each cumulative radius (px), as AA masks, via the distance transform."""
    a = np.asarray(mask)
    inv = (a < 128).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    out = []
    for r in radii:
        soft = np.clip(r + 0.5 - dist, 0, 1)
        out.append(Image.fromarray((np.maximum(soft, a / 255.0) * 255).round().astype(np.uint8), "L"))
    return out


# ---- patterns ----------------------------------------------------------------------------------------------------

def _star_pts(cx, cy, r, rot):
    return [(cx + math.cos(rot + i * math.pi / 5 - math.pi / 2) * (r if i % 2 == 0 else r * 0.42),
             cy + math.sin(rot + i * math.pi / 5 - math.pi / 2) * (r if i % 2 == 0 else r * 0.42)) for i in range(10)]


def pattern_image(spec: PatternSpec, size: tuple[int, int], pal: list[str], rng: np.random.Generator) -> Image.Image:
    """A generated pattern (never generated imagery): flat, halftone dot gradient, stars, tiled glyph or stripes."""
    w, h = size
    cols = [colour(c, pal) for c in (spec.colours or ["pal:0", "pal:1"])]
    if len(cols) < 2:
        cols.append(cols[0])
    short = min(w, h)
    if spec.kind == "flat":
        return Image.new("RGBA", size, cols[0])
    if spec.kind in ("halftone", "stripes"):
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        t = math.radians(spec.angle)
        u = xx * math.cos(t) + yy * math.sin(t)
        v = -xx * math.sin(t) + yy * math.cos(t)
        phase = rng.uniform(0, 1, 2).astype(np.float32)
        if spec.kind == "stripes":
            cell = short / (3 + spec.density * 14)
            on = ((u / cell + phase[0]) % 1.0) < 0.5
            m = on.astype(np.float32)
        else:
            cell = short / (14 + spec.density * 46)
            du = ((u / cell + phase[0]) % 1.0) - 0.5
            dv = ((v / cell + phase[1]) % 1.0) - 0.5
            g = np.clip((v - v.min()) / max(float(v.max() - v.min()), 1.0), 0, 1)  # dots grow along the screen
            radius = 0.10 + 0.52 * g
            m = np.clip((radius - np.hypot(du, dv)) * cell, 0, 1)
        arr = np.empty((h, w, 4), np.float32)
        for c in range(4):
            arr[..., c] = cols[0][c] * (1 - m) + cols[1][c] * m
        return Image.fromarray(arr.round().astype(np.uint8), "RGBA")
    base = Image.new("RGBA", size, cols[0])
    if spec.kind == "stars":
        n = 4 + round(spec.density * 10)
        cell = short / n
        big = Image.new("RGBA", (w * 2, h * 2), (0, 0, 0, 0))
        d = ImageDraw.Draw(big)
        for gy in range(int(h / cell) + 2):
            for gx in range(int(w / cell) + 2):
                cx = (gx + 0.5 + rng.uniform(-0.25, 0.25)) * cell * 2
                cy = (gy + 0.5 + rng.uniform(-0.25, 0.25)) * cell * 2
                r = cell * rng.uniform(0.18, 0.38) * 2
                d.polygon(_star_pts(cx, cy, r, rng.uniform(-0.4, 0.4)), fill=cols[1 + (gx + gy) % max(1, len(cols) - 1)])
        base.alpha_composite(big.resize(size, Image.LANCZOS))
        return base
    if spec.kind == "glyph":
        n = 5 + round(spec.density * 9)
        cell = short / n
        font = ImageFont.truetype(str(T.SANS_BOLD), max(8, round(cell * 0.8)))
        d = ImageDraw.Draw(base)
        for gy in range(int(h / cell) + 2):
            for gx in range(int(w / cell) + 2):
                off = cell / 2 if gy % 2 else 0
                d.text((gx * cell + off, gy * cell), spec.glyph or "+", font=font, fill=cols[1])
        return base
    raise ValueError(f"unknown pattern '{spec.kind}'")


def _pattern_of(layer: Layer) -> PatternSpec:
    return PatternSpec(kind=layer.kind or "flat", colours=layer.colours, density=layer.density, angle=layer.angle,
                       glyph=layer.glyph)


# ---- compositing helpers -----------------------------------------------------------------------------------------

def _fill_image(fill, size, pal, rng, default) -> Image.Image:
    if fill is None:
        fill = default
    if isinstance(fill, PatternSpec):
        return pattern_image(fill, size, pal, rng)
    return Image.new("RGBA", size, colour(fill, pal))


def _shaped(content: Image.Image, shape: str, outlines: list[Outline], pal: list[str], canvas_short: int,
            tail: Point | None = None) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """`content` (RGBA, layer-sized) clipped to `shape`, with offset outlines (innermost first) inside the layer box.
    The shape is inset by the outlines' total width so the whole thing stays within the box. Also returns the inner
    text box of the shape, in layer pixels."""
    w, h = content.size
    widths = [max(1, round(o.width_frac * canvas_short)) for o in outlines]
    inset = sum(widths)
    sw, sh = max(4, w - 2 * inset), max(4, h - 2 * inset)
    small, inner = shape_mask(shape, sw, sh, tail)
    mask = Image.new("L", (w, h), 0)
    mask.paste(small, (inset, inset))
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    if outlines:
        cum, radii = 0, []
        for wd in widths:
            cum += wd
            radii.append(cum)
        regs = _regions(mask, radii)
        for o, reg in reversed(list(zip(outlines, regs))):  # outermost first, so inner strokes sit on top
            out.paste(Image.new("RGBA", (w, h), colour(o.colour, pal)), (0, 0), reg)
    out.paste(content, (0, 0), mask)
    return out, (inner[0] + inset, inner[1] + inset, inner[2] + inset, inner[3] + inset)


def _to_cv(im: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)


MaskFn = Callable[[Image.Image], np.ndarray]  # RGB image -> float 0..1 subject mask of the same size


def default_mask_fn(models_dir: Path | None, kind: str = "object") -> MaskFn:
    """The mask_cutout engine (ONNX, CPU). `mk.subject` is looked up at call time so tests can substitute it."""
    def fn(im: Image.Image) -> np.ndarray:
        from wonhee.edit import mask as mk
        if models_dir is None:
            raise RuntimeError("cutout needs the models directory (compose is run without one)")
        return mk.subject(_to_cv(im), models_dir, kind)  # type: ignore[arg-type]
    return fn


@dataclass
class Notes:
    """What render() did that a reviewer should see: layers moved into a safe zone, and the clip windows."""
    adjusted: list[str] = field(default_factory=list)
    windows: list[tuple[int, int, int, int]] = field(default_factory=list)  # px boxes of src="clip" layers
    no_subject: list[str] = field(default_factory=list)  # focus: subject layers that found no subject (FOCUS_NO_SUBJECT)


def _placed(src: Image.Image, scale: float, cx: float, cy: float, size: tuple[int, int], clamp: bool = False) -> Image.Image:
    """`src` scaled, with its point (cx, cy) at the centre of a `size` RGBA image (clamped inside the picture)."""
    sw, sh = size
    nim = src.resize((max(1, round(src.width * scale)), max(1, round(src.height * scale))), Image.LANCZOS)
    x, y = sw / 2 - cx * scale, sh / 2 - cy * scale
    if clamp:
        x = min(0.0, max(x, sw - nim.width)) if nim.width >= sw else (sw - nim.width) / 2
        y = min(0.0, max(y, sh - nim.height)) if nim.height >= sh else (sh - nim.height) / 2
    out = Image.new("RGBA", size, (0, 0, 0, 0))
    _paste_clipped(out, nim, round(x), round(y))
    return out


def _photo(layer: Layer, inputs: list[Image.Image], size: tuple[int, int], pal, canvas_short: int,
           mask_fn: MaskFn | None, notes: Notes | None = None, index: int = 0) -> tuple[Image.Image, tuple]:
    w, h = size
    idx = layer.src if isinstance(layer.src, int) else 0
    if idx >= len(inputs) or idx < 0:
        raise ValueError(f"photo layer wants input {idx}, but only {len(inputs)} image(s) were given")
    src = ImageOps.exif_transpose(inputs[idx]).convert("RGBA")
    alpha = None
    box = None
    if layer.cutout or layer.focus == "subject":
        if mask_fn is None:
            raise RuntimeError("cutout/focus needs a mask engine")
        m = np.clip(mask_fn(src.convert("RGB")), 0, 1)
        if layer.cutout:
            alpha = Image.fromarray((m * 255).round().astype(np.uint8), "L")
        ys, xs = np.where(m > 0.5)
        if len(xs):
            box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        elif layer.focus == "subject" and notes is not None:
            notes.no_subject.append(f"layer {index}: no subject found, cropped to the centre")
    if alpha is not None:
        src.putalpha(alpha)
    widths = sum(max(1, round(o.width_frac * canvas_short)) for o in layer.outlines)
    sw, sh = max(4, w - 2 * widths), max(4, h - 2 * widths)
    cover = max(sw / src.width, sh / src.height)
    contain = min(sw / src.width, sh / src.height)
    content = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    off = ((w - sw) // 2, (h - sh) // 2)
    if layer.fit == "contain":
        content.alpha_composite(_placed(src, contain, src.width / 2, src.height / 2, (sw, sh)), off)
    else:
        cx, cy = src.width / 2, src.height / 2
        if box is not None and layer.focus == "subject":
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        scale = cover
        if layer.focus == "subject" and box is not None and layer.shape != "rect":
            # keep the subject's box inside the shape: zoom out from cover, over a blurred copy, until it fits
            sm = np.asarray(shape_mask(layer.shape, sw, sh)[0])
            corners = [(box[0], box[1]), (box[2], box[1]), (box[0], box[3]), (box[2], box[3])]
            for _ in range(40):
                pts = [(sw / 2 + (x - cx) * scale, sh / 2 + (y - cy) * scale) for x, y in corners]
                if all(0 <= x < sw and 0 <= y < sh and sm[int(y), int(x)] >= 200 for x, y in pts) or scale <= contain * 0.3:
                    break
                scale *= 0.96
            back = src.convert("RGB").resize((sw, sh), Image.BILINEAR).filter(ImageFilter.GaussianBlur(14)).convert("RGBA")
            back.alpha_composite(_placed(src, scale, cx, cy, (sw, sh)))
            content.alpha_composite(back, off)
        else:
            content.alpha_composite(_placed(src, scale, cx, cy, (sw, sh), clamp=True), off)
    if layer.shape == "rect" and not layer.outlines:
        return content, (0, 0, w, h)
    return _shaped(content, layer.shape, layer.outlines, pal, canvas_short)


# ---- stickers ----------------------------------------------------------------------------------------------------

def _text_spec(layer: Layer, pal: list[str], preset: str, default_fill: str, default_strokes: tuple[str, str],
               content: str, align: str | None = None) -> T.TextSpec:
    fill = layer.colour or default_fill
    strokes = layer.stroke or list(default_strokes)
    cols = tuple("#%02x%02x%02x" % colour(c, pal)[:3] for c in [fill, *strokes])
    band = "light"  # band preset is black text; sticker/outline presets carry their own colours
    return T.TextSpec(content, preset, "center", layer.size_frac, 1.0, band, cols, layer.slant, 0.06,
                      align or layer.align)  # type: ignore[arg-type]


def _text_into(img: Image.Image, spec: T.TextSpec, inner: tuple[int, int, int, int], ref_h: int) -> None:
    x0, y0, x1, y1 = inner
    if x1 - x0 < 8 or y1 - y0 < 8:
        return
    block = T.render_block(spec, (x1 - x0, y1 - y0), ref_h)
    img.alpha_composite(block, (x0, y0))


def _checkbox(d: ImageDraw.ImageDraw, x: int, y: int, s: int, ink, tick) -> None:
    d.rounded_rectangle((x, y, x + s, y + s), radius=max(2, s // 5), outline=ink, width=max(2, s // 8))
    d.line([(x + s * 0.2, y + s * 0.55), (x + s * 0.42, y + s * 0.78), (x + s * 0.82, y + s * 0.2)], fill=tick,
           width=max(3, s // 6), joint="curve")


def _sticker(layer: Layer, size: tuple[int, int], pal, rng, ref_h: int, canvas_short: int) -> Image.Image:
    w, h = size
    kind = layer.kind or ""
    cols = layer.colours or []
    fill = layer.fill if layer.fill is not None else (cols[0] if cols else "pal:0")
    ink = cols[1] if len(cols) > 1 else "pal:2"
    accent = cols[2] if len(cols) > 2 else "pal:1"
    outlines = layer.outlines or [Outline(width_frac=0.007, colour=ink)]
    text = layer.content or ""
    if kind in ("bubble", "badge", "burst"):
        shape = {"bubble": "bubble", "badge": "burst", "burst": "burst"}[kind]
        # a badge is a many-pointed scalloped disc; shape_mask's burst has 14 points, so badge uses circle+ring look
        if kind == "badge":
            shape = "circle"
            outlines = layer.outlines or [Outline(width_frac=0.008, colour=ink), Outline(width_frac=0.012, colour=accent)]
        content = _fill_image(fill, (w, h), pal, rng, "pal:0")
        img, inner = _shaped(content, shape, outlines, pal, canvas_short, layer.tail if kind == "bubble" else None)
        spec = _text_spec(layer, pal, layer.preset or "sticker", "pal:3" if kind != "bubble" else "pal:0", (ink, accent),
                          text)
        if spec.preset == "sticker" and not layer.colour:  # readable on the fill: the fill colour is the stroke's neighbour
            spec = _text_spec(layer, pal, "sticker", accent, (ink, "#ffffff"), text)
        _text_into(img, spec, inner, ref_h)
        return img
    if kind == "label":
        content = _fill_image(fill, (w, h), pal, rng, "pal:1")
        img, inner = _shaped(content, "rounded", layer.outlines or [Outline(width_frac=0.005, colour=ink)], pal,
                             canvas_short)
        spec = _text_spec(layer, pal, layer.preset or "band", ink, (), text)
        _text_into(img, spec, inner, ref_h)
        return img
    if kind == "note":
        paper = _fill_image(fill if layer.fill is not None else "#fff27a", (w, h), pal, rng, "#fff27a")
        pad = max(3, round(canvas_short * 0.006))
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(shadow).rectangle((pad * 2, pad * 2, w - 1, h - 1), fill=(0, 0, 0, 70))
        img.alpha_composite(shadow)
        card = Image.new("RGBA", (w - pad * 2, h - pad * 2), (0, 0, 0, 0))
        card.paste(paper.crop((0, 0, card.width, card.height)), (0, 0))
        fold = max(6, round(min(card.size) * 0.16))
        cd = ImageDraw.Draw(card)
        cd.polygon([(card.width - fold, card.height), (card.width, card.height - fold), (card.width, card.height)],
                   fill=(0, 0, 0, 0))
        cd.polygon([(card.width - fold, card.height - fold), (card.width, card.height - fold),
                    (card.width - fold, card.height)], fill=(0, 0, 0, 55))
        img.alpha_composite(card, (0, 0))
        spec = _text_spec(layer, pal, layer.preset or "script", "#1b1b1b", (), text, layer.align if layer.align != "center" else "left")
        if spec.preset == "script":
            spec = T.TextSpec(text, "script", "center", layer.size_frac, 1.0, "dark", None, 0, 0, spec.align)
        _text_into(img, spec, (round(w * 0.08), round(h * 0.1), round(w * 0.92), round(h * 0.9)), ref_h)
        return img
    if kind == "checklist":
        content = _fill_image(fill if layer.fill is not None else "#ffffff", (w, h), pal, rng, "#ffffff")
        img, inner = _shaped(content, "rounded", layer.outlines or [Outline(width_frac=0.006, colour=ink)], pal,
                             canvas_short)
        rows = layer.items or ([text] if text else [])
        if not rows:
            return img
        x0, y0, x1, y1 = inner
        rh = (y1 - y0) / len(rows)
        s = max(8, round(min(rh * 0.62, (x1 - x0) * 0.16)))
        d = ImageDraw.Draw(img)
        ink_c, tick_c = colour(ink, pal), colour(accent, pal)
        for i, row in enumerate(rows):
            cy = y0 + i * rh
            _checkbox(d, x0, round(cy + (rh - s) / 2), s, ink_c, tick_c)
            spec = T.TextSpec(row, "band", "center", layer.size_frac, 1.0, "light", None, 0, 0, "left")
            _text_into(img, spec, (x0 + s + round(s * 0.4), round(cy), x1, round(cy + rh)), ref_h)
        return img
    raise ValueError(f"unknown sticker kind '{kind}'")


# ---- safe zones --------------------------------------------------------------------------------------------------

def _extent(box: tuple[int, int, int, int], rotate: float) -> tuple[float, float, float, float]:
    """(cx, cy, half-width, half-height) of the box's rotated bounding extent."""
    c, sn = abs(math.cos(math.radians(rotate))), abs(math.sin(math.radians(rotate)))
    w, h = box[2] - box[0], box[3] - box[1]
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, (w * c + h * sn) / 2, (w * sn + h * c) / 2


def safe_box(box: tuple[int, int, int, int], W: int, H: int,
             rotate: float = 0.0) -> tuple[tuple[int, int, int, int], bool]:
    """DESIGN Stage 4 safe zones, the same rule as edit/text.py, for tall canvases only (h/w >= TALL): the layer's
    *rotated extent* moves inside 12%..76% of the height; if it reaches the button strip (35..88% of the height) it is
    kept inside the centred 72% of the width (BUTTONS_WRAP), shifted if it fits there and scaled down if not. Feed
    shapes (1:1, 4:5) have no UI over the image, so nothing moves. Returns the new (unrotated) box and whether it moved."""
    if H / W < T.TALL:
        return box, False
    cx, cy, hw, hh = _extent(box, rotate)
    w, h = box[2] - box[0], box[3] - box[1]
    top, bottom = H * T.SAFE_TOP, H * T.SAFE_BOTTOM
    if 2 * hh > bottom - top:  # taller than the band: scale into it
        f = (bottom - top) / (2 * hh)
        w, h, hw, hh = w * f, h * f, hw * f, hh * f
    cy = min(max(cy, top + hh), bottom - hh)
    if cy + hh > H * T.BUTTONS_Y[0] and cy - hh < H * T.BUTTONS_Y[1]:
        lo, hi = W * (1 - T.BUTTONS_WRAP) / 2, W * (1 + T.BUTTONS_WRAP) / 2
        if 2 * hw > hi - lo:
            f = (hi - lo) / (2 * hw)
            w, h, hw, hh = w * f, h * f, hw * f, hh * f
        cx = min(max(cx, lo + hw), hi - hw)
    new = (round(cx - w / 2), round(cy - h / 2), round(cx + w / 2), round(cy + h / 2))
    return new, new != tuple(box)


# ---- render ------------------------------------------------------------------------------------------------------

def _paste_rotated(canvas: Image.Image, layer_img: Image.Image, box: tuple[int, int, int, int], rotate: float,
                   opacity: float) -> None:
    if opacity < 1:
        a = layer_img.getchannel("A").point(lambda v: round(v * opacity))
        layer_img = layer_img.copy()
        layer_img.putalpha(a)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    if rotate:
        layer_img = layer_img.rotate(-rotate, resample=Image.BICUBIC, expand=True)
    _paste_clipped(canvas, layer_img, round(cx - layer_img.width / 2), round(cy - layer_img.height / 2))


def _paste_clipped(canvas: Image.Image, im: Image.Image, x: int, y: int) -> None:
    """alpha_composite with a negative offset (layers may overhang the canvas, as collage stickers do)."""
    sx, sy = max(0, -x), max(0, -y)
    if sx >= im.width or sy >= im.height or x >= canvas.width or y >= canvas.height:
        return
    crop = im.crop((sx, sy, min(im.width, canvas.width - x), min(im.height, canvas.height - y)))
    canvas.alpha_composite(crop, (max(0, x), max(0, y)))


def render_ex(layout: Layout, inputs: list[Image.Image] | None = None, seed: int | None = None, *,
              mask_fn: MaskFn | None = None) -> tuple[Image.Image, Notes]:
    inputs = inputs or []
    W, H = layout.canvas.w, layout.canvas.h
    short = min(W, H)
    pal = palette_of(layout)
    base_seed = layout.seed if seed is None else seed
    notes = Notes()
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    bg = layout.canvas.bg
    canvas.alpha_composite(_fill_image(bg, (W, H), pal, np.random.default_rng(base_seed), "#ffffff"))
    for i, ly in enumerate(layout.layers):
        rng = np.random.default_rng([base_seed, i])
        box = (round(ly.box.x * W), round(ly.box.y * H), round((ly.box.x + ly.box.w) * W), round((ly.box.y + ly.box.h) * H))
        if ly.type == "band":
            bh = round(ly.h_frac * H)
            y0 = 0 if ly.side == "top" else H - bh
            band = Image.new("RGBA", (W, bh), colour(ly.colour or "#000000", pal))
            _paste_rotated(canvas, band, (0, y0, W, y0 + bh), 0, ly.opacity)
            continue
        if ly.type in ("text", "sticker") and not (ly.content or "").strip() and not ly.items:
            continue  # an empty slot draws nothing
        if ly.type in ("text", "sticker"):
            box, moved = safe_box(box, W, H, ly.rotate)
            if moved:
                notes.adjusted.append(f"layer {i} ({ly.type}) moved into the safe zone")
        bw, bh = max(1, box[2] - box[0]), max(1, box[3] - box[1])
        if ly.type == "photo":
            if ly.src == "clip":
                hole = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
                canvas.paste(hole, (box[0], box[1]))  # paste (not composite): the window becomes transparent
                notes.windows.append(box)
                continue
            img, _ = _photo(ly, inputs, (bw, bh), pal, short, mask_fn, notes, i)
        elif ly.type == "pattern":
            img = pattern_image(_pattern_of(ly), (bw, bh), pal, rng)
        elif ly.type == "shape":
            content = _fill_image(ly.fill, (bw, bh), pal, rng, "pal:0")
            img, _ = _shaped(content, ly.kind or "rect", ly.outlines, pal, short, ly.tail)
        elif ly.type == "sticker":
            img = _sticker(ly, (bw, bh), pal, rng, H, short)
        else:  # text
            preset = ly.preset or "outline"
            strokes = ly.stroke or ["#000000", "#000000"]
            cols = tuple("#%02x%02x%02x" % colour(c, pal)[:3] for c in [ly.colour or "#ffffff", *strokes])
            spec = T.TextSpec(ly.content or "", preset, "center", ly.size_frac, 1.0,  # type: ignore[arg-type]
                              "light" if (ly.colour or "").lower() in ("#000000", "black") else "dark", cols, ly.slant,
                              0.06, ly.align)
            img = T.render_block(spec, (bw, bh), H)  # colours go through text._style, so emoji keep their own
        _paste_rotated(canvas, img, box, ly.rotate, ly.opacity)
    return canvas, notes


def render(layout: Layout, inputs: list[Image.Image] | None = None, seed: int | None = None, *,
           mask_fn: MaskFn | None = None) -> Image.Image:
    """Layout + input images + seed -> RGBA image. Deterministic: the same arguments give identical bytes."""
    return render_ex(layout, inputs, seed, mask_fn=mask_fn)[0]


def clip_windows(layout: Layout) -> list[tuple[int, int, int, int]]:
    """Pixel boxes of the layout's `src="clip"` photo layers (the transparent windows a video is scaled into)."""
    W, H = layout.canvas.w, layout.canvas.h
    return [(round(l.box.x * W), round(l.box.y * H), round((l.box.x + l.box.w) * W), round((l.box.y + l.box.h) * H))
            for l in layout.layers if l.type == "photo" and l.src == "clip"]


# ---- templates ---------------------------------------------------------------------------------------------------

def _template_files() -> dict[str, Path]:
    return {p.stem: p for p in sorted(TEMPLATES.glob("*.json"))}


def templates_digest() -> str:
    """sha256 over the shipped templates' bytes (sorted by name): part of the nodes' cache salt, so editing a template
    is a cache miss."""
    import hashlib
    h = hashlib.sha256()
    for name, p in _template_files().items():
        h.update(name.encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def list_templates() -> list[dict]:
    """[{name, doc, slots: {slot: {default, doc}}}] for every shipped template."""
    out = []
    for name, p in _template_files().items():
        d = json.loads(p.read_text(encoding="utf-8"))
        out.append({"name": name, "doc": d.get("doc", ""), "slots": d.get("slots", {})})
    return out


def describe_templates() -> str:
    lines = []
    for t in list_templates():
        slots = "; ".join(f"{k} (default {json.dumps(v.get('default'))}): {v.get('doc', '')}" for k, v in t["slots"].items())
        lines.append(f"{t['name']}: {t['doc']} Slots: {slots}")
    return "\n".join(lines)


def _fill_slots(node, slots: dict):
    if isinstance(node, str) and node.startswith("slot:"):
        return slots[node[5:]]
    if isinstance(node, list):
        return [_fill_slots(x, slots) for x in node]
    if isinstance(node, dict):
        return {k: _fill_slots(v, slots) for k, v in node.items()}
    return node


def load_template(name: str, slots: dict | None = None) -> Layout:
    files = _template_files()
    if name not in files:
        raise ValueError(f"unknown template '{name}'; known: {', '.join(files)}")
    d = json.loads(files[name].read_text(encoding="utf-8"))
    declared = d.get("slots", {})
    given = dict(slots or {})
    unknown = set(given) - set(declared)
    if unknown:
        raise ValueError(f"template '{name}' has no slot(s) {sorted(unknown)}; slots: {sorted(declared)}")
    merged = {k: v.get("default") for k, v in declared.items()} | given
    return Layout.model_validate(_fill_slots(d["layout"], merged))


# ---- captions (frame_video) --------------------------------------------------------------------------------------

def caption_groups(lines: list[tuple[int, int, str]], max_words: int = 3) -> list[tuple[float, float, str]]:
    """(start_s, end_s, text) groups of 1..max_words words from (start_ms, end_ms, text) lines.

    Each word gets an even slice of its line's time: `subtitles` carries line-level timing only (nodes/subtitles.py;
    analyze/speech.py keeps segment times, not word times). A group appears at its first word's time and stays until the
    next group starts (or its line ends)."""
    max_words = max(1, min(3, max_words))
    groups: list[tuple[float, float, str]] = []
    for start, end, text in lines:
        words = text.split()
        if not words or end <= start:
            continue
        per = (end - start) / len(words)
        for i in range(0, len(words), max_words):
            chunk = words[i:i + max_words]
            g0 = start + i * per
            g1 = start + (i + len(chunk)) * per
            groups.append((g0 / 1000, g1 / 1000, " ".join(chunk)))
    return groups


def caption_image(text: str, style: str, size: tuple[int, int], fill: str = "#c9a0e6", canvas_h: int = 1920) -> Image.Image:
    """One caption group as an RGBA image of `size`: `box` = dark text in an opaque coloured box; `highlight` = white
    outlined text over a translucent marker bar. Text goes through edit/text.py (emoji and CJK work)."""
    w, h = size
    out = Image.new("RGBA", size, (0, 0, 0, 0))
    rgba = ImageColor.getcolor(fill, "RGBA")
    if style == "box":
        spec = T.TextSpec(text, "band", "center", 0.034, 1.0, "light")
    else:
        spec = T.TextSpec(text, "outline", "center", 0.034, 1.0)
    block = T.render_block(spec, (round(w * 0.92), round(h * 0.86)), canvas_h)
    bb = block.getbbox()
    if bb is None:
        return out
    padx, pady = round(h * 0.18), round(h * 0.08)
    bx0, by0 = max(0, (w - block.width) // 2 + bb[0] - padx), max(0, (h - block.height) // 2 + bb[1] - pady)
    bx1, by1 = min(w, (w - block.width) // 2 + bb[2] + padx), min(h, (h - block.height) // 2 + bb[3] + pady)
    d = ImageDraw.Draw(out)
    if style == "box":
        d.rounded_rectangle((bx0, by0, bx1, by1), radius=round(h * 0.12), fill=rgba)
    else:
        d.rounded_rectangle((bx0, by0 + (by1 - by0) * 0.3, bx1, by1), radius=round(h * 0.08), fill=rgba[:3] + (170,))
    out.alpha_composite(block, ((w - block.width) // 2, (h - block.height) // 2))
    return out


# ---- grid plans --------------------------------------------------------------------------------------------------

GRID_PATTERNS = ("halftone", "stars", "stripes")


@dataclass(frozen=True)
class TilePlan:
    index: int  # position in the grid, reading order (row-major, 3 columns)
    row: int
    col: int
    template: str
    palette: str
    pattern: str


def plan_grid(n: int, palettes: list[str], template: str = "popart_cutout") -> list[TilePlan]:
    """Assign a template, palette and pattern to each tile of a 3, 6 or 9 grid: one palette per row (cycling), and a
    pattern that differs from its left and upper neighbours, so the profile grid reads as one set."""
    if n not in (3, 6, 9):
        raise ValueError(f"a grid set is 3, 6 or 9 posts, got {n}")
    if not palettes:
        raise ValueError("grid_plan needs at least one palette")
    plan = []
    for i in range(n):
        row, col = divmod(i, 3)
        plan.append(TilePlan(i, row, col, template, palettes[row % len(palettes)], GRID_PATTERNS[(row + col) % 3]))
    return plan
