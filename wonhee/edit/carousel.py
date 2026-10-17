"""Carousel layout: fit photos to a slide size; wide images span several slides, cut on exact pixel seams.

Adapted from slopcut/carousel.py.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageFilter, ImageOps

SIZES = {"1:1": (1080, 1080), "4:5": (1080, 1350), "9:16": (1080, 1920)}
INSTAGRAM_MAX_SLIDES = 20


def parse_size(text: str) -> tuple[int, int]:
    if text in SIZES:
        return SIZES[text]
    try:
        w, h = text.lower().split("x")
        size = (int(w), int(h))
    except ValueError:
        raise ValueError(f"bad size {text!r}: use WxH (1080x1350) or one of {sorted(SIZES)}") from None
    if min(size) < 64 or max(size) > 4096:
        raise ValueError(f"bad size {text!r}")
    return size


def load(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def span_for(width: int, height: int, slide: tuple[int, int]) -> int:
    """How many slides wide an image should be: its aspect ratio over the slide's, rounded."""
    return max(1, round((width / height) / (slide[0] / slide[1])))


def fit_to_canvas(img: Image.Image, size: tuple[int, int], fit: str) -> Image.Image:
    """'cover' centre-crops to fill; 'contain' letterboxes over a blurred copy of itself."""
    if fit == "cover":
        return ImageOps.fit(img, size, Image.LANCZOS, centering=(0.5, 0.5))
    canvas = ImageOps.fit(img, size, Image.LANCZOS).filter(ImageFilter.GaussianBlur(40))
    inner = ImageOps.contain(img, size, Image.LANCZOS)
    canvas.paste(inner, ((size[0] - inner.width) // 2, (size[1] - inner.height) // 2))
    return canvas


def slice_columns(big: Image.Image, slide_w: int, n: int) -> list[Image.Image]:
    return [big.crop((i * slide_w, 0, (i + 1) * slide_w, big.height)) for i in range(n)]


def layout(img: Image.Image, slide: tuple[int, int], fit: str, span: bool) -> list[Image.Image]:
    """One photo -> its slides (several if it is wide and spanning is on)."""
    n = span_for(img.width, img.height, slide) if span else 1
    return slice_columns(fit_to_canvas(img, (slide[0] * n, slide[1]), fit), slide[0], n)
