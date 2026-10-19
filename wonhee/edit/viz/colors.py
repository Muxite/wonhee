"""Deterministic name -> colour hashing, shared by the HTML graph (JS) and the PIL charts (Python), so a
language keeps one colour no matter how many languages are present or the order they were found in.

The JS mirror of `hash_hue` lives inline in depgraph_html.py's template. Keep them identical (same djb2-style
hash, same modulo) or the language-mix bar and the graph will disagree.
"""
from __future__ import annotations


def hash_hue(name: str) -> int:
    h = 5381
    for ch in name:
        h = ((h * 33) + ord(ch)) & 0xFFFFFFFF
    # golden-angle scatter: spreads nearby hashes across the hue circle instead of clumping
    return int((h * 137.508) % 360)


def hash_color(name: str, saturation: int = 65, lightness: int = 55) -> str:
    return f"hsl({hash_hue(name)}, {saturation}%, {lightness}%)"


def hsl_to_rgb(h: float, s: float, l: float) -> tuple[int, int, int]:
    """CSS hsl() -> 8-bit RGB, so PIL charts match the browser's colour for the same language."""
    import colorsys
    r, g, b = colorsys.hls_to_rgb(h / 360, l, s)
    return round(r * 255), round(g * 255), round(b * 255)


def language_rgb(name: str, lightness: float) -> tuple[int, int, int]:
    return hsl_to_rgb(hash_hue(name), 0.65, lightness)
