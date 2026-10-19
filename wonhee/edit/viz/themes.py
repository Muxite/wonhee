"""Colour themes for the viz renderers (HTML graphs, their PNG exports and the PIL charts).

`dark` is the original vizkit look. `paper` matches muksite's light palette (cream card #F4F2EB,
near-black ink) so a render can sit in a site page without a black box around it.
Node hues still come from colors.hash_hue; a theme only sets the lightness, so the same
language/extension keeps the same hue in every theme.
"""
from __future__ import annotations

THEMES: dict[str, dict[str, str]] = {
    "dark": {
        "bg": "#0b0d12",
        "hint": "#8a93a6",
        "edge": "#3a4157",
        "edge_opacity": "0.5",
        "stroke": "#0b0d12",
        "label": "#0b0d12",  # text on coloured cells
        "ink": "#e6e8ee",    # text on the background (PIL charts)
        "node_lightness": "55%",
    },
    "paper": {
        "bg": "#F4F2EB",
        "hint": "#2A2A2A",
        "edge": "#0A0A0A",
        "edge_opacity": "0.25",
        "stroke": "#F4F2EB",
        "label": "#0A0A0A",
        "ink": "#0A0A0A",
        "node_lightness": "45%",
    },
}


def theme(name: str) -> dict[str, str]:
    try:
        return THEMES[name]
    except KeyError:
        raise ValueError(f"unknown theme {name!r}; choose from {', '.join(THEMES)}") from None


def parse_size(value: str) -> tuple[int, int]:
    """'1200x600' -> (1200, 600)."""
    w, sep, h = value.lower().partition("x")
    if not sep or not w.isdigit() or not h.isdigit():
        raise ValueError(f"size must look like WIDTHxHEIGHT, got {value!r}")
    return int(w), int(h)
