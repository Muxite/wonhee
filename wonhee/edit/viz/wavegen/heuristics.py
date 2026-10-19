"""Map mood/color-theme/chaos/linearity knobs into a randomized WaveSim params.json.
Nothing here is rigorous — it's a rough, fast mapping from vibes to numbers, meant to give
the vision judge (generate.py) a reasonable starting point to adjust from, not a final answer.
"""
from __future__ import annotations

import random
from typing import Any

GENERATOR_TYPES = ["sine", "sawtooth", "falling_sawtooth", "square"]

# name -> (lo, mid, hi) RGB triples for the 3-stop colorize palette
COLOR_THEMES: dict[str, tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]] = {
    "ocean": ((0, 40, 120), (0, 10, 30), (0, 180, 220)),
    "fire": ((40, 0, 0), (20, 5, 0), (255, 120, 0)),
    "forest": ((0, 60, 20), (5, 15, 5), (140, 220, 60)),
    "neon": ((120, 0, 200), (10, 0, 20), (0, 230, 220)),
    "calm": ((60, 90, 140), (20, 25, 35), (200, 210, 230)),
    "storm": ((10, 10, 40), (5, 5, 10), (255, 60, 40)),
    "sunset": ((80, 0, 60), (30, 5, 20), (255, 140, 40)),
}
DEFAULT_THEME = ((0, 0, 255), (0, 0, 0), (255, 0, 0))


def _palette_for(color_theme: str | None, mood: str) -> tuple[tuple[int, int, int], ...]:
    key = (color_theme or "").strip().lower()
    if key in COLOR_THEMES:
        return COLOR_THEMES[key]
    mood_lower = mood.lower()
    for name, palette in COLOR_THEMES.items():
        if name in mood_lower:
            return palette
    return DEFAULT_THEME


def _source_count(chaos: float) -> int:
    lo, hi = 1, 8
    return round(lo + chaos * (hi - lo))


def _positions(n: int, width: int, height: int, linearity: float, rng: random.Random) -> list[tuple[int, int]]:
    if n == 1:
        return [(width // 2, height // 2)]
    # Linear: interpolate along a line through the canvas. Scattered: fully random, jittered
    # by (1 - linearity) around that same line so linearity is a continuous knob, not a switch.
    x0, y0 = rng.randint(0, width), rng.randint(0, height)
    x1, y1 = rng.randint(0, width), rng.randint(0, height)
    jitter_scale = (1.0 - linearity) * max(width, height) * 0.35
    positions = []
    for i in range(n):
        t = i / (n - 1)
        lx = x0 + t * (x1 - x0)
        ly = y0 + t * (y1 - y0)
        lx += rng.uniform(-jitter_scale, jitter_scale)
        ly += rng.uniform(-jitter_scale, jitter_scale)
        positions.append((int(max(0, min(width, lx))), int(max(0, min(height, ly)))))
    return positions


def draw_params(mood: str, color_theme: str | None, resolution: str, chaos: float, linearity: float,
                 seed: int | None = None) -> dict[str, Any]:
    rng = random.Random(seed)
    width_s, height_s = resolution.lower().split("x")
    width, height = int(width_s), int(height_s)

    chaos = max(0.0, min(1.0, chaos))
    linearity = max(0.0, min(1.0, linearity))

    lo, mid, hi = _palette_for(color_theme, mood)

    n_sources = _source_count(chaos)
    positions = _positions(n_sources, width, height, linearity, rng)

    sources = []
    for (x, y) in positions:
        gen_type = rng.choice(GENERATOR_TYPES) if chaos > 0.5 else "sine"
        amplitude = rng.uniform(20, 40) + chaos * rng.uniform(0, 120)
        frequency = rng.uniform(0.01, 0.03) + chaos * rng.uniform(0, 0.15)
        offset = rng.uniform(0, 6.283) if chaos > 0.3 else 0.0
        sources.append({
            "type": gen_type, "x": x, "y": y,
            "amplitude": round(amplitude, 2), "frequency": round(frequency, 4), "offset": round(offset, 3),
            "start": 1, "end": 99999,
        })

    speed = round(0.3 + chaos * 0.15, 3)
    frame_count = round(120 + (1.0 - chaos) * 120)  # calmer scenes settle better with more steps

    # Scale the colorize range to the actual amplitudes in play, not a fixed guess — otherwise
    # calm/low-amplitude scenes read as nearly black (most values stay far under a fixed bound).
    max_amplitude = max((s["amplitude"] for s in sources), default=64.0)
    colorize_bound = round(max_amplitude * 0.5, 1)

    return {
        "width": width, "height": height,
        "speed": speed, "frame_count": frame_count, "upscale": 1.0,
        "colorize_min": -colorize_bound, "colorize_max": colorize_bound,
        "palette": {"lo": list(lo), "mid": list(mid), "hi": list(hi)},
        "method": "cuda",
        "sources": sources,
    }
