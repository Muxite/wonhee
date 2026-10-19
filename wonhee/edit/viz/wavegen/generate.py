"""One wavegen run: draw WaveSim params from the knobs, render, let a vision judge look at the actual frame and
nudge the knobs, retry up to MAX_RETRIES more times. The judge is passed in (None = keep the first render), so
this module never talks to a model itself."""
from __future__ import annotations

from collections.abc import Callable

from wonhee.edit.viz.wavegen.heuristics import COLOR_THEMES, draw_params
from wonhee.edit.viz.wavegen.judge import Verdict
from wonhee.edit.viz.wavegen.render import render

MAX_RETRIES = 2
Judge = Callable[[bytes, str, float, float, "str | None"], Verdict]


def generate(mood: str, color_theme: str | None, resolution: str, chaos: float, linearity: float, *,
             image: str, seed: int | None = None, judge: Judge | None = None,
             log: Callable[[str], None] = lambda _m: None) -> tuple[bytes, list[str]]:
    """Returns (png bytes, one note per attempt)."""
    notes: list[str] = []
    attempt = 0
    while True:
        params = draw_params(mood, color_theme, resolution, chaos, linearity,
                             seed=(seed + attempt) if seed is not None else None)
        log(f"attempt {attempt + 1}/{MAX_RETRIES + 1}: {len(params['sources'])} sources, chaos={chaos:.2f}, "
            f"linearity={linearity:.2f}, theme={color_theme!r}")
        png = render(params, image=image)
        if judge is None or attempt >= MAX_RETRIES:
            notes.append(f"attempt {attempt + 1}: kept" + (" (retries exhausted)" if judge else ""))
            return png, notes
        v = judge(png, mood, chaos, linearity, color_theme)
        notes.append(f"attempt {attempt + 1}: accept={v.accept} {v.reasoning[:160]}")
        if v.accept:
            return png, notes
        a = v.adjust
        chaos = a.chaos if a.chaos is not None else chaos
        linearity = a.linearity if a.linearity is not None else linearity
        if a.color_theme in COLOR_THEMES:
            color_theme = a.color_theme
        mood = a.mood or mood
        attempt += 1
