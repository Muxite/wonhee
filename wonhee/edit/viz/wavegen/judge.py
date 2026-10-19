"""What the wavegen judge is asked, and the shape of its answer. The call itself goes through wonhee's vision path
(local or remote engine, see nodes/viz.py), so this module holds no client or key handling.

The judge nudges the high-level knobs (chaos, linearity, colour theme, mood), never the raw WaveSim params:
asking a model for a fully valid low-level sim config is far more fragile than letting it move a few knobs and
re-drawing from heuristics.draw_params().
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from wonhee.edit.viz.wavegen.heuristics import COLOR_THEMES

RUBRIC = """You are judging a still frame from a 2D wave-equation simulation, rendered for a
report/presentation — the goal is "looks cool and matches the vibe", not scientific accuracy.

Loosely, here's what generally reads well per axis (guidelines, not hard rules):
- mood/color: the color palette and overall brightness should evoke the requested mood
  (e.g. "calm ocean" -> cool blues, soft transitions; "chaotic storm" -> harsh contrast,
  jagged patterns).
- chaos (0=very calm/orderly, 1=very chaotic): higher chaos should look busier, more
  overlapping ripples, sharper/messier patterns. Lower chaos should look like a few clean,
  legible wave rings.
- linearity (0=scattered sources, 1=sources aligned in a line): higher linearity should show
  a visible linear/symmetric structure to the wave pattern; lower should look more scattered.
- A good result fills the frame with visible structure — avoid a mostly-flat/empty image or
  one single blown-out solid-color blob with no visible wave detail.
"""


class Adjust(BaseModel):
    model_config = ConfigDict(extra="ignore")
    chaos: float | None = Field(default=None, ge=0, le=1)
    linearity: float | None = Field(default=None, ge=0, le=1)
    color_theme: str | None = None
    mood: str | None = None


class Verdict(BaseModel):
    model_config = ConfigDict(extra="ignore")
    accept: bool
    reasoning: str = ""
    adjust: Adjust = Adjust()


def prompt(mood: str, chaos: float, linearity: float, color_theme: str | None) -> str:
    return (f"Requested mood: \"{mood}\"\nRequested color theme: {color_theme or '(none given)'}\n"
            f"Requested chaos: {chaos} (0-1)\nRequested linearity: {linearity} (0-1)\n\n"
            "Look at the attached image and decide whether it matches. Only set the adjust fields you want changed. "
            f"color_theme must be exactly one of: {', '.join(COLOR_THEMES)}; anything else is ignored.")
