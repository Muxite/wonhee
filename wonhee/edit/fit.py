"""Can this photo carry this line of text? A verdict (ok, tight, reject), what was measured, and one concrete fix per
failure ("leave more empty space above the head", "background is too busy behind the text; try a plainer wall").

The judge is `place.place`: the best position it finds is the one the check looks at, so a verdict is about the spot
the text would really go, not about the photo as a whole. On top of that the check measures the frame's brightness
against the text colour and the share of the frame the subject covers. Pure: no DB, no model. The subject mask is an
argument; without one the overlap and area are None (not checked), never 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import ImageColor

from wonhee.edit import place as pl
from wonhee.edit import text as T

OK_SCORE, TIGHT_SCORE = 0.5, 0.3  # place's best score: at or above OK_SCORE is ok, TIGHT_SCORE..OK_SCORE is tight
SUBJECT_OVERLAP = 0.02  # the same bar place uses for "clear of the subject"
SUBJECT_AREA = 0.55  # a subject mask covering more of the frame leaves no room for a line
HEAD_HIGH = 0.35  # the subject's top edge above this (fraction of frame height) leaves no room above the head
BUSY_FIX = 0.5  # busyness of the best box above this is named as the problem
LIGHT_LUMA = 0.5  # text fill luma above this is "light" text
BRIGHT_MAX = 0.7  # frame mean luma above which light text loses contrast
DARK_MIN = 0.3  # frame mean luma below which dark text loses contrast


@dataclass(frozen=True)
class Fit:
    verdict: str  # ok | tight | reject
    score: float | None  # best position's score 0..1; None when there is no position (empty line)
    brightness: float  # frame mean luma 0..1
    text_light: bool | None  # True when the text fill is light; None when the colour could not be read
    subject_area: float | None  # share of the frame the subject covers; None without a mask
    best: pl.Candidate | None
    reasons: list[str]  # what was measured and failed (or was not measured)
    fixes: list[str]  # one concrete fix per failure; empty when ok

    def to_dict(self) -> dict:
        def r(v):
            return round(v, 4) if isinstance(v, float) else v
        return {"verdict": self.verdict, "score": r(self.score), "brightness": r(self.brightness),
                "text_light": self.text_light, "subject_area": r(self.subject_area),
                "reasons": self.reasons, "fixes": self.fixes,
                "best": self.best.to_dict() if self.best else None}


def _luma(colour: str) -> float | None:
    try:
        r, g, b = ImageColor.getrgb(colour)[:3]
    except ValueError:
        return None
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0


def fit(im, spec: T.TextSpec, subject_mask: np.ndarray | None = None) -> Fit:
    """Verdict for `spec` on `im` (PIL image or BGR/gray array); `subject_mask` as for `place.place`."""
    bgr = pl._as_bgr(im)
    H, W = bgr.shape[:2]
    brightness = float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean()) / 255.0
    fill = _luma(T._style(spec, 1).fill)
    light = None if fill is None else fill > LIGHT_LUMA

    area = top = None
    if subject_mask is not None:
        m = cv2.resize(np.asarray(subject_mask, np.float32), (W, H), interpolation=cv2.INTER_LINEAR) > 0.5
        area = float(m.mean())
        rows = np.where(m.any(axis=1))[0]
        top = float(rows[0]) / H if len(rows) else None

    cands = pl.place(bgr, spec, subject_mask, n=1)
    if not cands:
        return Fit("reject", None, brightness, light, area, None, ["the line is empty"],
                   ["the line is empty; add text before placing it"])
    best = cands[0]

    hard: list[tuple[str, str]] = []  # (what was measured, fix): a failure that rejects whatever the score
    soft: list[tuple[str, str]] = []  # a failure that lowers the verdict only through the score
    reasons: list[str] = []
    if area is not None and area > SUBJECT_AREA:
        msg = f"subject covers {area:.0%} of the frame"
        hard.append((msg, "the subject fills most of the frame; use a looser shot so the line has room"))
    if best.subject_overlap is not None and best.subject_overlap > SUBJECT_OVERLAP:
        msg = f"best spot covers {best.subject_overlap:.0%} of the subject"
        fix = ("leave more empty space above the head" if top is not None and top < HEAD_HIGH
               else "the text will sit over the subject; use a looser shot")
        hard.append((msg, fix))
    if light is True and brightness > BRIGHT_MAX:
        hard.append((f"frame is bright (luma {brightness:.2f}) for light text",
                     "photo is too bright for light text; use darker text or darken the photo"))
    if light is False and brightness < DARK_MIN:
        hard.append((f"frame is dark (luma {brightness:.2f}) for dark text",
                     "photo is too dark for dark text; use lighter text or brighten the photo"))
    if best.busyness > BUSY_FIX:
        soft.append((f"busy behind the text ({best.busyness:.2f})",
                     "background is too busy behind the text; try a plainer wall or dim the photo"))
    if best.score < OK_SCORE and not hard and not soft and "leaves the safe zone" in best.why:
        soft.append(("the block leaves the safe area", "the line is too big for the safe area; shorten it or use a "
                                                       "smaller size"))

    if best.score < TIGHT_SCORE or hard:
        verdict = "reject"
    elif best.score < OK_SCORE:
        verdict = "tight"
    else:
        verdict = "ok"
    failures = hard + soft
    reasons += [m for m, _ in failures]
    if subject_mask is None:
        reasons.append("no subject mask; the line was not checked against the subject")
    fixes = [f for _, f in failures] if verdict != "ok" else []
    return Fit(verdict, float(best.score), brightness, light, area, best, reasons, fixes)
