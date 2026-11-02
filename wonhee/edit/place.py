"""Where a text block should sit on a photo: the quietest spot, clear of the subject, inside the safe zones.

The block is measured with the real text engine (`text._place`, so fonts, wrapping, boxes and the band pad are not
guessed), then slid over a grid of positions. Each position costs: how busy the pixels under it are (local variance
and edge density, both from integral images so every box is exact), how much of it covers the subject mask, how far it
leaves the safe band or the side margins, minus a small bonus for the reference habits (centred over a dark quiet
area, right-aligned above the subject, one short line in empty space). Deterministic: no model, no randomness; the
subject mask is an argument (`edit/mask.subject` makes one; no mask means face avoidance was not checked).

Fractions: every candidate is the block's centre (x, y) and size (w, h) as fractions of the frame. To draw it, pass
the centre as the exact `at=(x_frac, y_frac)` knob of `text_image` when it exists; `position` is only the nearest of
the five presets (a fallback that moves y but not x). `width_frac` is the wrap width the block was measured at: a
tall-frame block that reaches the button strip was measured narrower (text.BUTTONS_WRAP), so pass that width too.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import cv2
import numpy as np
from PIL import Image

from wonhee.edit import text as T

WORK = 256  # analysis long side in px: busyness is a coarse property, and the grid stays cheap
MARGIN = 0.04  # side margin as a fraction of the frame width
BUTTON_EDGE = 0.87  # right edge of a block that overlaps the button strip on a tall frame (strip is the right 13%)
STEP_X, STEP_Y = 0.025, 0.02
# cost weights (see the report in the commit / task notes): subject overlap dominates because covering the face is
# the one failure the owner rejects outright; busyness is next; a violation is a hard-ish penalty per unit overflow.
W_BUSY, W_SUBJECT, W_VIOLATION = 1.0, 3.0, 4.0
EDGE_GRAD = 40.0  # gradient magnitude (0..255 scale) that counts as an edge pixel
EDGE_FULL = 0.15  # edge fraction that counts as fully busy
STD_FULL = 50.0  # luma std that counts as fully busy
DARK_LUMA, QUIET = 0.38, 0.25  # a "dark quiet area": mean luma below, busyness below
HABIT = {"centred_dark": 0.08, "right_above_subject": 0.08, "short_in_empty": 0.06}
MAX_IOU = 0.2  # candidates in the top list may overlap this much at most


@dataclass(frozen=True)
class Candidate:
    x: float  # block centre, fraction of frame width
    y: float  # block centre, fraction of frame height
    w: float  # block size as fractions
    h: float
    score: float  # 0..1, higher is better (1 - cost, clipped)
    why: str
    busyness: float  # 0..1 under the box
    subject_overlap: float | None  # 0..1 of the box covered by the subject; None when no mask was given
    position: str  # nearest text.Position preset (fallback when exact placement is unavailable)
    width_frac: float  # wrap width the block was measured at
    align: str  # suggested line alignment for the block's side of the frame

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def _as_bgr(im) -> np.ndarray:
    if isinstance(im, Image.Image):
        return cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)
    a = np.asarray(im)
    if a.ndim == 2:
        a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    return a[..., :3]


def _integral(a: np.ndarray) -> np.ndarray:
    return cv2.integral(a.astype(np.float64))


def _box_mean(ii: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    n = max((x1 - x0) * (y1 - y0), 1)
    return float(ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]) / n


def measure(spec: T.TextSpec, size: tuple[int, int]) -> tuple[float, float, int]:
    """Block (width, height) in px and its line count, from the real engine. A band spans the frame, pad included."""
    W, _ = size
    lines, px, gap, block_h, _y0 = T._place(spec, size)
    if not lines:
        return 0.0, 0.0, 0
    if spec.preset == "band":
        return float(W), float(block_h + 2 * round(px * 0.55)), len(lines)
    return float(max(l.width for l in lines)), float(block_h), len(lines)


def _nearest_position(y: float) -> str:
    anchors = {"top": T.SAFE_TOP, **T.ANCHOR_Y, "bottom": T.SAFE_BOTTOM}
    return min(anchors, key=lambda k: abs(anchors[k] - y))


def place(im, spec: T.TextSpec, subject_mask: np.ndarray | None = None, n: int = 3) -> list[Candidate]:
    """The `n` best distinct positions for `spec`'s block on `im` (PIL image or BGR/gray array), best first.
    `subject_mask`: float 0..1 array of the subject (any size; resized), or None when not available."""
    bgr = _as_bgr(im)
    H, W = bgr.shape[:2]
    if not spec.content.strip() or n < 1:
        return []
    tall = H / W >= T.TALL
    k = WORK / max(H, W)
    wh = (max(8, round(W * k)), max(8, round(H * k)))
    small = cv2.resize(bgr, wh, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gx, gy = cv2.Sobel(blur, cv2.CV_32F, 1, 0), cv2.Sobel(blur, cv2.CV_32F, 0, 1)
    edges = (np.hypot(gx, gy) / 4.0 > EDGE_GRAD / 4.0).astype(np.float32)  # Sobel gain is 4 on a unit step
    i_g, i_g2, i_e = _integral(gray / 255.0), _integral((gray / 255.0) ** 2), _integral(edges)
    mask = None
    i_m = None
    sub_top = None
    if subject_mask is not None:
        mask = cv2.resize(np.asarray(subject_mask, np.float32), wh, interpolation=cv2.INTER_LINEAR) > 0.5
        i_m = _integral(mask.astype(np.float32))
        rows = np.where(mask.any(axis=1))[0]
        sub_top = float(rows[0]) / wh[1] if len(rows) else None
    ww, hh = wh

    wide = measure(spec, (W, H))
    narrow = None
    if tall and spec.width_frac > T.BUTTONS_WRAP and spec.preset != "band":
        narrow = measure(replace(spec, width_frac=T.BUTTONS_WRAP), (W, H))
    lines = wide[2]

    def evaluate(cx: float, cy: float, bw: float, bh: float, wf: float) -> Candidate:
        x0, x1 = cx - bw / 2, cx + bw / 2
        y0, y1 = cy - bh / 2, cy + bh / 2
        px0, px1 = max(0, int(x0 * ww)), min(ww, max(int(x0 * ww) + 1, round(x1 * ww)))
        py0, py1 = max(0, int(y0 * hh)), min(hh, max(int(y0 * hh) + 1, round(y1 * hh)))
        n_px = max((px1 - px0) * (py1 - py0), 1)
        mean = _box_mean(i_g, px0, py0, px1, py1)
        var = max(_box_mean(i_g2, px0, py0, px1, py1) - mean * mean, 0.0)
        std_n = min(var ** 0.5 * 255.0 / STD_FULL, 1.0)
        edge_n = min(_box_mean(i_e, px0, py0, px1, py1) / EDGE_FULL, 1.0)
        busy = 0.5 * std_n + 0.5 * edge_n
        overlap = _box_mean(i_m, px0, py0, px1, py1) if i_m is not None else 0.0
        # violation: how far the box leaves the safe band / margins, as a fraction of the frame
        viol = max(0.0, T.SAFE_TOP - y0) + max(0.0, y1 - T.SAFE_BOTTOM)
        if bw < 0.999:
            viol += max(0.0, MARGIN - x0) + max(0.0, x1 - (1 - MARGIN))
        if tall and y1 > T.BUTTONS_Y[0] and y0 < T.BUTTONS_Y[1]:
            viol += max(0.0, x1 - BUTTON_EDGE)
        cost = W_BUSY * busy + W_SUBJECT * overlap + W_VIOLATION * viol
        why = [f"busyness {busy:.2f}"]
        habit = None
        if abs(cx - 0.5) <= 0.03 and mean < DARK_LUMA and busy < QUIET:
            habit = "centred_dark"
            why.append("centred over a dark, quiet area")
        elif sub_top is not None and cx - bw / 2 >= 0.5 - 1e-6 and y1 <= sub_top + 0.02 and x1 >= 0.85:
            habit = "right_above_subject"
            why.append("right-aligned above the subject")
        elif lines == 1 and busy < 0.08 and bw < 0.6:
            habit = "short_in_empty"
            why.append("short line in empty space")
        if habit and overlap < 0.02 and viol == 0:
            cost -= HABIT[habit]
        if mask is not None:
            why.append("clear of the subject" if overlap < 0.02 else f"covers {overlap:.0%} subject")
        if viol > 0:
            why.append("leaves the safe zone")
        align = "left" if cx < 0.35 else "right" if cx > 0.65 else "center"
        return Candidate(cx, cy, bw, bh, float(np.clip(1.0 - cost, 0.0, 1.0)), ", ".join(why), busy,
                         overlap if i_m is not None else None, _nearest_position(cy), wf, align)

    def grid(dims: tuple[float, float, int], wf: float, only_strip: bool | None) -> list[Candidate]:
        bw, bh = dims[0] / W, dims[1] / H
        if bh <= 0 or bw <= 0:
            return []
        lo, hi = T.SAFE_TOP + bh / 2, T.SAFE_BOTTOM - bh / 2
        ys = [(T.SAFE_TOP + T.SAFE_BOTTOM) / 2] if hi < lo else list(np.arange(lo, hi + 1e-9, STEP_Y))
        xlo, xhi = MARGIN + bw / 2, 1 - MARGIN - bw / 2
        xs = [0.5] if xhi < xlo or bw >= 0.999 else list(np.arange(xlo, xhi + 1e-9, STEP_X))
        if bw < 0.999 and xhi >= xlo:
            xs.append(0.5)  # the centred habit must be on the grid whatever the step
        out = []
        for cy in ys:
            hits_strip = tall and (cy + bh / 2 > T.BUTTONS_Y[0] and cy - bh / 2 < T.BUTTONS_Y[1])
            if only_strip is not None and hits_strip != only_strip:
                continue
            out += [evaluate(float(cx), float(cy), bw, bh, wf) for cx in sorted(set(xs))]
        return out

    # a narrow re-wrap only for rows that reach the button strip; elsewhere the requested width stands
    cands = grid(wide, spec.width_frac, False if narrow else None)
    if narrow:
        cands += grid(narrow, T.BUTTONS_WRAP, True)
    cands.sort(key=lambda c: (-c.score, c.y, c.x))  # ties: higher on the frame, then left: stable and deterministic

    def iou(a: Candidate, b: Candidate) -> float:
        ix = max(0.0, min(a.x + a.w / 2, b.x + b.w / 2) - max(a.x - a.w / 2, b.x - b.w / 2))
        iy = max(0.0, min(a.y + a.h / 2, b.y + b.h / 2) - max(a.y - a.h / 2, b.y - b.h / 2))
        inter = ix * iy
        return inter / max(a.w * a.h + b.w * b.h - inter, 1e-9)

    chosen: list[Candidate] = []
    for c in cands:
        if all(iou(c, o) <= MAX_IOU for o in chosen):
            chosen.append(c)
            if len(chosen) == n:
                break
    return chosen
