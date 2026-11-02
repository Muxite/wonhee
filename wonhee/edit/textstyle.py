"""Measure the on-photo text style of a reference image: colours, relative size, lines, alignment, box or outline,
dimming and where the block sits. Plain numpy/cv2 on one image, no DB and no model, so it is deterministic and testable
on synthetic frames. The result is a `TextSpec`-shaped profile (fields the renderer already has) plus the raw readings.

Text is found as a cluster of small, high-contrast, light (or dark) strokes: a morphological top-hat removes anything
wider than a stroke, so sky, walls and boxes drop out and only glyph-sized detail stays. Anything that cannot be read is
None, never 0. A font role is not measured here; the caller supplies it (see nodes/textstyle.py)."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

LONG_SIDE = 1000  # analysis resolution; every output is a fraction of the frame so this only bounds the work
TOPHAT_MIN = 50  # stroke contrast over its surroundings (0..255) after a light blur
LIGHT_MIN, DARK_MAX = 150, 105  # absolute luma a light / dark stroke must have
HEIGHT_PER_EM = 0.74  # p90 glyph height over font size: caps 0.71, ascenders 0.76, descender letters 0.78
MIN_COMPONENTS = 3  # fewer glyph-sized strokes than this is not text


@dataclass
class TextStyle:
    found: bool
    fill: str | None = None  # "#rrggbb"
    stroke: str | None = None  # outline colour, None when there is none
    stroke_em: float | None = None
    box: bool | None = None
    box_colour: str | None = None
    size_frac: float | None = None  # font size as a fraction of frame height
    lines: int | None = None
    align: str | None = None  # left | center | right
    align_certain: bool | None = None  # False for one line: alignment is then a guess from where the block sits
    x: float | None = None  # block centre and size as fractions of the frame
    y: float | None = None
    w: float | None = None
    h: float | None = None
    dim: float | None = None  # 0..1: how dark the photo is overall (1 - median luma / 128)
    dim_local: float | None = None  # 0..1: how much darker the area around the text is than the rest of the frame
    polarity: str | None = None  # light | dark text
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _hex(bgr) -> str:
    b, g, r = (int(round(float(v))) for v in bgr)
    return f"#{r:02x}{g:02x}{b:02x}"


def _prep(im: np.ndarray) -> np.ndarray:
    h, w = im.shape[:2]
    s = LONG_SIDE / max(h, w)
    if s < 1:
        im = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return im


def _stroke_mask(gray: np.ndarray, light: bool) -> np.ndarray:
    h = gray.shape[0]
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    k = max(9, int(0.04 * h) | 1)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    if light:
        top = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, se)
        m = (top > TOPHAT_MIN) & (g > LIGHT_MIN)
    else:
        top = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, se)
        m = (top > TOPHAT_MIN) & (g < DARK_MAX)
    m = m.astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def _glyphs(mask: np.ndarray) -> tuple[list[tuple[int, int, int, int, int, int]], np.ndarray]:
    """Glyph-sized components of the stroke mask as (x, y, w, h, area, label), with streaks and specks dropped, and
    the label image."""
    H, W = mask.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_h, min_area = max(5, int(0.008 * H)), max(12, int(0.00003 * H * W))
    out = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in st[i])
        if h < min_h or h > 0.2 * H or a < min_area or w > 4 * h or a < 0.12 * w * h:
            continue
        out.append((x, y, w, h, a, i))
    return out, lab


def _same_size(boxes: list[tuple[int, int, int, int, int, int]]) -> list[tuple[int, int, int, int, int, int]]:
    """The glyph-sized comps that share the commonest height (within the ascender/x-height spread of one font):
    highlights and edges in a photo come in every size, text comes in one."""
    best, best_n = [], 0
    for h in {b[3] for b in boxes}:
        keep = [b for b in boxes if 0.7 * h <= b[3] <= 1.45 * h]
        if len(keep) > best_n:
            best, best_n = keep, len(keep)
    return best


def _cluster(boxes: list[tuple[int, int, int, int, int, int]]) -> list[int]:
    """Indices of the heaviest group of boxes that sit close enough to read as one text block."""
    n = len(boxes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        xi, yi, wi, hi = boxes[i][:4]
        for j in range(i + 1, n):
            xj, yj, wj, hj = boxes[j][:4]
            m = max(hi, hj)
            gx = max(xi, xj) - min(xi + wi, xj + wj)
            gy = max(yi, yj) - min(yi + hi, yj + hj)
            if gx < 1.5 * m and gy < 2.0 * m:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return max(groups.values(), key=lambda g: sum(boxes[i][4] for i in g))


def _lines(grp: list[tuple[int, int, int, int, int, int]], px: float) -> list[tuple[int, int, int, int]]:
    """Text lines of the block as (x0, y0, x1, y1). Glyph centres within one line spread by about a third of an em
    (x-height against ascenders and descenders), a line break moves them by a full pitch, so a 1-D split of the sorted
    centres at 0.6 em finds the lines even when descenders touch the next line's ascenders."""
    big = [g for g in grp if g[3] >= 0.6 * np.percentile([g[3] for g in grp], 90)]  # dots and punctuation do not vote
    big.sort(key=lambda g: g[1] + g[3] / 2)
    groups: list[list] = [[big[0]]]
    for g in big[1:]:
        if (g[1] + g[3] / 2) - (groups[-1][-1][1] + groups[-1][-1][3] / 2) > 0.6 * px:
            groups.append([g])
        else:
            groups[-1].append(g)
    return [(min(g[0] for g in ln), min(g[1] for g in ln), max(g[0] + g[2] for g in ln), max(g[1] + g[3] for g in ln))
            for ln in groups]


def _align(lines: list[tuple[int, int, int, int]], W: int, cx: float) -> tuple[str, bool]:
    if len(lines) < 2:
        x0, x1 = lines[0][0], lines[0][2]
        if abs(cx - 0.5) < 0.06:
            return "center", False
        return ("left" if cx < 0.5 else "right"), False
    bw = max(l[2] for l in lines) - min(l[0] for l in lines)
    spread = {"left": np.ptp([l[0] for l in lines]), "right": np.ptp([l[2] for l in lines]),
              "center": np.ptp([(l[0] + l[2]) / 2 for l in lines])}
    best = min(spread, key=lambda k: spread[k])
    if spread[best] > 0.12 * bw:  # no edge lines up: ragged text, call it by where the block sits
        return ("center" if abs(cx - 0.5) < 0.1 else ("left" if cx < 0.5 else "right")), False
    return best, True


def _uniform(px_bgr: np.ndarray, tol: float = 22.0) -> tuple[float, np.ndarray]:
    """Share of pixels within `tol` of their median colour, and that median."""
    med = np.median(px_bgr, axis=0)
    d = np.abs(px_bgr.astype(float) - med).max(axis=1)
    return float((d < tol).mean()), med


def measure(img_bgr: np.ndarray, text: str | None = None) -> TextStyle:
    """Measure one image. `text` (the stored on-screen text) is only used to sanity-check the line count."""
    im = _prep(img_bgr)
    H, W = im.shape[:2]
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    luma = gray.astype(float)

    found: dict[bool, tuple[int, np.ndarray, list, np.ndarray]] = {}
    for light in (True, False):
        mask = _stroke_mask(gray, light)
        gl, lab = _glyphs(mask)
        gl = _same_size(gl) if gl else gl
        if len(gl) >= MIN_COMPONENTS:
            grp = [gl[i] for i in _cluster(gl)]
            if len(grp) >= MIN_COMPONENTS:
                found[light] = (sum(g[4] for g in grp), mask, grp, lab)
    # Both polarities can fire. Outlined text: the light fill and the heavier dark outline around it, so light glyphs
    # sitting inside dark boxes means light text. Dark text on a bright photo also lights up the bright gaps between
    # letters, but those sit between the boxes; there the heavier dark strokes win. Otherwise light is the default.
    def inside(a: list, b: list) -> float:
        hit = sum(any(q[0] <= g[0] + g[2] / 2 <= q[0] + q[2] and q[1] <= g[1] + g[3] / 2 <= q[1] + q[3] for q in b)
                  for g in a)
        return hit / len(a)

    best = None
    if found:
        light = True in found
        if len(found) == 2:
            outlined = inside(found[True][2], found[False][2]) >= 0.85  # light glyphs sit inside the dark boxes
            light = outlined or found[False][0] < 1.3 * found[True][0]
        best = (found[light][0], light, found[light][1], found[light][2], found[light][3])
    if best is None:
        return TextStyle(found=False, notes=["no text-like strokes found"])
    _, light, mask, grp, lab = best

    x0 = min(g[0] for g in grp)
    y0 = min(g[1] for g in grp)
    x1 = max(g[0] + g[2] for g in grp)
    y1 = max(g[1] + g[3] for g in grp)
    hs = np.sort([g[3] for g in grp])
    px = float(np.percentile(hs, 90)) / HEIGHT_PER_EM
    cx, cy = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H

    # keep only the cluster's own strokes for line and colour work (other text elsewhere must not leak in)
    own = np.isin(lab, [g[5] for g in grp]).astype(np.uint8)
    lines = _lines(grp, px)
    align, certain = _align(lines, W, cx)
    notes: list[str] = []
    if text and len([l for l in text.splitlines() if l.strip()]) not in (0, len(lines)):
        notes.append(f"measured {len(lines)} lines, stored text has {len([l for l in text.splitlines() if l.strip()])}")

    # fill colour: the brightest (or darkest) interior strokes, where antialiasing has not blended it with the photo
    ys, xs = np.nonzero(own)
    fp = im[ys, xs]
    fl = luma[ys, xs]
    keep = fl >= np.percentile(fl, 60) if light else fl <= np.percentile(fl, 40)
    fill = np.median(fp[keep], axis=0)

    # box and outline are read from three bands around the fill: a thin ring hugging it (an outline's colour), the
    # padded block area (a box's colour) and a band further out (the photo). One flat colour unlike the photo is it.
    def grow(r: int) -> np.ndarray:
        return cv2.dilate(own, np.ones((2 * r + 1,) * 2, np.uint8)) > 0

    thin = max(3, int(round(0.05 * px)))
    near = grow(max(4, int(round(0.2 * px))))
    ring = grow(thin) & ~grow(1)
    band = grow(max(12, int(round(1.1 * px)))) & ~grow(max(10, int(round(0.7 * px))))  # clear of any box
    box, box_colour, stroke, stroke_em = False, None, None, None

    pad = max(4, int(round(0.3 * px)))  # a box pads the text by about a quarter em
    inside = np.zeros((H, W), bool)
    inside[max(0, y0 - pad):min(H, y1 + pad), max(0, x0 - pad):min(W, x1 + pad)] = True
    inside &= ~grow(max(2, int(round(0.1 * px))))
    if inside.sum() > 50 and band.sum() > 50:
        px_in = im[inside]
        q = (px_in // 32).astype(int)  # the modal colour: a box is the biggest flat colour here, even if the padded
        key = q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2]  # area also catches a little of the photo at its corners
        mode = np.bincount(key).argmax()
        sel = px_in[key == mode]
        med = np.median(sel, axis=0)
        if len(sel) > 0.3 * len(px_in) and _uniform(sel)[0] > 0.8 \
                and np.abs(med - np.median(im[band], axis=0)).max() > 25:
            box, box_colour = True, _hex(med)
    if not box and ring.sum() > 30 and band.sum() > 30:
        share, med = _uniform(im[ring], tol=30.0)
        if share > 0.6 and np.abs(med - np.median(im[band], axis=0)).max() > 30 and np.abs(med - fill).max() > 40:
            stroke = _hex(med)
            ink = (np.abs(im.astype(float) - med).max(axis=2) < 40) & near & (own == 0)
            edge = own - cv2.erode(own, np.ones((3, 3), np.uint8))
            if edge.sum() > 0:
                stroke_em = round(float(ink.sum() / edge.sum()) / px, 3)

    # dimming: global darkness, and the area around the text against the rest of the frame
    dim = float(np.clip(1 - np.median(luma) / 128.0, 0, 1))
    wide = grow(max(8, int(round(px))))
    local, rest = wide & ~near, ~wide
    dim_local = None
    if rest.sum() > 0.1 * H * W and local.sum() > 50 and not box:
        a, b = float(np.median(luma[rest])), float(np.median(luma[local]))
        dim_local = round(float(np.clip((a - b) / max(a, 1.0), 0, 1)), 3)

    return TextStyle(found=True, fill=_hex(fill), stroke=stroke, stroke_em=stroke_em, box=box, box_colour=box_colour,
                     size_frac=round(px / H, 4), lines=len(lines), align=align, align_certain=certain,
                     x=round(cx, 4), y=round(cy, 4), w=round((x1 - x0) / W, 4), h=round((y1 - y0) / H, 4),
                     dim=round(dim, 3), dim_local=dim_local, polarity="light" if light else "dark", notes=notes)


def to_profile(s: TextStyle, role: str | None = None) -> dict:
    """The measured style as a `TextSpec`-shaped profile (preset, colours, size_frac, align, box, at, width_frac, role)
    plus a `measured` block with the raw readings. Unknowns stay None."""
    if not s.found:
        return {"preset": None, "colours": None, "size_frac": None, "align": None, "box": None, "box_colour": None,
                "stroke_em": None, "at": None, "width_frac": None, "role": role, "measured": s.to_dict()}
    colours = [s.fill, s.stroke] if s.stroke else [s.fill]
    return {"preset": "outline", "colours": colours, "size_frac": s.size_frac, "align": s.align,
            "box": "box" if s.box else "none", "box_colour": s.box_colour,
            "stroke_em": s.stroke_em if s.stroke else (0.0 if s.box is not None else None),
            "at": [s.x, s.y], "width_frac": s.w, "role": role, "measured": s.to_dict()}
