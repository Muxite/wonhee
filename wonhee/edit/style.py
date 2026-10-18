"""Measure a photo's look, pool references into a profile, and move another photo toward it (no generation).

"Different subject, different place, similar vibe": the vibe is mostly tone (where the darks, mids and lights
sit), colour per tone band (cool shadows, pink highlights, ...), how saturated it is, and lens traits (soft edges,
vignette). All of these are measured here in CIELAB, so a profile is numbers you can read, and `match` is a
deterministic per-band transfer whose strength a person or a cheap model picks. Nothing is generated.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel

PCTS = (1, 5, 25, 50, 75, 95, 99)
BANDS = ("shadow", "mid", "high")
HUES = ("red", "orange", "yellow", "lime", "green", "teal", "cyan", "azure", "blue", "violet", "magenta", "pink")
WORK_PX = 1000  # stats are taken on a copy this size, so they do not depend on resolution


class Band(BaseModel):
    a: float  # green(-) .. magenta(+)
    b: float  # blue(-) .. yellow(+)
    share: float  # of the frame's pixels (soft weights)


class Stats(BaseModel):
    lum: list[float]  # L* at PCTS, 0..100
    bands: dict[str, Band]
    chroma: float  # mean colourfulness, sqrt(a*a + b*b)
    edge_sharpness: float  # edge / centre detail; < 0.6 = soft edges (fisheye, motion, wide aperture)
    vignette: float  # corner / centre brightness; < 0.75 = a visible vignette
    hues: dict[str, float]  # share of saturated pixels per hue family (top 3)


class Profile(BaseModel):
    name: str
    n: int
    stats: Stats  # the median across the references
    hints: list[str]  # lens and finishing traits worth copying, from the stats
    members: list[str] = []


def load(path: Path | str) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError(f"{path}: not an image cv2 can read")
    return im


def _small(im: np.ndarray, px: int = WORK_PX) -> np.ndarray:
    h, w = im.shape[:2]
    s = px / max(h, w)
    return cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else im


def to_lab(im: np.ndarray) -> np.ndarray:
    """float32 L* 0..100, a*, b* centred on 0."""
    lab = cv2.cvtColor(im.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    return lab


def from_lab(lab: np.ndarray) -> np.ndarray:
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR) * 255 + 0.5, 0, 255).astype(np.uint8)


def band_weights(L: np.ndarray) -> dict[str, np.ndarray]:
    """Soft shadow / mid / high membership that sums to 1 (no hard edges where a grade changes)."""
    sh = np.clip((50 - L) / 30, 0, 1)
    hi = np.clip((L - 50) / 30, 0, 1)
    return {"shadow": sh, "mid": 1 - sh - hi, "high": hi}


def measure(im: np.ndarray) -> Stats:
    im = _small(im)
    lab = to_lab(im)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    bands = {}
    for k, w in band_weights(L).items():
        tot = float(w.sum())
        bands[k] = Band(a=round(float((a * w).sum() / tot), 2) if tot else 0.0,
                        b=round(float((b * w).sum() / tot), 2) if tot else 0.0, share=round(tot / L.size, 3))
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F))
    h, w = g.shape
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot((yy - h / 2) / (h / 2), (xx - w / 2) / (w / 2))
    centre, edge = r < 0.5, r > 0.9
    sharp = float(lap[edge].mean() / max(lap[centre].mean(), 1e-3))
    vig = float(L[r > 1.1].mean() / max(L[centre].mean(), 1)) if (r > 1.1).any() else 1.0
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0][hsv[..., 1] > 90].astype(int)
    hist = np.bincount(hue // 15, minlength=12)[:12] / max(len(hue), 1)
    return Stats(lum=[round(float(v), 2) for v in np.percentile(L, PCTS)], bands=bands,
                 chroma=round(float(np.hypot(a, b).mean()), 2), edge_sharpness=round(sharp, 3),
                 vignette=round(vig, 3), hues={HUES[i]: round(float(hist[i]), 3) for i in np.argsort(-hist)[:3]
                                               if hist[i] > 0})


def hints(s: Stats) -> list[str]:
    out = []
    if s.lum[3] < 25 and s.lum[6] > 75:
        out.append("low-key: dark frame lit by bright practical lights (bloom)")
    elif s.lum[3] > 65:
        out.append("high-key: bright, lifted frame (direct flash or window light)")
    if s.lum[0] > 5:
        out.append("lifted blacks (faded shadows)")
    # soft edges alone are ambiguous (shallow focus, motion, or a sharp caption in the centre); with dark corners
    # they are the fisheye / ultra-wide signature of the K-pop night references
    if s.edge_sharpness < 0.6 and s.vignette < 0.75:
        out.append("soft edges and dark corners: fisheye / ultra-wide lens (fisheye, edge_blur, vignette)")
    elif s.edge_sharpness < 0.45:
        out.append("soft background: shallow focus or motion blur")
    elif s.vignette < 0.75:
        out.append("strong vignette")
    cast = {k: (v.a, v.b) for k, v in s.bands.items() if v.share >= 0.05}  # an empty band says nothing
    if cast and all(b < -5 for _, b in cast.values()):
        out.append("cool all over (blue / cyan cast in every band)")
    elif cast and all(b > 5 for _, b in cast.values()):
        out.append("warm all over")
    if cast.get("high", (0, 0))[0] > 8 or cast.get("mid", (0, 0))[0] > 12:  # skin alone sits around +8..10 in the mids
        out.append("pink / magenta tint")
    out.append("saturated" if s.chroma > 22 else "muted colour" if s.chroma < 10 else "moderate colour")
    return out


def profile(images: list[np.ndarray], name: str = "refs", members: list[str] | None = None) -> Profile:
    if not images:
        raise ValueError("a profile needs at least one reference image")
    st = [measure(im) for im in images]
    med = lambda xs: round(float(np.median(xs)), 3)  # noqa: E731
    bands = {k: Band(a=med([s.bands[k].a for s in st]), b=med([s.bands[k].b for s in st]),
                     share=med([s.bands[k].share for s in st])) for k in BANDS}
    hue_tot: dict[str, float] = {}
    for s in st:
        for k, v in s.hues.items():
            hue_tot[k] = hue_tot.get(k, 0) + v / len(st)
    pooled = Stats(lum=[med([s.lum[i] for s in st]) for i in range(len(PCTS))], bands=bands,
                   chroma=med([s.chroma for s in st]), edge_sharpness=med([s.edge_sharpness for s in st]),
                   vignette=med([s.vignette for s in st]),
                   hues={k: round(v, 3) for k, v in sorted(hue_tot.items(), key=lambda kv: -kv[1])[:3]})
    return Profile(name=name, n=len(st), stats=pooled, hints=hints(pooled), members=members or [])


def _tone_lut(src: list[float], tgt: list[float], k: float) -> tuple[np.ndarray, np.ndarray]:
    """A monotone L* curve that moves the source percentiles `k` of the way to the target's."""
    xs = np.array([0.0, *src, 100.0])
    ys = np.array([0.0, *[s + k * (t - s) for s, t in zip(src, tgt)], 100.0])
    xs = np.maximum.accumulate(xs + np.arange(len(xs)) * 1e-4)  # strictly increasing for interp
    return xs, np.clip(np.maximum.accumulate(ys), 0, 100)


def match(im: np.ndarray, target: Stats, strength: float = 0.7, *, tone: float = 1.0, colour: float = 1.0,
          mask: np.ndarray | None = None, subject_strength: float = 0.4, subject_lift: float = 0.0) -> np.ndarray:
    """Move a photo's tone curve, per-band colour and colourfulness toward `target`.

    strength scales everything (0 = untouched, 1 = the reference's numbers). `tone` and `colour` scale those two
    parts on their own (a daylight photo pushed fully to a night profile goes muddy; tone=0.5 keeps it readable).
    With a subject mask (0..1 float, same size), the subject gets `subject_strength` of the grade instead (skin
    stays closer to true) and `subject_lift` brightens it like a key light or on-camera flash.
    """
    src = measure(im)
    lab = to_lab(im)

    def grade(k: float) -> np.ndarray:
        out = lab.copy()
        xs, ys = _tone_lut(src.lum, target.lum, k * tone)
        out[..., 0] = np.interp(out[..., 0], xs, ys)
        w = band_weights(out[..., 0])
        for band in BANDS:
            da = (target.bands[band].a - src.bands[band].a) * k * colour
            db = (target.bands[band].b - src.bands[band].b) * k * colour
            out[..., 1] += w[band] * da
            out[..., 2] += w[band] * db
        c = np.clip((target.chroma / max(src.chroma, 1.0)) ** (k * colour), 0.5, 2.2)
        out[..., 1:] *= c
        return out

    bg = grade(strength)
    if mask is None:
        return from_lab(bg)
    subj = grade(strength * subject_strength)
    if subject_lift:
        subj[..., 0] = np.clip(subj[..., 0] * (1 + subject_lift) + 4 * subject_lift, 0, 100)
    m = mask.astype(np.float32)[..., None]
    return from_lab(bg * (1 - m) + subj * m)
