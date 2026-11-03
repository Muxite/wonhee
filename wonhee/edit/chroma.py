"""Chroma key without a real green screen: find the backdrop colour, key it, and keep the person safe.

Why it exists: takes shot against a whitish wall can't use a plain `chromakey` (it needs a saturated colour you
already know). So the backdrop is *detected* from the top and side edges, keyed in Lab, removed only where it
connects to the frame edge (a white shirt on a white wall stays), and guarded by the person mask. Spec:
docs/specs/2026-11-01-chroma-key-design.md. Plain functions on files and arrays; no DB access.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

RING = 0.06           # ring width, x the frame side
SIDE_ROWS = 0.6       # side strips and edge seeds stop here: the subject stands at the bottom
UNEVEN = 14.0         # Lab spread above which the wall is called uneven
HOLE_MIN = 0.003      # enclosed backdrop smaller than this (x frame area) is kept: a button, not a gap
GUARD_EVERY = 6       # the person mask runs every Nth frame and is held between
PROC_LONG = 960       # the matte is computed at this long side, then scaled up


@dataclass(frozen=True)
class Backdrop:
    kind: str                    # green | blue | white | other
    bgr: tuple[int, int, int]
    lab: np.ndarray
    spread: float                # 80th percentile Lab distance of the ring to its colour
    ring_coverage: float         # fraction of ring pixels within 12 Lab units of the colour
    ok: bool                     # spread is small enough to call the wall one colour


def _lab(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)


def _kind(bgr: tuple[int, int, int]) -> str:
    h, s, v = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
    if s > 0.35 * 255 and 35 <= h <= 85:
        return "green"
    if s > 0.35 * 255 and 95 <= h <= 130:
        return "blue"
    if s < 0.15 * 255 and v > 0.75 * 255:
        return "white"
    return "other"


def _ring_pixels(im: np.ndarray) -> np.ndarray:
    h, w = im.shape[:2]
    r, rs = max(2, int(RING * h)), max(2, int(RING * w))
    sr = int(SIDE_ROWS * h)
    return np.concatenate([im[:r].reshape(-1, 3), im[:sr, :rs].reshape(-1, 3), im[:sr, -rs:].reshape(-1, 3)])


def _parse_hex(hint: str) -> tuple[int, int, int]:
    s = hint.lstrip("#")
    return int(s[4:6], 16), int(s[2:4], 16), int(s[0:2], 16)


def detect_backdrop(frames: list[np.ndarray], hint: str = "auto") -> Backdrop:
    """Backdrop colour from the top and side ring of up to 9 frames. `hint`: auto, green, blue, white or #rrggbb."""
    px = np.concatenate([_ring_pixels(f)[::7] for f in frames[:9]])
    med = tuple(int(v) for v in np.median(px, axis=0))
    if hint.startswith("#"):
        med = _parse_hex(hint)
    lab = _lab(np.uint8([[med]]))[0, 0]
    d = np.linalg.norm(_lab(px.reshape(1, -1, 3))[0] - lab, axis=1)
    spread = float(np.percentile(d, 80))
    cover = float((d < 12).mean())
    kind = hint if hint in ("green", "blue", "white") else _kind(med)
    return Backdrop(kind, med, lab, spread, cover, spread <= UNEVEN)


def _smooth(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def matte(im: np.ndarray, bd: Backdrop, tolerance: float = 0.5, softness: float = 0.5,
          guard: np.ndarray | None = None) -> np.ndarray:
    """Float32 0..1 alpha (1 = keep). `guard` is a 0..1 person mask the size of `im`, or None."""
    h, w = im.shape[:2]
    weight = np.array([0.35, 1, 1], np.float32) if bd.kind in ("green", "blue") else np.ones(3, np.float32)
    d = np.linalg.norm((_lab(im) - bd.lab.astype(np.float32)) * weight, axis=2)
    lo = 4 + 20 * tolerance + 1.5 * min(bd.spread, UNEVEN)
    hi = lo + 4 + 24 * softness
    a = _smooth((d - lo) / (hi - lo)).astype(np.float32)

    cand = (a < 0.5).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=4)
    sr = int(SIDE_ROWS * h)
    seeds = np.unique(np.concatenate([lbl[0], lbl[:sr, 0], lbl[:sr, -1]]))
    seeds = seeds[seeds > 0]
    remove = np.isin(lbl, seeds)
    if n > 1:  # enclosed backdrop: big enough, and close to the backdrop colour on average
        mean_d = np.bincount(lbl.ravel(), weights=d.ravel(), minlength=n)[1:] / np.maximum(stats[1:, cv2.CC_STAT_AREA], 1)
        inside = (stats[1:, cv2.CC_STAT_AREA] > HOLE_MIN * h * w) & (mean_d < lo)
        enclosed = np.flatnonzero(inside) + 1
        remove |= np.isin(lbl, enclosed) & (cand > 0)
    zone = cv2.dilate(remove.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    alpha = cv2.GaussianBlur(np.where(zone, a, 1.0).astype(np.float32), (0, 0), 0.8)

    if guard is not None:
        k = max(3, int(0.02 * h)) | 1
        near = cv2.GaussianBlur(cv2.dilate((guard > 0.5).astype(np.float32), np.ones((k, k), np.uint8)), (0, 0), 2)
        core = cv2.erode((guard > 0.9).astype(np.uint8), np.ones((k, k), np.uint8)).astype(np.float32)
        alpha = np.maximum(alpha * np.clip(near, 0, 1), core)
    return np.clip(alpha, 0, 1)


def despill(im: np.ndarray, alpha: np.ndarray, kind: str, amount: float,
            bgr: tuple[int, int, int] = (235, 235, 235)) -> np.ndarray:
    """Remove backdrop colour from edge pixels. Green/blue: cap that channel at the larger of the other two.
    White/other: undo the blend with the backdrop colour (F = (I - (1-a) B) / a)."""
    f = im.astype(np.float32)
    a = alpha[..., None] if alpha.ndim == 2 else alpha
    w = amount * np.maximum(4 * a * (1 - a), 0.25 * (a > 0))
    if kind in ("green", "blue"):
        c = 1 if kind == "green" else 0
        others = [i for i in (0, 1, 2) if i != c]
        capped = f.copy()
        capped[..., c] = np.minimum(f[..., c], np.maximum(f[..., others[0]], f[..., others[1]]))
        out = f + (capped - f) * w
    else:
        b = np.array(bgr, np.float32)
        cleaned = np.clip((f - (1 - a) * b) / np.maximum(a, 0.05), 0, 255)
        out = f + (cleaned - f) * (w * (a > 0.05))
    return np.clip(out, 0, 255).astype(np.uint8)


def empty_report() -> dict:
    return {"backdrop": {"kind": None, "bgr": None, "spread": None, "ring_coverage": None}, "guard_used": None,
            "frames": None, "mean_subject_fraction": None, "flags": []}


def _cover(im: np.ndarray, w: int, h: int) -> np.ndarray:
    s = max(w / im.shape[1], h / im.shape[0])
    r = cv2.resize(im, (max(w, round(im.shape[1] * s)), max(h, round(im.shape[0] * s))), interpolation=cv2.INTER_AREA)
    y, x = (r.shape[0] - h) // 2, (r.shape[1] - w) // 2
    return r[y:y + h, x:x + w]


class _Fill:
    """Background provider: a still, a looping video, a flat colour or the drawn cabin."""

    def __init__(self, w: int, h: int, fill: str, background: Path | None):
        self.w, self.h, self.fill, self.cap, self.still = w, h, fill, None, None
        if background is not None:
            img = cv2.imread(str(background))
            if img is not None:
                self.still = _cover(img, w, h)
            else:
                self.cap = cv2.VideoCapture(str(background))
                if not self.cap.isOpened():
                    raise ValueError(f"cannot read background {background}")
        elif fill == "cabin":
            from wonhee.edit import cabin
            self.cabin = cabin
            self.still = cabin.render(w, h)
        elif fill == "window":
            from wonhee.edit import cabin
            self.still = cabin.render_window(w, h)
        elif fill == "matte":
            self.still = np.zeros((h, w, 3), np.uint8)
        else:
            self.still = np.full((h, w, 3), _parse_hex(fill), np.uint8)

    def at(self, i: int, fps: float) -> np.ndarray:
        if self.cap is not None:
            ok, fr = self.cap.read()
            if not ok:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, fr = self.cap.read()
            return _cover(fr, self.w, self.h)
        if self.fill == "cabin" and hasattr(self, "cabin"):
            return self.cabin.sway(self.still, i / fps)
        return self.still


def key_video(src: Path, out: Path, backdrop: str = "auto", tolerance: float = 0.5, softness: float = 0.5,
              guard: str = "auto", fill: str = "cabin", background: Path | None = None, spill: float = 0.7,
              models_dir: Path | None = None, subject_fn: Callable[[np.ndarray], np.ndarray] | None = None,
              show: str = "composite") -> dict:
    """Key `src` into `out` (mp4, audio kept). Returns the report described in the spec."""
    rep = empty_report()
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise ValueError(f"cannot read video {src}")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample = []
    for idx in np.linspace(0, max(total - 1, 0), 9).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, fr = cap.read()
        if ok:
            sample.append(fr)
    if not sample:
        raise ValueError(f"no frames in {src}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    bd = detect_backdrop(sample, backdrop)
    use_guard = guard == "person" or (guard == "auto" and bd.kind not in ("green", "blue"))
    if use_guard and subject_fn is None:
        if models_dir is None:
            raise ValueError("the person guard needs models_dir (or pass guard='none')")
        from wonhee.edit import mask as mk
        subject_fn = lambda im: mk.subject(im, models_dir, "person", 0.004)  # noqa: E731
    rep["backdrop"] = {"kind": bd.kind, "bgr": list(bd.bgr), "spread": round(bd.spread, 2),
                       "ring_coverage": round(bd.ring_coverage, 3)}
    rep["guard_used"] = bool(use_guard)
    if not bd.ok:
        rep["flags"].append("BACKDROP_UNEVEN")

    scale = min(1.0, PROC_LONG / max(w, h))
    pw, ph = max(2, round(w * scale)), max(2, round(h * scale))
    filler = _Fill(w, h, fill, background)
    enc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
         "-r", f"{fps:.4f}", "-i", "-", "-i", str(src), "-map", "0:v", "-map", "1:a?", "-c:v", "libx264",
         "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest",
         str(out)], stdin=subprocess.PIPE)
    prev, g, n, frac = None, None, 0, 0.0
    try:
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            small = cv2.resize(fr, (pw, ph), interpolation=cv2.INTER_AREA) if scale < 1 else fr
            if use_guard and n % GUARD_EVERY == 0:
                g = np.clip(subject_fn(small), 0, 1).astype(np.float32)
            a = matte(small, bd, tolerance, softness, g if use_guard else None)
            if prev is not None:  # steady the uncertain band only, so motion does not ghost
                band = (a > 0.05) & (a < 0.95)
                a = np.where(band, 0.7 * a + 0.3 * prev, a)
            prev = a
            A = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR) if scale < 1 else a
            frac += float(A.mean())
            if show == "matte" or fill == "matte":
                comp = np.repeat((A * 255).astype(np.uint8)[..., None], 3, axis=2)
            else:
                fg = despill(fr, A, bd.kind, spill, bd.bgr)
                bg = filler.at(n, fps).astype(np.float32)
                comp = (fg.astype(np.float32) * A[..., None] + bg * (1 - A[..., None])).round().astype(np.uint8)
            enc.stdin.write(comp.tobytes())
            n += 1
    finally:
        enc.stdin.close()
        code = enc.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg failed encoding {out}")
    rep["frames"] = n
    if n:
        rep["mean_subject_fraction"] = round(frac / n, 4)
        if rep["mean_subject_fraction"] < 0.03:
            rep["flags"].append("SUBJECT_TOO_SMALL")
        if rep["mean_subject_fraction"] > 0.85:
            rep["flags"].append("SUBJECT_TOO_BIG")
    return rep
