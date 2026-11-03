"""Frame sequences for the ``motion_*`` effects (no effect is registered here).

A *sequence* is a list of float32 RGB frames (H, W, 3) in 0..1 plus nothing
else: the effects turn it into one still (long exposure, strobe, slit-scan,
frame differences, flow trails). Where the frames come from is chosen by the
``scene`` parameter and is the only part that changes when the same look is run
on real video later:

    scene="frames"   a directory of images, a video file (cv2 / ffmpeg) or
                     ctx.data = list / (N, H, W, 3) array of frames
    scene="pan"      a camera move over ctx.source (pan / push-in, parallax by
                     luminance depth, seeded handheld shake)
    scene="orbs"     bright bodies on seeded smooth paths over a dark ground
    scene="plankton" | "clouds" | "aurora"   the library's own nature effects
                     rendered over their ``time`` parameter
    scene="effect:<name>"  any registered effect (time param if it has one,
                     else the seed is advanced per frame: restless flicker)
    scene="auto"     pan if a source exists, else orbs

Temporal coherence: every random quantity is a pure function of (seed, x, y, t)
through ``natkit.tnoise`` or of (seed, body index) and t, never of the frame
count, so frame ``t`` is identical whether you render 12 or 120 frames, and a
look developed on stills runs unchanged per frame on video.
"""
from __future__ import annotations

import glob
import math
import os
import subprocess

import cv2
import numpy as np

from . import features as F
from . import natkit as N

SEQ_PARAMS = {
    "scene": ("auto", "SOURCE: auto | pan | orbs | plankton | clouds | aurora | frames | effect:<name>"),
    "frames": ("", "SOURCE: for scene=frames: directory of images or a video file (also accepts ctx.data = list of frames)"),
    "n_frames": (24, "SOURCE: number of frames sampled over the time window"),
    "t0": (0.0, "COHERENCE: start of the time window, in 0..1 of the clip (or seconds for synthetic scenes)"),
    "span": (1.0, "COHERENCE: length of the time window (frame t is independent of n_frames)"),
    "travel": (0.30, "SOURCE: pan / orb travel in frame widths over the window"),
    "pan_angle": (-0.12, "SOURCE: camera move direction in radians"),
    "shake": (0.0, "COHERENCE: 0..1 seeded handheld shake (space-time noise)"),
    "bodies": (7, "SOURCE: number of orbs for scene=orbs"),
}


def with_seq(params: dict) -> dict:
    out = dict(params)
    for k, v in SEQ_PARAMS.items():
        out.setdefault(k, v)
    return out


# ---------------------------------------------------------------- reading
def _evenly(n_total: int, n: int, t0: float, span: float) -> list[int]:
    ts = np.linspace(t0, min(1.0, t0 + span), n)
    return [int(np.clip(round(t * (n_total - 1)), 0, n_total - 1)) for t in ts]


def read_frames(path: str, size, n: int, t0: float = 0.0, span: float = 1.0) -> list[np.ndarray]:
    """Frames from a directory of images or a video file, resized (cover) to size (W, H)."""
    W, H = size
    out: list[np.ndarray] = []
    if os.path.isdir(path):
        files = sorted(f for ext in ("png", "jpg", "jpeg", "webp", "tif", "tiff") for f in glob.glob(os.path.join(path, f"*.{ext}")))
        if not files:
            raise FileNotFoundError(f"no images in {path}")
        for i in _evenly(len(files), n, t0, span):
            im = cv2.imread(files[i], cv2.IMREAD_COLOR)
            out.append(F.fit_cover(im[..., ::-1].astype(np.float32) / 255, (W, H), "center"))
        return out
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    if total > 1:
        for i in _evenly(total, n, t0, span):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, im = cap.read()
            if ok:
                out.append(F.fit_cover(im[..., ::-1].astype(np.float32) / 255, (W, H), "center"))
        cap.release()
        if out:
            return out
    # ffmpeg fallback: decode evenly spaced frames to raw rgb
    pr = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames",
                         "-of", "csv=p=0", path], capture_output=True, text=True)
    total = int(pr.stdout.strip() or 0)
    if total < 2:
        raise RuntimeError(f"cannot read frames from {path}")
    idx = _evenly(total, n, t0, span)
    sel = "+".join(f"eq(n\\,{i})" for i in idx)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"select='{sel}',scale={W}:{H}", "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True).stdout
    arr = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3).astype(np.float32) / 255
    return list(arr)


# ---------------------------------------------------------------- synthetic scenes
def _ease(t):
    return t * t * (3 - 2 * t)


def pan_frames(ctx, size, n, t0, span, travel, angle, shake, parallax=0.5):
    """Camera move over the source photo. The photo is fitted to a canvas larger
    than the frame by ``travel`` so the window never leaves it; luminance
    (blurred) acts as a pseudo depth map for parallax."""
    W, H = size
    z = 1.0 + abs(travel) * 1.15 + 0.05
    BW, BH = int(W * z), int(H * z)
    base = F.fit_cover(np.asarray(ctx.source, np.float32)[..., :3], (BW, BH), ctx.p("crop", "center"))
    depth = F.blur(F.stretch(F.luminance(base), 2, 98), 0.05 * BW)
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    ca, sa = math.cos(angle), math.sin(angle)
    out = []
    for i in range(n):
        t = t0 + span * (i / max(1, n - 1))
        p = _ease(np.clip(t, 0, 1)) if span <= 1.0 else t
        # travel is measured on the clip timeline so frame t does not depend on n
        dx = (p - 0.5) * travel * W * ca
        dy = (p - 0.5) * travel * W * sa
        if shake > 0:
            dx += shake * 0.012 * W * N.tnoise(np.array([0.0]), np.array([0.0]), t * 9.0, ctx.seed, 1.0)[0]
            dy += shake * 0.012 * W * N.tnoise(np.array([3.3]), np.array([1.1]), t * 9.0, ctx.seed + 5, 1.0)[0]
        cx = (BW - W) / 2 + dx
        cy = (BH - H) / 2 + dy
        # parallax: nearer (brighter) areas move further
        dsh = cv2.remap(depth, (gx + np.float32(cx)), (gy + np.float32(cy)), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        par = parallax * (dsh - 0.5) * 0.10 * W * (p - 0.5) * 2
        mx = (gx + np.float32(cx) + par).astype(np.float32)
        my = (gy + np.float32(cy)).astype(np.float32)
        fr = cv2.remap(base, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        out.append(np.clip(fr, 0, 1))
    return out


def orbs_frames(ctx, size, n, t0, span, travel, bodies):
    """Bright bodies on smooth seeded paths. Each body's position is a pure
    function of (seed, body, t): sum of a linear sweep and two sines."""
    W, H = size
    rng = np.random.default_rng(ctx.seed + 4242)
    R = N.ramp(ctx.palette, 256, "mono", 0.0, 1.0, False)
    dark = ctx.palette.polarity == "dark"
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    ground = 0.05 + 0.05 * N.up(N.fbm(rng, max(8, H // 8), max(8, W // 8), 9, 3), (W, H)) if dark else 0.0 * xs
    spec = []
    for b in range(int(bodies)):
        ang = rng.uniform(-0.5, 0.5) + (math.pi if rng.random() < 0.3 else 0)
        spec.append(dict(x=rng.uniform(0.05, 0.95), y=rng.uniform(0.1, 0.9), a=ang, v=rng.uniform(0.5, 1.2),
                         r=rng.uniform(0.018, 0.05), w=rng.uniform(1.0, 3.0), ph=rng.uniform(0, 6.28), amp=rng.uniform(0.02, 0.09),
                         tone=rng.uniform(0.55, 1.0)))
    out = []
    for i in range(n):
        t = t0 + span * (i / max(1, n - 1))
        L = np.array(ground, np.float32) if dark else np.zeros((H, W), np.float32)
        for s in spec:
            x = (s["x"] - 0.5 * s["v"] * travel * math.cos(s["a"]) + math.cos(s["a"]) * s["v"] * travel * t + s["amp"] * math.sin(s["w"] * 2 * math.pi * t + s["ph"])) * W
            y = (s["y"] - 0.5 * s["v"] * travel * math.sin(s["a"]) * 0.6 + math.sin(s["a"]) * s["v"] * travel * t * 0.6 + s["amp"] * math.cos(s["w"] * 1.7 * math.pi * t + s["ph"])) * H
            r = s["r"] * W
            x0, x1 = int(max(0, x - 4.5 * r)), int(min(W, x + 4.5 * r + 1))
            y0, y1 = int(max(0, y - 4.5 * r)), int(min(H, y + 4.5 * r + 1))
            if x1 <= x0 or y1 <= y0:
                continue
            d2 = (xs[y0:y1, x0:x1] - x) ** 2 + (ys[y0:y1, x0:x1] - y) ** 2
            L[y0:y1, x0:x1] = np.maximum(L[y0:y1, x0:x1], s["tone"] * np.exp(-(d2 / (2 * (r * 0.55) ** 2)) ** 1.5))
        out.append(N.paint(np.clip(L, 0, 1), R))
    return out


def effect_frames(ctx, name, size, n, t0, span, travel):
    from .compose import render_effect
    from .registry import get
    eff = get(name)
    has_time = "time" in eff.params
    out = []
    for i in range(n):
        t = t0 + span * (i / max(1, n - 1))
        params = {"time": t * 6.0 * max(0.2, travel / 0.3)} if has_time else {}
        if name == "nature-flow-wash":
            params["time"] = t * 6.0
        seed = ctx.seed if has_time else ctx.seed + i
        extra = {k: v for k, v in (ctx.p("effect_params", {}) or {}).items()}
        params.update(extra)
        out.append(render_effect(name, size=size, seed=seed, palette=ctx.palette, params=params, source=ctx.source))
    return out


_SCENE_EFFECT = {"plankton": "nature-plankton-ribbons", "clouds": "nature-cloudfield", "aurora": "nature-aurora-sheets"}


def sequence(ctx, max_w: int = 720) -> list[np.ndarray]:
    """Frames for ctx at a working size (<= max_w wide) according to ``scene``."""
    W, H = ctx.size
    w = min(W, max_w if str(ctx.p("scene")) != "orbs" else min(max_w, 560))
    size = (w, max(8, int(round(w * H / W))))
    n = int(ctx.p("n_frames"))
    t0, span = float(ctx.p("t0")), float(ctx.p("span"))
    scene = str(ctx.p("scene"))
    if scene == "auto":
        scene = "pan" if ctx.source is not None else "orbs"
    travel = float(ctx.p("travel"))
    if scene == "frames":
        data = ctx.data
        if data is not None and not isinstance(data, str):
            arr = [np.asarray(f, np.float32) for f in data]
            return [F.fit_cover(f[..., :3] / (255.0 if f.max() > 1.5 else 1.0), size, "center") for f in arr]
        return read_frames(str(data or ctx.p("frames")), size, n, t0, span)
    if scene == "pan":
        if ctx.source is None:
            ctx = _with_source(ctx, size)
        return pan_frames(ctx, size, n, t0, span, travel, float(ctx.p("pan_angle")), float(ctx.p("shake")))
    if scene == "orbs":
        return orbs_frames(ctx, size, max(n, 160), t0, span, travel * 2.6, int(ctx.p("bodies")))
    if scene in _SCENE_EFFECT or scene.startswith("effect:"):
        size = (min(w, 480), max(8, int(round(min(w, 480) * H / W))))
    if scene in _SCENE_EFFECT:
        return effect_frames(ctx, _SCENE_EFFECT[scene], size, n, t0, span, travel)
    if scene.startswith("effect:"):
        return effect_frames(ctx, scene.split(":", 1)[1], size, n, t0, span, travel)
    raise ValueError(f"unknown scene {scene!r}")


def _with_source(ctx, size):
    from dataclasses import replace
    c = type(ctx)(size=ctx.size, seed=ctx.seed, palette=ctx.palette, params=ctx.params, source=F.synthetic_source(size, ctx.seed), data=ctx.data)
    return c


# ---------------------------------------------------------------- tools
def luma(a):
    return a @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def background(frames):
    """Per-pixel temporal median (the static scene, moving things removed)."""
    return np.median(np.stack(frames, 0), axis=0).astype(np.float32)


def flow_between(a, b, scale=0.5):
    """Dense optical flow a -> b (Farneback), (H, W, 2) in pixels of a."""
    ga = (np.clip(luma(a), 0, 1) * 255).astype(np.uint8)
    gb = (np.clip(luma(b), 0, 1) * 255).astype(np.uint8)
    if scale != 1:
        ga = cv2.resize(ga, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gb = cv2.resize(gb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    fl = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 4, 21, 3, 7, 1.5, 0)
    if scale != 1:
        fl = cv2.resize(fl, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR) / scale
    return fl.astype(np.float32)


def align_translation(frames, ref=None):
    """Compensate global camera translation (phase correlation on luminance)."""
    ref = len(frames) // 2 if ref is None else ref
    g0 = luma(frames[ref]).astype(np.float32)
    win = cv2.createHanningWindow((g0.shape[1], g0.shape[0]), cv2.CV_32F)
    out = []
    for f in frames:
        (dx, dy), _ = cv2.phaseCorrelate(g0, luma(f).astype(np.float32), win)
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        out.append(cv2.warpAffine(f, M, (f.shape[1], f.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT))
    return out


def colourise(ctx, rgb, lum=None, mode=None, extra=None):
    """Colour layer shared by the motion effects: ``photo`` keeps the frame colours
    (re-graded by contrast / tint), ``ramp`` paints a scalar through the palette
    ramp (ink on paper for light palettes). ``lum`` is the scalar for ramp mode."""
    mode = mode or str(ctx.p("colour_mode", "photo"))
    if mode == "ramp":
        t = lum if lum is not None else F.stretch(luma(rgb), 1, 99)
        if ctx.palette.polarity != "dark":
            pass
        R = N.ramp(ctx.palette, 640, ctx.p("ramp", "mono"), float(ctx.p("tint", 0.0)), float(ctx.p("contrast", 1.0)), bool(ctx.p("invert", False)))
        return N.paint(np.clip(t, 0, 1), R)
    c = float(ctx.p("contrast", 1.0))
    out = np.clip((rgb - 0.5) * c + 0.5, 0, 1)
    th = float(ctx.p("tint", 0.0))
    if th:
        lab = F.to_oklab(out)
        h = math.radians(th)
        a2, b2 = lab[..., 1].copy(), lab[..., 2].copy()
        lab[..., 1] = a2 * math.cos(h) - b2 * math.sin(h)
        lab[..., 2] = a2 * math.sin(h) + b2 * math.cos(h)
        out = F.from_oklab(lab)
    return out


def finish(ctx, rgb):
    W, H = ctx.size
    if rgb.shape[1] != W or rgb.shape[0] != H:
        rgb = cv2.resize(np.ascontiguousarray(rgb, np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    return N.surface(rgb, ctx)


MOTION_COLOUR = {
    "colour_mode": ("photo", "COLOUR: photo (keep the frames' colour) | ramp (scalar through the palette ramp; dark palette = light on dark, light palette = ink on paper)"),
}


def motion_params(params: dict) -> dict:
    out = N.with_common(with_seq(params), source=False)
    for k, v in MOTION_COLOUR.items():
        out.setdefault(k, v)
    out.setdefault("crop", N.SOURCE_PARAMS["crop"])
    return out
