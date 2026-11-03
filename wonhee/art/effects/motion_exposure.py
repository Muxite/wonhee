"""Temporal accumulation: long exposures and stroboscopic repetition."""
from __future__ import annotations

import math

import cv2
import numpy as np

from .. import features as F
from .. import motionkit as M
from .. import natkit as N
from ..registry import effect


@effect("motion-long-exposure", family="motion",
        tags=["motion", "long-exposure", "trails", "sequence", "video-ready", "energetic", "dreamy", "cinematic", "mood:energetic", "mood:dreamy", "mood:cinematic"],
        params=M.motion_params({
            "accumulate": ("trail", "STRUCTURE: mean (smooth blur) | max (lighten: bright things leave full-strength trails) | trail (mean base plus fading lighten trails)"),
            "persistence": (0.6, "STRUCTURE: 0..1 how long trails last (1 = whole window, 0 = only the latest frames)"),
            "trail_amount": (0.8, "STRUCTURE: 0..1 weight of the lighten trails over the mean exposure"),
            "sharp": (0.35, "STRUCTURE: 0..1 rear-curtain flash: weight of the last frame laid sharp on top"),
            "gain": (1.0, "SURFACE: exposure gain on the accumulated image"),
            "bloom": (0.0, "SURFACE: 0..1 soft glow on the trails (lens / long exposure)"),
        }))
def long_exposure(ctx):
    """Long-exposure composite of a frame sequence: a mean exposure (broad,
    smooth directional blur of everything that moved) with optional lighten
    trails whose weight fades with age (persistence), and an optional
    rear-curtain sharp last frame. From a pan over a photo it yields a broad
    camera-motion blur with parallax; from orbs, light-painting trails; from a
    video, the classic long exposure. Colour is independent: keep the frame
    colour or push the exposure through the palette ramp. Temporal coherence:
    weights depend only on normalised time, so the same look runs on any frame
    count. Broad directional movement, not fine streaks: reads at 288x192."""
    fr = M.sequence(ctx)
    n = len(fr)
    persistence = float(ctx.p("persistence"))
    tau = max(0.5, (0.05 + 1.6 * persistence) * n)
    ws = np.exp(-(n - 1 - np.arange(n)) / tau).astype(np.float32)
    mean = sum(f * w for f, w in zip(fr, ws)) / ws.sum()
    mode = str(ctx.p("accumulate"))
    mx = fr[0] * ws[0]
    for f, w in zip(fr[1:], ws[1:]):
        mx = np.maximum(mx, f * w)
    if mode == "mean":
        out = mean
    elif mode == "max":
        out = mx / ws.max()
    else:
        a = float(ctx.p("trail_amount"))
        out = np.maximum(mean, mean * (1 - a) + mx * a * 1.15)
    sharp = float(ctx.p("sharp"))
    if sharp > 0:
        out = out * (1 - sharp) + fr[-1] * sharp
    out = np.clip(out * float(ctx.p("gain")), 0, 1.5)
    b = float(ctx.p("bloom"))
    if b > 0:
        s = max(1.0, out.shape[1] / 1500.0)
        gl = cv2.GaussianBlur(np.clip(out - 0.55, 0, 1), (0, 0), 6 * s) * 0.6 + cv2.GaussianBlur(np.clip(out - 0.55, 0, 1), (0, 0), 20 * s) * 0.4
        out = 1 - (1 - np.clip(out, 0, 1)) * (1 - np.clip(gl * b * 1.8, 0, 1))
    out = np.clip(out, 0, 1)
    lum = M.luma(out)
    t = F.stretch(lum if ctx.palette.polarity == "dark" else 1 - lum, 1, 99.5) if ctx.p("colour_mode") == "ramp" else None
    return M.finish(ctx, M.colourise(ctx, out, t))


@effect("motion-stroboscope", family="motion",
        tags=["motion", "stroboscope", "chronophotography", "sequence", "video-ready", "energetic", "restless", "mood:energetic", "mood:restless"],
        params=M.motion_params({
            "mode": ("auto", "STRUCTURE: auto (multi for pans of a photo, movers otherwise) | movers (repeat what differs from the static scene) | multi (multiple exposure: whole frames repeated with offsets, lighten blend)"),
            "echoes": (7, "STRUCTURE: number of repeated exposures laid over the static scene"),
            "threshold": (0.09, "STRUCTURE: motion mask threshold (higher = only strong movers repeat)"),
            "feather": (4.0, "STRUCTURE: mask smoothing in px at 1500 wide (noise suppression)"),
            "align": (True, "STRUCTURE: cancel camera translation first so only subject / parallax motion repeats"),
            "fade": (0.55, "STRUCTURE: 0..1 how much older echoes fade"),
            "temporal_hue": (0.6, "COLOUR: 0..1 echo colour steps through the palette ramp with time (0 = photo colour)"),
            "background": (0.55, "SURFACE: 0..1 brightness of the static scene under the echoes"),
        }))
def stroboscope(ctx):
    """Chronophotography: the static scene is the temporal median of the
    (translation-aligned) sequence; everything that differs from it in each
    selected frame is a motion mask; the masked frames are laid back in
    chronological order so the mover is repeated at regular intervals, older
    echoes fading and stepping through the colour ramp (temporal_hue). Works
    on orb scenes, pans with parallax (near layers repeat against a still far
    layer), the library's flow effects, and real video. Failure mode: a
    sequence with no relative motion yields no echoes (the static scene)."""
    fr = M.sequence(ctx)
    if bool(ctx.p("align")) and ctx.p("scene") not in ("orbs",):
        fr = M.align_translation(fr)
    n = len(fr)
    bg = M.background(fr)
    k = int(ctx.p("echoes"))
    idx = np.unique(np.round(np.linspace(0, n - 1, k)).astype(int))
    s = fr[0].shape[1] / 1500.0
    dark = ctx.palette.polarity == "dark"
    R = N.ramp(ctx.palette, 256, ctx.p("ramp"), float(ctx.p("tint")), float(ctx.p("contrast")), bool(ctx.p("invert")))
    out = bg * float(ctx.p("background")) if dark else 1 - (1 - bg) * float(ctx.p("background")) * 0 - 0 * bg
    if not dark:
        out = 1 - (1 - bg) * (0.3 + 0.7 * float(ctx.p("background")))
    th = float(ctx.p("threshold"))
    mode = str(ctx.p("mode"))
    if mode == "auto":
        mode = "multi" if (ctx.p("scene") == "pan" or (ctx.p("scene") == "auto" and ctx.source is not None)) else "movers"
    if mode == "multi":
        fr0 = M.sequence(ctx)
        n0 = len(fr0)
        idx0 = np.unique(np.round(np.linspace(0, n0 - 1, k)).astype(int))
        acc = None
        for j, i in enumerate(idx0):
            age = (len(idx0) - 1 - j) / max(1, len(idx0) - 1)
            w = 1 - float(ctx.p("fade")) * age
            f = fr0[i]
            th_mix = float(ctx.p("temporal_hue"))
            if th_mix > 0:
                lum = np.clip(M.luma(f), 0, 1) if dark else np.clip(1 - M.luma(f), 0, 1)
                tt = 0.2 + 0.8 * (j / max(1, len(idx0) - 1))
                f = f * (1 - th_mix) + N.paint(np.clip(lum * 0.6 + 0.4 * tt, 0, 1), R) * th_mix
            f = f * w
            acc = f if acc is None else (np.maximum(acc, f) * 0.55 + (acc + f) * 0.5 * 0.45 if dark else np.minimum(acc, 1 - (1 - f) * 1.0) * 0.55 + (acc + f) * 0.5 * 0.45)
        return M.finish(ctx, np.clip(acc, 0, 1))
    for j, i in enumerate(idx):
        d = np.abs(fr[i] - bg).max(-1)
        m = N.smooth(th, 3 * th, d)
        m = cv2.GaussianBlur(m, (0, 0), max(0.5, float(ctx.p("feather")) * s))
        m = np.clip(m * 1.4, 0, 1)
        age = (len(idx) - 1 - j) / max(1, len(idx) - 1)
        w = (1 - float(ctx.p("fade")) * age)
        f = fr[i]
        th_mix = float(ctx.p("temporal_hue"))
        if th_mix > 0:
            lum = np.clip(M.luma(f), 0, 1) if dark else np.clip(1 - M.luma(f), 0, 1)
            tt = 0.25 + 0.75 * (j / max(1, len(idx) - 1))
            col = N.paint(np.clip(lum * 0.55 + 0.45 * tt, 0, 1) if dark else np.clip(lum * 0.7 + 0.15 + 0.15 * tt, 0, 1), R)
            f = f * (1 - th_mix) + col * th_mix
        mm = (m * w)[..., None]
        out = out * (1 - mm) + f * mm
    return M.finish(ctx, M.colourise(ctx, np.clip(out, 0, 1), None, mode="photo" if ctx.p("colour_mode") == "photo" else "photo"))
