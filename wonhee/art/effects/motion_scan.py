"""Slit-scan, frame differences and flow trails from a frame sequence."""
from __future__ import annotations

import math

import cv2
import numpy as np

from .. import features as F
from .. import motionkit as M
from .. import natkit as N
from ..registry import effect


@effect("motion-slit-scan", family="motion",
        tags=["motion", "slit-scan", "time-displacement", "sequence", "video-ready", "dreamy", "restless", "cinematic", "mood:dreamy", "mood:restless", "mood:cinematic"],
        params=M.motion_params({
            "n_frames": (40, "SOURCE: frames in the time volume (more = smoother time gradient)"),
            "travel": (0.9, "SOURCE: pan / orb travel in frame widths over the clip (slit-scan wants a long move)"),
            "mode": ("warp", "STRUCTURE: warp (each pixel samples the sequence at a time that varies across the frame) | slit (one column per output column, scanning the clip)"),
            "angle": (0.0, "STRUCTURE: direction in which time increases across the frame (radians)"),
            "bands": (1.0, "STRUCTURE: how many times time sweeps the whole clip across the frame (1 = once; 2+ = repeated, mirrored bands)"),
            "warp": (0.35, "STRUCTURE: 0..1 wobble of the time map (smooth space-time noise, coordinate based)"),
            "slit": (0.5, "STRUCTURE: slit position 0..1 for mode=slit"),
            "time_gain": (0.9, "STRUCTURE: 0..1 how much of the clip the time map covers"),
        }))
def slit_scan(ctx):
    """Time-displacement slit-scan: out(x, y) = frame[T(x, y)](x, y), with T a
    smooth ramp across the frame (direction `angle`, `bands` sweeps) plus
    seeded space-time noise (`warp`). With a moving subject the result is
    stretched and sheared along time; with a camera pan the scene unrolls into
    long smooth gradients; static regions stay untouched, so a recognisable
    scene is bent rather than destroyed. mode=slit is the classic strip scan
    (a single column of every frame becomes one output column). The time map
    is a pure function of normalised position, so it applies unchanged to
    video. Broad structure, no fine streaks."""
    M.prefer_scene(ctx, "plankton")
    mode = str(ctx.p("mode"))
    if mode == "slit":
        ctx.params["n_frames"] = max(int(ctx.p("n_frames")), 200)
    fr = M.sequence(ctx)
    n = len(fr)
    H, W = fr[0].shape[:2]
    if mode == "slit":
        sx = int(np.clip(float(ctx.p("slit")), 0, 1) * (W - 1))
        cols = np.stack([f[:, sx, :] for f in fr], 1)                      # (H, n, 3)
        out = cv2.resize(cols, (W, H), interpolation=cv2.INTER_CUBIC)
        # keep a hint of the scene: lightly mix the mean frame so the colour stays tied to the photo
        out = out * 0.88 + np.mean(fr, 0) * 0.12
    else:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        ca, sa = math.cos(float(ctx.p("angle"))), math.sin(float(ctx.p("angle")))
        u = ((xx - W / 2) * ca + (yy - H / 2) * sa) / (abs(W * ca) + abs(H * sa) + 1e-6) + 0.5
        u = u * float(ctx.p("bands"))
        tri = np.abs(((u + 1) % 2) - 1)                                       # 0..1..0 mirrored bands
        wob = float(ctx.p("warp"))
        if wob > 0:
            tri = tri + wob * 0.35 * N.tnoise(xx / W * 3.0, yy / W * 3.0, 0.0, ctx.seed, 1.0) + wob * 0.12 * N.tnoise(xx / W * 8.0, yy / W * 8.0, 1.0, ctx.seed + 7, 1.0)
        T = np.clip(tri, 0, 1) * float(ctx.p("time_gain")) * (n - 1) + (1 - float(ctx.p("time_gain"))) * (n - 1) * 0.5
        out = np.zeros((H, W, 3), np.float32)
        for k in range(n):
            w = np.clip(1 - np.abs(T - k), 0, 1)
            if w.max() > 0:
                out += w[..., None] * fr[k]
    out = np.clip(out, 0, 1)
    lum = M.luma(out)
    t = F.stretch(lum if ctx.palette.polarity == "dark" else 1 - lum, 1, 99.5) if ctx.p("colour_mode") == "ramp" else None
    return M.finish(ctx, M.colourise(ctx, out, t))


@effect("motion-frame-diff", family="motion",
        tags=["motion", "frame-difference", "edges", "sequence", "video-ready", "energetic", "restless", "precise", "mood:energetic", "mood:restless", "mood:scientific"],
        params=M.motion_params({
            "lag": (3, "STRUCTURE: frame distance of each difference (larger = fatter ghosts)"),
            "window": (0.8, "STRUCTURE: 0..1 share of the clip whose differences are accumulated"),
            "accumulate": ("max", "STRUCTURE: max (crisp, every motion edge once) | sum (dense layering)"),
            "signed": (True, "COLOUR: brightening and darkening motion get different hues (structure / accent); False = magnitude only"),
            "gain": (1.0, "SURFACE: gain relative to auto exposure (1 = 97th percentile of the motion at full ink)"),
            "denoise": (3.5, "SURFACE: noise suppression, blur in px at 1500 wide applied to the difference before thresholding"),
            "floor": (0.05, "SURFACE: differences below this are zero (kills sensor / compression noise)"),
            "base": (0.25, "SURFACE: 0..1 faint mean frame kept under the differences (0 = differences only)"),
            "widen": (1.0, "SURFACE: line widening for thumbnails: minimum ghost width in px at 288 px wide (scaled up)"),
        }))
def frame_diff(ctx):
    """Motion as light: absolute (or signed) differences between frames `lag`
    apart, denoised and floored, accumulated with max or sum over the window,
    and drawn on the canvas colour through the palette ramp (signed: two hues,
    brighter-than-before vs darker-than-before). Static scene vanishes; only
    where things moved remains, as broad ghost outlines with direction
    (a pan makes every edge a double-edge). Differs from feature-edge-glitch
    (single image). The denoise/floor/widen controls keep it clean and
    readable at 288x192."""
    M.prefer_scene(ctx, "plankton")
    fr = M.sequence(ctx)
    n = len(fr)
    H, W = fr[0].shape[:2]
    lag = int(max(1, ctx.p("lag")))
    s = W / 1500.0
    k0 = int((1 - float(ctx.p("window"))) * (n - lag - 1) * 0.5)
    k1 = n - lag - k0
    acc_pos = np.zeros((H, W), np.float32)
    acc_neg = np.zeros((H, W), np.float32)
    sg = max(0.6, float(ctx.p("denoise")) * s)
    for i in range(k0, max(k0 + 1, k1)):
        a = M.luma(cv2.GaussianBlur(fr[i], (0, 0), sg))
        b = M.luma(cv2.GaussianBlur(fr[i + lag], (0, 0), sg))
        d = b - a
        pos, neg = np.clip(d - float(ctx.p("floor")), 0, 1), np.clip(-d - float(ctx.p("floor")), 0, 1)
        if ctx.p("accumulate") == "sum":
            acc_pos += pos; acc_neg += neg
        else:
            acc_pos = np.maximum(acc_pos, pos); acc_neg = np.maximum(acc_neg, neg)
    # auto exposure: gain 1 puts the 97th percentile of the motion response at full ink
    am = np.maximum(acc_pos, acc_neg)
    g = float(ctx.p("gain")) / (np.percentile(am, 97) + 1e-4)
    # line widening for thumbnails: grow the response so it survives downscale
    rad = max(0.0, float(ctx.p("widen")) - 1.0) * 1.2 * s + 0.0
    if rad > 0.4:
        kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(rad * 2) | 1, int(rad * 2) | 1))
        acc_pos, acc_neg = cv2.dilate(acc_pos, kk), cv2.dilate(acc_neg, kk)
    mag = np.clip(np.maximum(acc_pos, acc_neg) * g, 0, 1)
    base = F.blur(np.mean(fr, 0).mean(-1), 2.0 * s)
    dark = ctx.palette.polarity == "dark"
    R1 = N.ramp(ctx.palette, 512, ctx.p("ramp"), float(ctx.p("tint")), float(ctx.p("contrast")), bool(ctx.p("invert")))
    bb = float(ctx.p("base")) * 0.30
    floor_t = (bb * base) if dark else 1 - bb * 0.0
    if ctx.p("colour_mode") == "photo" and bool(ctx.p("signed")):
        R2 = N.ramp(ctx.palette, 512, ctx.p("ramp"), float(ctx.p("tint")) + 150.0, float(ctx.p("contrast")), bool(ctx.p("invert")))
        wpos = np.clip(acc_pos / (acc_pos + acc_neg + 1e-6), 0, 1)[..., None]
        t = np.clip(0.02 + floor_t + mag * 0.95, 0, 1) if dark else np.clip(mag * 0.95, 0, 1)
        out = N.paint(t, R1) * wpos + N.paint(t, R2) * (1 - wpos)
    else:
        t = np.clip(0.02 + floor_t + mag * 0.95, 0, 1) if dark else np.clip(mag * 0.95, 0, 1)
        out = N.paint(t, R1)
    return M.finish(ctx, out)


@effect("motion-flow-trails", family="motion",
        tags=["motion", "optical-flow", "motion-blur", "trails", "sequence", "video-ready", "energetic", "cinematic", "mood:energetic", "mood:cinematic"],
        params=M.motion_params({
            "length": (2.5, "STRUCTURE: trail length as a multiple of the per-frame motion (pixels moved between frames)"),
            "pair": (4, "STRUCTURE: frame distance used for the optical flow (larger = smoother, broader flow)"),
            "smooth": (0.6, "STRUCTURE: 0..1 smoothing of the flow field (broad directional movement, not fine streaks)"),
            "min_blur": (0.0, "STRUCTURE: px (at 1500 wide) of minimum blur so still areas stay soft too"),
            "freeze": (0.25, "SURFACE: 0..1 weight of the sharp frame on top (the subject stays legible)"),
        }))
def flow_trails(ctx):
    """Motion blur from the actual motion: dense optical flow (Farneback)
    between two frames of the sequence drives a line-integral smear of the
    middle frame along the local velocity, so each region streaks in its own
    direction and distance (a pan with parallax streaks near layers more than
    far ones; orbs leave trails along their own paths; a person moving left
    smears left). The smear is forward and backward (symmetric shutter) with a
    tapered weight; `freeze` keeps a sharp frame under the blur. Colour is
    independent (photo or palette ramp). Runs on video unchanged."""
    M.prefer_scene(ctx, "plankton")
    fr = M.sequence(ctx)
    n = len(fr)
    H, W = fr[0].shape[:2]
    i = n // 2
    j = int(np.clip(i + int(ctx.p("pair")), 0, n - 1))
    i0 = max(0, j - int(ctx.p("pair")))
    fl = M.flow_between(fr[i0], fr[j], 0.5) / max(1, j - i0)
    sm = float(ctx.p("smooth"))
    s = W / 1500.0
    if sm > 0:
        fl = cv2.GaussianBlur(fl, (0, 0), (2 + 20 * sm) * s)
    mid = fr[(i0 + j) // 2]
    L = float(ctx.p("length"))
    steps = 40
    vx, vy = fl[..., 0].copy(), fl[..., 1].copy()
    mb = float(ctx.p("min_blur")) * s
    if mb > 0:
        mg = np.hypot(vx, vy) + 1e-6
        k = np.maximum(mg, mb / max(L, 1e-3)) / mg
        vx, vy = vx * k, vy * k
    out = N.smear_along(mid, vx, vy, steps, L / steps * 1.0 * (steps / 20.0) * 0.5 * 2 / 2, keep_sign=False)
    fz = float(ctx.p("freeze"))
    out = out * (1 - fz) + mid * fz
    lum = M.luma(out)
    t = F.stretch(lum if ctx.palette.polarity == "dark" else 1 - lum, 1, 99.5) if ctx.p("colour_mode") == "ramp" else None
    return M.finish(ctx, M.colourise(ctx, np.clip(out, 0, 1), t))
