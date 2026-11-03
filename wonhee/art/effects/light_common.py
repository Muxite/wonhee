"""Shared light-rendering toolkit for the ``light_*`` effects (no effect is
registered here).

Pipeline used by every light effect: accumulate scalar HDR *exposure* in a
float buffer (additive splatting, like photons on a sensor), then
``develop`` it: multi-scale glow, filmic tone curve, lookup in a light ramp
built from the palette roles (canvas -> structure -> accent -> tinted
highlight), a faint tinted ground, fine grain. One hue family comes out of
this by construction; contrast is lightness only.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from scipy import ndimage

from ..palette import oklch_to_rgb, rgb_to_oklch


def scale(ctx) -> float:
    """Pixel scale relative to a 1500 px wide frame."""
    return ctx.size[0] / 1500.0


# ---------------------------------------------------------------- splatting
def splat(buf: np.ndarray, x, y, w) -> None:
    """Bilinear additive splat of weights ``w`` at float positions into buf (H, W)."""
    H, W = buf.shape
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    w = np.broadcast_to(np.asarray(w, np.float64), x.shape)
    x0 = np.floor(x).astype(np.int64); y0 = np.floor(y).astype(np.int64)
    fx = x - x0; fy = y - y0
    flat = buf.reshape(-1)
    for dx, dy, wt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi = x0 + dx; yi = y0 + dy
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        if ok.any():
            flat += np.bincount(yi[ok] * W + xi[ok], weights=(w * wt)[ok], minlength=W * H).astype(buf.dtype)


def splat_segments(buf: np.ndarray, p0, p1, w0, w1=None, step: float = 0.7) -> None:
    """Additive line segments. ``w`` is intensity per unit length (independent
    of sampling density); it is interpolated from w0 to w1 along each segment.
    p0, p1: (N, 2) arrays in pixel coords."""
    p0 = np.asarray(p0, np.float64).reshape(-1, 2); p1 = np.asarray(p1, np.float64).reshape(-1, 2)
    n = len(p0)
    if n == 0:
        return
    w0 = np.broadcast_to(np.asarray(w0, np.float64), (n,))
    w1 = w0 if w1 is None else np.broadcast_to(np.asarray(w1, np.float64), (n,))
    d = p1 - p0
    L = np.hypot(d[:, 0], d[:, 1])
    cnt = np.maximum(1, np.ceil(L / step).astype(np.int64))
    # chunk to bound memory
    start = 0
    total = cnt.sum()
    if total > 6_000_000:
        csum = np.cumsum(cnt)
        edges = [0]
        k = 6_000_000
        while edges[-1] < n:
            j = int(np.searchsorted(csum, (csum[edges[-1] - 1] if edges[-1] else 0) + k, side="right"))
            edges.append(min(n, max(j, edges[-1] + 1)))
        for a, b in zip(edges[:-1], edges[1:]):
            splat_segments(buf, p0[a:b], p1[a:b], w0[a:b], w1[a:b], step)
        return
    idx = np.repeat(np.arange(n), cnt)
    first = np.cumsum(cnt) - cnt
    k = np.arange(total) - first[idx]
    t = (k + 0.5) / cnt[idx]
    x = p0[idx, 0] + d[idx, 0] * t
    y = p0[idx, 1] + d[idx, 1] * t
    w = (w0[idx] + (w1[idx] - w0[idx]) * t) * (L[idx] / cnt[idx])
    splat(buf, x, y, w)


def splat_polyline(buf, pts, w, step: float = 0.7) -> None:
    """pts (N,2); w scalar or (N,) per-vertex intensity per unit length."""
    pts = np.asarray(pts, np.float64)
    w = np.broadcast_to(np.asarray(w, np.float64), (len(pts),))
    splat_segments(buf, pts[:-1], pts[1:], w[:-1], w[1:], step)


class SegAcc:
    """Collects segments from many small polylines and splats them in one
    call (one bincount instead of thousands: big speed-up)."""

    def __init__(self):
        self.a, self.b, self.w0, self.w1 = [], [], [], []

    def polyline(self, pts, w):
        pts = np.asarray(pts, np.float64)
        w = np.broadcast_to(np.asarray(w, np.float64), (len(pts),))
        self.a.append(pts[:-1]); self.b.append(pts[1:]); self.w0.append(w[:-1]); self.w1.append(w[1:])

    def segments(self, a, b, w0, w1=None):
        a = np.asarray(a, np.float64); n = len(a)
        w0 = np.broadcast_to(np.asarray(w0, np.float64), (n,))
        w1 = w0 if w1 is None else np.broadcast_to(np.asarray(w1, np.float64), (n,))
        self.a.append(a); self.b.append(np.asarray(b, np.float64)); self.w0.append(w0); self.w1.append(w1)

    def flush(self, buf, step: float = 0.7):
        if self.a:
            splat_segments(buf, np.concatenate(self.a), np.concatenate(self.b), np.concatenate(self.w0), np.concatenate(self.w1), step)
        self.__init__()


# --------------------------------------------------------------- smooth noise
def smooth_noise(rng, shape, sigma, octaves: int = 1, persistence: float = 0.5) -> np.ndarray:
    """Zero-mean unit-ish smooth noise (H, W) via filtered white noise."""
    H, W = shape
    out = np.zeros((H, W), np.float32)
    amp = 1.0
    tot = 0.0
    for o in range(octaves):
        s = max(sigma / (2 ** o), 0.8)
        n = rng.standard_normal((H, W)).astype(np.float32)
        n = ndimage.gaussian_filter(n, s, mode="wrap")
        n /= (n.std() + 1e-9)
        out += amp * n
        tot += amp * amp
        amp *= persistence
    return out / math.sqrt(tot)


def lowres_noise(rng, shape, cells: float, octaves: int = 1, persistence: float = 0.5) -> np.ndarray:
    """Smooth noise by upsampling a small random grid (cheap, any size)."""
    H, W = shape
    out = np.zeros((H, W), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        c = cells * (2 ** o)
        gw = max(3, int(round(c))) + 3
        gh = max(3, int(round(c * H / W))) + 3
        g = rng.standard_normal((gh, gw)).astype(np.float32)
        up = cv2.resize(g, (int(W * (gw - 1) / (gw - 3)) + 2, int(H * (gh - 1) / (gh - 3)) + 2), interpolation=cv2.INTER_CUBIC)
        oy, ox = (up.shape[0] - H) // 2, (up.shape[1] - W) // 2
        out += amp * up[oy:oy + H, ox:ox + W]
        tot += amp * amp
        amp *= persistence
    return out / math.sqrt(tot) / 1.0


# ----------------------------------------------------------------- development
def light_ramp(palette, n: int = 1024) -> np.ndarray:
    """canvas -> structure -> accent -> tinted highlight, n x 3, lightness monotone."""
    # one hue family: the accent role contributes lightness (and a bounded
    # 30% of its hue offset), so a complement-accent palette (ultramarine)
    # still reads as one hue with a pale warm-tinted highlight, not two hues.
    _, _, hs = rgb_to_oklch(palette.structure)
    La, Ca, ha = rgb_to_oklch(palette.accent)
    d = (ha - hs + 180) % 360 - 180
    he = (hs + 0.3 * d) % 360
    acc = oklch_to_rgb(La, Ca * 0.8, he)
    hi = oklch_to_rgb(0.97, 0.03, he)
    cols = [palette.canvas, palette.structure, acc, hi]
    pos = np.array([0.0, 0.42, 0.74, 1.0])
    t = np.linspace(0, 1, n)
    return np.stack([np.interp(t, pos, [c[i] for c in cols]) for i in range(3)], 1).astype(np.float32)


def ground(ctx, strength: float = 1.0) -> np.ndarray:
    """Tinted dark/light ground: canvas with a soft broad band-tint wander (never pure black)."""
    W, H = ctx.size
    rng = np.random.default_rng(ctx.seed + 7919)
    n = lowres_noise(rng, (H, W), 2.2, 2)
    n = np.clip(0.5 + 0.35 * n, 0, 1)[..., None] * 0.9 * strength
    p = ctx.palette
    return (p.canvas + (p.band - p.canvas) * n).astype(np.float32)


def develop(ctx, I: np.ndarray, *, exposure: float = 1.0, glow: float = 0.5, glow_sigmas=(2.0, 7.0, 22.0),
            gamma: float = 1.0, grain: float = 0.012, base=None, fringe: float = 0.0, ramp=None) -> np.ndarray:
    """HDR scalar exposure (H, W) -> float32 sRGB (H, W, 3).

    glow: strength of the multi-scale halo added to the exposure.
    fringe: 0..1 chromatic fringe, shifts the ramp lookup by a pixel or two
    per channel (kept subtle: it must stay inside one hue family)."""
    W, H = ctx.size
    sc = scale(ctx)
    I = np.maximum(I, 0).astype(np.float32)
    if glow > 0:
        g = np.zeros_like(I)
        for s, wt in zip(glow_sigmas, (0.5, 0.32, 0.18)):
            g += wt * cv2.GaussianBlur(I, (0, 0), max(0.6, s * sc))
        I = I + glow * g * 2.0
    t = 1.0 - np.exp(-exposure * I)
    if gamma != 1.0:
        t = t ** gamma
    ramp = light_ramp(ctx.palette) if ramp is None else ramp
    n = len(ramp)
    if fringe > 0:
        d = max(1, int(round(1.5 * sc)))
        chans = []
        for c, sh in enumerate((d, 0, -d)):
            tc = np.roll(t, sh, axis=1) * fringe + t * (1 - fringe)
            chans.append(np.interp(tc, np.linspace(0, 1, n), ramp[:, c]))
        out = np.stack(chans, -1)
    else:
        idx = np.clip(t * (n - 1), 0, n - 1)
        i0 = np.floor(idx).astype(np.int32); f = (idx - i0)[..., None]
        i1 = np.minimum(i0 + 1, n - 1)
        out = ramp[i0] * (1 - f) + ramp[i1] * f
    if base is not None:  # tinted ground only where it is dark: blend under the ramp
        gnd = base
        m = (1 - t)[..., None]
        out = out + (gnd - ramp[0]) * m
    if grain > 0:
        rng = np.random.default_rng(ctx.seed + 104729)
        gr = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 0.6)
        out = out + (gr * grain)[..., None]
    return np.clip(out, 0, 1).astype(np.float32)


# ------------------------------------------------------------------ real material
def source_lum(ctx):
    """ctx.source resampled to the frame as luminance (H, W) in 0..1, or None."""
    if ctx.source is None:
        return None
    W, H = ctx.size
    s = np.asarray(ctx.source, np.float32)
    if s.shape[:2] != (H, W):
        s = cv2.resize(s, (W, H), interpolation=cv2.INTER_AREA)
    return (s @ np.array([0.2126, 0.7152, 0.0722], np.float32)) if s.ndim == 3 else s


def _spring(n, edges, rng, iters=120):
    pos = rng.random((n, 2))
    k = math.sqrt(1.0 / max(n, 1))
    e = np.array(edges, int).reshape(-1, 2) if len(edges) else np.zeros((0, 2), int)
    for it in range(iters):
        d = pos[:, None, :] - pos[None, :, :]
        r = np.maximum(np.hypot(d[..., 0], d[..., 1]), 1e-3)
        disp = (d / r[..., None] * (k * k / r)[..., None]).sum(1)
        for a, b in e:
            dd = pos[a] - pos[b]
            rr = max(float(np.hypot(*dd)), 1e-3)
            f = dd / rr * rr * rr / k
            disp[a] -= f; disp[b] += f
        ln = np.maximum(np.hypot(disp[:, 0], disp[:, 1]), 1e-9)
        pos += disp / ln[:, None] * np.minimum(ln, 0.1 * (1 - it / iters))[:, None]
    return pos


def data_points(ctx, margin: float = 0.06):
    """Real material -> (points (N,2) in pixels, edges (M,2) int, weights (N,)) or None.

    Accepted ``ctx.data``:
      * (N,2) array / list of [x, y]
      * dict with "points" or "nodes": list of [x, y] / {"x","y"[,"w"/"size"]} /
        {"id"} (no coords -> spring layout); optional "edges"/"links": pairs of
        indices or ids (or {"source","target"}).
    Coordinates are fitted into the frame with a margin (aspect not preserved:
    the point set fills the whole frame, as the art rules demand)."""
    d = ctx.data
    if d is None:
        return None
    W, H = ctx.size
    rng = np.random.default_rng(ctx.seed + 31337)
    nodes, edges = d, []
    if isinstance(d, dict):
        nodes = d.get("points", d.get("nodes", []))
        edges = d.get("edges", d.get("links", []))
    if isinstance(nodes, np.ndarray):
        nodes = nodes.tolist()
    if not len(nodes):
        return None
    ids = {}
    pts, wts = [], []
    have_xy = True
    for i, nd in enumerate(nodes):
        if isinstance(nd, dict):
            ids[nd.get("id", i)] = i
            if "x" in nd and "y" in nd:
                pts.append((float(nd["x"]), float(nd["y"])))
            else:
                have_xy = False; pts.append((0.0, 0.0))
            wts.append(float(nd.get("w", nd.get("size", 1.0))))
        elif isinstance(nd, (list, tuple, np.ndarray)) and len(nd) >= 2:
            ids[i] = i; pts.append((float(nd[0]), float(nd[1]))); wts.append(float(nd[2]) if len(nd) > 2 else 1.0)
        else:
            ids[nd] = i; have_xy = False; pts.append((0.0, 0.0)); wts.append(1.0)
    ed = []
    for e in edges:
        if isinstance(e, dict):
            a, b = e.get("source"), e.get("target")
        else:
            a, b = e[0], e[1]
        a, b = ids.get(a, a), ids.get(b, b)
        if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)) and 0 <= a < len(pts) and 0 <= b < len(pts) and a != b:
            ed.append((int(a), int(b)))
    P = np.array(pts, np.float64)
    if not have_xy:
        P = _spring(len(P), ed, rng)
    lo, hi = P.min(0), P.max(0)
    P = (P - lo) / np.maximum(hi - lo, 1e-9)
    P = np.stack([(margin + P[:, 0] * (1 - 2 * margin)) * W, (margin + P[:, 1] * (1 - 2 * margin)) * H], 1)
    return P, np.array(ed, int).reshape(-1, 2), np.array(wts, np.float64)


def blue_noise(rng, n: int, W: int, H: int, margin: float = 0.0, tries: int = 12) -> np.ndarray:
    """n well-spread points (Mitchell best-candidate), pixels."""
    pts = np.zeros((n, 2))
    mx, my = margin * W, margin * H
    pts[0] = (mx + rng.random() * (W - 2 * mx), my + rng.random() * (H - 2 * my))
    for i in range(1, n):
        c = np.stack([mx + rng.random(tries * 2) * (W - 2 * mx), my + rng.random(tries * 2) * (H - 2 * my)], 1)
        d = ((c[:, None, :] - pts[None, :i, :]) ** 2).sum(-1).min(1)
        pts[i] = c[int(np.argmax(d))]
    return pts


def jitter_grid(rng, n: int, W: int, H: int, jitter: float = 0.8) -> np.ndarray:
    """~n points on a jittered grid matched to the aspect, shuffled, pixels."""
    gx = max(1, int(round(math.sqrt(n * W / H)))); gy = max(1, int(math.ceil(n / gx)))
    pts = [((i + 0.5 + (rng.random() - 0.5) * jitter) / gx * W, (j + 0.5 + (rng.random() - 0.5) * jitter) / gy * H)
           for i in range(gx) for j in range(gy)]
    pts = np.array(pts)
    rng.shuffle(pts)
    return pts
