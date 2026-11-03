"""Swiss-inspired composition engine (families ``swiss-*``).

Not decoration: a small layout engine.

* Grid        cols x rows modules (near square), gutters, optional margin. Everything snaps to it.
* Layout      merged-module blocks are packed (big anchor block first, one block in every
              quadrant, asymmetric sizes) and the rest stays negative space.
* Content     each block is a solid colour field, a large geometric form (circle, quarter, half,
              rings, bars, diagonal, dot grid, steps, cross), fine rules, an outline, or a crop of a photo
              (duotone / halftone). Forms may sit on tinted blocks so inks overprint.
* Type        a SEPARATE optional layer: real text from ``ctx.data`` (str, list of str, or dict
              ``{title, subtitle, caption}``) in a bundled/system sans (Liberation Sans = Helvetica metrics),
              fitted into the biggest free rectangle on grid lines. No text without data (never gibberish).
* Press       blocks are written into 4 named ink masks (tint, A, B, ink) and printed with
              :func:`printkit.press` (paper, fibre, grain, registration, wear; ``clean`` turns the surface off).

Called "Swiss-inspired": it borrows modular grids, flat colour, asymmetry and sans type, it does not copy a poster.
"""
from __future__ import annotations

import math
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import features as F
from . import palette as pal
from . import printkit as K

FONT_CANDIDATES = {
    "bold": ["/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
             "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf", "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"],
    "regular": ["/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf", "/usr/share/fonts/truetype/freefont/FreeSans.ttf"],
}


def font_path(weight: str = "bold", override: str = "") -> str | None:
    if override and os.path.exists(override):
        return override
    for p in FONT_CANDIDATES[weight]:
        if os.path.exists(p):
            return p
    return None


def load_font(px: int, weight: str = "bold", override: str = ""):
    p = font_path(weight, override)
    return ImageFont.truetype(p, px) if p else ImageFont.load_default()


# ---------------------------------------------------------------- grid

class Grid:
    def __init__(self, size, cols: int, margin: float, gutter: float):
        W, H = size
        self.W, self.H = W, H
        self.cols = cols
        m = int(round(margin * W))
        g = max(int(round(gutter * W)), 2 if W < 600 else 3) if gutter > 0 else 0
        mw = (W - 2 * m - (cols - 1) * g) / cols
        rows = max(2, int(round((H - 2 * m + g) / (mw + g))))
        mh = (H - 2 * m - (rows - 1) * g) / rows
        self.rows, self.m, self.g, self.mw, self.mh = rows, m, g, mw, mh
        self.xe = [m + c * (mw + g) for c in range(cols)] + [m + cols * (mw + g) - g]
        self.ye = [m + r * (mh + g) for r in range(rows)] + [m + rows * (mh + g) - g]

    def rect(self, c0, r0, c1, r1, bleed: bool = True):
        """Pixel rect of modules c0..c1-1 x r0..r1-1 (merged; spans gutters); edge blocks bleed to the frame."""
        x0 = self.xe[c0] if not (bleed and c0 == 0) else 0
        y0 = self.ye[r0] if not (bleed and r0 == 0) else 0
        x1 = (self.xe[c1 - 1] + self.mw) if not (bleed and c1 == self.cols) else self.W
        y1 = (self.ye[r1 - 1] + self.mh) if not (bleed and r1 == self.rows) else self.H
        return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))


# ---------------------------------------------------------------- layout

def pack(grid: Grid, rng, density: float = 0.7, big: float = 0.3, min_blocks: int = 8):
    """Merged-module blocks. Returns (blocks [(c0,r0,c1,r1)], occ)."""
    R, C = grid.rows, grid.cols
    occ = -np.ones((R, C), int)
    blocks: list[tuple[int, int, int, int]] = []

    def free(c0, r0, c1, r1):
        return c1 <= C and r1 <= R and (occ[r0:r1, c0:c1] < 0).all()

    def put(c0, r0, c1, r1):
        occ[r0:r1, c0:c1] = len(blocks)
        blocks.append((c0, r0, c1, r1))

    def place(w, h, region=None, tries=40):
        for _ in range(tries):
            c0, r0 = int(rng.integers(0, C - w + 1)), int(rng.integers(0, R - h + 1))
            if region and not (region[0] <= c0 < region[2] and region[1] <= r0 < region[3]):
                continue
            if free(c0, r0, c0 + w, r0 + h):
                put(c0, r0, c0 + w, r0 + h)
                return True
        return False

    # anchor: one big block (about `big` of the area), snapped to an edge/corner for asymmetry
    aw = int(np.clip(round(C * rng.uniform(0.34, 0.55)), 2, C - 1))
    ah = int(np.clip(round(R * rng.uniform(0.5, 1.0) * (big / 0.3) ** 0.5), 2, R))
    while aw * ah > max(big * 1.25 * C * R, 4) and ah > 2:
        ah -= 1
    c0 = int(rng.choice([0, C - aw]))
    r0 = int(rng.choice([0, R - ah]))
    put(c0, r0, c0 + aw, r0 + ah)
    # one block in each empty quadrant, so detail is spread over the whole frame
    for qc, qr in ((0, 0), (1, 0), (0, 1), (1, 1)):
        reg = (qc * C // 2, qr * R // 2, (qc + 1) * C // 2, (qr + 1) * R // 2)
        if (occ[reg[1]:reg[3], reg[0]:reg[2]] >= 0).sum() < 0.3 * (reg[2] - reg[0]) * (reg[3] - reg[1]):
            w, h = int(rng.integers(1, 3)), int(rng.integers(1, 3))
            place(w, h, reg)
    spans = [(1, 1), (2, 1), (1, 2), (2, 2), (3, 1), (1, 3), (3, 2), (2, 3), (4, 1), (1, 4), (C // 2, 1), (1, R)]
    wts = np.array([4, 4, 4, 3, 2, 2, 1.5, 1.5, 1, 1, 1, 0.8])
    wts = wts / wts.sum()
    for _ in range(220):
        if (occ >= 0).mean() >= density and len(blocks) >= min_blocks:
            break
        w, h = spans[int(rng.choice(len(spans), p=wts))]
        if w <= C and h <= R:
            place(min(w, C), min(h, R))
    return blocks, occ


def largest_free(grid: Grid, occ):
    """Largest all-free module rectangle (c0,r0,c1,r1) or None; ties prefer wide, top-left."""
    R, C = occ.shape
    best, ba = None, 0
    for r0 in range(R):
        for c0 in range(C):
            for r1 in range(r0 + 1, R + 1):
                for c1 in range(c0 + 1, C + 1):
                    if (occ[r0:r1, c0:c1] < 0).all():
                        a = (r1 - r0) * (c1 - c0) + 0.01 * (c1 - c0)
                        if a > ba:
                            best, ba = (c0, r0, c1, r1), a
    return best


# ---------------------------------------------------------------- forms (local masks, antialiased)

def _aa(d, w=1.0):
    return np.clip(0.5 - d / w, 0, 1).astype(np.float32)      # d = signed distance (neg inside)


def form_mask(kind: str, bw: int, bh: int, rng, gap: float = 1.0) -> np.ndarray:
    yy, xx = np.mgrid[0:bh, 0:bw].astype(np.float32) + 0.5
    m = min(bw, bh)
    if kind == "circle":
        d = m * 0.5 * rng.uniform(0.82, 1.0)
        cx = bw / 2 if bw < 1.4 * bh else rng.choice([bw * 0.5, bh * 0.5 + 0, bw - bh * 0.5])
        return _aa(np.hypot(xx - cx, yy - bh / 2) - d)
    if kind in ("quarter", "rings"):
        cx = rng.choice([0, bw]); cy = rng.choice([0, bh])
        r = np.hypot(xx - cx, yy - cy)
        if kind == "quarter":
            return _aa(r - m * rng.uniform(0.92, 1.0))
        n = int(rng.integers(3, 6))
        pitch = m / n
        ph = (r / pitch) % 1.0
        ring = (np.abs(ph - 0.5) < 0.25).astype(np.float32)
        return ring * (r < m * 1.0001)
    if kind == "half":
        side = rng.choice(["t", "b", "l", "r"])
        rad = min(bw, bh * 2) / 2 if side in "tb" else min(bw * 2, bh) / 2
        if side == "b":
            return _aa(np.hypot(xx - bw / 2, yy - bh) - min(bw / 2, bh))
        if side == "t":
            return _aa(np.hypot(xx - bw / 2, yy) - min(bw / 2, bh))
        if side == "l":
            return _aa(np.hypot(xx, yy - bh / 2) - min(bw, bh / 2))
        return _aa(np.hypot(xx - bw, yy - bh / 2) - min(bw, bh / 2))
    if kind == "bars":
        k = int(rng.integers(3, 8))
        vertical = bw >= bh if rng.random() < 0.7 else bw < bh
        t = (xx / bw if vertical else yy / bh) * k
        f = t % 1.0
        px = (bw if vertical else bh) / k
        w = 0.5 * rng.uniform(0.5, 0.8)
        return np.clip((w - np.abs(f - 0.5)) * px + 0.5, 0, 1).astype(np.float32)
    if kind == "diag":
        flip = rng.random() < 0.5
        t = (xx / bw) + ((yy / bh) if flip else (1 - yy / bh))
        return _aa((t - 1.0) * m * 0.7)
    if kind == "dots":
        n = int(rng.integers(3, 6))
        pitch = m / n
        gx, gy = (xx % pitch) - pitch / 2, (yy % pitch) - pitch / 2
        return _aa(np.hypot(gx, gy) - pitch * 0.36)
    if kind == "steps":
        n = int(rng.integers(3, 7))
        sx = np.floor(xx / bw * n)
        return (yy / bh > 1 - (sx + 1) / n).astype(np.float32)
    if kind == "cross":
        t = m * 0.22
        return np.maximum(_aa(np.maximum(np.abs(xx - bw / 2) - t, np.abs(yy - bh / 2) - m * 0.5)),
                          _aa(np.maximum(np.abs(xx - bw / 2) - m * 0.5, np.abs(yy - bh / 2) - t)))
    raise ValueError(kind)


FORMS = ["circle", "quarter", "half", "rings", "bars", "diag", "dots", "steps", "cross"]


# ---------------------------------------------------------------- photo crops

def pick_crop(img, energy_ii, aspect: float, rng, frac_range=(0.3, 0.7)):
    h, w = img.shape[:2]
    best, bs = None, -1
    for _ in range(8):
        f = rng.uniform(*frac_range)
        cw = int(min(w, h * aspect) * f) if w / h > aspect else int(w * f)
        cw = max(cw, 16)
        chh = max(int(cw / aspect), 16)
        if chh > h:
            chh = h
            cw = int(chh * aspect)
        x0 = int(rng.integers(0, max(w - cw, 1)))
        y0 = int(rng.integers(0, max(h - chh, 1)))
        s = (energy_ii[y0 + chh, x0 + cw] - energy_ii[y0, x0 + cw] - energy_ii[y0 + chh, x0] + energy_ii[y0, x0]) / (cw * chh)
        if s > bs:
            best, bs = (x0, y0, cw, chh), s
    return best


# ---------------------------------------------------------------- the composer

def scheme(ctx):
    """Swiss colours: paper, A (the block colour), B (small accent), tint (pale A), ink (type/rules)."""
    P = ctx.palette
    var = str(ctx.p("variant", "auto"))
    pol = P.polarity if var == "auto" else var
    h = pal.rgb_to_oklch(P.structure)[2]
    ah = pal.rgb_to_oklch(P.accent)[2]
    dark = pol == "dark"
    if dark:
        paper = K.color(ctx.p("paper"), P) if ctx.p("paper") else pal.oklch_to_rgb(0.16, 0.035, h)
        A, B = pal.oklch_to_rgb(0.56, 0.20, h), pal.oklch_to_rgb(0.80, 0.16, ah)
        tint, ink = pal.oklch_to_rgb(0.30, 0.08, h), pal.oklch_to_rgb(0.93, 0.025, h + 40)
    else:
        paper = K.color(ctx.p("paper"), P) if ctx.p("paper") else pal.oklch_to_rgb(0.955, 0.028, 88)
        A, B = pal.oklch_to_rgb(0.44, 0.20, h), pal.oklch_to_rgb(0.62, 0.20, ah)
        tint, ink = pal.oklch_to_rgb(0.86, 0.07, h), pal.oklch_to_rgb(0.23, 0.04, h)
    cols = {"A": A, "B": B, "tint": tint, "ink": ink}
    spec = ctx.p("inks", "")
    if spec:
        for k, c in zip(("A", "B", "tint", "ink"), K.colors(spec, P)):
            cols[k] = c
    return dark, paper, cols


def _fit_text(lines, rect, weight, override, max_px=None):
    """Largest font px so every line fits the rect width and the stack fits the height."""
    x0, y0, x1, y1 = rect
    bw, bh = x1 - x0, y1 - y0
    lo, hi = 6, int(max_px or bh)
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        f = load_font(mid, weight, override)
        wmax = max(f.getlength(t) for t in lines)
        hh = len(lines) * mid * 1.0
        if wmax <= bw and hh <= bh:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


def parse_text(data):
    if data is None:
        return None
    if isinstance(data, str):
        return {"title": data}
    if isinstance(data, (list, tuple)) and data and all(isinstance(x, str) for x in data):
        return {"title": data[0], "subtitle": data[1] if len(data) > 1 else "", "caption": " / ".join(data[2:])}
    if isinstance(data, dict) and any(k in data for k in ("title", "subtitle", "caption")):
        return {k: str(data.get(k, "")) for k in ("title", "subtitle", "caption")}
    return None


def draw_type(ink_mask, grid: Grid, occ, text: dict, size, override: str = "", scale: float = 1.0):
    """Title (bold, fitted to the largest free module rectangle) with optional subtitle and caption
    (regular, small, dropped when below 9 px). Returns the module rect used or None."""
    fr = largest_free(grid, occ)
    if fr is None or not text.get("title"):
        return None
    x0, y0, x1, y1 = grid.rect(*fr, bleed=False)
    pad = max(int(grid.g * 1.0), 2)
    rect = (x0 + pad, y0 + pad, x1 - pad, y1 - pad)
    W, H = size
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    title = text["title"].strip()
    words = title.split()
    # split a long title into up to 3 lines of balanced length
    lines = [title]
    if len(words) > 1:
        best = None
        for n in (1, 2, 3):
            if n > len(words):
                break
            k = math.ceil(len(words) / n)
            cand = [" ".join(words[i:i + k]) for i in range(0, len(words), k)]
            px = _fit_text(cand, rect, "bold", override)
            if best is None or px > best[0] * 1.08:
                best = (px, cand)
        lines = best[1]
    has_sub = bool(text.get("subtitle") or text.get("caption"))
    rh = rect[3] - rect[1]
    tr = (rect[0], rect[1], rect[2], rect[1] + int(rh * (0.68 if has_sub else 1.0)))
    px = int(_fit_text(lines, tr, "bold", override, max_px=(tr[3] - tr[1]) * 0.95) * min(scale, 1.0))
    f = load_font(px, "bold", override)
    y = rect[1]
    for t in lines:
        d.text((rect[0], y - int(px * 0.12)), t, font=f, fill=255)
        y += int(px * 0.98)
    y += max(int(px * 0.12), 2)
    for key, frac in (("subtitle", 0.30), ("caption", 0.22)):
        txt = text.get(key)
        if not txt:
            continue
        sp = max(int(px * frac), 9)
        ft = load_font(sp, "regular", override)
        while ft.getlength(txt) > rect[2] - rect[0] and sp > 9:
            sp -= 1
            ft = load_font(sp, "regular", override)
        if sp < 9 or y + sp * 1.25 > rect[3] or px < 20:
            continue
        d.text((rect[0], y), txt, font=ft, fill=255)
        y += int(sp * 1.45)
    m = np.asarray(img, np.float32) / 255
    ink_mask[:] = np.maximum(ink_mask, m)
    return fr


def compose(ctx, *, cols: int = 6, density: float = 0.7, margin: float = 0.0, gutter: float = 0.008, big: float = 0.3,
            kinds: dict | None = None, photo: float = 0.0, photo_style: str = "duo", type_on: bool = True,
            grid_lines: float = 0.6, forms: list[str] | None = None, accent: float = 0.15, full: bool = False):
    """-> dict(masks, scheme, grid, blocks, occ, text_rect). Masks: tint, A, B, ink (float 0..1)."""
    W, H = ctx.size
    rng = ctx.rng
    grid = Grid(ctx.size, cols, margin, gutter)
    if full:
        blocks, occ = pack(grid, rng, 1.0, big)
        # greedy fill of the remaining holes with 1x1 modules
        for r in range(grid.rows):
            for c in range(grid.cols):
                if occ[r, c] < 0:
                    occ[r, c] = len(blocks)
                    blocks.append((c, r, c + 1, r + 1))
    else:
        blocks, occ = pack(grid, rng, density, big)
    dark, paper, colr = scheme(ctx)
    masks = {k: np.zeros((H, W), np.float32) for k in ("tint", "A", "B", "ink", "type")}
    forms = forms or FORMS
    src = K.source_rgb(ctx, crop="center", zoom=0.0) if photo > 0 else None
    if src is not None:
        L = F.luminance(src)
        E = F.detail_energy(L, 12.0)
        ii = np.pad(np.cumsum(np.cumsum(E, 0), 1), ((1, 0), (1, 0)))
    nB = 0
    order = np.argsort([-(b[2] - b[0]) * (b[3] - b[1]) for b in blocks])
    kinds = kinds or {"solid": 0.22, "form": 0.40, "rules": 0.08, "outline": 0.08, "photo": photo * 0.9}
    ks = list(kinds)
    kp = np.array([kinds[k] for k in ks], float)
    kp /= kp.sum()
    used_photo = 0
    for rank, bi in enumerate(order):
        c0, r0, c1, r1 = blocks[bi]
        x0, y0, x1, y1 = grid.rect(c0, r0, c1, r1)
        bw, bh = x1 - x0, y1 - y0
        sl = (slice(y0, y1), slice(x0, x1))
        big_block = rank == 0
        kind = ks[int(rng.choice(len(ks), p=kp))]
        if big_block:
            kind = "photo" if (photo > 0 and rng.random() < 0.8) else "solid"
        if kind == "photo" and src is None:
            kind = "form"
        if kind == "photo" and used_photo >= max(2, len(blocks) // 2):
            kind = "solid"
        if kind == "solid":
            key = "A" if big_block else rng.choice(["A", "tint", "ink", "B"], p=[0.38, 0.40, 0.14, 0.08] if nB < 2 else [0.4, 0.45, 0.15, 0.0])
            if key == "B":
                nB += 1
            masks[str(key)][sl] = 1
        elif kind == "form":
            fk = forms[int(rng.integers(0, len(forms)))]
            fm = form_mask(fk, bw, bh, rng)
            bg = rng.random() < 0.45
            if bg:
                masks["tint"][sl] = 1
            key = rng.choice(["A", "ink", "B"], p=[0.62, 0.22, 0.16] if nB < 2 else [0.7, 0.3, 0.0])
            if key == "B":
                nB += 1
            masks[str(key)][sl] = np.maximum(masks[str(key)][sl], fm)
        elif kind == "rules":
            n = int(rng.integers(4, 9))
            vertical = rng.random() < 0.5
            t = ((np.mgrid[0:bh, 0:bw][1] if vertical else np.mgrid[0:bh, 0:bw][0]) / (bw if vertical else bh)) * n
            masks["A"][sl] = np.maximum(masks["A"][sl], (np.abs(t % 1 - 0.5) < 0.09).astype(np.float32))
        elif kind == "outline":
            t = max(int(0.004 * W), 1)
            o = np.zeros((bh, bw), np.float32)
            o[:t], o[-t:], o[:, :t], o[:, -t:] = 1, 1, 1, 1
            masks["A"][sl] = np.maximum(masks["A"][sl], o)
        elif kind == "photo":
            used_photo += 1
            x, y, cw, chh = pick_crop(src, ii, bw / bh, rng)
            crop = cv2.resize(src[y:y + chh, x:x + cw], (bw, bh), interpolation=cv2.INTER_AREA)
            Lc = F.stretch(F.local_equalize(F.luminance(crop), 2.0, tiles=3, mix=0.4), 3, 97)
            D = Lc if dark else 1 - Lc
            style = photo_style if photo_style != "mix" else str(rng.choice(["duo", "dot", "solid2"]))
            if style == "dot":
                masks["A"][sl] = np.maximum(masks["A"][sl], K.screen(D, (W, H), kind="dot", pitch=18, angle=45, gain=1.05))
            elif style == "solid2":
                masks["tint"][sl] = np.maximum(masks["tint"][sl], F.smoothstep(0.22, 0.28, D))
                masks["B"][sl] = np.maximum(masks["B"][sl], F.smoothstep(0.62, 0.68, D)) if nB < 2 else masks["B"][sl]
                nB += 1
            else:       # duotone: tint for mid tones, A for darks, ink for the deepest
                masks["tint"][sl] = np.maximum(masks["tint"][sl], F.smoothstep(0.18, 0.24, D))
                masks["A"][sl] = np.maximum(masks["A"][sl], F.smoothstep(0.50, 0.56, D))
                masks["ink"][sl] = np.maximum(masks["ink"][sl], F.smoothstep(0.80, 0.86, D))
    text_fr = None
    if type_on:
        text = parse_text(ctx.data)
        if text:
            text_fr = draw_type(masks["type"], grid, occ, text, ctx.size, str(ctx.p("font", "")), float(ctx.p("type_scale", 1.0)))
    # hairline module grid
    if grid_lines > 0:
        gl = np.zeros((H, W), np.float32)
        t = max(1, int(round(0.0009 * W)))
        for x in grid.xe[:-1]:
            if x < 2:
                continue
            cv2.line(gl, (int(round(x)), 0), (int(round(x)), H), 1.0, t)
        for xr in [grid.xe[c] + grid.mw for c in range(grid.cols) if grid.xe[c] + grid.mw < W - 2]:
            cv2.line(gl, (int(round(xr)), 0), (int(round(xr)), H), 1.0, t)
        for y in grid.ye[:-1]:
            if y < 2:
                continue
            cv2.line(gl, (0, int(round(y))), (W, int(round(y))), 1.0, t)
        for yr in [grid.ye[r] + grid.mh for r in range(grid.rows) if grid.ye[r] + grid.mh < H - 2]:
            cv2.line(gl, (0, int(round(yr))), (W, int(round(yr))), 1.0, t)
        masks["_grid"] = gl * grid_lines
    return dict(masks=masks, colors=colr, paper=paper, dark=dark, grid=grid, blocks=blocks, occ=occ, text_rect=text_fr)


def print_swiss(ctx, comp: dict) -> np.ndarray:
    """Press the composition: tint, A, B, ink (+ hairline grid in A) with ink colours compensated for paper."""
    m, c, paper = comp["masks"], comp["colors"], comp["paper"]
    mode = "over" if comp["dark"] else "multiply"

    def eff(x):
        return np.clip(np.asarray(x) / np.maximum(paper, 1e-3), 0, 1) if mode == "multiply" else np.asarray(x)

    covs, inks, ops = [m["tint"], m["A"], m["B"], m["ink"], m["type"]], [eff(c["tint"]), eff(c["A"]), eff(c["B"]), eff(c["ink"]), eff(c["ink"])], [1.0] * 5
    plain = [4]
    if "_grid" in m:
        covs.append(m["_grid"])
        inks.append(eff(c["A"] if not comp["dark"] else c["tint"] * 0 + K.lighten(c["A"], 0.35)))
        ops.append(0.7)
        plain.append(len(covs) - 1)
    return K.press(covs, inks, ctx.size, ctx.rng, paper_rgb=paper, screens=dict(kind="raw"), opacity=ops, mode=mode, plain=plain, **K.surface(ctx))
