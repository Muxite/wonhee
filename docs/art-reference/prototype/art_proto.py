"""Phase 0 prototype of wonhee's `art` engine: seeded abstract art that is laid out for the box it is asked to fill.

Throwaway. If a style is picked, it graduates into wonhee/edit/viz/art.py with tests.
"""
from __future__ import annotations

import colorsys
import math
import subprocess
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from wonhee.edit import fx
from wonhee.edit.compose import PatternSpec, pattern_image
from wonhee.edit.filters import PACK, apply as filter_apply
from wonhee.edit.viz.colors import hash_hue

LANG = {".py": "Python", ".ts": "TypeScript", ".tsx": "TypeScript", ".js": "JavaScript", ".cpp": "C++", ".cu": "CUDA",
        ".c": "C", ".h": "C", ".rs": "Rust", ".go": "Go", ".java": "Java", ".yaml": "YAML", ".sql": "SQL"}


# --- project fingerprint: what makes one repo's art differ from another's ---------------------------------------
def fingerprint(repo: Path | None, slug: str) -> dict:
    fp = {"slug": slug, "weeks": None, "langs": []}
    if not repo or not (repo / ".git").exists():
        return fp
    r = subprocess.run(["git", "-C", str(repo), "log", "--format=%ct"], capture_output=True, text=True)
    ts = sorted(int(t) for t in r.stdout.split())
    if len(ts) > 3:
        wk = Counter((t - ts[0]) // (7 * 86400) for t in ts)
        n = max(wk) + 1
        s = np.array([wk.get(i, 0) for i in range(n)], np.float32)
        fp["weeks"] = s / max(s.max(), 1)
    r = subprocess.run(["git", "-C", str(repo), "ls-files"], capture_output=True, text=True)
    c = Counter(LANG[Path(f).suffix] for f in r.stdout.splitlines() if Path(f).suffix in LANG)
    tot = sum(c.values()) or 1
    fp["langs"] = [(k, v / tot) for k, v in c.most_common(3)]
    return fp


def hsl(h, s, l):
    r, g, b = colorsys.hls_to_rgb((h % 360) / 360, l, s)
    return np.array([b, g, r], np.float32) * 255  # BGR


def hexs(c) -> str:
    b, g, r = [int(round(x)) for x in c]
    return f"#{r:02x}{g:02x}{b:02x}"


def hues(fp):
    h0 = hash_hue(fp["slug"])
    h1 = hash_hue(fp["langs"][0][0]) if fp["langs"] else h0 + 150
    h2 = hash_hue(fp["langs"][1][0]) if len(fp["langs"]) > 1 else h0 + 60
    return h0, h1, h2


# --- the field: smooth scalar art, laid out in short-side units so it never stretches with the box ---------------
def field(fp, W, H, seed, calm=1.0):
    S = min(W, H)
    scale = min(1.0, 720 / max(W, H))
    w, h = max(8, round(W * scale)), max(8, round(H * scale))
    rng = np.random.default_rng(seed)
    x = (np.linspace(0, W / S, w, dtype=np.float32))[None, :].repeat(h, 0)
    y = (np.linspace(0, H / S, h, dtype=np.float32))[:, None].repeat(w, 1)
    long_x = W >= H
    # domain warp, stronger along the short axis so wide boxes get long sweeping bands
    for _ in range(2):
        a, b = rng.uniform(0.6, 1.6) * calm, rng.uniform(0, 6.28, 2)
        x, y = x + 0.18 * np.sin(2 * math.pi * a * y + b[0]), y + 0.18 * np.sin(2 * math.pi * a * x + b[1])
    f = np.zeros((h, w), np.float32)
    for _ in range(4):
        ang = rng.normal(0 if long_x else math.pi / 2, 0.7)
        k = rng.uniform(0.7, 2.4) * calm
        f += np.sin(2 * math.pi * k * (x * math.cos(ang) + y * math.sin(ang)) + rng.uniform(0, 6.28)) * rng.uniform(.5, 1)
    cx, cy = rng.uniform(.25, .75) * W / S, rng.uniform(.3, .7) * H / S
    f += 1.6 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (0.35 + 0.25 * rng.random()))
    wk = fp["weeks"]
    if wk is not None and len(wk) > 2:  # commit rhythm bends the field along the long axis
        t = (x if long_x else y) / (W / S if long_x else H / S)
        series = cv2.GaussianBlur(wk[None, :], (0, 0), max(1, len(wk) / 60))[0]
        f += 1.1 * np.interp(t, np.linspace(0, 1, len(series)), series) * (1 - 0.5 * np.cos(4 * math.pi * y))
    f = (f - f.min()) / max(float(f.max() - f.min()), 1e-6)
    return cv2.resize(f, (W, H), interpolation=cv2.INTER_CUBIC).clip(0, 1)


def grey3(f):
    g = (f * 255).astype(np.uint8)
    return np.dstack([g, g, g])


def stops(f, cols):
    """Gradient map: f in 0..1 -> blend through the colour stops."""
    cols = np.array(cols, np.float32)
    pos = np.linspace(0, 1, len(cols))
    return np.dstack([np.interp(f, pos, cols[:, c]) for c in range(3)]).clip(0, 255).astype(np.uint8)


# --- styles ------------------------------------------------------------------------------------------------------
def riso(fp, W, H, seed, dark=False):
    h0, h1, h2 = hues(fp)
    f = field(fp, W, H, seed)
    ink, ink2 = hsl(h0, .85, .34), hsl(h0 + 160, .9, .62)
    paper = "#14161d" if dark else "#f2ede2"
    base = fx.duotone(grey3(f), ink=hexs(ink) if not dark else hexs(hsl(h0, .75, .62)), paper=paper, texture=.6, seed=seed)
    shifted = np.roll(f, (int(H * .012), int(W * .01)), (0, 1))
    dots = fx.halftone(grey3(1 - shifted), ink=hexs(ink2), paper="#ffffff", angle=15, cell=max(5, round(max(W, H) / 120)))
    mult = base.astype(np.float32) * (0.35 + 0.65 * dots.astype(np.float32) / 255)  # second ink only tints, so the inks stay clean
    out = mult if not dark else cv2.addWeighted(base, .75, dots, .25, 0)
    return fx.grain(out.clip(0, 255).astype(np.uint8), .28, seed)


def dusk(fp, W, H, seed, dark=False):
    h0, h1, h2 = hues(fp)
    f = field(fp, W, H, seed, calm=.8)
    cols = [hsl(h0 + 200, .55, .12 if dark else .22), hsl(h0 + 330, .7, .38), hsl(h0 + 20, .85, .58), hsl(h0 + 45, .8, .86)]
    im = stops(f, cols)
    im = fx.halftone(im, ink=hexs(cols[0]), paper=hexs(cols[3]), angle=45, amount=.5, cell=max(6, round(max(W, H) / 90)))
    return fx.grain(fx.vignette(im, .4), .3, seed)


def swiss(fp, W, H, seed, dark=False):
    h0, h1, h2 = hues(fp)
    f = field(fp, W, H, seed, calm=1.2)
    S = min(W, H)
    cell = S / 5
    cols_n, rows_n = max(3, round(W / cell)), max(3, round(H / cell))  # a grid that tiles the box exactly
    cw, ch = W / cols_n, H / rows_n
    pal = [hsl(h0, .8, .45), hsl(h2 + 30, .85, .6), hsl(h1 + 180, .6, .85), hsl(h0 + 180, .7, .25)]
    paper = np.array([29, 22, 20], np.float32) if dark else np.array([226, 235, 242], np.float32)
    SS = 2
    im = Image.new("RGB", (W * SS, H * SS), tuple(int(v) for v in paper[::-1]))
    d = ImageDraw.Draw(im)
    rng = np.random.default_rng(seed)
    for j in range(rows_n):
        for i in range(cols_n):
            v = float(f[min(H - 1, int((j + .5) * ch)), min(W - 1, int((i + .5) * cw))])
            x0, y0, x1, y1 = i * cw * SS, j * ch * SS, (i + 1) * cw * SS, (j + 1) * ch * SS
            c = tuple(int(q) for q in pal[int(v * 3.99)][::-1])
            kind = int(rng.integers(0, 3))
            r = min(x1 - x0, y1 - y0)
            if kind == 0:
                d.pieslice([x0, y0, x0 + 2 * (x1 - x0), y0 + 2 * (y1 - y0)], 180, 270, fill=c)
            elif kind == 1:
                m = (0.25 + 0.7 * v) * r / 2
                d.ellipse([(x0 + x1) / 2 - m, (y0 + y1) / 2 - m, (x0 + x1) / 2 + m, (y0 + y1) / 2 + m], fill=c)
            else:
                d.rectangle([x0, y0 + (y1 - y0) * (1 - v), x1, y1], fill=c)
    out = np.array(im.resize((W, H), Image.LANCZOS))[..., ::-1]
    return fx.grain(out, .12, seed)


def neon(fp, W, H, seed, dark=True):
    h0, h1, h2 = hues(fp)
    f = field(fp, W, H, seed, calm=.7)
    cols = [hsl(h0 + 230, .6, .06), hsl(h0 + 250, .8, .25), hsl(h0 + 300, .9, .5), hsl(h0 + 330, .9, .72), hsl(h0 + 40, .6, .95)]
    im = stops(f ** 1.4, cols)
    im = fx.bloom(im, .6, .6)
    return filter_apply(im, PACK["night-neon"][1], .8, seed)


def pop(fp, W, H, seed, dark=False):
    h0, h1, h2 = hues(fp)
    pal = [hexs(hsl(h0, .85, .6)), hexs(hsl(h0 + 180, .8, .45)), "#1a1a1a", "#fffdf5"]
    rng = np.random.default_rng(seed)
    kind = ["stripes", "halftone", "stars"][seed % 3]
    bg = pattern_image(PatternSpec(kind=kind, colours=["pal:0", "pal:3"], density=.55, angle=-20 + seed % 40), (W, H), pal, rng)
    f = field(fp, W, H, seed, calm=1.0)
    S = min(W, H)
    SS = 2
    big = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(big)
    for n in range(9):
        cx, cy = rng.uniform(0, 1) * W * SS, rng.uniform(0, 1) * H * SS
        r = S * SS * rng.uniform(.07, .26) * (0.5 + float(f[min(H - 1, int(cy / SS)), min(W - 1, int(cx / SS))]))
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=pal[n % 3 if n % 3 != 2 else 1], outline=pal[2], width=int(S * SS * .008))
    bg.alpha_composite(big.resize((W, H), Image.LANCZOS))
    out = np.array(bg.convert("RGB"))[..., ::-1]
    return fx.grain(out, .1, seed)


STYLES = {"riso": riso, "dusk": dusk, "swiss": swiss, "neon": neon, "pop": pop}


def coverage(im: np.ndarray, tiles=8) -> float:
    """Share of tiles with visible structure: a render with a dead flat margin fails."""
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = g.shape
    ok = 0
    for j in range(tiles):
        for i in range(tiles):
            t = g[j * H // tiles:(j + 1) * H // tiles, i * W // tiles:(i + 1) * W // tiles]
            ok += t.std() > 3.0
    return ok / tiles ** 2
