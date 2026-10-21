"""Composed 9:16 pieces: stacked structure layers at different contrast, backgrounds, echo (misregistration), post filters."""
import sys
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import cv2, numpy as np
from PIL import Image, ImageFilter
from art4 import *

PAL = {p["name"]: p for p in PALETTES}

def waves(cv, rng, P, k=1.0, stepped=False):
    """Interference lines: parallel rows displaced by a few sines; stepped=True quantises them into a digital waveform."""
    W, H, u = cv.W, cv.H, cv.u
    n = 46; f = rng.uniform(.8, 1.4, 3); ph = rng.uniform(0, 6.28, 3)
    ex = float(rng.choice([.12, .88])) * W; cy = float(rng.choice([.16, .84])) * H  # the energy sits in a corner, never the middle
    best = None; rows = []
    for i in range(n):
        y0 = 3 * u + (H - 6 * u) * i / (n - 1); pts = []; peak = 0
        for x in np.linspace(0, W, 260 if not stepped else 90):
            env = math.exp(-((y0 - cy) / (H * .30)) ** 2 - ((x - ex) / (W * .75)) ** 2)
            v = sum(math.sin(2 * math.pi * f[j] * (x / W * 2.2 + y0 / H * .8 * (j + 1)) + ph[j]) for j in range(3)) / 3
            y = y0 - v * env * 11 * u
            if stepped: y = round(y / (1.6 * u)) * 1.6 * u
            pts.append((x, y)); peak = max(peak, abs(v) * env)
        rows.append((pts, peak))
    hot = int(np.argmax([r[1] for r in rows]))
    for i, (pts, pk) in enumerate(rows):
        t = i / (n - 1); col = P["accent"] if i == hot else mix(P["edge"], P["edge2"], t)
        if stepped:
            st = []
            for a, b in zip(pts, pts[1:]): st += [a, (b[0], a[1])]
            pts = st
        cv.curve(pts, col, .75 if i == hot else .26)

def pixel_waves(cv, rng, P, k=1.0, outline=False):
    """Adaptive squares: big where the wave field is quiet, quartered into small dense ones where it peaks (peak off-centre)."""
    W, H, u = cv.W, cv.H, cv.u
    f = rng.uniform(.7, 1.3, 3); ph = rng.uniform(0, 6.28, 3)
    ex, ey = float(rng.choice([.1, .9])) * W, float(rng.choice([.15, .85])) * H
    def field(x, y):
        v = sum(math.sin(2 * math.pi * f[j] * (x / W * 1.6 + y / H * (.7 + .5 * j)) + ph[j]) for j in range(3)) / 3 * .5 + .5
        return v * math.exp(-((x - ex) / (W * .85)) ** 2 - ((y - ey) / (H * .55)) ** 2)
    top = max(field(x, y) for x in np.linspace(0, W, 40) for y in np.linspace(0, H, 70)) + 1e-6
    cells = []
    def split(x, y, s, depth):
        fv = field(x + s / 2, y + s / 2) / top
        if s > 2.2 * u and fv > (.22, .38, .52, .66)[min(depth, 3)] * (.8 + .4 * rng.random()):
            h = s / 2
            for dx in (0, h):
                for dy in (0, h): split(x + dx, y + dy, h, depth + 1)
        else:
            cells.append((x, y, s, fv))
    s0 = 16 * u
    for gx in range(int(W / s0) + 1):
        for gy in range(int(H / s0) + 1): split(gx * s0, gy * s0, s0, 0)
    for (x, y, s, fv) in cells:
        g = s * .09; col = mix(P["structure"], P["structure2"], fv) if P["ncol"] == 3 else P["structure"]
        if outline or fv < .12 and rng.random() < .7:
            cv.box(x + g, y + g, x + s - g, y + s - g, line=mix(P["edge"], P["edge2"], fv), lw=.22, r=.2)
        elif rng.random() < .3 + fv * .7:
            hot = fv > .93
            cv.box(x + g, y + g, x + s - g, y + s - g, fill=P["accent"] if hot else col, r=.2, a=int(110 + 145 * fv))

def wheel_off(fx, fy, Rf, nr=3):
    def f(cv, rng, P, k=1.0):
        wheel(cv, rng, P, fx * cv.W, fy * cv.H, Rf * cv.W, nrings=nr, ns=(48, 36, 24)[:nr] if nr == 3 else (44, 32))
    return f

def spiral_off(fx, fy, Rf):
    def f(cv, rng, P, k=1.0):
        W, H, u = cv.W, cv.H, cv.u; R = Rf * W; N = 190; turns = 4.6; pts = []
        for i in range(N):
            t = i / (N - 1); r = .07 + .93 * t; a = 2 * math.pi * turns * t - math.pi / 2
            pts.append((fx * W + R * r * math.cos(a), fy * H + R * r * math.sin(a)))
        head = int(rng.integers(N // 2, N)); size = int(N * .5); cv.curve(pts, P["edge"], .22, a=200)
        for i, p in enumerate(pts):
            q0, q1 = pts[max(i - 1, 0)], pts[min(i + 1, N - 1)]
            rad = float(np.clip(math.hypot(q1[0] - q0[0], q1[1] - q0[1]) / 2 / u * .36, .55, 2.0)); st, t = run_state(i, head, size, N)
            fill, line, a = cell_style(P, st, t)
            cv.dot(p, rad * .8, P["canvas"], ring=P["edge"]) if fill is None else cv.dot(p, rad * (1.5 if st == "head" else 1), fill, a=a)
    return f

ALL = {"wheel-tr": wheel_off(.95, .2, .66), "wheel-bl": wheel_off(.05, .86, .7), "wheel-br": wheel_off(.92, .9, .55, 2), "wheel-tl": wheel_off(.02, .14, .6),
       "spiral-br": spiral_off(.9, .85, .7), "spiral-bl": spiral_off(.1, .8, .7), **DAGS, **RINGS, **COMBOS, "waves": waves, "pixel waves": pixel_waves, "pixel outline": lambda cv, rng, P, k=1.0: pixel_waves(cv, rng, P, k, True), "digital waves": lambda cv, rng, P, k=1.0: waves(cv, rng, P, k, True)}
tohex = lambda c: "#%02x%02x%02x" % tuple(c)

def dim(P, d):
    """Pull structure and accent toward the canvas by d (0 = as designed, 1 = invisible)."""
    Q = dict(P); c = P["canvas"]
    for k, f in (("structure", 1), ("structure2", 1), ("edge", 1), ("edge2", 1)):
        Q[k] = mix(P[k], c, d * f)
    # a ghost layer's accent takes the structure hue first, so it never greys out into a muddy blob
    Q["accent"] = mix(mix(P["accent"], P["structure"], min(1.0, d * 1.1)), c, d * .55)
    return Q

def layer(name, P, d, W, H, seed, scale=1.0, dx=0, dy=0, blur=0.0, alpha=1.0, flip=False):
    rng = np.random.default_rng(seed); cv = Cv3(W, H); Q = dim(P, d)
    if name == "matrix": matrix_art(cv, rng, {}, Q, .9)
    else: ALL[name](cv, rng, Q, 1.0)
    im = cv.flat()
    if flip: im = im.transpose(Image.FLIP_LEFT_RIGHT)
    if scale != 1.0:
        im = im.resize((round(W * scale), round(H * scale)), Image.LANCZOS)
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    out.alpha_composite(im, (round((W - im.width) / 2 + dx * W), round((H - im.height) / 2 + dy * H))) if True else None
    if blur: out = out.filter(ImageFilter.GaussianBlur(blur))
    if alpha < 1: out.putalpha(out.getchannel("A").point(lambda v: int(v * alpha)))
    return out

def echo(lay, col, dx, dy, a):
    m = lay.getchannel("A").point(lambda v: int(v * a)); t = Image.new("RGBA", lay.size, col + (0,)); t.putalpha(m)
    out = Image.new("RGBA", lay.size, (0, 0, 0, 0)); out.alpha_composite(t, (dx, dy)); return out

def bg(P, W, H, kind, seed):
    c = np.array(P["canvas"], np.float32); im = np.empty((H, W, 3), np.float32); im[:] = c
    s = np.array(mix(P["canvas"], P["structure"], .16), np.float32)
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    if kind == "glow":
        rng = np.random.default_rng(seed); cx, cy = float(rng.choice([0., 1.])) * W, float(rng.choice([0., 1.])) * H
        g = np.exp(-(((xx - cx) / (W * .7)) ** 2 + ((yy - cy) / (H * .45)) ** 2))[..., None]; im = im + (s - c) * g
    elif kind == "dots":
        step = max(10, W // 24); m = ((xx % step < 2) & (yy % step < 2))[..., None]; im = np.where(m, np.array(mix(P["canvas"], P["structure"], .3), np.float32), im)
    elif kind == "halftone":
        rng = np.random.default_rng(seed); f = cv2.resize(rng.random((9, 5)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        f = cv2.GaussianBlur(f, (0, 0), W / 12); f = (f - f.min()) / (np.ptp(f) + 1e-6); g = (f * 255).astype(np.uint8)
        ht = fx.halftone(np.dstack([g, g, g]), ink=tohex(mix(P["canvas"], P["structure"], .55)), paper=tohex(P["canvas"]), amount=.9, angle=30, cell=max(6, W // 60))[..., ::-1].astype(np.float32)
        im = im * .35 + ht * .65
    elif kind == "grad":
        t = (yy / H)[..., None]; im = im + (s - c) * t * .9
    return im.clip(0, 255).astype(np.uint8)

def tint_vignette(arr, P, amt):
    H, W = arr.shape[:2]; yy, xx = np.mgrid[:H, :W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / 1.41
    m = (np.clip((r - .4) / .6, 0, 1) ** 2 * amt)[..., None]
    dark = pal.lightness(tohex(P["canvas"])) < .5
    v = np.array(mix(P["canvas"], (0, 0, 0), .55) if dark else mix(P["canvas"], P["structure"], .22), np.float32)
    return (arr * (1 - m) + v * m).clip(0, 255).astype(np.uint8)

def piece(pal_name, layers, bgk="flat", bloom=0.0, grain=0.0, vig=0.0, seed=1, W=1080, H=1920):
    P = PAL[pal_name]; out = Image.fromarray(bg(P, W, H, bgk, seed)).convert("RGBA")
    for L in layers:
        L = dict(L); nm = L.pop("fn"); ec = L.pop("echo", None); d = L.pop("d", 0.0); sd = L.pop("seed", seed)
        lay = layer(nm, P, d, W, H, sd, **L)
        if ec:
            col = {"accent": P["accent"], "structure": P["structure"], "ink": P["ink"]}[ec[0]]
            out.alpha_composite(echo(lay, col, ec[1], ec[2], ec[3]))
        out.alpha_composite(lay)
    arr = np.array(out.convert("RGB"))
    if vig: arr = tint_vignette(arr, P, vig)
    if bloom: arr = fx.bloom(arr, bloom, .62)
    if grain: arr = fx.grain(arr, grain, seed)
    return arr

SPEC = {
 "sunset-waves": ("Sunset trio", [dict(fn="pixel waves", d=.05, seed=3), dict(fn="wheel-bl", d=.55, seed=4, blur=1.5), dict(fn="pills", seed=9, scale=.82, dx=.09)], "glow", .5, .035, 0),
 "neon-digital": ("Neon trio", [dict(fn="pixel waves", d=.0, seed=4), dict(fn="digital waves", d=.5, seed=4), dict(fn="lattice", d=.15, seed=8, scale=.86, dx=-.07, dy=.03)], "dots", .8, .03, 0),
 "ember-waves-spiral": ("Ember", [dict(fn="waves", d=.35, seed=6), dict(fn="spiral-br", d=.45, seed=5), dict(fn="flow", seed=7, scale=.82, dx=-.09)], "glow", .6, .04, 0),
 "ocean-pixels": ("Ocean trio", [dict(fn="pixel waves", d=.05, seed=2), dict(fn="bundle", d=.1, seed=7, scale=.88, dx=.06)], "grad", .45, .03, 0),
 "ink-digital": ("Ink on paper", [dict(fn="digital waves", d=.3, seed=4), dict(fn="wheel-tr", d=.5, seed=5), dict(fn="bundle", seed=5, scale=.88, dx=-.06)], "halftone", 0, .09, 0),
 "red-pixels": ("Red poster", [dict(fn="pixel waves", d=.2, seed=3), dict(fn="wheel-br", d=.5, seed=3), dict(fn="lattice", seed=8, scale=.86, dx=-.06, echo=("accent", 5, 4, .55))], "flat", 0, .06, 0),
 "plum-olive-digital": ("Plum olive", [dict(fn="digital waves", d=.4, seed=2), dict(fn="pills", seed=9, scale=.8, dx=-.1, echo=("structure", 3, 3, .25))], "grad", 0, .07, 0),
 "blue-violet-whisper": ("Blue violet", [dict(fn="waves", d=.55, seed=3), dict(fn="matrix", d=.8, scale=1.0, seed=3), dict(fn="bundle", d=.25, seed=6, scale=.86, dx=.07)], "glow", .9, .03, 0),
 "green-gold-wave-spiral": ("Green gold", [dict(fn="waves", d=.45, seed=3), dict(fn="spiral-bl", d=.5, seed=3), dict(fn="lattice", d=.1, seed=8, scale=.86, dx=.07)], "glow", .55, .04, 0),
 "rose-gold-waves": ("Rose gold", [dict(fn="waves", d=.2, seed=2), dict(fn="swimlane", d=.1, seed=5, scale=.84, dx=.08)], "glow", .5, .04, 0),
 "white-dark-digital": ("White dark", [dict(fn="digital waves", d=.45, seed=2), dict(fn="lattice", seed=7, scale=.84, dx=-.08, echo=("accent", 4, 0, .35))], "flat", .4, .05, 0),
 "cobalt-waves-wheel": ("Cobalt poster", [dict(fn="waves", d=.3, seed=3), dict(fn="wheel-tr", d=.5, seed=3), dict(fn="flow", seed=4, scale=.84, dx=-.08, echo=("accent", 4, 4, .5))], "flat", 0, .05, 0),
 "forest-pixels": ("Forest poster", [dict(fn="pixel waves", d=.1, seed=2), dict(fn="bundle", seed=6, scale=.88, dx=.05)], "grad", .2, .05, 0),
 "cream-red-pixels": ("Red on cream", [dict(fn="pixel waves", d=.15, seed=2), dict(fn="pills", seed=9, scale=.76, dx=-.12)], "halftone", 0, .07, 0),
 "mint-pink-soft": ("Mint pink", [dict(fn="waves", d=.5, seed=3), dict(fn="matrix", d=.62, seed=3), dict(fn="lattice", d=.3, seed=4, scale=.85, dx=.06)], "grad", 0, .05, 0),
 "blue-violet-light-pixels": ("Blue violet light", [dict(fn="pixel waves", d=.3, seed=3), dict(fn="pills", d=.2, seed=9, scale=.8, dx=.1)], "glow", 0, .05, 0),
 "yellow-poster-bold": ("Yellow poster", [dict(fn="pixel waves", d=.1, seed=2), dict(fn="lattice", seed=8, scale=.82, dx=-.08, echo=("accent", 4, 4, .45))], "flat", 0, .07, 0),
 "black-cherry-bold": ("Black cherry", [dict(fn="waves", d=.45, seed=3), dict(fn="wheel-tl", d=.5, seed=3), dict(fn="flow", seed=6, scale=.84, dx=.08, echo=("structure", 5, 0, .5))], "glow", .45, .04, 0),
 "violet-poster-trio": ("Violet poster", [dict(fn="waves", d=.4, seed=2), dict(fn="wheel-br", d=.45, seed=3), dict(fn="swimlane", seed=7, scale=.84, dx=-.07, echo=("accent", 4, 4, .4))], "flat", 0, .05, 0),
 "teal-coral-soft": ("Teal coral light", [dict(fn="digital waves", d=.4, seed=4), dict(fn="snake", d=.6, seed=5), dict(fn="pills", seed=9, scale=.8, dx=.1)], "halftone", 0, .06, 0),

 "violet-pixels-only": ("Violet dusk", [dict(fn="pixel waves", d=0, seed=6), dict(fn="waves", d=.5, seed=2)], "glow", .5, .04, 0),
 "mint-pixel-outline": ("Mint pink", [dict(fn="pixel outline", d=.1, seed=7), dict(fn="pixel waves", d=.3, seed=8), dict(fn="pills", d=.3, seed=9, scale=.7, dx=.1)], "grad", 0, .05, 0),
}
if __name__ == "__main__":
    import glob
    names = sys.argv[1:] or list(SPEC)
    for n in names:
        p, ls, b, bl, gr, vg = SPEC[n]; Image.fromarray(piece(p, ls, b, bl, gr, vg)).save(f"c6_{n}.png")
    fs = [f"c6_{n}.png" for n in names]; ims = [Image.open(f).resize((300, 533)) for f in fs]
    cols = 5; rows = (len(ims) + cols - 1) // cols; s = Image.new("RGB", (300 * cols, 533 * rows))
    for i, im in enumerate(ims): s.paste(im, ((i % cols) * 300, (i // cols) * 533))
    s.save("sheet6.png")
