"""Optical and captured-screen effects: the picture as seen through a lens, a display or a
low-light camera. Spatial parts that the colour-only Y2K / VHS / CRT grades lack.

Every defect has its own amount (0 = off) and all are subtle by default. All noise and
textures are functions of (pixel coordinate, seed), so frames of a video with one seed do
not crawl. Sizes are px at a 1500 px master and scale with the frame; the screen lattice
fades out below ~2 px pitch so 288x192 never shows fake sub-pixels. The picture always fills
the frame (perspective and distortion are cropped in, no black borders).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .. import atmoskit as K
from .. import features as F
from ..registry import effect

COMMON = {"crop": ("center", "center | feature, crop of a source with another aspect")}


def _src(ctx):
    W, H = ctx.size
    if ctx.source is None:
        from .atmos_common import synth_night
        return synth_night((W, H), ctx.seed)
    return F.fit_cover(np.asarray(ctx.source, np.float32)[..., :3], (W, H), ctx.p("crop", "center"))


def _bloom(x, amount, thr=0.7, radius=14.0, tint=(1.0, 0.96, 0.92)):
    if amount <= 0:
        return x
    s = K.scale_of(x.shape)
    xl = K.lin(x)
    L = K.luma(x)
    bright = xl * K.smoothstep(thr - 0.12, thr + 0.12, L)[..., None]
    g = 0.5 * K.blur_big(bright, radius * s) + 0.5 * K.blur_big(bright, radius * 3.5 * s)
    return K.srgb(np.clip(1 - (1 - xl) * np.exp(-g * amount * 2.2 * np.asarray(tint, np.float32)), 0, 1))


def _vignette(x, amount, power=2.0):
    if amount <= 0:
        return x
    H, W = x.shape[:2]
    xx, yy = K.coords(x.shape)
    r = np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2)) / 1.414
    v = 1 - amount * np.clip(r, 0, 1) ** power
    return K.srgb(K.lin(x) * v[..., None])


def _chan_shift(x, px_x, px_y):
    """Channel displacement: red and blue move opposite ways (sub-pixel via affine)."""
    if abs(px_x) + abs(px_y) < 0.05:
        return x
    H, W = x.shape[:2]
    out = x.copy()
    for c, sgn in ((0, 1), (2, -1)):
        M = np.float32([[1, 0, sgn * px_x], [0, 1, sgn * px_y]])
        out[..., c] = cv2.warpAffine(np.ascontiguousarray(x[..., c]), M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out


def _grain(x, amount, seed):
    if amount <= 0:
        return x
    s = K.scale_of(x.shape)
    n = K.hash_noise(x.shape, seed + 3)
    n = K.blur(n, 0.45 + 0.4 * s)
    n = n / (n.std() + 1e-6)
    L = K.luma(x)
    return x + (n * amount * 0.035 * (0.4 + 0.6 * np.sqrt(np.clip(4 * L * (1 - L), 0, 1))) * min(1.0, 0.4 + s))[..., None]


@effect("optics-screen-photo", family="optics", tags=["source", "photo", "screen", "captured", "mediated", "nostalgic", "candid"],
        params={**COMMON,
                "tilt": (0.08, "0..0.2 off-axis perspective (screen corners pulled in, still filling the frame)"),
                "tilt_angle": (35.0, "degrees: which way the perspective leans"),
                "barrel": (0.09, "0..0.3 curved glass (radial distortion, cropped to fill)"),
                "softness": (0.5, "0..1 focus softness: sharp middle, soft corners"),
                "pitch": (3.4, "pixel pitch of the display in px at 1500 wide (lattice fades below ~2 px, so tiny frames show none)"),
                "grid": (0.7, "0..1 visible pixel grid / gaps between display pixels"),
                "subpixel": (0.35, "0..1 RGB sub-pixel stripes (only when the pitch is wide enough)"),
                "moire": (0.4, "0..1 slow beat between screen and sensor lattice (low frequency, survives small sizes)"),
                "bands": (0.25, "0..1 broad refresh / exposure bands (static, seed decides their place)"),
                "displace": (1.0, "px channel displacement (red and blue move apart) at 1500 wide"),
                "bloom": (0.35, "0..1 highlight bloom"),
                "glare": (0.25, "0..1 soft diagonal reflection on the glass"),
                "vignette": (0.3, "0..1 corner falloff"),
                "grain": (0.3, "0..1 sensor grain")})
def screen_photo(ctx):
    """A photo of a screen showing the picture: off-axis perspective, curved glass, soft
    corners, the display's pixel lattice and RGB stripes (when resolvable), a slow moire beat,
    exposure bands, a faint glass reflection, red/blue displacement, bloom, vignette and grain.
    The picture still fills the frame. Best on screenshots, UI, photos you want 'seen through
    a device'; mood: mediated, nostalgic. Fails at tilt > 0.12 on faces (leaning), and lattice
    terms are intentionally invisible below ~2 px pitch (small cards keep moire, bands, bloom)."""
    x = _src(ctx)
    H, W = x.shape[:2]
    s = K.scale_of(x.shape)
    g = lambda k: float(ctx.p(k))
    t = g("tilt")
    if t > 0:
        a = math.radians(g("tilt_angle"))
        ca, sa = math.cos(a), math.sin(a)
        # corner offsets (fraction of frame, inward): opposite corners pulled differently
        base = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], np.float32)
        lean = (base[:, 0] * ca + base[:, 1] * sa)
        off = np.zeros((4, 2), np.float32)
        off[:, 0] = np.clip(t * (0.4 + 0.6 * np.clip(lean, 0, 1)), 0, None) * 0.5
        off[:, 1] = np.clip(t * (0.4 + 0.6 * np.clip(-lean, 0, 1)), 0, None) * 0.5
        x = K.perspective_fill(x, off)
    if g("barrel") != 0 or g("displace") > 0:
        x = K.radial_warp(x, g("barrel"), 0.0, fill=True, ca=g("displace") * 0.6 * s * 2)
    soft = K.focus_field(x.shape, "edges", g("softness"))
    if g("softness") > 0:
        x = K.variable_blur(x, soft, 3.2)
    p = g("pitch") * s
    if g("grid") > 0 or g("subpixel") > 0:
        x = K.srgb(K.screen_grid(K.lin(x), p, 0.3 * g("grid"), g("subpixel")))
    xx, yy = K.coords(x.shape)
    if g("moire") > 0:
        ang = math.radians(7.0 + (ctx.seed % 5))
        per = 70.0 * max(s, 0.4)
        kx, ky = math.cos(ang) / per, math.sin(ang) / per
        m = 0.5 + 0.5 * np.cos(2 * math.pi * (xx * kx + yy * ky) + 0.5 * np.sin(2 * math.pi * yy / (H * 1.7)))
        m2 = 0.5 + 0.5 * np.cos(2 * math.pi * (xx * ky * 1.3 - yy * kx * 1.3) + 1.0)
        mo = (0.65 * m + 0.35 * m2)
        x = K.srgb(K.lin(x) * (1 - 0.14 * g("moire") + 0.28 * g("moire") * (mo - 0.5) * 2 * 0.5 + 0.0)[..., None])
    if g("bands") > 0:
        ph = (ctx.seed * 0.37) % 1.0
        b = 0.5 + 0.5 * np.sin(2 * math.pi * (yy / (H * 0.42) + ph) + 0.4 * np.sin(2 * math.pi * xx / (W * 1.3)))
        x = K.srgb(K.lin(x) * (1 - 0.18 * g("bands") * (1 - b) ** 1.5)[..., None])
    x = _chan_shift(x, g("displace") * s * 0.8, 0.0)
    x = _bloom(x, g("bloom"))
    if g("glare") > 0:
        d = (xx / W * 0.8 + yy / H * 0.6)
        sheen = np.exp(-((d - 0.42) / 0.16) ** 2) * 0.65 + np.exp(-((d - 0.84) / 0.07) ** 2) * 0.25
        x = x + (sheen * g("glare") * 0.12)[..., None] * (1 - 0.7 * x)
    x = _vignette(x, g("vignette"))
    x = _grain(x, g("grain"), ctx.seed)
    return np.clip(x, 0.004, 0.996).astype(np.float32)


@effect("optics-lens", family="optics", tags=["source", "photo", "lens", "mediated", "nostalgic", "candid", "intimate"],
        params={**COMMON,
                "distortion": (0.08, "-0.3..0.3 radial distortion k1 (+ barrel, - pincushion), cropped to fill"),
                "fringe": (1.6, "px lateral chromatic aberration at the corners (1500 wide)"),
                "focus": ("edges", "edges | plane | none: where the lens is soft"),
                "softness": (0.5, "0..1 amount of defocus outside the sharp zone"),
                "plane_angle": (0.0, "degrees, tilt of the sharp plane when focus=plane"),
                "plane_pos": (0.55, "centre of the sharp band 0..1 (focus=plane)"),
                "bloom": (0.3, "0..1 highlight bloom (veiling glare)"),
                "bloom_radius": (16.0, "bloom radius px at 1500 wide"),
                "vignette": (0.35, "0..1 corner falloff in linear light"),
                "grain": (0.2, "0..1 fine grain")})
def lens(ctx):
    """An ordinary imperfect lens: radial distortion (cropped, never a black corner), lateral
    colour fringing growing toward the corners, a sharp zone with soft edges or a tilted sharp
    plane, veiling bloom and vignetting. Every defect independent and subtle. Best on any photo,
    people included (distortion is gentle); fails with distortion > 0.2 on faces near the frame
    edge. Small cards: fringe and softness keep working, grain is attenuated."""
    x = _src(ctx)
    s = K.scale_of(x.shape)
    g = lambda k: float(ctx.p(k))
    x = K.radial_warp(x, g("distortion"), 0.0, fill=True, ca=g("fringe") * s * 1.0)
    f = str(ctx.p("focus"))
    if f != "none" and g("softness") > 0:
        fld = K.focus_field(x.shape, f, g("softness"), g("plane_angle"), g("plane_pos"), 0.25)
        x = K.variable_blur(x, fld, 4.5)
    x = _bloom(x, g("bloom"), 0.7, g("bloom_radius"))
    x = _vignette(x, g("vignette"), 2.4)
    x = _grain(x, g("grain"), ctx.seed)
    return np.clip(x, 0.004, 0.996).astype(np.float32)


@effect("optics-flash", family="optics", tags=["source", "photo", "flash", "candid", "intimate", "nocturnal", "nostalgic"],
        params={**COMMON,
                "falloff": (0.85, "0..1 direct-flash inverse-square falloff: near bright, far drops to ambient"),
                "centre_y": (0.55, "0..1 flash axis row (subject height)"),
                "reach": (0.45, "size of the lit zone as a fraction of the frame"),
                "ambient": (0.35, "0..1 how much of the original ambient exposure survives far away"),
                "depth": ("subject", "subject (lit zone around the flash axis) | gradient (bottom near, top far)"),
                "specular": (0.35, "0..1 hot highlights and shine on lit surfaces"),
                "cool": (0.25, "0..1 flash is cooler than the ambient (shadows go warm/dark)"),
                "skin": (0.9, "0..1 protect skin from the colour shift"),
                "noise": (0.3, "0..1 colour noise in the unlit shadows"),
                "grain": (0.25, "0..1 fine grain")})
def flash(ctx):
    """Direct on-camera flash: exposure follows 1/d^2 from a depth proxy, so the subject zone is
    punchy and flat, and the background falls toward the dark ambient exposure; hot specular
    shine; the flash is slightly cooler than ambient light (skin protected); noisy shadows. The
    fall-off is applied in linear light and is a smooth function of position, so it is stable
    across video frames. Best on people and close interiors; fails on landscapes (no near
    subject) and on photos that already have strong flash fall-off."""
    x = _src(ctx)
    H, W = x.shape[:2]
    s = K.scale_of(x.shape)
    g = lambda k: float(ctx.p(k))
    xx, yy = K.coords(x.shape)
    nx, ny = (xx - W / 2) / (W / 2), (yy - H * g("centre_y")) / (H / 2)
    if str(ctx.p("depth")) == "gradient":
        d = 0.5 + 0.8 * (1 - yy / H) + 0.1 * np.abs(nx)
        inv = 1.0 / (d ** 2 + 0.2)
        inv = inv / inv.max()
    else:
        r2 = (nx * 1.0) ** 2 + (ny * 1.25) ** 2
        inv = 1.0 / (1 + r2 / max(0.02, g("reach")) ** 2 * 1.4)
    amb = g("ambient")
    gain = (1 - g("falloff")) + g("falloff") * (amb * 0.55 + (1 - amb * 0.55) * (inv / inv.max()) * 1.5)
    xl = K.lin(x)
    lit = xl * gain[..., None]
    if g("cool") > 0:
        sk = np.clip(K.blur(K.skin_mask(x), 3.0) * 1.6, 0, 1) * g("skin")
        cool = np.array([0.94, 1.0, 1.06], np.float32)
        far = (1 - np.clip(inv / inv.max(), 0, 1))
        wmix = (g("cool") * 0.5 * (1 - sk) * (0.3 + 0.7 * far))[..., None]
        lit = lit * (1 + wmix * (np.array([1.05, 1.0, 0.93], np.float32) - 1))
    out = K.srgb(np.clip(lit, 0, 1.2))
    if g("specular") > 0:
        L = K.luma(out)
        hot = K.smoothstep(0.72, 0.98, L) * np.clip(inv / inv.max(), 0, 1)
        out = out + (K.blur(hot, 2.0 * s + 0.5) * g("specular") * 0.25)[..., None]
        out = _bloom(np.clip(out, 0, 1), g("specular") * 0.6, 0.82, 10.0)
    if g("noise") > 0:
        n = np.stack([K.blur(K.hash_noise(x.shape, ctx.seed + 20 + c), 1.2 * s + 0.6) for c in range(3)], -1)
        n = n / (n.std() + 1e-6)
        dark = (1 - K.smoothstep(0.1, 0.5, K.luma(out)))[..., None]
        out = out + n * g("noise") * 0.03 * dark * min(1.0, 0.3 + s)
    out = _grain(out, g("grain"), ctx.seed)
    return np.clip(out, 0.004, 0.996).astype(np.float32)


@effect("optics-lowlight-cam", family="optics", tags=["source", "photo", "lowlight", "noise", "nocturnal", "candid", "mediated", "nostalgic"],
        params={**COMMON,
                "iso": (0.6, "0..1 sensor noise (signal-dependent shot noise in linear light + chroma blotches)"),
                "denoise": (0.4, "0..1 waxy noise reduction (smooths flat areas, keeps edges): the phone look"),
                "push": (0.35, "0..1 shadow push (lifted shadows, the look of an over-brightened night shot)"),
                "smear": (0.3, "0..1 highlight smear: vertical charge bleed and a ghost below bright lights"),
                "clip": (0.3, "0..1 highlight clipping with a hard shoulder"),
                "sharpen": (0.3, "0..1 over-sharpened halos"),
                "vignette": (0.2, "0..1 corner falloff")})
def lowlight(ctx):
    """A small sensor in the dark: shadows pushed, shot noise that scales with signal in linear
    light (static pattern from coordinates, so it does not crawl), chroma blotches, waxy
    edge-preserving denoise, hard highlight clip, vertical charge smear under bright lights and
    crunchy sharpening. Best on night street, interiors, candid portraits (skin keeps a natural
    tone because only luminance is pushed); fails on bright daylight (nothing to push). At 288
    px noise is attenuated so the tile is not static."""
    x = _src(ctx)
    s = K.scale_of(x.shape)
    g = lambda k: float(ctx.p(k))
    xl = K.lin(x)
    if g("push") > 0:
        L = K.luma(x)
        k = 1 + g("push") * 2.2 * (1 - K.smoothstep(0.0, 0.5, L))
        xl = xl * k[..., None]
    out = K.srgb(np.clip(xl, 0, 1.5))
    if g("iso") > 0:
        amp = min(1.0, 0.25 + s * 1.6)
        L = K.luma(np.clip(out, 0, 1))
        sig = (0.35 + 0.65 * np.sqrt(L)) * 0.045 * g("iso") * amp        # shot noise grows with signal
        n = np.stack([K.hash_noise(x.shape, ctx.seed + 50 + c) for c in range(3)], -1)
        nl = K.blur(K.hash_noise(x.shape, ctx.seed + 60), 0.8 * s + 0.3)[..., None]
        nl = nl / (nl.std() + 1e-6)
        blot = np.stack([K.value_noise(x.shape, 14 * max(s, 0.3), ctx.seed + 70 + c, 2) - 0.5 for c in range(3)], -1)
        out = out + (0.35 * n + 0.6 * nl) * sig[..., None] + blot * g("iso") * 0.07 * amp * (0.3 + 0.7 * np.sqrt(L))[..., None]
    if g("denoise") > 0:
        u8 = (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)
        d = cv2.bilateralFilter(u8, 0, 22 * g("denoise") + 6, 3.0 * s + 1.5 + 2 * g("denoise"))
        d = cv2.bilateralFilter(d, 0, 18 * g("denoise") + 5, 2.5 * s + 1.5)
        out = out * (1 - g("denoise")) + d.astype(np.float32) / 255 * g("denoise") if False else d.astype(np.float32) / 255 * g("denoise") + out * (1 - g("denoise"))
    if g("smear") > 0:
        hot = K.lin(np.clip(out, 0, 1)) * K.smoothstep(0.8, 0.97, K.luma(out))[..., None]
        L = max(2.0, 120 * s)
        k = int(L * 4) | 1
        xk = np.arange(k, dtype=np.float32)
        ker = np.exp(-xk / L); ker /= ker.sum()
        kk = np.zeros(2 * k - 1, np.float32); kk[k - 1:] = ker
        sm = cv2.sepFilter2D(np.ascontiguousarray(hot), -1, np.array([1.0], np.float32).reshape(1, 1), kk.reshape(-1, 1))
        sm = K.blur(sm, 1.0 * s + 0.5)
        out = K.srgb(np.clip(K.lin(np.clip(out, 0, 1)) + sm * g("smear") * 1.1, 0, 1))
    if g("sharpen") > 0:
        out = out + (out - K.blur(out, 1.3 * s + 0.6)) * g("sharpen") * 1.2
    if g("clip") > 0:
        t = 1 - 0.18 * g("clip")
        out = np.where(out > t, t + (out - t) * (1 - 0.8 * g("clip")) + (out - t) * 0.9 * 0, out)
        out = np.clip(out * (1 + 0.12 * g("clip")), 0, 1)
    out = _vignette(np.clip(out, 0, 1), g("vignette"))
    return np.clip(out, 0.004, 0.996).astype(np.float32)
