"""A drawn plane cabin (one-point perspective: aisle, seat rows, overhead bins, windows with daylight).

Why it exists: the chroma key needs something to key onto before real cabin footage exists, and storyboards need a
set. It is deterministic (no randomness), so the same size always gives the same pixels. Plain function on arrays.
"""
from __future__ import annotations

import cv2
import numpy as np

HALF, AISLE, CEIL, CAM_Y = 1.6, 0.27, 2.15, 1.15   # metres: cabin half width, aisle half width, ceiling, eye height
SEAT_W, ROW_GAP, Z0, Z1 = 0.46, 0.86, 1.1, 15.0


def _proj(w: int, h: int):
    f, cx, cy = 0.8 * w, 0.5 * w, 0.40 * h

    def p(x, y, z):
        return (cx + f * x / z, cy - f * (y - CAM_Y) / z)
    return p


def _poly(img, pts, color):
    cv2.fillPoly(img, [np.round(np.array(pts)).astype(np.int32)], color, cv2.LINE_AA)


def _shade(color, z):
    k = 1.0 - min(z / 22.0, 0.55)
    return tuple(int(c * k) for c in color)


def render(w: int = 1080, h: int = 1920) -> np.ndarray:
    p = _proj(w, h)
    img = np.zeros((h, w, 3), np.uint8)
    far = Z1 + 4
    # ceiling, floor and side walls as four flat planes meeting at the far wall
    _poly(img, [p(-HALF, CEIL, Z0 * .3), p(HALF, CEIL, Z0 * .3), p(HALF, CEIL, far), p(-HALF, CEIL, far)], (226, 228, 232))
    _poly(img, [p(-HALF, 0, Z0 * .3), p(HALF, 0, Z0 * .3), p(HALF, 0, far), p(-HALF, 0, far)], (92, 74, 66))
    for s in (-1, 1):
        _poly(img, [p(s * HALF, 0, Z0 * .3), p(s * HALF, CEIL, Z0 * .3), p(s * HALF, CEIL, far), p(s * HALF, 0, far)],
              (214, 214, 210))
    # far wall: galley curtain
    _poly(img, [p(-HALF, 0, far), p(HALF, 0, far), p(HALF, CEIL, far), p(-HALF, CEIL, far)], (120, 96, 84))
    # aisle carpet stripe + floor light strip
    _poly(img, [p(-AISLE, 0.001, Z0 * .3), p(AISLE, 0.001, Z0 * .3), p(AISLE, 0.001, far), p(-AISLE, 0.001, far)],
          (128, 104, 90))
    rows = np.arange(Z0, Z1, ROW_GAP)[::-1]            # far to near
    for z in rows:
        for s in (-1, 1):
            # window on the wall, a little behind the seat back, bright daylight with a soft frame
            wz0, wz1 = z + 0.12, z + 0.62
            win = [p(s * HALF, 0.88, wz0), p(s * HALF, 1.36, wz0), p(s * HALF, 1.36, wz1), p(s * HALF, 0.88, wz1)]
            _poly(img, win, (170, 160, 150))
            cx = np.mean([q[0] for q in win]); cy = np.mean([q[1] for q in win])
            win_in = [(cx + (q[0] - cx) * .8, cy + (q[1] - cy) * .8) for q in win]
            _poly(img, win_in, _shade((255, 214, 150), z * 0.3))
            # seats: three per side, backs facing the camera
            for k in range(3):
                xa = s * (AISLE + k * SEAT_W)
                xb = s * (AISLE + (k + 1) * SEAT_W - 0.03)
                c = _shade((112, 70, 52), z)
                _poly(img, [p(xa, 0.42, z), p(xb, 0.42, z), p(xb, 1.06, z), p(xa, 1.06, z)], c)
                _poly(img, [p(xa, 0.86, z), p(xb, 0.86, z), p(xb, 1.08, z), p(xa, 1.08, z)], _shade((236, 236, 232), z))
                tray = [p(xa + s * .03, 0.62, z - .02), p(xb - s * .03, 0.62, z - .02), p(xb - s * .03, 0.74, z - .02),
                        p(xa + s * .03, 0.74, z - .02)]
                _poly(img, tray, _shade((150, 120, 100), z))
    # overhead bins: underside + front face, each side
    for s in (-1, 1):
        _poly(img, [p(s * .12, 1.72, Z0 * .3), p(s * HALF, 1.72, Z0 * .3), p(s * HALF, 1.72, far), p(s * .12, 1.72, far)],
              (232, 232, 230))
        _poly(img, [p(s * .12, 1.72, Z0 * .3), p(s * .12, 2.05, Z0 * .3), p(s * .12, 2.05, far), p(s * .12, 1.72, far)],
              (206, 208, 210))
        for z in np.arange(Z0, Z1, 1.7):
            _poly(img, [p(s * .12, 1.74, z), p(s * .12, 2.03, z), p(s * .12, 2.03, z + .03), p(s * .12, 1.74, z + .03)],
                  (170, 172, 176))
    # ceiling reading-light panel and a soft warm light spill on the aisle
    light = np.zeros((h, w), np.float32)
    cv2.ellipse(light, (w // 2, int(h * 0.62)), (int(w * .3), int(h * .22)), 0, 0, 360, 1, -1)
    light = cv2.GaussianBlur(light, (0, 0), w * 0.12)
    img = np.clip(img.astype(np.float32) + light[..., None] * np.array([18, 26, 34], np.float32), 0, 255)
    # vignette and a gentle cool grade
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    v = 1 - 0.32 * (((xx - w / 2) / (w / 2)) ** 2 + ((yy - h * .5) / (h / 2)) ** 2) / 2
    img = img * v[..., None] * np.array([1.03, 1.0, 0.96], np.float32)
    return np.clip(cv2.GaussianBlur(img, (0, 0), 1.0), 0, 255).astype(np.uint8)


def sway(img: np.ndarray, t: float, amp: float = 0.004) -> np.ndarray:
    """A slow handheld-in-turbulence drift: a few pixels of x/y shift and a hair of rotation."""
    h, w = img.shape[:2]
    dx, dy = amp * w * np.sin(2 * np.pi * 0.31 * t), amp * h * np.sin(2 * np.pi * 0.47 * t + 1.1)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), 0.25 * np.sin(2 * np.pi * 0.23 * t), 1.02)
    m[:, 2] += (dx, dy)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def render_seat(w: int = 1080, h: int = 1920) -> np.ndarray:
    """A window-seat close set for a seated 3/4-angle shot: cabin wall and window behind, headrest, overhead bin above.
    The subject sits in the lower middle, so the headrest is drawn behind that spot."""
    s = w / 540.0
    X = lambda v: int(round(v * s))  # noqa: E731  (design units are a 540x960 frame)
    img = np.zeros((h, w, 3), np.uint8)
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    img[:] = (np.array([226, 222, 214], np.float32) * (1 - yy) + np.array([190, 184, 176], np.float32) * yy)
    # window: frame, daylight, a half-lowered shade
    x0, y0, x1, y1 = X(330), X(250), X(505), X(560)
    cv2.rectangle(img, (x0 - X(14), y0 - X(14)), (x1 + X(14), y1 + X(14)), (178, 174, 168), -1, cv2.LINE_AA)
    sky = np.linspace(0, 1, y1 - y0, dtype=np.float32)[:, None, None]
    img[y0:y1, x0:x1] = np.array([255, 226, 170], np.float32) * (1 - sky) + np.array([255, 245, 225], np.float32) * sky
    cv2.rectangle(img, (x0, y0), (x1, y0 + X(95)), (205, 200, 192), -1)
    cv2.line(img, (x0, y0 + X(95)), (x1, y0 + X(95)), (160, 156, 150), X(3), cv2.LINE_AA)
    glow = np.zeros((h, w), np.float32)
    cv2.rectangle(glow, (x0, y0), (x1, y1), 1, -1)
    glow = cv2.GaussianBlur(glow, (0, 0), X(40))
    # overhead bin and reading-light panel along the top
    cv2.rectangle(img, (0, 0), (w, X(150)), (214, 216, 218), -1)
    cv2.rectangle(img, (0, X(150)), (w, X(168)), (150, 150, 152), -1)
    cv2.rectangle(img, (X(60), X(180)), (X(150), X(196)), (235, 235, 238), -1, cv2.LINE_AA)
    # headrest (blue seat, white cover) behind the subject
    cv2.ellipse(img, (X(270), X(690)), (X(185), X(330)), 0, 0, 360, (112, 70, 52), -1, cv2.LINE_AA)
    cv2.ellipse(img, (X(270), X(470)), (X(170), X(150)), 0, 180, 360, (236, 236, 232), -1, cv2.LINE_AA)
    cv2.rectangle(img, (X(100), X(470)), (X(440), X(520)), (236, 236, 232), -1)
    # armrest and tray edge at the bottom
    cv2.rectangle(img, (0, X(905)), (w, h), (92, 70, 58), -1)
    cv2.rectangle(img, (0, X(895)), (w, X(910)), (140, 112, 96), -1)
    out = img.astype(np.float32) + glow[..., None] * np.array([10, 22, 34], np.float32)
    xx, yv = np.mgrid[0:h, 0:w].astype(np.float32)[::-1]
    v = 1 - 0.3 * (((xx - w / 2) / (w / 2)) ** 2 + ((yv - h * .5) / (h / 2)) ** 2) / 2
    return np.clip(cv2.GaussianBlur(out * v[..., None], (0, 0), 1.0), 0, 255).astype(np.uint8)


def render_window(w: int = 1080, h: int = 1920) -> np.ndarray:
    """One big plane window on a cabin wall: sky, soft clouds, a wing, a half-lowered shade. Static and deterministic,
    meant as the backdrop behind a person seated at the window (the subject covers the middle and right)."""
    from PIL import Image, ImageDraw
    s = w / 540.0
    X = lambda v: int(round(v * s))  # noqa: E731  (design units are a 540x960 frame)
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    img = (np.array([226, 222, 214], np.float32) * (1 - yy) + np.array([188, 182, 174], np.float32) * yy) * np.ones((h, w, 3), np.float32)
    x0, y0, x1, y1 = X(40), X(170), X(500), X(830)
    iw, ih = x1 - x0, y1 - y0
    sky = np.linspace([235, 190, 120], [255, 240, 215], ih, dtype=np.float32)[:, None, :] * np.ones((ih, iw, 3), np.float32)
    cl = np.zeros((ih, iw), np.float32)
    r = np.random.default_rng(11)
    for _ in range(14):
        cv2.ellipse(cl, (int(r.integers(0, iw)), int(r.integers(ih // 4, ih))), (X(int(r.integers(40, 90))), X(int(r.integers(18, 40)))),
                    0, 0, 360, 1, -1)
    cl = np.clip(cv2.GaussianBlur(cl, (0, 0), X(10)) * 1.4, 0, 1)
    sky = sky * (1 - cl[..., None]) + 255.0 * cl[..., None]
    cv2.fillPoly(sky, [np.array([(0, int(ih * .70)), (int(iw * .62), int(ih * .60)), (int(iw * .62), int(ih * .66)), (0, int(ih * .80))])],
                 (204, 198, 196), cv2.LINE_AA)                                  # wing (BGR)
    cv2.rectangle(sky, (0, 0), (iw, X(150)), (192, 200, 205), -1)               # half-lowered shade
    cv2.line(sky, (0, X(150)), (iw, X(150)), (150, 156, 160), X(3), cv2.LINE_AA)
    m = Image.new("L", (iw, ih), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, iw - 1, ih - 1), X(150), fill=255)
    ma = np.asarray(m, np.float32)[..., None] / 255.0
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - ma) + sky * ma
    fr = Image.fromarray(img.astype(np.uint8)[..., ::-1])
    ImageDraw.Draw(fr).rounded_rectangle((x0 - X(14), y0 - X(14), x1 + X(14), y1 + X(14)), X(164), outline=(176, 172, 166), width=X(18))
    img = np.asarray(fr)[..., ::-1].astype(np.float32)
    yv, xv = np.mgrid[0:h, 0:w].astype(np.float32)
    v = 1 - 0.28 * (((xv - w / 2) / (w / 2)) ** 2 + ((yv - h / 2) / (h / 2)) ** 2) / 2
    return np.clip(img * v[..., None], 0, 255).astype(np.uint8)
