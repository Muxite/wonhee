"""Prototype 4: 9:16 post art. Layered DAG and ring buffer families on free-hue palettes (2 or 3 colours + canvas).

Palettes are OKLCH (L, C, h) per role, so contrast is set by lightness, not hue. A palette has canvas, structure, an
optional second structure colour (3-colour palettes), accent, and derived edge/band tints. Canvas may be dark, light,
or a saturated "poster" colour.
"""
from __future__ import annotations

import math
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from structures import Cv  # noqa: E402
from art2 import matrix as matrix_art, hx  # noqa: E402
from wonhee.edit import fx  # noqa: E402
from wonhee.edit.viz import palette as pal  # noqa: E402


def mix(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def _rgb(lch):
    return hx(pal.to_hex(*lch))


def make(name, fam, canvas, struct, accent, s2=None, ink=None):
    c, s, a = _rgb(canvas), _rgb(struct), _rgb(accent)
    s2c = _rgb(s2) if s2 else s
    if ink is None:
        ink = (.2, .02, canvas[2]) if canvas[0] > .6 else (.93, .01, canvas[2])
    P = dict(canvas=c, structure=s, structure2=s2c, accent=a, ink=_rgb(ink), edge=mix(s, c, .30), edge2=mix(s2c, c, .30), band=mix(c, s, .10))
    P["name"], P["fam"], P["ncol"] = name, fam, 3 if s2 else 2
    P["dL_struct"] = round(abs(pal.lightness(pal.to_hex(*struct)) - pal.lightness(pal.to_hex(*canvas))), 2)
    P["dL_accent"] = round(abs(pal.lightness(pal.to_hex(*accent)) - pal.lightness(pal.to_hex(*canvas))), 2)
    P["hex"] = [pal.to_hex(*canvas), pal.to_hex(*struct)] + ([pal.to_hex(*s2)] if s2 else []) + [pal.to_hex(*accent)]
    return P


D, Lt, Po = "dark", "light", "poster"
PALETTES = [
    make("Blue violet", D, (.15, .045, 285), (.52, .12, 268), (.76, .20, 325)),
    make("Violet dusk", D, (.15, .04, 295), (.50, .13, 285), (.84, .14, 200), s2=(.62, .15, 335)),
    make("Red yellow", D, (.15, .03, 25), (.52, .17, 27), (.88, .17, 95)),
    make("Ember", D, (.14, .025, 40), (.50, .16, 25), (.90, .17, 100), s2=(.66, .15, 55)),
    make("White dark", D, (.14, .004, 260), (.52, .01, 260), (.97, .004, 260)),
    make("Black cherry", D, (.13, .03, 10), (.47, .12, 10), (.95, .02, 85)),
    make("Teal coral", D, (.15, .03, 205), (.52, .09, 195), (.74, .17, 35)),
    make("Green gold", D, (.15, .03, 155), (.52, .11, 150), (.86, .16, 95)),
    make("Neon trio", D, (.14, .05, 280), (.62, .15, 215), (.90, .20, 125), s2=(.60, .22, 340)),
    make("Indigo lime", D, (.16, .05, 272), (.50, .13, 275), (.88, .20, 125)),
    make("Sunset trio", D, (.15, .04, 335), (.55, .16, 350), (.90, .15, 92), s2=(.70, .15, 50)),
    make("Ocean trio", D, (.15, .035, 240), (.52, .10, 235), (.86, .15, 165), s2=(.62, .10, 195)),
    make("Rose gold", D, (.15, .015, 30), (.55, .07, 30), (.83, .10, 70)),
    make("Forest peach", D, (.15, .03, 160), (.48, .09, 155), (.82, .12, 50)),
    make("Blue orange", D, (.15, .03, 255), (.50, .13, 255), (.76, .17, 55)),
    make("Magenta mint", D, (.15, .04, 330), (.52, .17, 335), (.88, .15, 170)),
    make("Ink on paper", Lt, (.965, .004, 90), (.50, .008, 90), (.15, .01, 90)),
    make("Blue violet light", Lt, (.965, .014, 290), (.46, .13, 270), (.52, .24, 320)),
    make("Red on cream", Lt, (.955, .06, 100), (.50, .20, 28), (.25, .06, 28), s2=(.62, .17, 60)),
    make("Teal coral light", Lt, (.965, .012, 190), (.45, .08, 200), (.58, .20, 30)),
    make("Plum olive", Lt, (.96, .02, 100), (.40, .10, 330), (.55, .20, 25), s2=(.50, .10, 130)),
    make("Cobalt cream", Lt, (.96, .02, 85), (.42, .16, 262), (.58, .22, 35)),
    make("Mint pink", Lt, (.96, .02, 165), (.42, .08, 170), (.55, .22, 350)),
    make("Red poster", Po, (.50, .19, 28), (.93, .03, 85), (.90, .17, 98)),
    make("Cobalt poster", Po, (.40, .19, 263), (.88, .06, 235), (.91, .19, 105)),
    make("Violet poster", Po, (.38, .17, 300), (.80, .12, 330), (.92, .18, 105), s2=(.85, .10, 215)),
    make("Yellow poster", Po, (.89, .16, 98), (.22, .04, 280), (.52, .22, 28)),
    make("Forest poster", Po, (.34, .10, 155), (.85, .10, 140), (.90, .17, 85)),
    make("Pink violet", Po, (.78, .12, 350), (.50, .20, 350), (.25, .08, 300)),
]


class Cv3(Cv):
    def poly(self, pts, col, a=255):
        self.d.polygon([self._p(p) for p in pts], fill=col + (a,))

    def outline(self, pts, col, w=.3, a=255):
        self.curve(list(pts) + [pts[0]], col, w, a)


def sector(cv, cx, cy, rx0, rx1, ry0, ry1, a0, a1, fill=None, line=None, lw=.3, a=255, n=7):
    ts = np.linspace(a0, a1, n)
    outer = [(cx + rx1 * math.cos(t), cy + ry1 * math.sin(t)) for t in ts]
    inner = [(cx + rx0 * math.cos(t), cy + ry0 * math.sin(t)) for t in ts[::-1]]
    pts = outer + inner
    if fill is not None:
        cv.poly(pts, fill, a)
    if line is not None:
        cv.outline(pts, line, lw)


def run_state(i, head, size, N):
    """Cell i of a ring of N whose run of `size` cells ends at `head` (the newest). Returns (state, t): t is 0 at the tail."""
    off = (head - i) % N
    if off >= size:
        return "free", 0.0
    return ("head" if off == 0 else "body"), 1 - off / max(size - 1, 1)


def cell_style(P, state, t):
    """fill, line, alpha for a ring cell."""
    if state == "free":
        return None, P["edge"], 255
    if state == "head":
        return P["accent"], P["accent"], 255
    fill = mix(P["structure2"], P["structure"], t) if P["ncol"] == 3 else P["structure"]
    return fill, fill, int(150 + 105 * t) if P["ncol"] == 2 else 255


# ---- DAG ----------------------------------------------------------------------------------------------------------
def layers_in(rng, region, widths, jit):
    x0, y0, x1, y1 = region
    L = len(widths)
    out = []
    for l, n in enumerate(widths):
        y = y0 + (y1 - y0) * l / (L - 1)
        out.append([(x0 + (x1 - x0) * (j + .5) / n + rng.normal(0, jit), y + rng.normal(0, jit * .4)) for j in range(n)])
    return out


def edges_for(rng, layers, extra=.45):
    E = []
    for l in range(len(layers) - 1):
        A, B = layers[l], layers[l + 1]
        parented = set()
        for i, p in enumerate(A):
            order = np.argsort([abs(q[0] - p[0]) for q in B])
            tg = [int(order[0])]
            if rng.random() < extra and len(B) > 1:
                tg.append(int(order[1 if rng.random() < .8 else min(2, len(B) - 1)]))
            for j in tg:
                E.append((l, i, j)); parented.add(j)
        for j, q in enumerate(B):
            if j not in parented:
                E.append((l, int(np.argmin([abs(p[0] - q[0]) for p in A])), j))
    return E


def hot_path(rng, layers, E):
    path = [int(rng.integers(0, len(layers[0])))]
    for l in range(len(layers) - 1):
        outs = [j for (ll, i, j) in E if ll == l and i == path[-1]]
        path.append(int(rng.choice(outs)))
    return path


def dag_draw(cv, layers, E, path, P, node="box", edge="s", ew=.32, hot_ew=.75, ns=1.0, bundle_x=None):
    u = cv.u
    L = len(layers)
    for hot_pass in (False, True):
        for (l, i, j) in E:
            hot = path[l] == i and path[l + 1] == j
            if hot != hot_pass:
                continue
            p, q = layers[l][i], layers[l + 1][j]
            t = l / max(L - 1, 1)
            col = P["accent"] if hot else mix(P["edge"], P["edge2"], t)
            w = hot_ew if hot else ew
            if edge == "s":
                cv.s_curve(p, q, col, w, horizontal=False)
            elif edge == "line":
                cv.line(p, q, col, w)
            else:
                bx = bundle_x if bundle_x is not None else cv.W / 2
                cv.bez(p, ((p[0] + q[0]) / 2 * .3 + bx * .7, (p[1] + q[1]) / 2), q, col, w)
    for l, lay in enumerate(layers):
        t = l / max(L - 1, 1)
        for j, p in enumerate(lay):
            hot = path[l] == j
            col = P["accent"] if hot else mix(P["structure"], P["structure2"], t)
            x, y = p
            if node == "box":
                hw, hh = 3.3 * u * ns, 1.55 * u * ns
                cv.box(x - hw, y - hh, x + hw, y + hh, fill=P["accent"] if hot else P["canvas"], line=col, lw=.38, r=.5)
                if not hot:
                    cv.box(x - hw * .55, y - .28 * u, x + hw * .55, y + .28 * u, fill=col, r=.2)
            elif node == "pill":
                hw, hh = 4.3 * u * ns, 1.5 * u * ns
                cv.box(x - hw, y - hh, x + hw, y + hh, fill=P["accent"] if hot else P["canvas"], line=col, lw=.38, r=1.5 * ns)
                cv.dot((x - hw + 1.6 * u, y), .45, P["canvas"] if hot else col)
                cv.dot((x + hw - 1.6 * u, y), .45, P["canvas"] if hot else col)
            elif node == "dot":
                cv.dot(p, (1.7 if hot else 1.05) * ns, col)
                if hot:
                    cv.dot(p, 2.6 * ns, P["canvas"], ring=P["accent"])
            elif node == "ring":
                r = (1.8 if hot else 1.25) * ns
                cv.dot(p, r, P["accent"] if hot else P["canvas"], ring=col)
            elif node == "sq":
                h = (1.7 if hot else 1.2) * u * ns
                cv.box(x - h, y - h, x + h, y + h, fill=col, r=.3)


def widths_rand(rng, L, lo, hi):
    return [int(v) for v in rng.integers(lo, hi + 1, L)]


def full_region(cv, mx=6, my=6):
    return (mx * cv.u, my * cv.u, cv.W - mx * cv.u, cv.H - my * cv.u)


def dag_flow(cv, rng, P, k=1.0):
    r = full_region(cv, 8, 6); w = widths_rand(rng, 11, 3, 6); w[0] = 2; w[-1] = 2
    ly = layers_in(rng, r, w, cv.u * .8); E = edges_for(rng, ly); dag_draw(cv, ly, E, hot_path(rng, ly, E), P, "box", "s")


def dag_pills(cv, rng, P, k=1.0):
    r = full_region(cv, 10, 6); w = widths_rand(rng, 9, 2, 4); w[0] = 1; w[-1] = 1
    ly = layers_in(rng, r, w, cv.u * .6); E = edges_for(rng, ly, .35); dag_draw(cv, ly, E, hot_path(rng, ly, E), P, "pill", "s", ew=.3)


def dag_lattice(cv, rng, P, k=1.0):
    r = full_region(cv, 5, 5); w = widths_rand(rng, 17, 5, 9)
    ly = layers_in(rng, r, w, cv.u * 1.0); E = edges_for(rng, ly, .6); dag_draw(cv, ly, E, hot_path(rng, ly, E), P, "dot", "s", ew=.22, hot_ew=.6)


def dag_diamond(cv, rng, P, k=1.0):
    r = full_region(cv, 5, 6); w = [1, 2, 3, 5, 7, 9, 7, 5, 3, 2, 1]
    ly = layers_in(rng, r, w, cv.u * .5); E = edges_for(rng, ly, .55); dag_draw(cv, ly, E, hot_path(rng, ly, E), P, "ring", "s", ew=.26, ns=1.1)


def dag_bundle(cv, rng, P, k=1.0):
    r = full_region(cv, 5, 6); w = widths_rand(rng, 12, 4, 8)
    ly = layers_in(rng, r, w, cv.u * .5); E = edges_for(rng, ly, .7); dag_draw(cv, ly, E, hot_path(rng, ly, E), P, "dot", "bundle", ew=.2, hot_ew=.55)


def dag_swimlane(cv, rng, P, k=1.0):
    u, W, H = cv.u, cv.W, cv.H
    lanes, L = 5, 12; mx = 6 * u; lw = (W - 2 * mx) / lanes
    for i in range(lanes):
        cv.box(mx + i * lw + .6 * u, 3 * u, mx + (i + 1) * lw - .6 * u, H - 3 * u, fill=mix(P["canvas"], P["structure"], .06 + .05 * (i % 2)), r=.6)
    layers = []
    for l in range(L):
        y = 8 * u + (H - 16 * u) * l / (L - 1)
        ks = sorted(rng.choice(lanes, int(rng.integers(2, 4)), replace=False).tolist())
        layers.append([(mx + (c + .5) * lw, y) for c in ks])
    E = edges_for(rng, layers, .5); dag_draw(cv, layers, E, hot_path(rng, layers, E), P, "sq", "s", ew=.3)


def dag_tiered(cv, rng, P, k=1.0):
    u, W, H = cv.u, cv.W, cv.H
    r = full_region(cv, 10, 7); w = widths_rand(rng, 9, 2, 5)
    ly = layers_in(rng, r, w, cv.u * .5)
    for l, lay in enumerate(ly):
        t = l / 8
        cv.box(0, lay[0][1] - 6.5 * u, W, lay[0][1] + 6.5 * u, fill=mix(P["canvas"], mix(P["structure"], P["structure2"], t), .08 + .05 * (l % 2)), r=0)
    E = edges_for(rng, ly, .4); dag_draw(cv, ly, E, hot_path(rng, ly, E), P, "box", "s", ew=.3)


def dag_mirror(cv, rng, P, k=1.0):
    u, W, H = cv.u, cv.W, cv.H
    L = 11; widths = widths_rand(rng, L, 2, 4)
    left = layers_in(rng, (5 * u, 6 * u, W / 2 - 9 * u, H - 6 * u), widths, u * .6)
    Eh = edges_for(rng, left, .5)
    layers = []
    for l in range(L):
        n = widths[l]
        lay = left[l] + [(W / 2, left[l][0][1])] + [(W - x, y) for (x, y) in left[l]]
        layers.append(lay)
    E = []
    for (l, i, j) in Eh:
        n, n2 = widths[l], widths[l + 1]
        E.append((l, i, j)); E.append((l, n + 1 + i, n2 + 1 + j))
    for l in range(L - 1):
        n, n2 = widths[l], widths[l + 1]
        E.append((l, n, n2)); E.append((l, n, int(rng.integers(0, n2)))); E.append((l, n, n2 + 1 + int(rng.integers(0, n2))))
    path = [widths[l] for l in range(L)]
    dag_draw(cv, layers, E, path, P, "box", "s", ew=.3, ns=.9)


DAGS = {"flow": dag_flow, "pills": dag_pills, "lattice": dag_lattice, "diamond": dag_diamond, "bundle": dag_bundle,
        "swimlane": dag_swimlane, "tiered": dag_tiered, "mirror": dag_mirror}


# ---- ring buffers -------------------------------------------------------------------------------------------------
def draw_cell_rect(cv, P, x0, y0, x1, y1, state, t, rng):
    fill, line, a = cell_style(P, state, t)
    cv.box(x0, y0, x1, y1, fill=fill, line=line, lw=.3, r=.45, a=a)


def ring_snake(cv, rng, P, k=1.0):
    W, H, u = cv.W, cv.H, cv.u; m = 4 * u
    cols = 9; cw = (W - 2 * m) / cols; rows = round((H - 2 * m) / cw); ch = (H - 2 * m) / rows; N = cols * rows
    head = int(rng.integers(0, N)); size = int(N * rng.uniform(.4, .62))
    order = []
    for r in range(rows):
        order += [(r, c) for c in (range(cols) if r % 2 == 0 else range(cols - 1, -1, -1))]
    for idx, (r, c) in enumerate(order):
        st, t = run_state(idx, head, size, N)
        draw_cell_rect(cv, P, m + c * cw + cw * .08, m + r * ch + ch * .08, m + (c + 1) * cw - cw * .08, m + (r + 1) * ch - ch * .08, st, t, rng)


def wheel(cv, rng, P, cx, cy, R, nrings=3, thick=None, gap=1.4, ns=(48, 36, 24)):
    u = cv.u
    thick = thick or R * .17
    heads = []
    for ri in range(nrings):
        r1 = R - ri * (thick + gap * u); r0 = r1 - thick
        n = ns[ri]; head = int(rng.integers(0, n)); size = int(n * rng.uniform(.4, .7))
        for i in range(n):
            a0 = -math.pi / 2 + 2 * math.pi * i / n; a1 = -math.pi / 2 + 2 * math.pi * (i + 1) / n
            gp = .018
            st, t = run_state(i, head, size, n)
            fill, line, a = cell_style(P, st, t)
            sector(cv, cx, cy, r0, r1, r0, r1, a0 + gp, a1 - gp, fill=fill, line=line if fill is None else None, a=a)
        ha = -math.pi / 2 + 2 * math.pi * (head + .5) / n
        heads.append((cx + (r0 + r1) / 2 * math.cos(ha), cy + (r0 + r1) / 2 * math.sin(ha)))
    return heads


def tape(cv, rng, P, x0, y0, x1, y1, rows, cols):
    cw, ch = (x1 - x0) / cols, (y1 - y0) / rows; N = cols * rows
    head = int(rng.integers(N // 3, N)); size = int(N * rng.uniform(.4, .7)); cells = {}
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            st, t = run_state(idx, head, size, N)
            draw_cell_rect(cv, P, x0 + c * cw + cw * .1, y0 + r * ch + ch * .1, x0 + (c + 1) * cw - cw * .1, y0 + (r + 1) * ch - ch * .1, st, t, rng)
            if st == "head":
                cells["head"] = (x0 + (c + .5) * cw, y0 + (r + .5) * ch)
    return cells


def ring_wheel(cv, rng, P, k=1.0):
    W, H, u = cv.W, cv.H, cv.u; cx, cy = W / 2, H / 2; R = W / 2 - 4 * u
    heads = wheel(cv, rng, P, cx, cy, R)
    cv.dot((cx, cy), 2.4, P["accent"])
    top = tape(cv, rng, P, 4 * u, 4 * u, W - 4 * u, cy - R - 3 * u, 3, 14)
    bot = tape(cv, rng, P, 4 * u, cy + R + 3 * u, W - 4 * u, H - 4 * u, 3, 14)
    if "head" in top:
        cv.bez(top["head"], (top["head"][0], cy - R * .8), heads[0], P["accent"], .45, a=200)
    if "head" in bot:
        cv.bez(bot["head"], (bot["head"][0], cy + R * .8), heads[-1], P["accent"], .45, a=200)


def ring_lanes(cv, rng, P, k=1.0):
    W, H, u = cv.W, cv.H, cv.u; m = 4 * u; lanes = 16; lh = (H - 2 * m) / lanes; hot = int(rng.integers(0, lanes))
    for ln in range(lanes):
        n = int(rng.choice([14, 16, 18, 20])); cw = (W - 2 * m) / n; head = int(rng.integers(0, n)); size = int(n * rng.uniform(.3, .75))
        for i in range(n):
            st, t = run_state(i, head, size, n)
            if st == "head" and ln != hot:
                st = "body"
            draw_cell_rect(cv, P, m + i * cw + cw * .12, m + ln * lh + lh * .22, m + (i + 1) * cw - cw * .12, m + (ln + 1) * lh - lh * .22, st, t, rng)


def ring_spiral(cv, rng, P, k=1.0):
    W, H, u = cv.W, cv.H, cv.u; cx, cy = W / 2, H / 2; rx, ry = W / 2 - 5 * u, H / 2 - 5 * u
    N = 190; turns = 4.6; pts = []
    for i in range(N):
        t = i / (N - 1); r = .07 + .93 * t; a = 2 * math.pi * turns * t - math.pi / 2
        pts.append((cx + rx * r * math.cos(a), cy + ry * r * math.sin(a)))
    head = int(rng.integers(N // 2, N)); size = int(N * rng.uniform(.4, .6))
    cv.curve(pts, P["edge"], .22, a=200)
    for i, p in enumerate(pts):
        q0, q1 = pts[max(i - 1, 0)], pts[min(i + 1, N - 1)]
        ds = math.hypot(q1[0] - q0[0], q1[1] - q0[1]) / 2
        rad = float(np.clip(ds / u * .36, .55, 2.0))
        st, t = run_state(i, head, size, N)
        fill, line, a = cell_style(P, st, t)
        if fill is None:
            cv.dot(p, rad * .8, P["canvas"], ring=P["edge"])
        else:
            cv.dot(p, rad * (1.5 if st == "head" else 1), fill, a=a)


def ring_gauges(cv, rng, P, k=1.0):
    W, H, u = cv.W, cv.H, cv.u; cols, rows = 2, 5; m = 4 * u
    gw, gh = (W - 2 * m) / cols, (H - 2 * m) / rows; hot = int(rng.integers(0, cols * rows))
    for g in range(cols * rows):
        gx, gy = m + (g % cols + .5) * gw, m + (g // cols + .5) * gh; R = min(gw, gh) / 2 - 1.2 * u; n = int(rng.choice([24, 28, 32])); th = R * .26
        head = int(rng.integers(0, n)); size = int(n * rng.uniform(.3, .85))
        for i in range(n):
            a0 = -math.pi / 2 + 2 * math.pi * i / n + .02; a1 = -math.pi / 2 + 2 * math.pi * (i + 1) / n - .02
            st, t = run_state(i, head, size, n)
            if st == "head" and g != hot:
                st = "body"
            fill, line, a = cell_style(P, st, t)
            sector(cv, gx, gy, R - th, R, R - th, R, a0, a1, fill=fill, line=line if fill is None else None, a=a, n=4)
        cv.dot((gx, gy), .7, P["accent"] if g == hot else P["edge"])


def ring_trace(cv, rng, P, k=1.0):
    W, H, u = cv.W, cv.H, cv.u; m = 6 * u
    cols = 8; sp = (W - 2 * m) / (cols - 1); rows = int((H - 2 * m) / sp) + 1; sy = (H - 2 * m) / (rows - 1); N = cols * rows
    order = []
    for r in range(rows):
        order += [(r, c) for c in (range(cols) if r % 2 == 0 else range(cols - 1, -1, -1))]
    pts = [(m + c * sp, m + r * sy) for (r, c) in order]
    head = int(rng.integers(N // 2, N)); size = int(N * rng.uniform(.4, .6))
    cv.curve(pts, P["edge"], .22)
    for i, p in enumerate(pts):
        st, t = run_state(i, head, size, N)
        if st in ("body", "head") and i + 1 < N and run_state(i + 1, head, size, N)[0] in ("body", "head"):
            cv.line(p, pts[i + 1], mix(P["structure2"], P["structure"], t) if P["ncol"] == 3 else P["structure"], .9)
    for i, p in enumerate(pts):
        st, t = run_state(i, head, size, N)
        fill, line, a = cell_style(P, st, t)
        if fill is None:
            cv.dot(p, .85, P["canvas"], ring=P["edge"])
        else:
            cv.dot(p, 1.9 if st == "head" else 1.3, fill, a=a)
    cv.dot(pts[head], 3.4, P["canvas"], ring=P["accent"])


RINGS = {"snake": ring_snake, "wheel": ring_wheel, "lanes": ring_lanes, "spiral": ring_spiral, "gauges": ring_gauges, "trace": ring_trace}


# ---- combos -------------------------------------------------------------------------------------------------------
def combo_dag_ring(cv, rng, P, k=1.0):
    W, H, u = cv.W, cv.H, cv.u
    w = widths_rand(rng, 8, 2, 5); w[0] = 1
    ly = layers_in(rng, (8 * u, 5 * u, W - 8 * u, H * .52), w, u * .6); E = edges_for(rng, ly, .45); path = hot_path(rng, ly, E)
    dag_draw(cv, ly, E, path, P, "dot", "s", ew=.28, hot_ew=.7, ns=1.1)
    R = W / 2 - 5 * u; cy = H * .52 + 7 * u + R * .98
    heads = wheel(cv, rng, P, W / 2, min(cy, H - R - 4 * u), R, nrings=2, thick=R * .2, ns=(44, 32))
    last = ly[-1][path[-1]]
    cv.bez(last, (last[0], (last[1] + heads[0][1]) / 2), heads[0], P["accent"], .6)


def combo_ring_dag(cv, rng, P, k=1.0):
    W, H, u = cv.W, cv.H, cv.u
    R = W / 2 - 5 * u; cy = 5 * u + R
    heads = wheel(cv, rng, P, W / 2, cy, R, nrings=2, thick=R * .2, ns=(44, 32))
    top = cy + R + 8 * u
    w = widths_rand(rng, 8, 2, 5); w[-1] = 1
    ly = layers_in(rng, (8 * u, top, W - 8 * u, H - 5 * u), w, u * .6); E = edges_for(rng, ly, .45); path = hot_path(rng, ly, E)
    dag_draw(cv, ly, E, path, P, "dot", "s", ew=.28, hot_ew=.7, ns=1.1)
    cv.bez(heads[0], (heads[0][0], (top + heads[0][1]) / 2), ly[0][path[0]], P["accent"], .6)


def combo_chord(cv, rng, P, k=1.0):
    W, H, u = cv.W, cv.H, cv.u; cx, cy = W / 2, H / 2; R = W / 2 - 4 * u; n = 44; thick = R * .09
    head = int(rng.integers(0, n)); size = int(n * .55); r0 = R - thick
    pos = []
    for i in range(n):
        a0 = -math.pi / 2 + 2 * math.pi * i / n; a1 = -math.pi / 2 + 2 * math.pi * (i + 1) / n
        st, t = run_state(i, head, size, n); fill, line, a = cell_style(P, st, t)
        sector(cv, cx, cy, r0, R, r0, R, a0 + .02, a1 - .02, fill=fill, line=line if fill is None else None, a=a)
        am = (a0 + a1) / 2; pos.append((cx + (r0 - 1.5 * u) * math.cos(am), cy + (r0 - 1.5 * u) * math.sin(am)))
    hot = int(rng.integers(0, n))
    for _ in range(34):
        i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
        if i == j:
            continue
        hotc = i == head or j == head
        cv.bez(pos[i], (cx + (pos[i][0] + pos[j][0] - 2 * cx) * .12, cy + (pos[i][1] + pos[j][1] - 2 * cy) * .12), pos[j],
               P["accent"] if hotc else mix(P["edge"], P["edge2"], rng.random()), .5 if hotc else .24)
    top = tape(cv, rng, P, 4 * u, 4 * u, W - 4 * u, cy - R - 3 * u, 3, 14)
    bot = tape(cv, rng, P, 4 * u, cy + R + 3 * u, W - 4 * u, H - 4 * u, 3, 14)


def combo_frame(cv, rng, P, k=1.0):
    W, H, u = cv.W, cv.H, cv.u; cx, cy = W / 2, H / 2; rx, ry = W / 2 - 3 * u, H / 2 - 3 * u; th = 4.6 * u; n = 96
    head = int(rng.integers(0, n)); size = int(n * rng.uniform(.4, .6))
    for i in range(n):
        a0 = -math.pi / 2 + 2 * math.pi * i / n + .008; a1 = -math.pi / 2 + 2 * math.pi * (i + 1) / n - .008
        st, t = run_state(i, head, size, n); fill, line, a = cell_style(P, st, t)
        sector(cv, cx, cy, rx - th, rx, ry - th, ry, a0, a1, fill=fill, line=line if fill is None else None, a=a, n=3)
    rix, riy = rx - th - 4 * u, ry - th - 4 * u
    L = 11; layers = []
    for l in range(L):
        y = cy - riy * .94 + (2 * riy * .94) * l / (L - 1)
        chord = 2 * rix * math.sqrt(max(1 - ((y - cy) / riy) ** 2, .02))
        n_l = int(np.clip(round(chord / (13 * u)), 1, 6))
        layers.append([(cx - chord / 2 + chord * (j + .5) / n_l * 1.0 - (chord * 0 if n_l > 1 else 0), y) for j in range(n_l)])
    layers = [[(x + rng.normal(0, .5 * u), y) for (x, y) in lay] for lay in layers]
    E = edges_for(rng, layers, .5); dag_draw(cv, layers, E, hot_path(rng, layers, E), P, "box", "s", ew=.3, ns=.85)


COMBOS = {"dag over ring": combo_dag_ring, "ring over dag": combo_ring_dag, "chord ring": combo_chord, "dag in a ring frame": combo_frame}


# ---- rendering ----------------------------------------------------------------------------------------------------
def background(P, W, H, kind, seed):
    im = np.empty((H, W, 3), np.uint8); im[:] = np.array(P["canvas"], np.uint8)
    if kind == "dots":
        step = max(10, W // 22); ed = np.array(mix(P["canvas"], P["structure"], .22), np.uint8)
        for y in range(step // 2, H, step):
            for x in range(step // 2, W, step):
                im[max(y - 1, 0):y + 1, max(x - 1, 0):x + 1] = ed
    elif kind == "rules":
        step = max(12, W // 18); ed = np.array(mix(P["canvas"], P["structure"], .10), np.int32)
        im[::step] = ed
    return im


def render(kind, name, P, seed=7, W=450, H=800, bg="flat", grain=0.0):
    rng = np.random.default_rng(seed)
    cv = Cv3(W, H)
    ({**DAGS, **RINGS, **COMBOS, "matrix": matrix_art}[name])(cv, rng, P, 1.0) if name != "matrix" else matrix_art(cv, rng, {}, P, .9)
    out = Image.fromarray(background(P, W, H, bg, seed)).convert("RGBA")
    out.alpha_composite(cv.flat())
    arr = np.array(out.convert("RGB"))
    return fx.grain(arr[..., ::-1], grain, seed)[..., ::-1] if grain else arr


if __name__ == "__main__":
    import io
    for nm, P in [("flow", PALETTES[0]), ("wheel", PALETTES[2]), ("frame", PALETTES[8])]:
        pass
    for kind, nm, pi in [("dag", "flow", 0), ("ring", "wheel", 2), ("ring", "spiral", 9), ("dag", "mirror", 8), ("combo", "dag in a ring frame", 23),
                         ("combo", "chord ring", 24), ("ring", "gauges", 4), ("ring", "trace", 25), ("dag", "swimlane", 17), ("matrix", "matrix", 26)]:
        Image.fromarray(render(kind, nm, PALETTES[pi])).save(f"t4_{kind}_{nm.replace(chr(32), chr(95))}.png")
    for P in PALETTES:
        print(f"{P['name']:18s} {P['fam']:7s} {P['ncol']}c  dL struct {P['dL_struct']:.2f}  accent {P['dL_accent']:.2f}")
