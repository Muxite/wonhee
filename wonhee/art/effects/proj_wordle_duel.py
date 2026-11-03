"""proj-wordle-duel: Wordle feedback tiles as a dot-matrix mosaic, built as duels.
Each board is a REAL simulated Wordle game: a solver guesses 5-letter words that stay
consistent with all earlier feedback until it finds the target. Every tile shows its
letter as a 5x7 dot-matrix glyph on a 7x9 dot tile, coloured green / gold / grey by the
true Wordle feedback rule. Boards tile edge to edge in a brick pattern (alternate board columns are shifted
by half a board) so two players' boards sit side by side everywhere: no centre.
Deterministic. ctx.data (optional): {"games": [[guess, ..., target], ...]}: real games shown on every other board."""
from __future__ import annotations

import numpy as np

from .. import dotkit as dk
from ..palette import oklch_to_rgb
from ..registry import effect

WORDS = ("CRANE SLATE TRAIN HOUSE PLANT GHOST BRICK FLAME STONE WORLD MOUSE NIGHT LIGHT "
         "SHARE POINT SMILE BEACH CLOUD DREAM FIELD GRAPE HEART JOLLY KNIFE LEMON MAPLE "
         "NOBLE OCEAN PIANO QUEEN RIVER SOLID TIGER UNITY VIVID WATER YOUNG ZEBRA BLAZE "
         "CHAIR DRIVE EAGLE FROST GLOBE HONEY IVORY JUICE LUNAR MARCH NORTH PEARL RAISE "
         "SCOUT TOAST ULTRA VOICE WHEAT PRIZE SPARK SWEET TRACK BLOOM CRISP DANCE GRAIN").split()


def feedback(g, t):
    r = [0] * 5
    left = {}
    for i in range(5):
        if g[i] == t[i]:
            r[i] = 2
        else:
            left[t[i]] = left.get(t[i], 0) + 1
    for i in range(5):
        if r[i] == 0 and left.get(g[i], 0) > 0:
            r[i] = 1
            left[g[i]] -= 1
    return tuple(r)


def real_game(words):
    # words: guesses ending in the target (e.g. the PRIDE/CHASE/SHARE game in the Run Wrapped poster)
    t = words[-1].upper()
    return [(w.upper(), feedback(w.upper(), t)) for w in words]


def play(rng, rows=6):
    target = WORDS[int(rng.integers(len(WORDS)))]
    cand = list(WORDS)
    out = []
    for _ in range(rows):
        pool = cand if (len(out) >= 4 or len(cand) < 3) else [w for w in WORDS if w != target]
        g = pool[int(rng.integers(len(pool)))]
        fb = feedback(g, target)
        out.append((g, fb))
        if g == target:
            break
        cand = [c for c in cand if feedback(g, c) == fb]
    return out


@effect("proj-wordle-duel", family="dots", tags=["dots", "wordle", "tiles", "project", "all-over"],
        params={"d": (0.0078, "dot pitch, fraction of width"), "hue": (135, "green hue"), "gold": (0.16, "gold weight"),
                "rows": (6, "guesses per board")})
def proj_wordle_duel(ctx):
    W, H = ctx.size
    rng = ctx.rng
    P = ctx.palette
    d = float(ctx.p("d")) * W
    hue = float(ctx.p("hue"))
    nrows = int(ctx.p("rows"))
    green = oklch_to_rgb(0.62, 0.17, hue)
    gold = oklch_to_rgb(0.80, 0.14, 92)
    grey = oklch_to_rgb(0.44, 0.025, hue)
    cols_c = {2: green, 1: gold, 0: grey}
    glyph_c = {2: oklch_to_rgb(0.86, 0.15, hue), 1: oklch_to_rgb(0.92, 0.11, 95), 0: oklch_to_rgb(0.68, 0.03, hue)}
    TW, TH = 8, 10                      # tile pitch in dots
    cv = dk.new_canvas(ctx.size, P.canvas)
    nTc = int(np.ceil(W / (TW * d))) + 7
    nTr = int(np.ceil(H / (TH * d))) + 2 * (nrows + 1)
    BW, BH = 5 + 1, nrows + 1           # board in tiles incl. gutter
    ox, oy = -0.45 * TW * d, -0.5 * TH * d
    boards = {}
    xs, ys, rs, cs, al = [], [], [], [], []
    def push(x, y, r, c, a):
        xs.append(x); ys.append(y); rs.append(r); cs.append(c); al.append(a)
    for tr in range(nTr):
        for tc in range(nTc):
            bi = tc // BW
            trs = tr + (BH // 2 if bi % 2 else 0)
            bj = trs // BH
            lc, lr = tc % BW, trs % BH
            key = (bi, bj)
            if key not in boards:
                sub = np.random.default_rng(ctx.seed * 7919 + bi * 131 + bj * 17)
                real = ctx.data.get("games") if isinstance(ctx.data, dict) else None
                boards[key] = real_game(real[(bi + bj) // 2 % len(real)]) if real and (bi + bj) % 2 == 0 else play(sub, nrows)
            game = boards[key]
            if lc == 5 or lr == nrows:
                ccol, state, ch = None, None, None
            else:
                cc = lc
                if lr < len(game):
                    g, fb = game[lr]
                    state, ch = fb[cc], g[cc]
                    ccol = cols_c[state]
                else:
                    ccol, state, ch = None, None, None
            x0 = ox + tc * TW * d
            y0 = oy + tr * TH * d
            if x0 > W or y0 > H or x0 < -TW * d or y0 < -TH * d:
                continue
            glyph = dk.text_matrix(ch)[:, :5] if ch else None
            for j in range(9):
                for i in range(7):
                    px, py = x0 + (i + 0.5) * d, y0 + (j + 0.5) * d
                    if ccol is None:
                        push(px, py, d * (0.16 if lr < nrows and lc < 5 else 0.10), green if lr < nrows and lc < 5 else P.structure * 0.5 + P.canvas * 0.5, 0.22)
                        continue
                    on = glyph is not None and 1 <= i <= 5 and 1 <= j <= 7 and glyph[j - 1, i - 1]
                    if on:
                        push(px, py, d * 0.44, glyph_c[state], 1.0)
                    else:
                        push(px, py, d * (0.30 if state else 0.26), ccol, 0.62 if state != 0 else 0.45)
    dk.draw_dots(cv, np.array(xs), np.array(ys), np.array(rs), np.array(cs), np.array(al), "round")
    return cv
