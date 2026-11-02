"""Cut / temporal / geometry fx (kind "cut").

Conventions:
- Register with @fx("name", "cut", tags=("slop", ...)); the function takes `ctx` and returns a fragment.
- Fragment: linear chain ("shuffleframes='0 0 0 0 4 4 4 4'", no labels) OR a graph with [IN] (once), [OUT]
  and private labels [_x] (rewritten unique), e.g. "[IN]split[_a][_b];[_a]crop=iw/2:ih:0:0,split[_l1][_l2];
  [_l1][_l2]hstack[OUT]" (hstack of l + hflip(l)). Use [IN] once; split it for multiple uses.
- Ctx fields: w, h, fps, frames, seed (int per clip/slot; offset hashes/random params), intensity (0.5..2), dur.
- Output size/frame count may differ: the renderer re-scales to w x h and pads (clone) / trims to `frames`.
- Time in expressions: crop/geq/eq use t (seconds) or n (frame); commas inside expressions go in '...'.
- shuffleframes maps within a window of N frames (indices 0-based, -1 = drop); shufflepixels needs size
  divisible by block (use width/height parameters, not guesses). geq is slow: keep tests at 270x480.
- Helpers: fx/_expr.py (hash2, rgb_geq). Test: pytest tests/test_montage_fx.py -k <name>.
"""
from . import fx
from ._expr import hash2, rgb_geq


def _shuffle(pattern: str) -> str:
    """shuffleframes only emits complete windows: a clip shorter than one window (or a tail) would
    vanish, and an empty stream stalls the whole xfade chain. Clone-pad one window so it can't."""
    return f"tpad=stop_mode=clone:stop={len(pattern.split())},shuffleframes='{pattern}'"


@fx("stutter", "cut", tags=("slop", "hype", "glitch"))
def stutter(ctx):
    return _shuffle("0 0 0 0 4 4 4 4")


def _divisor(n: int, target: int) -> int:
    """Divisor of n closest to target (>=2 when possible)."""
    ds = [d for d in range(2, n + 1) if n % d == 0] or [n]
    return min(ds, key=lambda d: abs(d - target))


@fx("rewind_jitter", "cut", tags=("slop", "hype", "glitch"))
def rewind_jitter(ctx):
    pats = ("0 1 2 3 4 5 4 3 2 3 4 5", "0 1 2 3 2 1 0 1 2 3 4 5", "0 1 2 3 4 5 3 2 1 2 3 4")
    return _shuffle(pats[ctx.seed % 3])


@fx("echo_trails", "cut", tags=("slop", "hype"))
def echo_trails(ctx):
    lag = max(2, round(3 * ctx.intensity))
    ws = " ".join(w if i % lag == 0 else "0" for i in range(lag * 2 + 1)
                  for w in [str(3 - i // lag)])
    return f"tmix=frames={lag * 2 + 1}:weights='{ws}'"


@fx("slice_shift", "cut", tags=("slop", "glitch"))
def slice_shift(ctx):
    n = 5
    hh = max(2, int(ctx.h / n) // 2 * 2)
    parts, labels = [], []
    for i in range(n):
        sp = (0.01 + 0.02 * ((ctx.seed * 7 + i * 13) % 5) / 4) * ctx.intensity * (1 if (ctx.seed + i) % 2 else -1)
        parts.append(f"[_s{i}]crop=iw:{hh}:0:{i * hh},scroll=horizontal={sp:.4f}[_o{i}]")
        labels.append(f"[_o{i}]")
    split = f"[IN]split={n}" + "".join(f"[_s{i}]" for i in range(n))
    return split + ";" + ";".join(parts) + ";" + "".join(labels) + f"vstack=inputs={n}[OUT]"


@fx("pixel_scramble", "cut", tags=("slop", "glitch"))
def pixel_scramble(ctx):
    bw = _divisor(ctx.w, max(4, round(ctx.w / 14 / ctx.intensity)))
    bh = _divisor(ctx.h, max(4, round(ctx.h / 24 / ctx.intensity)))
    return (f"shufflepixels=mode=block:width={bw}:height={bh}:seed={ctx.seed % 100000}"
            f":enable='lt(mod(t+{(ctx.seed % 7) / 10},0.6),0.3)'")


@fx("tile_grid", "cut", tags=("slop", "hype"))
def tile_grid(ctx):
    n = 2 if ctx.intensity < 0.8 else 3
    sc = f"scale=trunc(iw/{2 * n})*2:trunc(ih/{2 * n})*2"
    ls = "".join(f"[_t{i}]" for i in range(n))
    rows = ""
    for r in range(n):
        flip = ("", "hflip,", "vflip,")[r % 3]
        rows += f"[_r{r}]{flip}null[_q{r}];"
    out = f"[IN]{sc},split={n}{ls};"
    out += f"{ls}hstack=inputs={n},split={n}" + "".join(f"[_r{r}]" for r in range(n)) + ";"
    out += rows + "".join(f"[_q{r}]" for r in range(n)) + f"vstack=inputs={n}[OUT]"
    return out


@fx("mirror_kaleido", "cut", tags=("slop", "glitch"))
def mirror_kaleido(ctx):
    sw = 0.3 + 0.2 * (ctx.seed % 3)
    return ("[IN]crop=trunc(iw/4)*2:ih:x='(iw-ow)/2*(1+0.9*sin(" + f"{sw * 3 * ctx.intensity:.2f}*t+{ctx.seed % 6}" + "))':y=0,"
            "split[_a][_b];[_b]hflip[_c];[_a][_c]hstack,"
            "crop=iw:trunc(ih/4)*2:0:'(ih-oh)/2*(1+0.9*cos(" + f"{sw * 2:.2f}*t" + "))',"
            "split[_d][_e];[_e]vflip[_f];[_d][_f]vstack[OUT]")


@fx("crash_zoom_jitter", "cut", tags=("slop", "hype"))
def crash_zoom_jitter(ctx):
    amp = 0.9 * ctx.intensity
    z = f"(1+{amp:.2f}*pow(mod(t,0.6)/0.6,3))"
    jx = hash2("n", f"{ctx.seed % 97}+1")
    jy = hash2("n", f"{ctx.seed % 89}+7")
    return (f"scale=w='trunc(iw*{z}/2)*2':h='trunc(ih*{z}/2)*2':eval=frame,"
            f"crop={ctx.w}:{ctx.h}:x='(iw-ow)/2+({jx}-0.5)*{ctx.w * 0.04:.0f}':"
            f"y='(ih-oh)/2+({jy}-0.5)*{ctx.h * 0.04:.0f}'")
