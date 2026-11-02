"""Warp fx: earthquake, heat haze, zoom blur and other geometry/colour bends. Fragment convention is in
overlay.py / cut.py."""
from . import Ctx, fx
from ._expr import hash2, rgb_geq  # noqa: F401


def _ev(x: int) -> int:
    return max(2, int(x) // 2 * 2)


@fx("earthquake", "cut", tags=("slop", "hype", "glitch"))
def earthquake(ctx):
    k = ctx.intensity
    amp = min(0.07, 0.04 * k) * ctx.w          # max shift stays inside the 1.15x margin (0.075 w)
    ampy = min(0.07, 0.04 * k) * ctx.h
    jx = hash2("n", f"{ctx.seed % 97}+1")
    jy = hash2("n", f"{ctx.seed % 89}+7")
    return (f"scale=w='trunc(iw*1.15/2)*2':h='trunc(ih*1.15/2)*2',"
            f"crop={ctx.w}:{ctx.h}:x='clip((iw-ow)/2+({jx}-0.5)*2*{amp:.1f}+sin(n*2.1)*{amp * 0.3:.1f},0,iw-ow)':"
            f"y='clip((ih-oh)/2+({jy}-0.5)*2*{ampy:.1f}+sin(n*1.7)*{ampy * 0.3:.1f},0,ih-oh)',"
            f"tmix=frames={2 if k < 1.3 else 3}")


@fx("heat_haze", "filter", tags=("slop", "glitch"))
def heat_haze(ctx):
    n = 12
    hh = _ev(ctx.h / n)
    n = max(2, ctx.h // hh)
    last = ctx.h - hh * (n - 1)
    amp = min(0.9, 0.5 * ctx.intensity)
    w2 = _ev(ctx.w * 1.06)
    parts, labels = [], []
    for i in range(n):
        hi = last if i == n - 1 else hh
        ph = (i * 0.9 + (ctx.seed % 7))
        parts.append(f"[_s{i}]crop={ctx.w}:{hi}:x='(iw-ow)/2*(1+{amp:.2f}*sin(t*6+{ph:.2f}))':y={i * hh}[_o{i}]")
        labels.append(f"[_o{i}]")
    return (f"[IN]scale={w2}:{ctx.h},split={n}" + "".join(f"[_s{i}]" for i in range(n)) + ";"
            + ";".join(parts) + ";" + "".join(labels) + f"vstack=inputs={n}[OUT]")


@fx("chroma_boom", "filter", tags=("slop", "hype", "glitch"))
def chroma_boom(ctx):
    base = max(4, round(ctx.w * 0.05 * ctx.intensity))
    off = (ctx.seed % 4) / 10 * 0.2
    chain = []
    for lo, hi, f in ((0.0, 0.07, 1.0), (0.07, 0.15, 0.55), (0.15, 0.28, 0.25)):
        s = max(1, round(base * f))
        chain.append(f"rgbashift=rh={-s}:bh={s}:rv={s // 3}:bv={-s // 3}"
                     f":enable='between(mod(t+{off:.2f},0.4),{lo},{hi})'")
    return ",".join(chain)


@fx("zoom_blur", "filter", tags=("slop", "hype"))
def zoom_blur(ctx):
    zs = (1.0, 1.04 + 0.01 * ctx.intensity, 1.08 + 0.025 * ctx.intensity, 1.12 + 0.05 * ctx.intensity)
    n = len(zs)
    parts = []
    for i, z in enumerate(zs):
        sc = f"scale={_ev(ctx.w * z)}:{_ev(ctx.h * z)}," if i else ""
        parts.append(f"[_c{i}]{sc}crop={ctx.w}:{ctx.h}[_z{i}]")
    return (f"[IN]format=gbrp,split={n}" + "".join(f"[_c{i}]" for i in range(n)) + ";"
            + ";".join(parts) + ";" + "".join(f"[_z{i}]" for i in range(n))
            + f"mix=inputs={n}:weights='2 1.5 1 1',eq=contrast=1.08[OUT]")


@fx("strobe_invert", "filter", tags=("slop", "hype", "glitch"))
def strobe_invert(ctx):
    b = min(0.35, 0.18 * ctx.intensity)
    return (f"negate=enable='lt(mod(n+{ctx.seed % 6},6),3)',"
            f"eq=brightness='{b:.2f}*lt(mod(n+{ctx.seed % 6},6),1)':contrast=1.25:eval=frame")


@fx("datamosh", "cut", tags=("slop", "glitch"))
def datamosh(ctx):
    bw, bh = _ev(ctx.w / 12), _ev(ctx.h / 12)
    decay = min(0.97, 0.9 + 0.03 * ctx.intensity)
    return (f"[IN]format=gbrp,split=3[_a][_b][_c];"
            f"[_b]scale={bw}:{bh}:flags=neighbor,scale={ctx.w}:{ctx.h}:flags=neighbor,lagfun=decay={decay:.2f}[_p];"
            f"[_c]tblend=all_mode=difference,eq=brightness=0.1:contrast=2.5[_d];"
            f"[_a][_p]blend=all_mode=average[_m];"
            f"[_m][_d]blend=all_mode=addition:all_opacity=0.7[OUT]")


@fx("jpeg_crush", "filter", tags=("slop", "glitch"))
def jpeg_crush(ctx):
    div = 6 * (1 if ctx.intensity < 1.5 else 1.5)
    sw, sh = _ev(ctx.w / div), _ev(ctx.h / div)
    return (f"scale={sw}:{sh}:flags=bicubic,"
            f"noise=alls=14:allf=t:all_seed={ctx.seed % 10000 + 1},"
            f"scale={ctx.w}:{ctx.h}:flags=neighbor,"
            "eq=contrast=1.35:saturation=1.5:brightness=0.04,"
            "chromashift=crh=3:cbh=-3:crv=1:edge=smear,"
            f"noise=alls=8:allf=t:all_seed={ctx.seed % 10000 + 2}")


@fx("bass_boost", "cut", tags=("slop", "hype"))
def bass_boost(ctx):
    amp = min(0.4, 0.25 * ctx.intensity)
    p = "pow(1-mod(t,0.5)/0.5,2.5)"
    z = f"(1+{amp:.2f}*{p})"
    return (f"scale=w='trunc(iw*{z}/2)*2':h='trunc(ih*{z}/2)*2':eval=frame,"
            f"crop={ctx.w}:{ctx.h},"
            f"eq=brightness='0.07*{p}':saturation='1+0.6*{p}':contrast='1+0.15*{p}':eval=frame")
