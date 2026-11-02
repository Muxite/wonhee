"""Colour / look filters (kind "filter").

Conventions:
- Register with @fx("name", "filter", tags=("slop", ...)); the function takes `ctx` and returns a fragment.
  Tags choose style pools: "slop" (all), "hype", "glitch" (calm never gets fx).
- Fragment: a linear chain ("negate,eq=contrast=1.5", no labels) OR a graph using [IN], [OUT] and private
  labels [_x] (rewritten to be unique); use [IN] once (split it). See fx/__init__.py.
- Ctx fields: w, h, fps, frames (exact), seed (int, per clip/slot), intensity (0.5..2), dur (seconds).
- Size/pix_fmt/frame count may change; the renderer re-normalises afterwards.
- Commas inside expressions need '...' quotes; no ':' or ',' unquoted in values. geq is slow: prefer
  lutrgb/colorchannelmixer/eq/hue. Helpers: fx/_expr.py (hash2, rgb_geq).
- Test: pytest tests/test_montage_fx.py -k <name>; eyeball with ffmpeg -ss T -i out.mp4 -frames:v 1 f.png.
"""
from . import fx


@fx("negative", "filter", tags=("slop", "glitch"))
def negative(ctx):
    return "negate"


def _i(ctx, base, lo=0.0, hi=1e9):
    return max(lo, min(hi, base * ctx.intensity))


@fx("vhs", "filter", tags=("slop", "glitch"))
def vhs(ctx):
    sh = int(round(_i(ctx, 4, 2, 10)))
    ns = int(_i(ctx, 14, 6, 40))
    return (f"chromashift=crh={-sh}:crv=1:cbh={sh}:edge=smear,"
            f"noise=alls={ns}:allf=t+u:all_seed={ctx.seed % 10000 + 1},"
            "gblur=sigma=0.9:sigmaV=0.3,"
            "eq=saturation=0.8:contrast=1.08:gamma_r=1.06:gamma_b=0.92,"
            "geq=lum='lum(X,Y)*(0.78+0.22*lt(mod(Y,4),2))':cb='cb(X,Y)':cr='cr(X,Y)'")


@fx("deepfry", "filter", tags=("slop", "hype", "glitch"))
def deepfry(ctx):
    k = ctx.intensity
    return (f"eq=saturation={min(2.0 + k, 4):.2f}:contrast={min(1.6 + 0.4 * k, 3):.2f}:brightness=0.04,"
            "unsharp=7:7:2.5:7:7:0,"
            "noise=alls=18:allf=t:all_seed=%d,"
            "lutrgb=r='floor(val/32)*32':g='floor(val/32)*32':b='floor(val/32)*32'" % (ctx.seed % 10000 + 1))


@fx("thermal", "filter", tags=("slop", "glitch", "hype"))
def thermal(ctx):
    # gray replicated into planar RGB, then a per-channel ramp: black-purple-red-yellow-white
    return ("format=gray,format=gbrp,"
            "lutrgb=r='clip(val*3-70,0,255)':"
            "g='clip(val*3-400,0,255)':"
            "b='clip(if(lt(val,90),val*2.5,(170-val)*2.6+55),0,255)',"
            "format=yuv420p")


@fx("nightvision", "filter", tags=("slop", "glitch"))
def nightvision(ctx):
    ns = int(_i(ctx, 28, 12, 60))
    return ("format=gbrp,colorchannelmixer=rr=0.1:rg=0.2:rb=0.1:gr=0.4:gg=0.9:gb=0.3:br=0.05:bg=0.1:bb=0.05,"
            "eq=brightness=0.08:contrast=1.35,"
            f"noise=alls={ns}:allf=t:all_seed={ctx.seed % 10000 + 1},"
            "vignette=angle=PI/3.5,format=yuv420p")


@fx("sepia", "filter", tags=("slop", "calm"))
def sepia(ctx):
    return ("format=gbrp,colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131,"
            "vignette=angle=PI/4,format=yuv420p")


@fx("posterize", "filter", tags=("slop", "hype"))
def posterize(ctx):
    step = 64 if ctx.intensity < 1.5 else 85
    return f"lutrgb=r='floor(val/{step})*{step}':g='floor(val/{step})*{step}':b='floor(val/{step})*{step}'"


@fx("bw_crush", "filter", tags=("slop", "calm", "hype"))
def bw_crush(ctx):
    c = _i(ctx, 2.2, 1.6, 3.5)
    return f"hue=s=0,eq=contrast={c:.2f}:brightness=-0.05"


@fx("rainbow", "filter", tags=("slop", "hype", "glitch"))
def rainbow(ctx):
    sp = int(_i(ctx, 540, 300, 1080))
    return f"hue=h='t*{sp}':s=2.2"


@fx("edges", "filter", tags=("slop", "glitch"))
def edges(ctx):
    return ("edgedetect=mode=colormix:high=0.2:low=0.05,"
            f"eq=brightness={_i(ctx, 0.1, 0.05, 0.3):.2f}:saturation=1.8:contrast=1.3")


@fx("bloom", "filter", tags=("slop", "calm", "hype"))
def bloom(ctx):
    sig = int(_i(ctx, 20, 10, 40))
    return ("[IN]split[_a][_b];"
            f"[_b]eq=brightness=0.05:contrast=1.2,gblur=sigma={sig}[_c];"
            "[_a]format=gbrp[_d];[_c]format=gbrp[_e];"
            "[_d][_e]blend=all_mode=screen:all_opacity=0.8,format=yuv420p[OUT]")


@fx("lofi_pixel", "filter", tags=("slop", "glitch", "hype"))
def lofi_pixel(ctx):
    blk = int(_i(ctx, 8, 5, 16))
    sw, sh = max(2, ctx.w // blk), max(2, ctx.h // blk)
    return (f"scale={sw}:{sh}:flags=area,scale={ctx.w}:{ctx.h}:flags=neighbor,"
            "lutrgb=r='floor(val/24)*24':g='floor(val/24)*24':b='floor(val/24)*24'")
