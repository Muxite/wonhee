"""Generated overlays blended over the clip (kind "overlay").

Conventions:
- Register with @fx("name", "overlay", tags=("slop", ...)); the function takes `ctx` and returns a graph fragment.
- Graph fragment: [IN] (once), [OUT], private labels [_x] (rewritten unique). Source filters are allowed:
  "color=black:s={w}x{h}:r={fps}:d={dur:.3f},format=gbrp,geq=r='..':g='..':b='..'[_f];
   [IN]format=gbrp[_i];[_i][_f]blend=all_mode=screen,format=yuv420p[OUT]". Always give sources s=, r=, d= (d=ctx.dur) so
  they match the clip; geq T is seconds, N frame number, X/Y/W/H pixel position and size.
- Ctx fields: w, h, fps, frames, seed (int, per clip/slot; offset hashes with it), intensity (0.5..2), dur.
- GOTCHA: blend in RGB (format=gbrp on BOTH inputs, format=yuv420p after). Blending yuv420p applies screen/add
  to the U/V planes too and tints everything purple. all_mode=screen/addition/multiply. geq on yuv needs
  lum/cb/cr, so generate in gbrp (see _expr.rgb_geq). geq is slow (~40 cpu-s for 7s@270x480 when 3 channels
  are heavy): compute one gray `lum` geq at 1/4 res, colour with lutrgb, scale up (see fire).
- Commas in expressions go inside '...'. geq is slow: keep tests at 270x480. The renderer normalises the output.
- Helpers: fx/_expr.py (hash2, rgb_geq). Test: pytest tests/test_montage_fx.py -k <name>.
"""
from . import fx
from ._expr import hash2, rgb_geq  # noqa: F401


@fx("fire", "overlay", tags=("slop", "hype"))
def fire(ctx):
    # Flame intensity computed by one cheap gray geq at 1/4 res (geq is slow), coloured by lutrgb,
    # scaled up and screen-blended. Cells scroll upward (Y + T*speed); strongest in the bottom ~45%.
    k, w, h = ctx.seed % 997, max(2, ctx.w // 4), max(2, ctx.h // 4)
    cx, cy = f"floor((X+sin(Y/6+T*9+{k})*2)/2)", f"floor((Y+T*{50 * ctx.intensity:.0f})/2)+{k}"
    mask = "pow(max(0,(Y/H-0.45)/0.55),1.1)"
    i = f"clip(255*{mask}*(0.3+1.2*{hash2(cx, cy)})*(0.8+0.2*sin(T*37+{k}))*1.5-30,0,255)"
    ramp = "r='clip(val*2.4,0,255)':g='pow(val/255,1.8)*255':b='pow(val/255,5)*150'"
    return (f"color=black:s={w}x{h}:r={ctx.fps}:d={ctx.dur:.3f},format=gray,geq=lum='{i}',format=gbrp,"
            f"lutrgb={ramp},scale={ctx.w}:{ctx.h}:flags=bilinear[_f];"
            f"[IN]format=gbrp[_i];[_i][_f]blend=all_mode=screen,format=yuv420p[OUT]")


def _screen(ctx, ramp, src, flags="bilinear"):
    """Gray source `src` (a filter chain ending in a gray geq) -> lutrgb ramp -> scaled -> screen-blend over [IN]."""
    return (f"{src},format=gbrp,lutrgb={ramp},scale={ctx.w}:{ctx.h}:flags={flags}[_f];"
            f"[IN]format=gbrp[_i];[_i][_f]blend=all_mode=screen,format=yuv420p[OUT]")


def _gray_src(ctx, div, lum):
    w, h = max(2, ctx.w // div), max(2, ctx.h // div)
    return f"color=black:s={w}x{h}:r={ctx.fps}:d={ctx.dur:.3f},format=gray,geq=lum='{lum}'"


@fx("explosion", "overlay", tags=("slop", "hype"))
def explosion(ctx):
    # Shake the clip (crop jitter, decaying over 0.5s), then screen a white flash + expanding orange ring.
    k = ctx.seed % 997
    cx, cy = 0.3 + 0.4 * ((k * 7) % 10) / 10, 0.35 + 0.3 * ((k * 3) % 10) / 10
    m = max(2, int(ctx.w * 0.04 * ctx.intensity)) // 2 * 2
    fall = "max(0,1-t/0.5)"
    shake = (f"[IN]crop=w=iw-{2*m}:h=ih-{2*m}:x='{m}+{m}*{fall}*sin(t*97+{k})':"
             f"y='{m}+{m}*{fall}*cos(t*83+{k})',scale={ctx.w}:{ctx.h},setsar=1,format=gbrp[_i];")
    d = f"hypot(X-W*{cx:.2f},Y-H*{cy:.2f})"
    r = f"(T*W*1.1*{ctx.intensity:.2f}+4)"
    ring = f"exp(-pow(({d}-{r})/(W*0.07),2))*max(0,1-T/1.3)"
    core = f"if(lt({d},{r}),0.35*max(0,1-T/0.9),0)"
    flash = "max(0,1-T/0.25)"
    lum = f"clip(255*({ring}*1.3+{core}+{flash}*{1.2*ctx.intensity:.2f}),0,255)"
    ramp = "r='clip(val*1.3,0,255)':g='pow(val/255,1.5)*230':b='pow(val/255,4)*220'"
    return (shake + _gray_src(ctx, 4, lum) + f",format=gbrp,lutrgb={ramp},scale={ctx.w}:{ctx.h}:flags=bilinear[_f];"
            "[_i][_f]blend=all_mode=screen,format=yuv420p[OUT]")


@fx("lightning", "overlay", tags=("slop", "hype", "glitch"))
def lightning(ctx):
    # Random strobe pairs of frames: negate + bluish-white flash box, picked by hash of frame number.
    k = ctx.seed % 997
    p = min(0.5, 0.12 * ctx.intensity)
    hit = f"gt({hash2('floor(n/2)', str(k))},{1 - p:.3f})"
    neg = f"{hit}*eq(mod(n,2),0)"
    return (f"[IN]negate=enable='{neg}',"
            f"drawbox=x=0:y=0:w=iw:h=ih:color=0xaecbff@0.65:t=fill:enable='{hit}*eq(mod(n,2),1)'[OUT]")


@fx("confetti", "overlay", tags=("slop", "hype"))
def confetti(ctx):
    # Coloured square flakes falling at per-column speeds. Gray value encodes hue, lutrgb decodes it.
    k = ctx.seed % 997
    s = 5
    c = f"floor(X/{s})"
    speed = f"(40+90*{hash2(c + f'+{k}', '3.7')})*{ctx.intensity:.2f}"
    cy = f"floor((Y-T*{speed})/{s})"
    h = hash2(f"{c}+{k}", cy)
    hue = hash2(f"{c}*3.1", f"{cy}*1.7+{k}")
    lum = f"if(gt({h},{1 - 0.05 * ctx.intensity:.3f}),24+{hue}*230,0)"
    ramp = ("r='if(lt(val,12),0,40+215*abs(sin(val*0.045)))':"
            "g='if(lt(val,12),0,40+215*abs(sin(val*0.045+2.1)))':"
            "b='if(lt(val,12),0,40+215*abs(sin(val*0.045+4.2)))'")
    return _screen(ctx, ramp, _gray_src(ctx, 2, lum), flags="neighbor")


@fx("lens_flare", "overlay", tags=("slop",))
def lens_flare(ctx):
    # Moving bright blob + horizontal streak + two ghost blobs mirrored through the centre.
    k = ctx.seed % 997
    ph = (k % 10) / 10
    fx_ = f"(W*(0.15+0.7*mod(T/{max(ctx.dur, 0.1):.3f}+{ph:.1f},1)))"
    fy = f"(H*(0.2+0.15*sin(T*1.7+{k})))"

    def blob(cx, cy, sig, a):
        return f"{a}*exp(-(pow(X-{cx},2)+pow(Y-{cy},2))/(2*pow(W*{sig},2)))"
    gx, gy = f"(W-{fx_})", f"(H-{fy})"
    hx, hy = f"(W/2+({fx_}-W/2)*-0.5)", f"(H/2+({fy}-H/2)*-0.5)"
    streak = f"0.8*exp(-pow((Y-{fy})/2,2))*exp(-pow((X-{fx_})/(W*0.45),2))"
    v = f"{blob(fx_, fy, 0.09, 1.2)}+{streak}+{blob(gx, gy, 0.05, 0.5)}+{blob(hx, hy, 0.03, 0.6)}"
    lum = f"clip(255*({v})*{ctx.intensity:.2f},0,255)"
    ramp = "r='clip(val*1.15,0,255)':g='pow(val/255,1.3)*235':b='pow(val/255,1.9)*255'"
    return _screen(ctx, ramp, _gray_src(ctx, 4, lum))


@fx("scanlines_crt", "overlay", tags=("slop", "glitch"))
def scanlines_crt(ctx):
    # Dark horizontal lines + vignette + barrel distortion + light noise. Native filters only.
    a = min(0.85, 0.4 * ctx.intensity)
    k = -0.12 * ctx.intensity
    return (f"[IN]lenscorrection=k1={k:.3f}:k2={k:.3f},drawgrid=w=iw:h=3:t=1:c=black@{a:.2f},"
            "vignette=angle=PI/3.5,noise=alls=7:allf=t,eq=contrast=1.1:saturation=1.15[OUT]")


@fx("film_burn", "overlay", tags=("slop", "hype"))
def film_burn(ctx):
    # Orange-white blotch with a wobbly edge growing from a seed-chosen corner over the clip.
    k = ctx.seed % 997
    cx, cy = ("0", "0") if k % 4 == 0 else ("W", "0") if k % 4 == 1 else ("0", "H") if k % 4 == 2 else ("W", "H")
    d = f"(hypot(X-{cx},Y-{cy})+W*0.05*sin(X/6+T*4+{k})*sin(Y/8-T*3))"
    r = f"(W*1.5*pow(T/{max(ctx.dur, 0.1):.3f},0.9)*{ctx.intensity:.2f}+1)"
    lum = f"clip(255*({r}-{d})/({r}*0.3+1),0,255)*(0.85+0.15*sin(T*40+{k}))"
    ramp = "r='clip(val*2.4,0,255)':g='pow(val/255,1.7)*255':b='pow(val/255,4.5)*230'"
    return _screen(ctx, ramp, _gray_src(ctx, 4, lum))


@fx("sparkles", "overlay", tags=("slop",))
def sparkles(ctx):
    # Sparse 4-point stars on a hash grid, each twinkling with its own sin(T) phase/rate.
    k = ctx.seed % 997
    s = 14
    gx, gy = f"floor(X/{s})", f"floor(Y/{s})"
    h = hash2(f"{gx}+{k}", gy)
    h2 = hash2(f"{gx}*1.3", f"{gy}*0.7+{k}")
    dx, dy = f"abs(X-({gx}*{s}+{s // 2}))", f"abs(Y-({gy}*{s}+{s // 2}))"
    star = f"exp(-{dx}*{dy}/1.2)*max(0,1-hypot({dx},{dy})/{s // 2})"
    tw = f"pow(max(0,sin(T*(3+9*{h2})+{h}*40)),2)"
    lum = f"if(gt({h},{1 - 0.35 * min(ctx.intensity, 2) / 1.0 * 0.8:.3f}),255*{star}*{tw}*1.4,0)"
    ramp = "r='val':g='val*0.95':b='val*0.75'"
    return _screen(ctx, ramp, _gray_src(ctx, 2, lum))
