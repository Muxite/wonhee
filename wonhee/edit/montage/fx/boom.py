"""Explosion-themed overlays (the "boom" category): gray geq at low res -> lutrgb ramp -> scale up -> blend in
gbrp. Fragment convention is in overlay.py. No st()/ld()."""
from . import Ctx, fx
from ._expr import hash2, rgb_geq  # noqa: F401

_TAGS = ("slop", "hype")
_PI2 = "6.2831853"


def _pre(ctx, amp=0.0, fall=0.6, extra="", k=0):
    """[IN] -> [_i] gbrp, optionally with crop-jitter shake decaying over `fall` seconds."""
    m = max(0, int(ctx.w * amp * ctx.intensity)) // 2 * 2
    if m:
        f = f"max(0,1-t/{fall})"
        s = (f"crop=w=iw-{2*m}:h=ih-{2*m}:x='{m}+{m}*{f}*sin(t*97+{k})':y='{m}+{m}*{f}*cos(t*83+{k})',"
             f"scale={ctx.w}:{ctx.h},setsar=1,")
    else:
        s = ""
    return f"[IN]{s}{extra}format=gbrp[_i0];"


def _layers(ctx, pre, layers):
    """layers: [(div, lum, ramp, mode)] -> chained blends over [_i0]; returns the full fragment."""
    out, cur = pre, "_i0"
    for n, (div, lum, ramp, mode) in enumerate(layers):
        w, h = max(2, ctx.w // div), max(2, ctx.h // div)
        out += (f"color=black:s={w}x{h}:r={ctx.fps}:d={ctx.dur:.3f},format=gray,geq=lum='{lum}',format=gbrp,"
                f"lutrgb={ramp},scale={ctx.w}:{ctx.h}:flags=bilinear[_l{n}];")
        nxt = "OUT" if n == len(layers) - 1 else f"_b{n}"
        tail = ",format=yuv420p" if nxt == "OUT" else ""
        out += f"[{cur}][_l{n}]blend=all_mode={mode}{tail}[{nxt}];"
        cur = nxt
    return out.rstrip(";")


def _centre(ctx, lo=0.3, hi=0.7):
    k = ctx.seed % 997
    return lo + (hi - lo) * ((k * 7) % 10) / 10, 0.3 + 0.3 * ((k * 3) % 10) / 10, k


@fx("nuke", "overlay", tags=("slop", "hype"))
def nuke(ctx):
    I = ctx.intensity
    k = ctx.seed % 997
    flash = "if(lt(T,0.2),1,max(0,1-(T-0.2)/0.35))"
    s = "clip((T-0.15)/1.3,0,1)"
    cx = "(W/2)"
    dy = f"(H*(1.02-0.5*pow({s},0.7)))"           # dome centre rises
    R = f"(W*0.36*pow({s},0.55)*{min(I, 1.5):.2f}+1)"
    noise = f"(0.8+0.2*sin(X/2.5+T*6+{k})*sin(Y/3-T*4)+0.1*sin(X/1.3-T*9))"
    dome = f"clip((1-hypot((X-{cx})/{R},(Y-{dy})/({R}*0.7)))*2.4,0,1)*lt(Y,{dy}+{R}*0.25)"
    stem = (f"clip(1-abs(X-{cx})/(W*0.055+W*0.03*(Y/H)),0,1)*gt(Y,{dy})*clip({s}*8,0,1)*"
            f"(0.75+0.25*sin(Y/2+T*8))")
    skirt = f"clip(1-hypot((X-{cx})/(W*0.5*{s}+1),(Y-H)/(H*0.06+1)),0,1)*0.8"
    lum = f"clip(255*(max(max({dome},{stem}),{skirt})*{noise}*1.15+{flash}*1.5),0,255)"
    ramp = "r='clip(val*1.8,0,255)':g='pow(val/255,1.7)*255':b='pow(val/255,5)*255'"
    tint = "255*min(1,T/0.7)*if(lt(T,0.2),0,1)"
    tramp = "r='255':g='255-val*0.38':b='255-val*0.8'"
    return _layers(ctx, _pre(ctx, 0.05, 1.5, k=k),
                   [(2, tint, tramp, "multiply"), (4, lum, ramp, "screen")])


@fx("shockwave", "overlay", tags=("slop", "hype", "glitch"))
def shockwave(ctx):
    I = ctx.intensity
    cx, cy, k = _centre(ctx)
    d = f"hypot(X-W*{cx:.2f},Y-H*{cy:.2f})"
    r = f"(T*W*1.25+3)"
    ring = f"exp(-pow(({d}-{r})/(W*0.018),2))*max(0,1-T/1.0)"
    halo = f"exp(-pow(({d}-{r}+W*0.03)/(W*0.06),2))*0.35*max(0,1-T/0.8)"
    flash = "max(0,1-T/0.18)*0.9"
    lum = f"clip(255*(({ring}+{halo})*{min(I, 1.5):.2f}+{flash}),0,255)"
    ramp = "r='clip(val*0.9,0,255)':g='clip(val*1.0,0,255)':b='clip(val*1.25,0,255)'"
    z = f"(0.12*{I:.2f}*exp(-pow((t-0.12)/0.18,2)))"
    pulse = (f"scale=w='trunc(iw*(1+{z})/2)*2':h='trunc(ih*(1+{z})/2)*2':eval=frame,"
             f"crop={ctx.w}:{ctx.h},setsar=1,rgbashift=rh=-5:bh=5:enable='lt(t,0.7)',")
    return _layers(ctx, _pre(ctx, 0.0, extra=pulse), [(4, lum, ramp, "screen")])


@fx("fireball", "overlay", tags=("slop", "hype"))
def fireball(ctx):
    I = ctx.intensity
    cx, cy, k = _centre(ctx, 0.35, 0.65)
    a = f"atan2(Y-H*{cy:.2f},X-W*{cx:.2f})"
    R = (f"(W*0.5*{min(I, 1.6):.2f}*pow(min(1,T/1.1),0.6)*(1+0.14*sin({a}*5+T*8+{k})"
         f"+0.08*sin({a}*11-T*13)+0.05*sin({a}*23+T*21)))")
    d = f"hypot(X-W*{cx:.2f},Y-H*{cy:.2f})"
    life = f"clip((T/{ctx.dur:.2f}-0.7)/0.3,0,1)"
    v = f"clip(({R}-{d})/({R}*0.55+1)*1.5,0,1)*(1-0.6*{life})"
    flick = f"(0.85+0.15*sin(T*41+{k})+0.08*sin(T*67))"
    lum = f"clip(255*{v}*{flick}*(0.9+0.2*sin(X/2+T*10)*sin(Y/3-T*7)),0,255)"
    ramp = "r='clip(val*2.2,0,255)':g='pow(val/255,2.0)*255':b='pow(val/255,6)*255'"
    return _layers(ctx, _pre(ctx, 0.02, 0.8, k=k), [(4, lum, ramp, "screen")])


@fx("debris", "overlay", tags=("slop", "hype"))
def debris(ctx):
    I = ctx.intensity
    cx, cy, k = _centre(ctx)
    nb = 26
    dx, dy = f"(X-W*{cx:.2f})", f"(Y-H*{cy:.2f}-W*0.9*T*T)"
    b = f"floor((atan2({dy},{dx})+3.14159)/{_PI2}*{nb})"
    h1, h2, h3 = hash2(f"{b}", f"{k}+1"), hash2(f"{b}*1.7", f"{k}+5"), hash2(f"{b}*2.3", f"{k}+9")
    ang = f"(({b}+0.2+0.6*{h2})/{nb}*{_PI2}-3.14159)"
    rr = f"((0.35+1.3*{h1})*T*W*{min(I, 1.6):.2f}+2)"
    sz = f"(W*(0.02+0.035*{h3}))"
    ins = f"lt(hypot({dx}-{rr}*cos({ang}),{dy}-{rr}*sin({ang})),{sz})"
    lum = f"255*(1-0.92*{ins}*gt(T,0.03))"
    ramp = "r='val':g='val*0.9':b='val*0.78'"
    flash = "255*max(0,1-T/0.12)*0.7"
    fr = "r='val':g='val*0.9':b='val*0.6'"
    return _layers(ctx, _pre(ctx, 0.025, 0.5, k=k),
                   [(3, lum, ramp, "multiply"), (4, flash, fr, "screen")])


@fx("smoke_plume", "overlay", tags=("slop", "hype"))
def smoke_plume(ctx):
    I = ctx.intensity
    cx, cy, k = _centre(ctx, 0.4, 0.6)
    g = f"min(1,T/({ctx.dur:.2f}*0.75))"
    spread = f"(W*(0.12+0.55*(1-Y/H)*{g}+0.2*{g}))"
    m = f"exp(-pow((X-W*{cx:.2f}+W*0.08*sin(Y/H*5-T*2+{k}))/{spread},2))"
    front = f"clip((Y/H-(1.05-{g}*1.15))*7+0.2,0,1)"
    n = (f"(0.55+0.2*sin(X/4+Y/6-T*3+{k})+0.15*sin(X/7-Y/3+T*4)+0.1*sin(X/2.3+Y/2.9-T*7))")
    a = f"clip(({m}*{front}*{n}*0.95+0.3*{g})*{min(I, 1.6):.2f},0,0.93)"
    lum = f"255*(1-{a})"
    sc = f"255*{a}*0.28"
    return _layers(ctx, _pre(ctx, 0.0, k=k),
                   [(4, lum, "r='val*0.96':g='val':b='val*1.02'", "multiply"),
                    (4, sc, "r='val*0.9':g='val*0.88':b='val*0.85'", "screen")])


@fx("sparks_burst", "overlay", tags=("slop", "hype"))
def sparks_burst(ctx):
    I = ctx.intensity
    cx, cy, k = _centre(ctx)
    nb = 64
    dx, dy = f"(X-W*{cx:.2f})", f"(Y-H*{cy:.2f}-W*0.7*T*T)"
    ba = f"((atan2({dy},{dx})+3.14159)/{_PI2}*{nb})"
    b = f"floor({ba})"
    h1, h2, h3 = hash2(b, f"{k}+2"), hash2(f"{b}*1.9", f"{k}+6"), hash2(f"{b}*3.1", f"{k}+8")
    perp = f"abs(hypot({dx},{dy})*({ba}-{b}-0.2-0.6*{h2})/{nb}*{_PI2})"
    d = f"hypot({dx},{dy})"
    rr = f"((0.3+1.2*{h1})*T*W*{min(I, 1.6):.2f}+2)"
    ln = f"(W*(0.05+0.07*{h3}))"
    on = f"lt({perp},1.1)*gt({d},{rr}-{ln})*lt({d},{rr})*gt({h3},0.25)"
    fade = f"max(0,1-T/(0.9+1.2*{h2}))"
    lum = f"255*{on}*({fade})*(0.6+0.4*({d}-({rr}-{ln}))/{ln})*1.3"
    ramp = "r='clip(val*1.3,0,255)':g='clip(val*0.95,0,255)':b='pow(val/255,2.2)*255'"
    return _layers(ctx, _pre(ctx, 0.01, 0.3, k=k), [(3, lum, ramp, "screen")])


@fx("screen_shatter", "overlay", tags=("slop", "hype", "glitch"))
def screen_shatter(ctx):
    I = ctx.intensity
    cx, cy, k = _centre(ctx, 0.3, 0.7)
    nb = 16
    dx, dy = f"(X-W*{cx:.2f})", f"(Y-H*{cy:.2f})"
    d = f"hypot({dx},{dy})"
    ba = f"((atan2({dy},{dx})+3.14159)/{_PI2}*{nb})"
    b = f"floor({ba})"
    h1, h2 = hash2(b, f"{k}+3"), hash2(f"{b}*1.3", f"{k}+7")
    t0 = "gte(T,0.1)"
    reveal = f"lt({d},W*10*(T-0.1)+1)"
    ray = (f"lt(abs({d}*({ba}-{b}-0.15-0.7*{h2})/{nb}*{_PI2}),0.9)*lt({d},W*(0.35+0.9*{h1}))")
    rings = "+".join(
        f"lt(abs({d}-W*{r}*(0.9+0.2*{hash2(b + f'+{j * 5}', f'{k}+{j}')})),1)*gt({hash2(b + f'*0.7+{j}', f'{k}+11')},0.35)"
        for j, r in enumerate((0.1, 0.22, 0.38)))
    crack = f"min(1,{ray}+{rings})*{t0}*{reveal}"
    flash = "if(lt(T,0.1),0,max(0,1-(T-0.1)/0.14))*0.9"
    lum = f"clip(255*({crack}*{min(I, 1.5):.2f}+{flash}),0,255)"
    ramp = "r='val':g='val':b='clip(val*1.05,0,255)'"
    band = f"floor({d}/(W*0.14))"
    shard = hash2(f"{b}+{band}*37", f"{k}+13")
    dark = f"255*(1-{t0}*(0.1+0.3*{shard}*{min(I, 1.5):.2f}))"
    off = max(2, int(ctx.w * 0.012 * I)) // 2 * 2
    pre = _pre(ctx, 0.0, extra=(f"crop=w=iw-{2*off}:h=ih-{2*off}:x='{off}+{off}*gt(t,0.1)':y='{off}-{off}*gt(t,0.1)',"
                                f"scale={ctx.w}:{ctx.h},setsar=1,rgbashift=rh=-3:bh=3:enable='gt(t,0.1)',"))
    return _layers(ctx, pre, [(8, dark, "r='val':g='val':b='val'", "multiply"),
                              (3, lum, ramp, "screen")])
