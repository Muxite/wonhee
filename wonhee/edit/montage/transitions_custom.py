"""Custom xfade transitions: CUSTOM maps name -> expr for `xfade=transition=custom:expr='...'`.

These exist because ffmpeg's built-in xfade names have no spin, card flip, shatter or explosion looks. Each one is a
per-pixel expression, so there is nothing to install and the result is deterministic.

Conventions:
- Add entries as `CUSTOM["name"] = <expr string>` (no quotes inside; the renderer wraps it in '...').
  Commas are fine. Don't use ':' or single quotes. st()/ld() are fine within a pixel (transitions run single-threaded).
- xfade P goes 1 -> 0 (P=1 shows A, P=0 shows B). X,Y,W,H are PER PLANE (yuv420p chroma planes are half size),
  so work in normalised coords (X/W, Y/H) or ratios, never absolute pixels. PLANE is 0 (Y), 1 (U), 2 (V).
- Reading pixels: a0/a1/a2(x,y) and b0/b1/b2(x,y) read plane 0/1/2 of A and B at plane coords. Use `sample`, which
  picks the plane and clamps coords. BLACK is the colour for "outside" (Y=16, UV=128). Output is a sample 0..255.
- Chroma planes: displace in normalised units (e.g. x + dx*W) so luma and chroma move together.
"""
BLACK = "if(eq(PLANE,0),16,128)"


def hash2(a: str, b: str) -> str:
    """Pseudo-random 0..1 from two expressions (sin hash). Pure, so safe to nest and repeat."""
    return f"abs(mod(sin(({a})*12.9898+({b})*78.233)*43758.5453,1))"


def sample(src: str, x: str, y: str, clamp: bool = True) -> str:
    """Sample plane-appropriate pixel of 'a' or 'b' at plane coords (x, y)."""
    if clamp:
        x, y = f"clip({x},0,W-1)", f"clip({y},0,H-1)"
    return (f"if(eq(PLANE,0),{src}0({x},{y}),if(eq(PLANE,1),{src}1({x},{y}),{src}2({x},{y})))")


CUSTOM: dict[str, str] = {}

# 32 square blocks across flip to B in hash order.
CUSTOM["block_dissolve"] = (
    f"if(gt({hash2('floor(X*32/W)', 'floor(Y*32/W)')},P),{sample('b', 'X', 'Y', False)},"
    f"{sample('a', 'X', 'Y', False)})")


# ---------------------------------------------------------------------------------------------------------------
# helpers. st()/ld() scratch variables are allowed: xfade shares one AVExpr across slice threads, so the
# renderer runs every transition with -filter_complex_threads 1 (and many transitions in parallel instead).

_PX = "((X/W-0.5)*W/H)"  # aspect-correct centred coords (height units)
_PY = "(Y/H-0.5)"


def _dim(v: str, f: str) -> str:
    """Darken sample expression v by factor f (luma scales, chroma pulls toward 128)."""
    return f"if(eq(PLANE,0),({v})*({f}),128+(({v})-128)*({f}))"


def _uv(px: str, py: str) -> tuple[str, str]:
    """Aspect-space centred coords back to normalised 0..1 texture coords."""
    return f"(({px})*H/W+0.5)", f"(({py})+0.5)"


def _rot(src: str, ang: str, scale: str, bg: str | None) -> str:
    """Sample src rotated by ang (radians) and zoomed by scale about the centre, aspect-correct.
    scale>1 magnifies. bg=None clamps at the border, else out-of-bounds pixels become bg."""
    c, s, k = f"cos({ang})", f"sin({ang})", f"max({scale},0.001)"
    u, v = _uv(f"({_PX}*{c}+{_PY}*{s})/{k}", f"({_PY}*{c}-{_PX}*{s})/{k}")
    px = sample(src, f"{u}*W", f"{v}*H")
    if bg is None:
        return px
    return f"if(between({u},0,1)*between({v},0,1),{px},{bg})"


def _warp_mix(u: str, v: str) -> str:
    """Crossfade of A and B both sampled at warped texture coords (u, v)."""
    return f"{sample('a', f'{u}*W', f'{v}*H')}*P+{sample('b', f'{u}*W', f'{v}*H')}*(1-P)"


# 1. A squeezes into the left strip [0, W*P] and darkens, B grows in from the right.
_s = "max(W*P,1)"
_t = "max(W*(1-P),1)"
CUSTOM["cube_left"] = (
    f"if(lt(X,W*P),{_dim(sample('a', f'X*W/{_s}', 'Y'), '0.35+0.65*P')},"
    f"{sample('b', f'(X-W*P)*W/{_t}', 'Y')})")

# 2. vertical version.
_s = "max(H*P,1)"
_t = "max(H*(1-P),1)"
CUSTOM["cube_up"] = (
    f"if(lt(Y,H*P),{_dim(sample('a', 'X', f'Y*H/{_s}'), '0.35+0.65*P')},"
    f"{sample('b', 'X', f'(Y-H*P)*H/{_t}')})")

# 3. A spins a full turn about the centre while shrinking away, revealing B.
CUSTOM["spin"] = _rot("a", "(1-P)*2*PI", "P", sample("b", "X", "Y"))

# 4. A rotates out 90 degrees over the first half, B rotates in over the second.
CUSTOM["rotate_swap"] = (
    f"if(gt(P,0.5),{_rot('a', '(1-P)*PI', 'P', BLACK)},"
    f"{_rot('b', '-P*PI', '1-P', BLACK)})")

# 5. Zoom into the centre with rotation, swap at the midpoint, zoom back out on B.
CUSTOM["zoom_spin"] = (
    f"if(gt(P,0.5),{_rot('a', '(1-P)*PI', '1+(1-P)*6', None)},"
    f"{_rot('b', '-P*PI', '1+P*6', None)})")

# 6. Twirl distortion peaking mid-transition, crossfading A to B.
_r = f"hypot({_PX},{_PY})"
_th = f"(6*sin(PI*P)*pow(max(1-{_r}/0.8,0),2))"
_u, _v = _uv(f"{_PX}*cos({_th})+{_PY}*sin({_th})", f"{_PY}*cos({_th})-{_PX}*sin({_th})")
CUSTOM["swirl"] = _warp_mix(_u, _v)

# 7. Circle closes on A down to black, then opens on B.
CUSTOM["circle_out_in"] = (
    f"if(gt(P,0.5),if(lt({_r},0.75*(2*P-1)),{sample('a', 'X', 'Y', False)},{BLACK}),"
    f"if(lt({_r},0.75*(1-2*P)),{sample('b', 'X', 'Y', False)},{BLACK}))")

# 9. Horizontal bands slide sideways, each flipping A->B at its own time.
_band = "floor(Y/H*24)"
_off = f"((2*{hash2(_band, '3.7')}-1)*0.3*sin(PI*P)*W)"
CUSTOM["glitch_slice"] = (
    f"if(gt({hash2(_band, '1.3')},P),{sample('b', f'X+{_off}', 'Y')},{sample('a', f'X+{_off}', 'Y')})")

# 10. Horizontal squash by |cos| showing A, then B.
_c = "max(abs(cos(PI*(1-P))),0.01)"
_cu = f"((X/W-0.5)/{_c}+0.5)"
CUSTOM["card_flip"] = (
    f"if(between({_cu},0,1),if(gt(P,0.5),{sample('a', f'{_cu}*W', 'Y')},{sample('b', f'{_cu}*W', 'Y')}),{BLACK})")


# 11. A collapses vertically to a bright line, then B expands from it.
def _tv(src: str, h: str, t: str) -> str:
    hh = f"max({h},0.008)"
    v = f"((Y/H-0.5)/{hh}+0.5)"
    px = sample(src, "X", f"{v}*H")
    return f"if(between({v},0,1),if(eq(PLANE,0),min(235,{px}+230*pow({t},4)),{px}),{BLACK})"


CUSTOM["tv_off"] = f"if(gt(P,0.5),{_tv('a', '2*P-1', '2-2*P')},{_tv('b', '1-2*P', '2*P')})"

# 12. Radial sine ripple displacement peaking mid-transition, with crossfade.
_r2 = f"(hypot({_PX},{_PY})+0.0001)"
_k = f"(1+0.04*sin(PI*P)*sin({_r2}*45-(1-P)*25)/{_r2})"
_u, _v = _uv(f"{_PX}*{_k}", f"{_PY}*{_k}")
CUSTOM["ripple"] = _warp_mix(_u, _v)


# ---------------------------------------------------------------------------------------------------------------
# Explosion-themed transitions. Colours are per plane (Y, U, V).
_WHITE = "if(eq(PLANE,0),235,128)"
_ORANGE = "if(eq(PLANE,0),154,if(eq(PLANE,1),58,187))"
_HOT = "if(eq(PLANE,0),205,if(eq(PLANE,1),70,150))"


def _lerp(v: str, c: str, w: str) -> str:
    """Mix pixel expression v toward colour c by amount w (0..1)."""
    return f"(({v})+(({c})-({v}))*clip({w},0,1))"


def _zoomblur(src: str, e: str, amt: str) -> str:
    """3-tap radial zoom blur of src about the centre; taps pulled toward centre by up to amt*e."""
    def tap(f: str) -> str:
        s = f"(1-{amt}*({e})*{f})"
        return sample(src, f"((X/W-0.5)*{s}+0.5)*W", f"((Y/H-0.5)*{s}+0.5)*H")
    return f"(({tap('0')}+{tap('0.5')}+{tap('1')})/3)"


# 1. A zoom-blurs and blows out to white at P=0.5, B fades in from white with a settling zoom blur.
_ea = "(2-2*P)"        # 0..1 over first half
_eb = "(2*P)"          # 1..0 over second half
CUSTOM["boom_cut"] = (
    f"if(gt(P,0.5),{_lerp(_zoomblur('a', _ea, '0.45'), _WHITE, f'pow({_ea},2)')},"
    f"{_lerp(_zoomblur('b', _eb, '0.45'), _WHITE, f'pow({_eb},2)')})")


# 2. A shatters into 8x14 cells that fly outward and fall, revealing B. Cell offsets depend on the SOURCE cell;
# the source of an output pixel is found by two fixed-point iterations, then checked for self-consistency.
# Uses st()/ld() (safe: the renderer runs transitions single-threaded); inlined it was a 25k-char expr
# re-deriving the same hashes hundreds of times per pixel.
def _shatter() -> str:
    e = "(1-P)"

    def off() -> str:
        """Offset of cell (ld(2), ld(3)) -> vars 4, 5."""
        i, j = "ld(2)", "ld(3)"
        h1 = hash2(f"({i})+0.37", f"({j})+1.91")
        h2 = hash2(f"({i})+5.13", f"({j})+2.71")
        h3 = hash2(f"({i})+9.77", f"({j})+7.31")
        k = f"{e}*(0.4+2.2*{h1})"
        return (f"st(4,((({i})+0.5)/8-0.5)*{k}+({h2}-0.5)*{e}*0.4);"
                f"st(5,((({j})+0.5)/14-0.5)*{k}+{e}*{e}*(0.3+1.6*{h3}))")

    u2, v2 = "ld(6)", "ld(7)"
    ok = f"eq(floor({u2}*8),ld(2))*eq(floor({v2}*14),ld(3))*between({u2},0,1)*between({v2},0,1)"
    a = sample("a", f"{u2}*W", f"{v2}*H")
    return ("st(0,X/W);st(1,Y/H);st(2,floor(ld(0)*8));st(3,floor(ld(1)*14));" + off() + ";"
            "st(2,floor((ld(0)-ld(4))*8));st(3,floor((ld(1)-ld(5))*14));" + off() + ";"
            "st(6,ld(0)-ld(4));st(7,ld(1)-ld(5));"
            f"if({ok},{_dim(a, f'1-0.5*{e}')},{sample('b', 'X', 'Y', False)})")


CUSTOM["shatter"] = _shatter()


# 3. Noise-threshold burn: A is eaten away (to B) where noise < threshold, with a glowing orange/yellow edge band.
def _burn() -> str:
    n = (f"(0.2*(1-Y/H)+0.8*(0.6*{hash2('floor(X/W*10)', 'floor(Y/H*18)')}"
         f"+0.4*{hash2('floor(X/W*45)+3', 'floor(Y/H*80)')}))")
    th = "((1-P)*1.3)"
    wd = f"(0.14*min(1,{th}*6)+0.0001)"
    d = f"(({n}-{th})/{wd})"
    a = sample("a", "X", "Y", False)
    edge = _lerp(_HOT, _ORANGE, f"{d}*1.6")
    char = _dim(a, f"0.25+0.75*{d}")
    glow = _lerp(char, edge, f"1-pow({d},3)")
    return f"if(lt({n},{th}),{sample('b', 'X', 'Y', False)},if(lt({n},{th}+{wd}),{glow},{a}))"


CUSTOM["burn_away"] = _burn()

# 4. A is sucked into the centre while twisting, white flash at P=0.5, B bursts out from the centre.
_qa = "(2*P-1)"   # 1..0 over first half
_qb = "(1-2*P)"   # 0..1 over second half
CUSTOM["implode"] = (
    f"if(gt(P,0.5),"
    f"{_lerp(_rot('a', f'(1-{_qa})*PI*1.5', f'0.03+0.97*pow({_qa},2)', BLACK), _WHITE, f'pow(1-{_qa},3)')},"
    f"{_lerp(_rot('b', f'-(1-{_qb})*PI*1.5', f'0.03+0.97*pow({_qb},0.5)', BLACK), _WHITE, f'pow(1-{_qb},3)')})")


# 5. Shockwave ring expands from the centre: inside is B, outside is shaking A, ring is orange/white with
# radial displacement.
def _blast() -> str:
    r = f"hypot({_PX},{_PY})"
    R = "((1-P)*1.25)"
    wd = "(0.09*min(1,(1-P)*8)+0.0001)"
    d = f"({r}-{R})"
    shx = f"((({hash2('floor(P*37)', '1.1')})-0.5)*0.03*min(1,(1-P)*6))"
    shy = f"((({hash2('floor(P*37)', '2.3')})-0.5)*0.03*min(1,(1-P)*6))"
    inring = f"lt(abs({d}),{wd})"
    k = f"(1+0.25*cos({d}/{wd}*PI/2)*min(1,(1-P)*6))"
    u, v = _uv(f"{_PX}*{k}", f"{_PY}*{k}")
    ua, va = f"({u}+{shx})", f"({v}+{shy})"
    a_ring = sample("a", f"{ua}*W", f"{va}*H")
    b_ring = sample("b", f"{u}*W", f"{v}*H")
    g = f"(1-abs({d})/{wd})"
    base = f"if(lt({d},0),{b_ring},{a_ring})"
    ring = _lerp(_lerp(base, _ORANGE, f"{g}*0.85"), _WHITE, f"pow({g},3)")
    a_out = sample("a", f"X+{shx}*W", f"Y+{shy}*H")
    return f"if({inring},{ring},if(lt({d},0),{sample('b', 'X', 'Y', False)},{a_out}))"


CUSTOM["blast_wave"] = _blast()
