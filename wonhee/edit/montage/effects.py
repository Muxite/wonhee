"""Per-image motion/effect presets, expressed as ffmpeg filter fragments (a still has no motion of its own).

Each effect is (zoompan z, x, y expressions, post-filter chain). `N` is the clip's frame
count and `fps` the output rate; both are substituted when the filter graph is built.
"""

CENTER_X = "iw/2-(iw/zoom/2)"
CENTER_Y = "ih/2-(ih/zoom/2)"

# name -> (z, x, y, post)
EFFECTS: dict[str, tuple[str, str, str, str]] = {
    "zoom_in": ("1+0.25*on/{N}", CENTER_X, CENTER_Y, ""),
    "zoom_out": ("1.25-0.25*on/{N}", CENTER_X, CENTER_Y, ""),
    "pan_left": ("1.25", "(iw-iw/zoom)*(1-on/{N})", CENTER_Y, ""),
    "pan_right": ("1.25", "(iw-iw/zoom)*on/{N}", CENTER_Y, ""),
    "punch": ("1+0.3*(1-pow(1-min(1,on/{N}),4))", CENTER_X, CENTER_Y, ""),
    "shake": ("1.1", "(iw-iw/zoom)/2+sin(on*1.7)*iw*0.012",
              "(ih-ih/zoom)/2+cos(on*2.3)*ih*0.012", ""),
    "glitch": ("1+0.15*on/{N}", CENTER_X, CENTER_Y,
               "rgbashift=rh=-10:bh=10:gv=6:enable='lt(mod(t,0.6),0.12)'"),
    "flash": ("1+0.2*on/{N}", CENTER_X, CENTER_Y,
              "eq=brightness='0.7*max(0,1-t*5)':eval=frame"),
    "hue": ("1+0.2*on/{N}", CENTER_X, CENTER_Y, "hue=h='t*90'"),
}

STYLES: dict[str, list[str]] = {
    "calm": ["zoom_in", "zoom_out", "pan_left", "pan_right"],
    "hype": ["punch", "shake", "flash", "zoom_in", "glitch", "hue", "pan_left", "pan_right"],
    "glitch": ["glitch", "shake", "hue", "flash", "glitch"],
    "slop": sorted(EFFECTS),
}


def filter_for(effect: str, frames: int, fps: int, size: tuple[int, int]) -> str:
    """Filter chain turning one still frame into `frames` frames of the given effect."""
    if effect not in EFFECTS:
        raise ValueError(f"unknown effect {effect!r}; choose from {sorted(EFFECTS)}")
    z, x, y, post = EFFECTS[effect]
    def sub(s: str) -> str:
        return s.replace("{N}", str(frames))

    chain = (f"zoompan=z='{sub(z)}':x='{sub(x)}':y='{sub(y)}':d={frames}"
             f":s={size[0]}x{size[1]}:fps={fps}")
    if post:
        chain += "," + post
    return chain
