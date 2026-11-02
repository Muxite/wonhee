"""xfade transition pools per style: ffmpeg's built-in names plus the expression-based customs.

A style ("calm", "hype", "glitch", "slop") is a pool a seeded plan draws from, so one `--style` word picks a mood
without the caller naming every transition. `ALL` is the validation set for hand-edited plans.
"""
from .transitions_custom import CUSTOM

BUILTIN = [
    "fade", "wipeleft", "wiperight", "wipeup", "wipedown", "slideleft", "slideright",
    "slideup", "slidedown", "circlecrop", "rectcrop", "distance", "fadeblack",
    "fadewhite", "radial", "smoothleft", "smoothright", "smoothup", "smoothdown",
    "circleopen", "circleclose", "vertopen", "vertclose", "horzopen", "horzclose",
    "dissolve", "pixelize", "diagtl", "diagtr", "diagbl", "diagbr", "hlslice",
    "hrslice", "vuslice", "vdslice", "hblur", "fadegrays", "wipetl", "wipetr", "wipebl",
    "wipebr", "squeezeh", "squeezev", "zoomin", "fadefast", "fadeslow", "hlwind",
    "hrwind", "vuwind", "vdwind", "coverleft", "coverright", "coverup", "coverdown",
    "revealleft", "revealright", "revealup", "revealdown",
]
ALL = sorted(BUILTIN + list(CUSTOM))

HYPE_CUSTOM = ["cube_left", "spin", "card_flip", "circle_out_in"]
GLITCH_CUSTOM = ["block_dissolve", "glitch_slice"]


def _have(names: list[str]) -> list[str]:
    """Only customs that exist, so a pool never names a transition the renderer cannot build."""
    return [n for n in names if n in CUSTOM]


POOLS: dict[str, list[str]] = {
    "calm": ["fade", "dissolve", "smoothleft", "smoothright", "fadegrays"],
    "hype": ["wipeleft", "slideup", "circleopen", "pixelize", "radial", "zoomin",
             "fadeblack", "squeezeh", "slideleft"] + _have(HYPE_CUSTOM),
    "glitch": ["pixelize", "hblur", "distance", "squeezeh", "radial", "wiperight"] + _have(GLITCH_CUSTOM),
    "slop": ALL,
}
