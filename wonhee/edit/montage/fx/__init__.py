"""Named ffmpeg effects for the montage engine: stackable per-clip or whole-video filtergraph fragments.

Ported from slopcut (`video/fx/`). Each effect is a pure function `build(ctx) -> str` returning filtergraph text;
nothing here runs ffmpeg or touches the DB, so the renderer decides how to run it and the tests can run any
fragment alone. Two fragment shapes:

- linear chain, no labels: "negate,eq=contrast=1.5" (`expand` wraps it as [src]chain[dst])
- graph with [IN] / [OUT] and private labels [_name]: "[IN]split[_a][_b];[_a]...[_c];[_b][_c]blend=...[OUT]".
  [IN] may appear once (split it yourself); source filters (color=, gradients=) are fine inside. `expand` rewrites
  "[_" to "[{prefix}_" so labels stay unique across clips and slots.

The renderer normalises size, fps, pix_fmt and frame count after the stack, so an effect may change them (but must
still emit at least a few frames; see tests). Expressions containing commas go inside '...'.

Categories (one module each): overlay, boom, color, warp, cut. `kind` (filter / overlay / cut) says how the
effect reads its input; `category` says which module holds it. List by category with `by_category()` or
`names(category=...)`.

Name overlap with `wonhee.edit.fx` (the pixel-array looks): `bloom` exists in both. This registry is a separate
namespace and is never merged into that module; `wonhee.edit.fx.bloom` works on numpy images (stills), the one
here is an ffmpeg filter fragment for video. Always import it as `wonhee.edit.montage.fx`. The other names there
(fisheye, vignette, grain, duotone, halftone, motion_blur) have no counterpart here.
"""
from dataclasses import dataclass
from typing import Callable

CATEGORIES = ("overlay", "boom", "color", "warp", "cut")
KINDS = ("filter", "overlay", "cut")


@dataclass(frozen=True)
class Ctx:
    w: int
    h: int
    fps: int
    frames: int                 # exact frame count of the clip
    seed: int = 0               # per (item, fx slot); offsets the hashes so replays are identical
    intensity: float = 1.0      # 0.5..2, scales effect parameters

    @property
    def dur(self) -> float:
        return self.frames / self.fps


@dataclass(frozen=True)
class Fx:
    name: str
    kind: str                   # "filter" | "overlay" | "cut"
    build: Callable[[Ctx], str]
    tags: frozenset             # style pools it joins: {"hype", "glitch", "slop", ...}
    category: str = ""          # overlay | boom | color | warp | cut: the module that defines it


FX: dict[str, Fx] = {}


def fx(name: str, kind: str, tags=("slop",)):
    """Decorator: register `build(ctx) -> fragment` as effect `name`."""
    if kind not in KINDS:
        raise ValueError(f"bad fx kind {kind!r}")

    def deco(f):
        category = f.__module__.rsplit(".", 1)[-1]
        if name in FX:
            raise ValueError(f"fx {name!r} registered twice")
        FX[name] = Fx(name, kind, f, frozenset(tags), category if category in CATEGORIES else "")
        return f
    return deco


def names(category: str | None = None) -> list[str]:
    """Sorted effect names, optionally only those of one category."""
    if category is not None and category not in CATEGORIES:
        raise ValueError(f"unknown fx category {category!r}; choose from {list(CATEGORIES)}")
    return sorted(n for n, e in FX.items() if category is None or e.category == category)


def by_category() -> dict[str, list[str]]:
    """{category: sorted names} for every category, in CATEGORIES order."""
    return {c: names(c) for c in CATEGORIES}


def get(name: str) -> Fx:
    """The registered effect, or ValueError listing the known names."""
    if name not in FX:
        raise ValueError(f"unknown fx {name!r}; choose from {names()}")
    return FX[name]


def slot_seed(plan_seed: int, item: int, slot: int) -> int:
    """Ctx.seed for fx slot `slot` of item `item` (item == len(items) for global fx)."""
    return (int(plan_seed) * 1_000_003 + item * 1009 + slot) % 2**31


def expand(name: str, ctx: Ctx, src: str, dst: str, prefix: str) -> str:
    """Filtergraph text turning label `src` into label `dst` through effect `name`."""
    frag = get(name).build(ctx)
    if "[IN]" in frag or "[OUT]" in frag:
        return frag.replace("[IN]", f"[{src}]").replace("[OUT]", f"[{dst}]").replace("[_", f"[{prefix}_")
    return f"[{src}]{frag}[{dst}]"


from . import boom, color, cut, overlay, warp  # noqa: E402,F401  (register on import)
