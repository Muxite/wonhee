"""Build and validate the render plan: a plain dict you can dump, hand-edit and re-render exactly.

The plan is the contract between choosing and rendering. `build_plan` makes every random choice (effects,
transitions, durations, fx picks) from one seed and writes it down, so the same seed gives the same plan and the
render itself needs no randomness. Hand edits to the JSON are checked by `validate_plan` before ffmpeg runs.

Fx names are checked against `montage/fx` and sound names against `edit/sfx`, so a plan naming either one that does
not exist fails here, at plan time, before any ffmpeg runs.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from wonhee.edit.text import TextSpec

from ..sfx.sounds import SOUNDS
from .effects import EFFECTS, STYLES
from .fx import FX
from .transitions import ALL as ALL_TRANSITIONS, POOLS

MIN_CLIP = 0.4
FX_DEPTH = {"calm": (0, 0), "hype": (0, 1), "glitch": (1, 2), "slop": (2, 4)}

# Caption keys the renderer owns (timing and layering); every other key is a wonhee.edit.text.TextSpec field.
CAPTION_TIMING = ("start", "end", "fade", "layer")
LAYERS = ("under", "top")


@dataclass
class Still:
    """One image going into the montage, with optional per-item overrides
    (duration, effect, text, fx, sfx: the same keys a plan item carries)."""
    path: Path
    overrides: dict = field(default_factory=dict)


def _check_fx(names, where: str = "") -> list[str]:
    names = list(names or [])
    bad = [n for n in names if n not in FX]
    if bad:
        raise ValueError(f"{where}unknown fx {bad[0]!r}; choose from {sorted(FX)}")
    return names


def _check_sounds(names, where: str = "") -> list[str]:
    names = list(names or [])
    bad = [n for n in names if n not in SOUNDS]
    if bad:
        raise ValueError(f"{where}unknown sound {bad[0]!r}; choose from {sorted(SOUNDS)}")
    return names


def normalise_caption(spec) -> dict | None:
    """A bare string or a dict -> a checked caption dict, or None for no caption.

    Dict keys are `content`, the timing keys (start, end, fade, layer) and any `TextSpec` field. The text itself
    is drawn by wonhee.edit.text, so presets, fonts and safe areas match every other wonhee caption.
    """
    if spec in (None, "", False):
        return None
    if isinstance(spec, str):
        spec = {"content": spec}
    spec = dict(spec)
    if not str(spec.get("content", "")).strip():
        return None
    spec.setdefault("position", "bottom")
    if spec.get("layer", "under") not in LAYERS:
        raise ValueError(f"caption layer {spec['layer']!r}: choose from {LAYERS}")
    if spec.get("fade") is not None and float(spec["fade"]) < 0:
        raise ValueError("caption fade must be >= 0 seconds")
    to_text_spec(spec)   # raises ValueError on a bad preset/role/box/etc.
    return spec


def to_text_spec(caption: dict) -> TextSpec:
    fields = {k: v for k, v in caption.items() if k not in CAPTION_TIMING}
    for key in ("colours", "at"):   # JSON has lists; TextSpec is frozen and wants tuples
        if fields.get(key) is not None:
            fields[key] = tuple(fields[key])
    try:
        return TextSpec(**fields)
    except TypeError as e:
        raise ValueError(f"bad caption: {e}") from None


def build_plan(items: list[Still], *, length: float | None, beat: float | None,
               size: tuple[int, int], fps: int, style: str, seed: int,
               texts: list[str] | None = None, fx: list[str] | None = None,
               fx_depth: int | None = None, global_fx: list[str] | None = None,
               sfx: bool = True, sfx_gain: float = 1.0) -> dict:
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}; choose from {sorted(STYLES)}")
    if (length is None) == (beat is None):
        raise ValueError("give exactly one of length or beat")
    if not items:
        raise ValueError("a montage needs at least one image")
    rng = random.Random(seed)
    fx_rng = random.Random(f"{seed}-fx")   # separate stream: effect/transition picks stay stable
    forced, global_fx = _check_fx(fx), _check_fx(global_fx)
    pool = sorted(n for n, f in FX.items() if style in f.tags)
    if fx_depth is not None and not pool:
        pool = sorted(FX)
    lo, hi = FX_DEPTH[style] if fx_depth is None else (fx_depth, fx_depth)
    n = len(items)

    # Transitions first: each overlap eats time, so clip lengths are solved afterwards.
    span = (length or beat * n) / n
    transitions = [
        {"type": rng.choice(POOLS[style]), "duration": round(min(rng.uniform(0.3, 0.5), span * 0.4), 3)}
        for _ in range(n - 1)
    ]
    overlap = sum(t["duration"] for t in transitions)

    # Per-item duration overrides are honoured; the rest share what remains.
    fixed = {i: float(it.overrides["duration"]) for i, it in enumerate(items) if "duration" in it.overrides}
    if beat is not None:
        free = {i: beat for i in range(n) if i not in fixed}
    else:
        remaining = length + overlap - sum(fixed.values())
        free_ids = [i for i in range(n) if i not in fixed]
        if free_ids and remaining / len(free_ids) < MIN_CLIP:
            raise ValueError(f"length {length}s is too short for {n} images (clips would be under {MIN_CLIP}s)")
        free = {i: remaining / len(free_ids) for i in free_ids}
    durations = {**free, **fixed}

    out_items = []
    for i, it in enumerate(items):
        effect = it.overrides.get("effect") or rng.choice(STYLES[style])
        if effect not in EFFECTS:
            raise ValueError(f"{Path(it.path).name}: unknown effect {effect!r}")
        cli_text = texts[i] if texts and i < len(texts) else None
        text = normalise_caption(cli_text or it.overrides.get("text"))
        picks = fx_rng.sample(pool, min(fx_rng.randint(lo, hi), len(pool)))
        if "fx" in it.overrides:
            picks = _check_fx(it.overrides["fx"], f"{Path(it.path).name}: ")
        sounds = (_check_sounds(it.overrides.get("sfx"), f"{Path(it.path).name}: ")
                  if "sfx" in it.overrides else None)
        clip_fx = picks + [f for f in forced if f not in picks]
        out_items.append({
            "src": str(it.path),
            "effect": effect,
            "duration": round(max(MIN_CLIP, durations[i]), 3),
            **({"fx": clip_fx} if clip_fx else {}),
            **({"sfx": sounds} if sounds is not None else {}),
            **({"text": text} if text else {}),
            **({"transition": transitions[i]} if i < n - 1 else {}),
        })
    if style == "slop":   # slop always gets one random whole-video filter
        filters = sorted(n for n, f in FX.items() if f.kind == "filter" and style in f.tags)
        if filters:
            global_fx = global_fx + [fx_rng.choice(filters)]
    plan = {"size": list(size), "fps": fps, "seed": seed, "style": style, "sfx": sfx, "items": out_items}
    if sfx_gain != 1.0:
        plan["sfx_gain"] = sfx_gain
    if global_fx:
        plan["global_fx"] = global_fx
    return plan


def total_length(plan: dict) -> float:
    items = plan["items"]
    return sum(i["duration"] for i in items) - sum(i["transition"]["duration"] for i in items[:-1])


def validate_plan(plan: dict) -> dict:
    """Check a (possibly hand-edited) plan and normalise captions in place. Raises ValueError."""
    if not plan.get("items"):
        raise ValueError("plan has no items")
    for key in ("size", "fps"):
        if key not in plan:
            raise ValueError(f"plan is missing {key!r}")
    items = plan["items"]
    for i, it in enumerate(items):
        if it.get("effect") not in EFFECTS:
            raise ValueError(f"unknown effect {it.get('effect')!r} in plan")
        if not it.get("duration", 0) > 0:
            raise ValueError(f"plan item {i}: duration must be > 0")
        if "text" in it:
            it["text"] = normalise_caption(it["text"])
            if it["text"] is None:
                del it["text"]
        _check_fx(it.get("fx"), "plan: ")
        _check_sounds(it.get("sfx"), "plan: ")
        if "transition" in it and it["transition"]["type"] not in ALL_TRANSITIONS:
            raise ValueError(f"unknown transition {it['transition']['type']!r} in plan")
        if i < len(items) - 1 and "transition" not in it:
            raise ValueError(f"plan item {i} needs a transition (every item but the last has one)")
    _check_fx(plan.get("global_fx"), "plan: ")
    return plan


def dump_plan(plan: dict) -> str:
    """The canonical JSON text of a plan: sorted keys, so equal plans are equal bytes."""
    return json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def load_plan(path: str | Path) -> dict:
    return validate_plan(json.loads(Path(path).read_text(encoding="utf-8")))
