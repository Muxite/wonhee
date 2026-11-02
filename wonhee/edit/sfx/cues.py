"""Plan -> sound-effect cues -> one stereo WAV on the video timeline.

Cues are derived at render time from the plan, so edited durations stay in sync. Each fx / motion effect / transition
maps to a rule `(seed, dur, fps) -> [(offset, sound, gain, length)]` that mirrors the visual's own timing: nuke's
crack lands on its flash, bass_boost thumps on its 0.5 s pulse, chroma_boom glitches on its seeded 0.4 s grid,
lightning zaps on the very frames the hash2 strobe fires. An item may set "sfx": ["boom", ...] to replace its
automatic fx sounds (played at clip start). Numpy only; the same plan (seed included) gives the same samples.
"""
from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ..montage.fx import slot_seed
from ..montage.plan import total_length
from .dsp import SR, place
from .sounds import make

MAX_FX_SOUNDS = 2          # per clip; slop stacks 4 fx and 4 simultaneous booms is mud
CENTRED = {"boom", "nuke", "sub_drop", "rumble", "suck", "riser"}   # low/huge sounds stay mono

Rule = Callable[[int, float, int], list[tuple[float, str, float, float]]]   # (seed, dur, fps)


@dataclass(frozen=True)
class Cue:
    t: float          # video time (s) where the sound's anchor lands
    sound: str
    gain: float = 1.0
    length: float = 1.0   # requested length for sustained sounds


def hit(sound: str, at: float = 0.0, gain: float = 1.0) -> Rule:
    return lambda seed, dur, fps: [(at, sound, gain, dur)]


def hold(sound: str, gain: float = 1.0) -> Rule:
    return lambda seed, dur, fps: [(0.0, sound, gain, dur)]


def every(sound: str, period: float, length: float, gain: float = 1.0,
          phase: Callable[[int], float] = lambda seed: 0.0) -> Rule:
    """Hits on the fx's own `mod(t + phase, period)` grid."""
    def rule(seed, dur, fps):
        t0 = (-phase(seed)) % period
        return [(float(t), sound, gain, length) for t in np.arange(t0, dur - 0.05, period)]
    return rule


def _hash2(a: float, b: float) -> float:
    """Python twin of fx._expr.hash2 (ffmpeg mod is floor-based, like this)."""
    v = math.sin(a * 12.9898 + b * 78.233) * 43758.5453
    return abs(v - math.floor(v))


def _lightning(seed: int, dur: float, fps: int, p: float = 0.12) -> list:
    """Same strike test as overlay.lightning: hash2(floor(n/2), k) > 1 - p, per frame pair."""
    k, out, last = seed % 997, [], -1.0
    for pair in range(int(dur * fps) // 2 + 1):
        t = pair * 2 / fps
        if _hash2(pair, k) > 1 - p and t - last > 0.3 and len(out) < 4:
            out.append((t, "zap", 0.9, 1.0))
            last = t
    return out


def _combo(*rules: Rule) -> Rule:
    return lambda seed, dur, fps: [c for r in rules for c in r(seed, dur, fps)]


# fx name -> (priority, rule). Higher priority wins the MAX_FX_SOUNDS slots.
FX_RULES: dict[str, tuple[int, Rule]] = {
    "nuke": (10, hit("nuke")),
    "explosion": (9, hit("boom")),
    "fireball": (9, hit("boom", gain=0.85)),
    "shockwave": (8, hit("shockwave", 0.05)),
    "screen_shatter": (8, hit("glass", 0.1)),
    "lightning": (7, _lightning),
    "debris": (7, hit("debris")),
    "sparks_burst": (6, hit("sparks")),
    "earthquake": (6, hold("rumble")),
    "bass_boost": (6, every("sub_drop", 0.5, 0.5, 0.8)),
    "crash_zoom_jitter": (5, every("sub_drop", 0.6, 0.6, 0.6)),
    "fire": (5, hold("crackle")),
    "rewind_jitter": (5, hold("rewind", 0.8)),
    "stutter": (5, hold("stutter")),
    "strobe_invert": (4, hold("stutter", 0.6)),
    "chroma_boom": (4, every("glitch", 0.4, 0.15, 0.6, lambda s: (s % 4) / 10 * 0.2)),
    "pixel_scramble": (4, every("glitch", 0.6, 0.3, 0.5, lambda s: (s % 7) / 10)),
    "datamosh": (4, hit("glitch", gain=0.7)),
    "slice_shift": (3, hit("glitch", gain=0.6)),
    "film_burn": (4, hold("crackle", 0.6)),
    "confetti": (4, _combo(hit("pop"), hit("pop", 0.06, 0.7), hit("chime", 0.1, 0.5))),
    "smoke_plume": (3, hold("rumble", 0.45)),
    "sparkles": (2, hit("chime", gain=0.7)),
    "lens_flare": (2, hit("chime", 0.2, 0.5)),
}

# base motion effects (effects.py); glitch's rgbashift fires on lt(mod(t,0.6),0.12)
EFFECT_RULES: dict[str, Rule] = {
    "punch": hit("sub_drop", gain=0.7),
    "flash": hit("shutter"),
    "glitch": every("glitch", 0.6, 0.12, 0.45),
}

TRANSITION_SOUNDS: dict[str, tuple[str, float]] = {
    "boom_cut": ("boom", 0.9), "shatter": ("glass", 0.9), "burn_away": ("crackle", 0.8),
    "implode": ("suck", 1.0), "blast_wave": ("shockwave", 1.0), "tv_off": ("tv_off", 0.9),
    "glitch_slice": ("glitch", 0.7), "block_dissolve": ("glitch", 0.6), "pixelize": ("glitch", 0.5),
    "zoomin": ("riser", 0.7), "zoom_spin": ("riser", 0.8), "fadewhite": ("shutter", 0.8),
    "fadeblack": ("sub_drop", 0.4),
}
_WHOOSH_PREFIX = ("wipe", "slide", "smooth", "cover", "reveal", "diag", "squeeze", "horz", "vert",
                  "circle", "cube", "spin", "rotate", "swirl", "card", "ripple", "radial", "rect",
                  "hl", "hr", "vu", "vd", "hblur")


def transition_sound(name: str) -> tuple[str, float] | None:
    if name in TRANSITION_SOUNDS:
        return TRANSITION_SOUNDS[name]
    if name.startswith(_WHOOSH_PREFIX):
        return ("whoosh", 0.75)
    return None   # fades / dissolves stay quiet


def _fx_cues(names: list[str], seed_of: Callable[[int], int], dur: float, fps: int, start: float,
             limit: int | None) -> list[Cue]:
    ranked = sorted(((FX_RULES[n][0], k, n) for k, n in enumerate(names) if n in FX_RULES),
                    key=lambda r: (-r[0], r[1]))[:limit]
    return [Cue(start + off, snd, g, ln)
            for _, k, n in ranked for off, snd, g, ln in FX_RULES[n][1](seed_of(k), dur, fps)]


def cues_for(plan: dict) -> list[Cue]:
    items, seed, fps = plan["items"], plan.get("seed", 0), plan["fps"]
    cues, start = [], 0.0
    for i, it in enumerate(items):
        d = it["duration"]
        if "sfx" in it:
            cues += [Cue(start, s, 1.0, d) for s in it["sfx"]]
        else:
            cues += _fx_cues(it.get("fx") or [], lambda k: slot_seed(seed, i, k), d, fps, start,
                              int(plan.get("sfx_max", MAX_FX_SOUNDS)))
        if it["effect"] in EFFECT_RULES:
            cues += [Cue(start + off, s, g, ln) for off, s, g, ln in EFFECT_RULES[it["effect"]](0, d, fps)]
        if it.get("text"):
            cues.append(Cue(start + float(it["text"].get("start", 0.0)), "pop", 0.45))
        if "transition" in it:
            tr = it["transition"]
            nxt = start + d - tr["duration"]          # the next clip starts as the xfade begins
            snd = transition_sound(tr["type"])
            if snd:
                cues.append(Cue(nxt + tr["duration"] / 2, snd[0], snd[1], tr["duration"] * 2))
            start = nxt
    total = total_length(plan)
    cues += _fx_cues(plan.get("global_fx") or [], lambda k: slot_seed(seed, len(items), k), total, fps, 0.0, None)
    return sorted((c for c in cues if c.t < total), key=lambda c: c.t)


def mix(plan: dict, gain: float = 1.0) -> np.ndarray:
    """(2, samples) float stereo mix of every cue, soft-limited to +-0.9."""
    total = total_length(plan)
    buf = np.zeros((2, int(math.ceil(total * SR))))
    rng = np.random.default_rng(int(plan.get("seed", 0)) + 0x5FC)
    for j, c in enumerate(cues_for(plan)):
        x, anchor = make(c.sound, int(rng.integers(2**31)), c.length)
        pan = 0.0 if c.sound in CENTRED else rng.uniform(-0.45, 0.45)
        a = (pan + 1) * math.pi / 4                      # equal-power pan, unity in the centre
        at = int(round((c.t - anchor) * SR))
        place(buf[0], x, at, c.gain * math.cos(a) * math.sqrt(2))
        place(buf[1], x, at, c.gain * math.sin(a) * math.sqrt(2))
    return 0.9 * np.tanh(buf * 0.8 * gain)


def write_wav(path: Path, stereo: np.ndarray) -> Path:
    pcm = (np.clip(stereo.T, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return path


def render_track(plan: dict, dest_wav: Path) -> Path | None:
    """WAV of the plan's sound effects, or None when sfx are off or nothing makes a sound."""
    if not plan.get("sfx", True) or not cues_for(plan):
        return None
    return write_wav(Path(dest_wav), mix(plan, float(plan.get("sfx_gain", 1.0))))
