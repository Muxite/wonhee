"""Procedural sound effects: each sound is a `build(rng, dur) -> mono array` registered with @sound.

Why it exists: the sound half of wonhee's slop edits (booms, whooshes, glitches, UI blips) without shipping any
third-party sample. Each sound is generated from oscillators and noise, so the same name and seed give the same
samples on any machine.

`anchor` is where the hit sits inside the sound (seconds, or "mid"/"end"): a cue at time t starts the sound at
t - anchor, so whooshes peak and risers land on the visual event. `dur` is the requested length for sustained sounds
(fire, rumble...); one-shots may ignore it. Every sound is normalised to peak 1 then scaled by `level`, so cue gains
are comparable.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .dsp import (SR, crush, curve, drive, env, fades, filt, n_of, noise, normalise, osc, place,
                  resample, retime, reverb, times)


@dataclass(frozen=True)
class Sound:
    name: str
    build: Callable[[np.random.Generator, float], np.ndarray]
    anchor: float | str
    level: float
    desc: str


SOUNDS: dict[str, Sound] = {}


def sound(name: str, anchor: float | str = 0.0, level: float = 1.0, desc: str = ""):
    def deco(f):
        SOUNDS[name] = Sound(name, f, anchor, level, desc)
        return f
    return deco


def names() -> list[str]:
    return sorted(SOUNDS)


def make(name: str, seed: int = 0, dur: float = 1.0, sr: int = SR) -> tuple[np.ndarray, float]:
    """Render sound `name` -> (float64 samples at `sr`, anchor in seconds). Same name and seed, same samples."""
    if name not in SOUNDS:
        raise ValueError(f"unknown sound {name!r}; choose from {names()}")
    s = SOUNDS[name]
    x = normalise(s.build(np.random.default_rng(seed), max(0.05, dur)))
    loud = np.flatnonzero(np.abs(x) > 1e-3)                      # trim the inaudible reverb tail
    x = fades(x[:loud[-1] + 1 if len(loud) else len(x)], 0.001, 0.01) * s.level
    a = s.anchor
    anchor = len(x) / SR / 2 if a == "mid" else len(x) / SR if a == "end" else float(a)
    x = retime(x, sr)
    # The build functions cannot produce NaN/inf on finite input, but a guard here keeps the output contract
    # (finite, peak <= 1) even if a future sound gets a divide wrong.
    return np.clip(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0), anchor


def render(name: str, seed: int = 0, sr: int = SR, dur: float = 2.0) -> np.ndarray:
    """Render sound `name` as float32 mono at `sr`, peak <= 1. `dur` (seconds) only matters for the sustained
    sounds; one-shots ignore it."""
    return make(name, seed, dur, sr)[0].astype(np.float32)


def write_wav(path, samples: np.ndarray, sr: int) -> None:
    """Write mono float samples in [-1, 1] as 16-bit PCM WAV, using only the stdlib `wave` module."""
    import wave

    pcm = (np.clip(np.nan_to_num(np.asarray(samples, float)), -1.0, 1.0) * 32767).round().astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())


# --- building blocks shared by several sounds ------------------------------------------------

def _thump(rng, f0=90.0, f1=38.0, sweep=0.12, decay=0.45, n=None) -> np.ndarray:
    """Sine kick: fast pitch drop + exponential tail, lightly driven for harmonics."""
    n = n or n_of(decay * 5)
    f = f1 + (f0 - f1) * np.exp(-times(n) / sweep)
    return drive(osc(f, n) * env(n, 0.002, decay), 1.8)


def _burst(rng, n, lo=None, hi=None, decay=0.02, color="white") -> np.ndarray:
    return filt(noise(n, rng, color), lo, hi) * env(n, 0.0005, decay)


def _clicks(rng, n, rate, decay=0.004, lo=1500.0, hi=None, sharp=1.6) -> np.ndarray:
    """Poisson clicks, heavy-tailed loudness (a few big pops among small ticks)."""
    out = np.zeros(n)
    k = rng.poisson(rate * n / SR)
    grain = n_of(decay * 6)
    for at in rng.integers(0, n, k):
        g = noise(grain, rng) * env(grain, 0.0002, decay * rng.uniform(0.5, 1.5))
        place(out, g, int(at), min(1.0, 0.15 * rng.pareto(sharp)))
    return filt(out, lo, hi)


def _rumble(rng, n, cutoff=140.0, wobble=0.35) -> np.ndarray:
    """Brown noise below `cutoff` with a slow random swell."""
    t = times(n)
    lfo = 1 + wobble * sum(np.sin(2 * np.pi * rng.uniform(0.3, 2.5) * t + rng.uniform(0, 6)) for _ in range(3)) / 3
    return filt(noise(n, rng, "brown"), 25, cutoff) * lfo


# --- impacts ---------------------------------------------------------------------------------

@sound("boom", desc="explosion: crack, low-passed blast, sub drop, room tail")
def boom(rng, dur):
    n = n_of(3.0)
    p = rng.uniform(0.85, 1.15)
    sub = _thump(rng, 75 * p, 30 * p, 0.25, 0.55, n)
    cut = curve(5000 * p, 120, n, 0.35)
    body = filt(noise(n, rng, "brown") * 0.6 + noise(n, rng) * 0.4, 30, cut) * (
        env(n, 0.003, 0.3) + 0.35 * env(n, 0.01, 0.6))
    crack = _burst(rng, n, 1200, None, 0.012)
    x = sub * 0.9 + normalise(body) * 0.8 + crack * 0.6
    return reverb(fades(drive(x, 2.2), 0.0, 0.6), rng, 1.4, 0.18, 2500)


@sound("nuke", level=1.0, desc="huge detonation: crack, very long sub, rolling thunder")
def nuke(rng, dur):
    n = n_of(4.5)
    p = rng.uniform(0.9, 1.1)
    crack = _burst(rng, n, 800, None, 0.03) + 0.5 * _burst(rng, n, 80, 1500, 0.15, "brown")
    sub = _thump(rng, 55 * p, 24 * p, 0.5, 1.0, n)
    roll = normalise(_rumble(rng, n, curve(600, 70, n, 0.5), 0.6)) * env(n, 0.08, 0.8, 0.3)
    debris = _clicks(rng, n, 40, 0.01, 200, 1800) * env(n, 0.3, 1.0)
    x = crack * 0.7 + sub + roll * 0.9 + normalise(debris) * 0.15
    return reverb(drive(x, 2.0) * np.exp(-np.maximum(0, times(n) - 3.0) / 0.4), rng, 2.0, 0.25, 1800)


@sound("shockwave", desc="pressure thump then a fast air whoosh tearing past")
def shockwave(rng, dur):
    n = n_of(1.6)
    th = _thump(rng, 110, 40, 0.06, 0.25, n)
    centre = curve(4500, 250, n, 0.4)
    air = filt(noise(n, rng), centre * 0.5, centre * 1.6, res=0.6) * env(n, 0.01, 0.35)
    return th + normalise(air) * 0.7


@sound("glass", desc="window shatter: crack plus a spray of ringing shards")
def glass(rng, dur):
    n = n_of(1.8)
    x = _burst(rng, n, 2500, None, 0.025) * 0.8
    for _ in range(int(rng.integers(45, 70))):
        at = int(rng.exponential(0.18) * SR)
        m = n_of(rng.uniform(0.08, 0.4))
        f = rng.uniform(2200, 8500)
        tone = sum(osc(f * r, m, phase=rng.random()) / (k + 1) for k, r in enumerate((1, 2.32, 4.25)))
        place(x, fades(tone * env(m, 0.0005, rng.uniform(0.02, 0.12)), 0, 0.03), at, rng.uniform(0.05, 0.3))
    x += _clicks(rng, n, 70, 0.002, 3000) * env(n, 0.05, 0.4) * 0.5
    return reverb(x, rng, 1.0, 0.2, 9000)


@sound("debris", desc="rubble raining down: thuds and rattles thinning out")
def debris(rng, dur):
    n = n_of(1.8)
    x = _burst(rng, n, 40, 700, 0.12, "brown")
    for _ in range(int(rng.integers(30, 50))):
        at = int(rng.gamma(2.0, 0.18) * SR)
        m = n_of(0.08)
        knock = osc(rng.uniform(140, 520), m) * env(m, 0.0005, rng.uniform(0.008, 0.025))
        grit = _burst(rng, m, 200, rng.uniform(800, 4000), rng.uniform(0.004, 0.015))
        place(x, knock * 0.6 + grit, at, rng.uniform(0.1, 0.5) * np.exp(-at / SR / 0.9))
    return reverb(x, rng, 0.8, 0.15, 3000)


@sound("sparks", desc="electric fizz with crackling spark pops")
def sparks(rng, dur):
    n = n_of(1.4)
    fizz = filt(noise(n, rng), 4000, 11000) * env(n, 0.005, 0.35)
    pops = _clicks(rng, n, 120, 0.003, 2500) * env(n, 0.0, 0.5)
    pop0 = _burst(rng, n, 1500, None, 0.008)
    return normalise(fizz) * 0.4 + normalise(pops) * 0.8 + pop0 * 0.6


@sound("zap", desc="lightning: white crack, buzzing arc, thunder tail")
def zap(rng, dur):
    n = n_of(1.6)
    crack = _burst(rng, n, 1800, None, 0.015)
    m = n_of(0.22)
    gate = np.repeat(rng.random(m // 220 + 1) > 0.35, 220)[:m]
    arc = filt(osc(rng.uniform(70, 130) * (1 + 0.05 * noise(m, rng)), m, "saw"), 200, 6000) * gate
    x = crack * 0.9
    place(x, normalise(arc) * env(m, 0.001, 0.12), 0, 0.6)
    thunder = normalise(_rumble(rng, n, 260, 0.5)) * env(n, 0.06, 0.6)
    return drive(x + thunder * 0.55, 1.5)


# --- sustained -------------------------------------------------------------------------------

@sound("crackle", level=0.7, desc="fire: low roar under random wood crackles (sustained)")
def crackle(rng, dur):
    n = n_of(dur)
    roar = normalise(filt(noise(n, rng, "pink"), 60, 700)) * 0.35
    pops = normalise(_clicks(rng, n, 30, 0.005, 700, 7000))
    hiss = filt(noise(n, rng), 3000, 9000) * 0.05
    return fades(roar + pops + hiss, 0.12, 0.25)


@sound("rumble", level=0.75, desc="earthquake: sub rumble with crumbling bursts (sustained)")
def rumble(rng, dur):
    n = n_of(dur)
    base = normalise(_rumble(rng, n, 160, 0.5))
    hum = osc(rng.uniform(32, 44) * (1 + 0.04 * np.sin(times(n) * 7)), n) * 0.4
    crumbles = normalise(_clicks(rng, n, 18, 0.03, 80, 900, 1.3)) * 0.5
    return fades(drive(base + hum + crumbles, 1.6), 0.15, 0.3)


@sound("stutter", level=0.55, desc="gated buzz chopping on and off (sustained)")
def stutter(rng, dur):
    n = n_of(dur)
    period = rng.uniform(0.11, 0.16)
    t = times(n)
    hops = rng.choice([1, 1.5, 2, 0.75], size=int(dur / period) + 2)   # pitch hops per chop
    f = rng.uniform(90, 160) * hops[(t // period).astype(int)]
    src = osc(f, n, "saw") * 0.6 + osc(f * 2.01, n, "square") * 0.3 + noise(n, rng) * 0.3
    gate = (t % period) < period * 0.55
    return crush(filt(src, 120, 5000) * gate, 6, 3)


# --- movement --------------------------------------------------------------------------------

@sound("whoosh", anchor="mid", level=0.8, desc="air swish swelling to a peak and away")
def whoosh(rng, dur):
    length = float(np.clip(dur * 1.8, 0.35, 1.1))
    n = n_of(length)
    u = np.linspace(-1, 1, n)
    bell = np.exp(-(u * rng.uniform(2.0, 2.8)) ** 2)
    up = rng.random() < 0.5
    peak = rng.uniform(1200, 3200)
    centre = peak * (0.35 + 0.65 * (bell if up else bell ** 0.5))
    x = filt(noise(n, rng, "pink"), centre * 0.55, centre * 1.5, res=0.8) * bell
    return x


@sound("riser", anchor="end", level=0.7, desc="tension riser: climbing noise + pitch, cut at the top")
def riser(rng, dur):
    n = n_of(rng.uniform(1.0, 1.4))
    amp = curve(0.02, 1.0, n, 1.6)
    x = filt(noise(n, rng), curve(300, 5000, n, 1.3), curve(1500, 14000, n, 1.3), res=0.5)
    pitch = osc(curve(220, 1100, n, 1.5), n, "saw") * (0.5 + 0.5 * osc(curve(4, 22, n), n))
    return (normalise(x) + filt(pitch, None, 4000) * 0.3) * amp


@sound("suck", anchor="end", level=0.85, desc="reverse swell sucking inward, ends on a soft thup")
def suck(rng, dur):
    n = n_of(1.1)
    cym = filt(noise(n, rng), 3000, 12000) * env(n, 0.001, 0.35)
    metal = sum(osc(rng.uniform(3000, 7000), n) for _ in range(5)) * env(n, 0.001, 0.25) * 0.15
    low = _thump(rng, 70, 40, 0.1, 0.3, n)
    x = (normalise(cym + metal) + low * 0.6)[::-1]
    end = _thump(rng, 160, 60, 0.02, 0.04, n_of(0.12))
    return np.concatenate([x, end * 0.6])


@sound("rewind", level=0.7, desc="tape scrubbed back and forth at speed (sustained-ish)")
def rewind(rng, dur):
    n = n_of(float(np.clip(dur, 0.4, 1.2)))
    t = times(n)
    src = filt(osc(rng.uniform(180, 260), n * 4, "saw") + noise(n * 4, rng) * 0.4, 200, 6000)
    rate = -3.5 + 2.8 * np.sin(2 * np.pi * rng.uniform(5, 9) * t) + rng.normal(0, 0.3, n)
    x = resample(src, rate)
    hiss = filt(noise(n, rng), 5000, None) * 0.15
    return fades(filt(x, 300, 5000) + hiss, 0.02, 0.06)


# --- digital / glitch ------------------------------------------------------------------------

@sound("glitch", level=0.65, desc="burst of crushed digital fragments")
def glitch(rng, dur):
    length = float(np.clip(dur, 0.12, 0.45))
    out = []
    while sum(map(len, out)) < n_of(length):
        m = n_of(rng.uniform(0.015, 0.06))
        kind = rng.integers(0, 5)
        if kind == 0:
            seg = osc(rng.uniform(150, 3000), m, "square")
        elif kind == 1:
            seg = noise(m, rng) * 2
        elif kind == 2:
            seg = osc(curve(rng.uniform(2000, 5000), rng.uniform(100, 400), m), m, "saw")
        elif kind == 3:
            seg = osc(rng.uniform(40, 120), m, "square") * osc(rng.uniform(800, 2500), m)
        else:
            seg = np.zeros(m)
        out.append(seg * rng.uniform(0.4, 1.0))
    x = np.concatenate(out)
    return crush(x, int(rng.integers(3, 6)), int(rng.integers(3, 12)))


@sound("tv_off", desc="CRT power-down: static snap, falling whine, final click")
def tv_off(rng, dur):
    n = n_of(0.9)
    static = filt(noise(n, rng), 800, 9000) * (times(n) < 0.08)
    whine = osc(curve(rng.uniform(2800, 4200), 120, n, 0.45), n) * env(n, 0.002, 0.22)
    thunk = _thump(rng, 120, 50, 0.02, 0.06, n)
    click = np.zeros(n)
    place(click, _burst(rng, n, 2000, None, 0.002), n_of(0.55), 1.0)
    return static * 0.7 + whine * 0.5 + thunk * 0.5 + click * 0.4


@sound("shutter", level=0.8, desc="camera: two-blade shutter click + flash recharge whine")
def shutter(rng, dur):
    n = n_of(0.8)
    x = np.zeros(n)
    gap = n_of(rng.uniform(0.05, 0.09))
    for at, g in ((0, 1.0), (gap, 0.7)):
        place(x, filt(noise(n_of(0.03), rng), 1500, 7000) * env(n_of(0.03), 0.0003, 0.004), at, g)
    whine = osc(curve(1800, 7500, n, 0.7), n) * curve(0.0, 1.0, n) * np.exp(-times(n) / 0.4)
    return x + whine * 0.06


# --- sparkle / ui ----------------------------------------------------------------------------

@sound("chime", level=0.55, desc="glittery bell arpeggio with shimmer")
def chime(rng, dur):
    n = n_of(1.8)
    x = np.zeros(n)
    root = rng.choice([1047, 1175, 1319, 1568])            # C6 D6 E6 G6
    steps = np.cumsum(rng.choice([2, 3, 4, 5], size=int(rng.integers(4, 8))))
    for k, st in enumerate(steps):
        f = root * 2 ** (st / 12)
        m = n_of(0.9)
        bell = sum(osc(f * r, m) * a for r, a in ((1, 1), (2.76, 0.4), (5.4, 0.15)))
        place(x, fades(bell * env(m, 0.001, 0.3), 0, 0.3), n_of(k * rng.uniform(0.05, 0.08)), 0.6)
    x += filt(noise(n, rng), 6000, 14000) * env(n, 0.02, 0.4) * 0.15
    return reverb(x, rng, 1.6, 0.3, 10000)


@sound("pop", level=0.7, desc="cork/bubble pop: upward blip + click")
def pop(rng, dur):
    n = n_of(0.2)
    f = curve(rng.uniform(250, 380), rng.uniform(900, 1400), n_of(0.04))
    blip = np.concatenate([f, np.full(n - len(f), f[-1])])
    return osc(blip, n) * env(n, 0.001, 0.03) + _burst(rng, n, 1000, None, 0.003) * 0.6


@sound("sub_drop", level=0.75, desc="808 hit: clicky sub kick that drops in pitch")
def sub_drop(rng, dur):
    n = n_of(1.6)
    k = _thump(rng, rng.uniform(140, 180), rng.uniform(38, 48), 0.05, 0.25, n)
    click = _burst(rng, n, 2000, 9000, 0.002)
    return drive(k * 1.4, 2.5) + click * 0.25
