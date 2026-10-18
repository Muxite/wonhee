"""Audio clean-up on the box: measure loudness and noise, then denoise, tame peaks and hit a consistent loudness.

All ffmpeg, all local. Denoise is RNNoise (`arnndn`, a small recurrent net, faster than real time on CPU) when its
model file is present, else FFT denoise (`afftdn`). Loudness is EBU R128 two-pass `loudnorm`: the first pass
measures, the second applies a linear gain where it can, so every post lands at the same level (-14 LUFS, true
peak -1.5 dBTP by default, what Instagram / TikTok / YouTube normalise to) without pumping.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel

from wonhee.procenv import clean_env

Preset = Literal["voice", "mixed", "music"]
RNN_URL = ("https://raw.githubusercontent.com/GregorR/rnnoise-models/master/"
           "somnolent-hogwash-2018-09-01/sh.rnnn")  # general-purpose speech model, BSD-3


class AudioStats(BaseModel):
    lufs: float  # integrated loudness
    true_peak: float  # dBTP
    lra: float  # loudness range, LU
    threshold: float
    noise_floor: float | None = None  # dBFS, from the quietest RMS windows
    rms: float | None = None  # dBFS, overall
    snr: float | None = None  # rms - noise_floor: how far the content sits above the noise (gain-independent)
    peak: float | None = None  # sample peak, dBFS
    clipped: bool = False


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-y", *args], env=clean_env(), capture_output=True,
                          text=True)


def has_audio(src: Path) -> bool:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of",
                          "csv=p=0", str(src)], env=clean_env(), capture_output=True, text=True)
    return bool(out.stdout.strip())


def _loudnorm_json(stderr: str) -> dict:
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", stderr, re.S)
    if not m:
        raise RuntimeError(f"loudnorm measurement missing: {stderr[-300:]}")
    return json.loads(m.group(0))


def measure(src: Path, pre: str = "", target: tuple[float, float, float] = (-14, -1.5, 11)) -> tuple[AudioStats, dict]:
    """Loudness and noise of the first audio stream; also returns loudnorm's raw first-pass numbers."""
    i, tp, lra = target
    chain = (pre + "," if pre else "") + (f"astats=metadata=0:measure_perchannel=none:measure_overall="
                                          "Noise_floor+Peak_level+Peak_count+RMS_level,"
                                          f"loudnorm=I={i}:TP={tp}:LRA={lra}:print_format=json")
    proc = _run(["-i", str(src), "-map", "0:a:0", "-af", chain, "-f", "null", "-"])
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg audio measure failed: {proc.stderr.strip()[-400:]}")
    ln = _loudnorm_json(proc.stderr)

    def stat(name: str) -> float | None:
        m = re.findall(rf"{name}: (-?[\d.]+|-inf)", proc.stderr)
        return None if not m or m[-1] == "-inf" else float(m[-1])

    peak = stat("Peak level dB")
    # astats has no clip counter: a full-scale peak hit more than a few times is clipping
    clipped = peak is not None and peak > -0.1 and (stat("Peak count") or 0) > 3
    st = AudioStats(lufs=float(ln["input_i"]), true_peak=float(ln["input_tp"]), lra=float(ln["input_lra"]),
                    threshold=float(ln["input_thresh"]), noise_floor=stat("Noise floor dB"),
                    peak=peak, clipped=clipped, rms=stat("RMS level dB"))
    if st.rms is not None and st.noise_floor is not None:
        st.snr = round(st.rms - st.noise_floor, 1)
    return st, ln


def rnn_model(models_dir: Path) -> Path | None:
    """The RNNoise weights, fetched once; None when offline (afftdn is used instead)."""
    p = models_dir / "rnnoise-sh.rnnn"
    if p.exists() and p.stat().st_size > 10_000:
        return p
    try:
        r = httpx.get(RNN_URL, follow_redirects=True, timeout=60)
        r.raise_for_status()
    except httpx.HTTPError:
        return None
    models_dir.mkdir(parents=True, exist_ok=True)
    p.write_bytes(r.content)
    return p


def pre_chain(preset: str, denoise: float, rnn: Path | None, de_ess: bool, gate: bool) -> str:
    """Everything before loudness: rumble cut, denoise, gate, de-ess, compression. `denoise` 0..1."""
    parts = ["highpass=f=70" if preset != "music" else "highpass=f=30"]
    if denoise > 0 and preset != "music":
        if rnn is not None and preset == "voice":
            # mix < 1 keeps that share of the noisy original: 0.8 removes ~10 dB, 1.0 ~20 dB (measured)
            parts.append(f"arnndn=m='{rnn}':mix={min(denoise, 1):.2f}")
            parts.append(f"afftdn=nr={4 + 8 * denoise:.0f}:nf=-50")  # residual hiss RNNoise leaves
        else:
            parts.append(f"afftdn=nr={6 + 18 * denoise:.0f}:nf=-45:tn=1")
    if gate and preset == "voice":
        parts.append("agate=threshold=0.01:ratio=2:attack=5:release=200")
    if de_ess and preset != "music":
        parts.append("deesser=i=0.4")
    if preset == "voice":
        parts.append("acompressor=threshold=-21dB:ratio=3:attack=8:release=150:makeup=2")
    elif preset == "mixed":
        parts.append("acompressor=threshold=-18dB:ratio=2:attack=15:release=250")
    return ",".join(parts)


def clean_chain(first_pass: dict, pre: str, target: tuple[float, float, float]) -> str:
    i, tp, lra = target
    ln = (f"loudnorm=I={i}:TP={tp}:LRA={lra}:measured_I={first_pass['input_i']}:"
          f"measured_TP={first_pass['input_tp']}:measured_LRA={first_pass['input_lra']}:"
          f"measured_thresh={first_pass['input_thresh']}:offset={first_pass['target_offset']}:linear=true")
    # loudnorm resamples to 192 kHz internally; bring it back, and a limiter catches inter-sample overs
    return ",".join(x for x in [pre, ln, "aresample=48000", f"alimiter=limit={10 ** (tp / 20):.3f}:level=false"] if x)
