"""audio_check / audio_clean nodes: measure a clip's loudness and noise; denoise, tame peaks and normalise it."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from wonhee.db import repo
from wonhee.edit import audio as au
from wonhee.media import path_for
from wonhee.nodes import registry
from wonhee.nodes.base import Node
from wonhee.nodes.subtitles import save_text

PLATFORM_LUFS = -14.0


class CheckParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


def verdicts(s: au.AudioStats, target: float = PLATFORM_LUFS) -> list[str]:
    out = []
    if abs(s.lufs - target) > 2:
        out.append(f"loudness {s.lufs:.1f} LUFS, {s.lufs - target:+.1f} from the {target:g} target: the app will "
                   "turn it " + ("down (and squash peaks)" if s.lufs > target else "up (and raise the noise)"))
    if s.true_peak > -1.0 or s.clipped:
        out.append(f"peaks hit {s.true_peak:.1f} dBTP" + (" with clipped samples" if s.clipped else "")
                   + ": limit to -1.5 dBTP")
    if s.snr is not None and s.snr < 35:
        out.append(f"content only {s.snr:.0f} dB above the noise floor: audible hiss or room tone, denoise it")
    if s.lra > 15:
        out.append(f"loudness range {s.lra:.0f} LU: quiet and loud parts far apart, compress it")
    return out or ["clean: nothing to fix"]


def audio_check_fn(ctx, inputs, p: CheckParams) -> dict[str, str]:
    src = path_for(ctx.settings.media_dir, inputs["video"])
    if not au.has_audio(src):
        raise ValueError("no audio stream")
    st, _ = au.measure(src)
    return {"report": save_text(ctx.conn, ctx.settings.media_dir,
                                json.dumps({"stats": st.model_dump(), "verdicts": verdicts(st)}, indent=1))}


class CleanParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    preset: au.Preset = "voice"  # voice: denoise + compress; mixed: light; music: loudness only
    denoise: float = Field(default=1.0, ge=0, le=1)
    de_ess: bool = False
    gate: bool = False  # duck the gaps between words (voice only)
    lufs: float = Field(default=PLATFORM_LUFS, ge=-30, le=-8)
    true_peak: float = Field(default=-1.5, ge=-6, le=-0.1)
    lra: float = Field(default=11, ge=3, le=20)


def audio_clean_fn(ctx, inputs, p: CleanParams) -> dict[str, str]:
    d = ctx.settings.media_dir
    src = path_for(d, inputs["video"]).resolve()
    if not au.has_audio(src):
        raise ValueError("no audio stream")
    target = (p.lufs, p.true_peak, p.lra)
    rnn = au.rnn_model(ctx.settings.home / "models") if p.preset == "voice" and p.denoise > 0 else None
    if p.preset == "voice" and p.denoise > 0 and rnn is None:
        ctx.flag("DENOISE_FALLBACK", "RNNoise model unavailable, used afftdn")
    pre = au.pre_chain(p.preset, p.denoise, rnn, p.de_ess, p.gate)
    before, _ = au.measure(src)
    _, first = au.measure(src, pre, target)  # pass 1 measures what the pre-chain produces
    with tempfile.TemporaryDirectory(prefix="wonhee-audio-") as tmp:
        out = Path(tmp) / "out.mp4"
        proc = au._run(["-i", str(src), "-map", "0:v?", "-map", "0:a:0", "-c:v", "copy", "-af",
                        au.clean_chain(first, pre, target), "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                        "-movflags", "+faststart", str(out)])
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(f"ffmpeg audio clean failed: {proc.stderr.strip()[-400:]}")
        after, _ = au.measure(out)
        sha = repo.save_file(ctx.conn, d, out, "video", "video/mp4")
    report = {"before": before.model_dump(), "after": after.model_dump(), "chain": pre,
              "denoiser": "rnnoise" if rnn else ("afftdn" if "afftdn" in pre else "none")}
    return {"video": sha, "report": save_text(ctx.conn, d, json.dumps(report, indent=1))}


registry.register(Node("audio_check", 2, {"video": "video"}, {"report": "text"}, CheckParams, audio_check_fn,
                       "Measure a clip's loudness (LUFS), true peak, range and noise floor, with what to fix."))
registry.register(Node("audio_clean", 2, {"video": "video"}, {"video": "video", "report": "text"}, CleanParams,
                       audio_clean_fn, "Local audio clean-up: rumble cut, RNNoise denoise, compression, two-pass "
                       "loudness to -14 LUFS / -1.5 dBTP; video copied untouched."))
