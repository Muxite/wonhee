"""join_videos / video_fx / montage / sfx nodes: slopcut's transitions, effects and synthesized sound as chain steps.

Specs: docs/specs/2026-10-28-slopcut-port-design.md. Engines live in edit/montage and edit/sfx; these nodes only
read stored refs, call them and store the result. Every node is seeded: the same inputs, params and seed give the
same bytes. Names (transitions, fx, sounds, styles) are checked when the params are validated, before any ffmpeg.

Ports worth knowing:
- montage takes `images` (or a hand-edited `plan` text, which wins) and an optional `audio` of type `file`; it gives
  `video` and `plan` (the JSON it rendered, canonical so equal plans are equal bytes). Feed `plan` back in to re-run.
- sfx has no inputs; its wav comes out as type `file` (there is no audio port type), which montage takes as `audio`.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from wonhee import media
from wonhee.db import repo
from wonhee.edit.montage import fx as fxmod
from wonhee.edit.montage import join as joinmod
from wonhee.edit.montage import plan as planmod
from wonhee.edit.montage import render as renderer
from wonhee.edit.montage.effects import STYLES
from wonhee.edit.montage.transitions import ALL as TRANSITIONS
from wonhee.edit.sfx import sounds
from wonhee.nodes import registry
from wonhee.nodes.base import Node
from wonhee.nodes.images import load_images
from wonhee.nodes.subtitles import save_text


def _known(value: str, allowed, what: str) -> str:
    if value not in allowed:
        raise ValueError(f"unknown {what} {value!r}; choose from {sorted(allowed)}")
    return value


class JoinParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transition: str = "fade"  # any name in edit/montage/transitions.ALL (ffmpeg xfade or a custom expression)
    duration_s: float = Field(default=0.5, gt=0.05, le=3)
    seed: int = 0

    @field_validator("transition")
    @classmethod
    def _transition(cls, v):
        return _known(v, TRANSITIONS, "transition")


class FxParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fx: str  # a name in edit/montage/fx (`fx.names()`)
    seed: int = 0
    intensity: float = Field(default=1.0, ge=0.5, le=2)

    @field_validator("fx")
    @classmethod
    def _fx(cls, v):
        return _known(v, fxmod.names(), "fx")


class MontageParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    style: str = "calm"  # calm | hype | glitch | slop
    length_s: float = Field(default=6.0, gt=0.5, le=300)
    seed: int = 0
    fps: int = Field(default=30, ge=12, le=60)
    size: tuple[int, int] = (1080, 1920)  # w, h; 9:16 by default
    sfx: bool = True  # synthesized sound cues on the visual hits (ignored when a `plan` input is given)

    @field_validator("style")
    @classmethod
    def _style(cls, v):
        return _known(v, STYLES, "style")

    @field_validator("size")
    @classmethod
    def _size(cls, v):
        if min(v) < 64 or max(v) > 4096 or v[0] % 2 or v[1] % 2:
            raise ValueError("size must be even and within 64..4096 per side")
        return v


class SfxParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str  # a name in edit/sfx/sounds (`sounds.names()`)
    seed: int = 0
    dur: float = Field(default=2.0, gt=0.05, le=30)  # seconds; only sustained sounds use it

    @field_validator("name")
    @classmethod
    def _name(cls, v):
        return _known(v, sounds.names(), "sound")


def _store_video(ctx, path: Path) -> str:
    return repo.save_file(ctx.conn, ctx.settings.media_dir, path, "video", "video/mp4")


def join_videos_fn(ctx, inputs, p: JoinParams) -> dict[str, str]:
    d = ctx.settings.media_dir
    with tempfile.TemporaryDirectory(prefix="wonhee-join-") as tmp:
        out = Path(tmp) / "joined.mp4"
        joinmod.join(media.path_for(d, inputs["video"]), media.path_for(d, inputs["video_b"]), out,
                     p.transition, p.duration_s, p.seed)
        return {"video": _store_video(ctx, out)}


def video_fx_fn(ctx, inputs, p: FxParams) -> dict[str, str]:
    d = ctx.settings.media_dir
    with tempfile.TemporaryDirectory(prefix="wonhee-fx-") as tmp:
        out = Path(tmp) / "fx.mp4"
        joinmod.apply_fx(media.path_for(d, inputs["video"]), out, p.fx, p.seed, p.intensity)
        return {"video": _store_video(ctx, out)}


def montage_fn(ctx, inputs, p: MontageParams) -> dict[str, str]:
    d = ctx.settings.media_dir
    if inputs.get("plan"):
        plan = json.loads(media.path_for(d, inputs["plan"]).read_text(encoding="utf-8"))
        planmod.validate_plan(plan)
    elif inputs.get("images"):
        slides = load_images(d, inputs["images"]).items
        if not slides:
            raise ValueError("no images to cut together")
        stills = [planmod.Still(media.path_for(d, s.sha).resolve(), {"text": s.text} if s.text else {})
                  for s in slides]
        plan = planmod.build_plan(stills, length=p.length_s, beat=None, size=p.size, fps=p.fps, style=p.style,
                                  seed=p.seed, sfx=p.sfx)
    else:
        raise ValueError("montage needs `images` (to build a plan) or a `plan` text")
    audio = media.path_for(d, inputs["audio"]) if inputs.get("audio") else None
    with tempfile.TemporaryDirectory(prefix="wonhee-montage-node-") as tmp:
        out = Path(tmp) / "montage.mp4"
        renderer.render(plan, out, audio=audio)
        sha = _store_video(ctx, out)
    return {"video": sha, "plan": save_text(ctx.conn, d, planmod.dump_plan(plan))}


def sfx_fn(ctx, inputs, p: SfxParams) -> dict[str, str]:
    samples = sounds.render(p.name, p.seed, dur=p.dur)
    with tempfile.TemporaryDirectory(prefix="wonhee-sfx-") as tmp:
        wav = Path(tmp) / f"{p.name}.wav"
        sounds.write_wav(wav, samples, sounds.SR)
        return {"file": repo.save_file(ctx.conn, ctx.settings.media_dir, wav, "audio", "audio/wav")}


registry.register(Node("join_videos", 1, {"video": "video", "video_b": "video"}, {"video": "video"}, JoinParams,
                       join_videos_fn, "Crossfade video into video_b with a named transition (ffmpeg xfade or "
                       "custom); length a+b-duration, audio crossfaded."))
registry.register(Node("video_fx", 1, {"video": "video"}, {"video": "video"}, FxParams, video_fx_fn,
                       "Apply one named montage effect (overlay / boom / color / warp / cut) to a whole video."))
registry.register(Node("montage", 1, {"images": "images?", "plan": "text?", "audio": "file?"},
                       {"video": "video", "plan": "text"}, MontageParams, montage_fn,
                       "Cut stills into a seeded 9:16 montage with transitions, motion, fx and sound cues; the plan "
                       "comes out as JSON text and can be hand-edited and fed back in."))
registry.register(Node("sfx", 1, {}, {"file": "file"}, SfxParams, sfx_fn,
                       "Synthesize one named sound effect to a wav (type file; no samples, seeded)."))
