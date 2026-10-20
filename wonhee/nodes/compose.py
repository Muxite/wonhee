"""compose / frame_video / grid_plan nodes: the layout engine (edit/compose.py) as chain steps.

Design: docs/specs/2026-10-20-meme-and-popart-formats-design.md ("Generalized design (approved form)").
- compose (images? -> image): a Layout dict or a named template plus slots, rendered to one PNG.
- frame_video (video, subtitles? -> video): the template's overlay (bands, title) as one PNG with a transparent window,
  the clip scaled into the window, optional word captions, all in ONE ffmpeg encode.
- grid_plan (images -> images): a template, palette and pattern per tile for a 3, 6 or 9 grid.
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

from wonhee import media
from wonhee.db import repo
from wonhee.edit import compose as C
from wonhee.edit.looks import Preset as LookPreset, filter_chain
from wonhee.fsutil import run_timed
from wonhee.inspire import dossier as dz
from wonhee.nodes import registry
from wonhee.nodes.base import Node
from wonhee.nodes.images import Images, Slide, load_images, save_images
from wonhee.nodes.subtitles import load_subs
from wonhee.procenv import clean_env

_TEMPLATES = C.describe_templates()


_COMPOSE_DOC = ("A picture as a canvas plus ordered layers. Give `layout` (a Layout dict: canvas, palette, seed, layers of "
                "type photo | pattern | shape | sticker | text | band) or `template` + `slots`. Photo layers read the "
                "`images` input by index (`src`). Shipped templates and their slots:\n" + _TEMPLATES)


class ComposeParams(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"description": _COMPOSE_DOC})
    layout: C.Layout | None = None
    template: str | None = Field(default=None, description="named template; one of the templates in this schema's description")
    slots: dict = Field(default_factory=dict, description="values for the template's slots")
    seed: int | None = Field(default=None, description="overrides the layout's seed (patterns, jitter)")

    @model_validator(mode="after")
    def _one(self):
        if (self.layout is None) == (self.template is None):
            raise ValueError("give exactly one of layout or template")
        return self


def _images(ctx, ref: str | None) -> list[Image.Image]:
    if not ref:
        return []
    d = ctx.settings.media_dir
    out = []
    for item in load_images(d, ref).items:
        with Image.open(media.path_for(d, item.sha)) as im:
            im.load()
            out.append(im.copy())
    return out


def _layout(p: ComposeParams) -> C.Layout:
    return p.layout if p.layout is not None else C.load_template(p.template, p.slots)  # type: ignore[arg-type]


def _store_png(ctx, im: Image.Image, tmpd: Path, name: str = "out.png") -> str:
    path = tmpd / name
    im.save(path)
    return repo.save_file(ctx.conn, ctx.settings.media_dir, path, "image", "image/png")


_OVERLAY = re.compile(r'^\s*overlay: "(.*)" \(', re.M)  # the inspiration digest's overlay lines (nodes/ideate.py)
COPY_WORDS = 3  # stickers are short, so a 3-word run counts as copying (ideate uses 5 for whole lines)


def copy_guard(layout: C.Layout, refs: str) -> list[str]:
    """Drop text/sticker content that shares a COPY_WORDS-word run with a saved reference overlay (the `ideate` copy
    check, docs/specs/2026-10-20-meme-and-popart-formats-design.md "No copying"). `refs` is an inspiration digest
    (overlay lines are read from it) or plain text, one reference line per row. Returns the notes for IDEA_COPY."""
    sources = _OVERLAY.findall(refs) or [ln for ln in refs.splitlines() if ln.strip()]
    notes = []
    for i, ly in enumerate(layout.layers):
        if ly.type not in ("text", "sticker"):
            continue
        for field_name, lines in (("content", [ly.content] if ly.content else []), ("items", ly.items or [])):
            for line in lines:
                if hit := dz.copies_source(line, sources, COPY_WORDS):
                    notes.append(f"layer {i}: line dropped, shares '{hit}' with a saved post")
                    setattr(ly, field_name, None)
                    break
    return notes


def compose_fn(ctx, inputs, p: ComposeParams) -> dict[str, str]:
    layout = _layout(p)
    if inputs.get("refs"):
        for note in copy_guard(layout, media.path_for(ctx.settings.media_dir, inputs["refs"]).read_text(encoding="utf-8")):
            ctx.flag("IDEA_COPY", note)
    mask_fn = C.default_mask_fn(ctx.settings.home / "models")
    img, notes = C.render_ex(layout, _images(ctx, inputs.get("images")), p.seed, mask_fn=mask_fn)
    if notes.adjusted:
        ctx.flag("LAYOUT_ADJUSTED", "; ".join(notes.adjusted))
    for note in notes.no_subject:
        ctx.flag("FOCUS_NO_SUBJECT", note)
    with tempfile.TemporaryDirectory(prefix="wonhee-compose-") as tmp:
        return {"image": _store_png(ctx, img, Path(tmp))}


# ---- compose_video / frame_video ---------------------------------------------------------------------------------

class Tighten(BaseModel):
    """Pause trimming (reel-styles spec §3): a pause of at least min_pause_ms between words is cut down to keep_ms."""
    model_config = ConfigDict(extra="forbid")
    min_pause_ms: int = Field(default=250, ge=50, le=5000)
    keep_ms: int = Field(default=80, ge=0, le=1000)


_VIDEO_DOC = ("A reel: a layout over time around a clip, in ONE ffmpeg encode (reel-styles spec §2, §3). Give `layout` "
              "or `template` + `slots`. The clip goes into the layout's `photo` layer whose `src` is `clip` (or `video`), or "
              "fills the canvas when there is none. Layers may carry `t` {in_ms, out_ms} and `anim` (none | fade | pop | type "
              "| rise); layer types `captions` (chunk | word | karaoke), `stack` and `list` read the `subtitles` words; "
              "photo layers read the `images` input by index (insets). `tighten` cuts pauses and re-times every layer.")


class ComposeVideoParams(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"description": _VIDEO_DOC + "\nTemplates:\n" + _TEMPLATES})
    layout: C.Layout | None = None
    template: str | None = Field(default=None, description="named template; one of the templates in this schema's description")
    slots: dict = Field(default_factory=dict)
    clip_slot: str = Field(default="photo", description="the template slot that becomes the clip window (if it has one)")
    tighten: Tighten | bool | None = Field(
        default=None, description="pause trimming. null = the template's own default (the talking-head styles turn it "
                                  "on), false = off, true = the defaults (250 ms / keep 80 ms), or {min_pause_ms, keep_ms}")
    look: LookPreset | None = Field(default=None, description="grade the clip (not the text) with a look preset, e.g. lofi_cam; same encode")
    seed: int | None = None
    encoder: Literal["auto", "nvenc", "x264"] = "auto"
    quality: int = Field(default=20, ge=12, le=35)

    @model_validator(mode="after")
    def _one(self):
        if (self.layout is None) == (self.template is None):
            raise ValueError("give exactly one of layout or template")
        return self


class FrameVideoParams(BaseModel):
    """The first form of compose_video (kept as an alias): a template's bands and title around the clip, plus one
    `captions` layer in the lower band unless the template has its own. Templates: see `compose`."""
    model_config = ConfigDict(extra="forbid")
    template: str = "meme_frame"
    slots: dict = Field(default_factory=dict)
    clip_slot: str = Field(default="photo", description="the template slot that becomes the transparent clip window")
    captions: Literal["box", "highlight", "none"] = "box"
    words_per_group: int = Field(default=3, ge=1, le=3)
    caption_colour: str = "#c9a0e6"
    caption_box: C.Box = C.Box(x=0.14, y=0.665, w=0.72, h=0.085)
    tighten: Tighten | None = None
    seed: int | None = None
    encoder: Literal["auto", "nvenc", "x264"] = "auto"
    quality: int = Field(default=20, ge=12, le=35)


def _ffmpeg(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return run_timed(["ffmpeg", "-loglevel", "error", "-y", *args], cwd=cwd, env=clean_env(),
                     capture_output=True, text=True)


def _even(v: int) -> int:
    return max(2, v - v % 2)


def silence_cuts(src: Path, t: Tighten) -> tuple[tuple[int, int], ...]:
    """Cuts from ffmpeg's silencedetect, for a take with no subtitles: an analysis pass (decode to null), not an encode."""
    proc = run_timed(["ffmpeg", "-nostats", "-vn", "-i", str(src), "-af", f"silencedetect=noise=-35dB:d={t.min_pause_ms / 1000}",
                      "-f", "null", "-"], env=clean_env(), capture_output=True, text=True)
    starts = [float(m) for m in re.findall(r"silence_start: (-?[\d.]+)", proc.stderr)]
    ends = [float(m) for m in re.findall(r"silence_end: ([\d.]+)", proc.stderr)]
    cuts = []
    for a, b in zip(starts, ends):
        a, b = max(0, round(a * 1000)), round(b * 1000)
        if a > 0 and b - a >= t.min_pause_ms and b - a > t.keep_ms:  # a leading silence is not a pause between speech
            cuts.append((a + t.keep_ms // 2, b - (t.keep_ms - t.keep_ms // 2)))
    return tuple(cuts)


def _render_video(ctx, inputs, layout: C.Layout, tighten: Tighten | None, seed, encoder: str, quality: int,
                  look: str | None = None) -> str:
    from wonhee.edit import reel as R
    from wonhee.edit import timeline as TL
    from wonhee.nodes.subtitles import line_words
    d = ctx.settings.media_dir
    src = media.path_for(d, inputs["video"]).resolve()
    info = media.probe(src)
    if not info.duration_ms:
        raise RuntimeError("cannot read the clip's duration")
    layout.canvas.w, layout.canvas.h = _even(layout.canvas.w), _even(layout.canvas.h)
    W, H = layout.canvas.w, layout.canvas.h
    words, estimated = [], False
    if inputs.get("subtitles"):
        for ln in load_subs(d, inputs["subtitles"]).lines:
            ws, est = line_words(ln)
            words += ws
            estimated = estimated or est
    word_driven = any(ly.type in C.WORD_TYPES for ly in layout.layers)
    if estimated and (word_driven or tighten):
        ctx.flag("WORDS_ESTIMATED", "subtitles carry line timing only; words timed evenly within each line")
    cuts: tuple[tuple[int, int], ...] = ()
    has_audio = media.has_audio(src)
    if tighten and not words and not has_audio:
        ctx.flag("TIGHTEN_SKIPPED", "no word timings and no audio to find pauses in; the take is left untrimmed")
        tighten = None
    if tighten:
        raw = TL.find_pauses(words, tighten.min_pause_ms, tighten.keep_ms) if words else silence_cuts(src, tighten)
        cuts = TL.snap_cuts(raw, info.fps, info.duration_ms)  # on the frame grid, so audio and video are cut together
    tl = TL.Timeline(cuts)
    plan = R.plan_reel(layout, words, info.duration_ms, _images(ctx, inputs.get("images")), seed,
                       mask_fn=C.default_mask_fn(ctx.settings.home / "models"), timeline=tl)
    if plan.notes.adjusted:
        ctx.flag("LAYOUT_ADJUSTED", "; ".join(plan.notes.adjusted))
    for note in plan.notes.no_subject:
        ctx.flag("FOCUS_NO_SUBJECT", note)
    if tighten:
        ctx.flag("PAUSES_TRIMMED", f"removed {tl.removed_ms() / 1000:.2f} s in {len(cuts)} cut(s): "
                                   f"{info.duration_ms / 1000:.2f} s -> {plan.duration_ms / 1000:.2f} s")
    enc = {"nvenc": ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", str(quality), "-b:v", "0"],
           "x264": ["-c:v", "libx264", "-preset", "medium", "-crf", str(quality)]}
    order = ["nvenc", "x264"] if encoder == "auto" else [encoder]
    with tempfile.TemporaryDirectory(prefix="wonhee-reel-") as tmp:
        tmpd = Path(tmp)
        args, graph, alabel = R.graph_for(plan, (W, H), tmpd, tl.kept(info.duration_ms) if cuts else None, has_audio,
                                         filter_chain(look, video=True) if look else None)
        audio = ["-map", alabel, "-c:a", "aac", "-b:a", "160k"] if alabel else (
            ["-map", "0:a?", "-c:a", "copy"] if not cuts else [])
        out = tmpd / "out.mp4"
        proc = None
        for name in order:
            proc = _ffmpeg(["-i", str(src), *args, "-filter_complex", graph, "-map", "[o]", *audio, *enc[name],
                            "-movflags", "+faststart", str(out)], tmpd)
            if proc.returncode == 0 and out.exists():
                if name != order[0]:
                    ctx.flag("ENCODER_FALLBACK", f"{order[0]} failed, used {name}")
                break
        else:
            raise RuntimeError(f"ffmpeg compose_video failed: {proc.stderr.strip()[-400:]}")
        return repo.save_file(ctx.conn, d, out, "video", "video/mp4")


def compose_video_fn(ctx, inputs, p: ComposeVideoParams) -> dict[str, str]:
    declared = {t["name"]: t["slots"] for t in C.list_templates()}.get(p.template or "", {})
    slots = {**p.slots, **({p.clip_slot: "clip"} if p.clip_slot in declared and p.clip_slot not in p.slots else {})}
    layout = p.layout if p.layout is not None else C.load_template(p.template, slots)  # type: ignore[arg-type]
    return {"video": _render_video(ctx, inputs, layout, _tighten_of(p), p.seed, p.encoder, p.quality, p.look)}


def _tighten_of(p: ComposeVideoParams) -> Tighten | None:
    """The pause trimming to use: the param when given, else the template's own default (reel styles spec §3, §6)."""
    t = p.tighten
    if t is None and p.template:
        d = C.template_params(p.template).get("tighten")
        t = Tighten.model_validate(d) if d else None
    return Tighten() if t is True else None if t is False else t


def frame_video_fn(ctx, inputs, p: FrameVideoParams) -> dict[str, str]:
    layout = C.load_template(p.template, {**p.slots, p.clip_slot: "clip"})
    if not C.clip_windows(layout):
        raise ValueError(f"template '{p.template}' has no clip window (a photo layer with src 'clip')")
    if p.captions != "none" and not any(ly.type == "captions" for ly in layout.layers):
        box = p.caption_box
        layout.layers.append(C.Layer(
            type="captions", mode="chunk", max_words=p.words_per_group, size_frac=0.034,
            box=C.Box(x=box.x, y=box.y, w=box.w, h=box.h), bg_box="box",
            bg_colour=p.caption_colour if p.captions == "box" else p.caption_colour + "aa" if len(p.caption_colour) == 7
            else p.caption_colour,
            colour="#000000" if p.captions == "box" else "#ffffff", preset="band" if p.captions == "box" else "outline"))
    elif p.captions == "none":
        layout.layers = [ly for ly in layout.layers if ly.type != "captions"]
    return {"video": _render_video(ctx, inputs, layout, p.tighten, p.seed, p.encoder, p.quality)}


# ---- grid_plan ---------------------------------------------------------------------------------------------------

class GridPlanParams(BaseModel):
    """Render a 3, 6 or 9 post set as one grid: each tile is a `template` (default popart_cutout) with the row's palette
    and a pattern that differs from its neighbours. Tiles come out in grid reading order (row by row, 3 columns); post
    them in reverse, because Instagram shows the newest first. With split=true the first input photo is cut into
    `tiles` pieces, one photo across the whole grid."""
    model_config = ConfigDict(extra="forbid")
    template: str = "popart_cutout"
    palettes: list[str] = Field(default_factory=lambda: ["popart-yellow", "popart-pink", "lilac"], min_length=1)
    slots: dict = Field(default_factory=dict, description="extra slot values for every tile")
    cutout: bool = Field(default=False, description="cut each subject out (CPU ONNX masks); needs a template slot 'cutout'")
    split: bool = False
    tiles: Literal[3, 6, 9] = 9  # split only: how many tiles to cut
    seed: int = 0


def _check_splittable(template: str) -> None:
    """`split` cuts one photo across the grid, so each tile's photo must be full-bleed: a `rect` photo layer covering the
    whole canvas, with no outlines and no cutout (spec amendment 2026-10-20)."""
    declared = {t["name"]: t["slots"] for t in C.list_templates()}.get(template, {})
    lay = C.load_template(template, {k: v for k, v in {"photo": 0, "cutout": False}.items() if k in declared})
    photos = [ly for ly in lay.layers if ly.type == "photo" and ly.src == 0]
    full = C.Box()
    ok = bool(photos) and all(ly.shape == "rect" and not ly.outlines and not ly.cutout and ly.box == full for ly in photos)
    if not ok:
        raise ValueError(f"split needs a template whose photo layer is a full-bleed rect without cutout; "
                         f"'{template}' is not (try brag_collage)")


def grid_plan_fn(ctx, inputs, p: GridPlanParams) -> dict[str, str]:
    d = ctx.settings.media_dir
    imgs = _images(ctx, inputs["images"])
    if not imgs:
        raise ValueError("no images in the set")
    n = p.tiles if p.split else len(imgs)
    plan = C.plan_grid(n, p.palettes, p.template)
    declared = {t["name"]: t["slots"] for t in C.list_templates()}.get(p.template)
    if declared is None:
        raise ValueError(f"unknown template '{p.template}'")
    for need in ("photo", "palette"):
        if need not in declared:
            raise ValueError(f"template '{p.template}' needs a '{need}' slot to be used in a grid")
    if p.split:
        _check_splittable(p.template)
    mask_fn = C.default_mask_fn(ctx.settings.home / "models") if p.cutout else None
    items = []
    with tempfile.TemporaryDirectory(prefix="wonhee-grid-") as tmp:
        tmpd = Path(tmp)
        pieces: list[Image.Image] = []
        if p.split:
            probe = C.load_template(p.template, {})
            tw, th = probe.canvas.w, probe.canvas.h
            from PIL import ImageOps
            src = ImageOps.exif_transpose(imgs[0]).convert("RGB")
            rows = n // 3
            full = ImageOps.fit(src, (tw * 3, th * rows), Image.LANCZOS)
            pieces = [full.crop(((t.col) * tw, t.row * th, (t.col + 1) * tw, (t.row + 1) * th)) for t in plan]
        for t in plan:
            slots = {**p.slots, "palette": t.palette, "photo": 0}
            if "pattern" in declared:
                slots.setdefault("pattern", t.pattern)
            if "cutout" in declared:
                slots["cutout"] = p.cutout
            layout = C.load_template(t.template, slots)
            tile_in = [pieces[t.index]] if p.split else [imgs[t.index]]
            img = C.render(layout, tile_in, p.seed + t.index, mask_fn=mask_fn)
            items.append(Slide(sha=_store_png(ctx, img, tmpd, f"tile{t.index}.png")))
    ctx.flag("GRID_PLANNED", json.dumps([t.__dict__ for t in plan]))
    return {"images": save_images(ctx.conn, d, Images(items=items))}


def _salt(_settings) -> dict:
    return {"mask": "u2net-onnx", "templates": C.templates_digest()}  # editing a template is a cache miss


registry.register(Node("compose", 3, {"images": "images?", "refs": "text?"}, {"image": "image"}, ComposeParams, compose_fn,
                       "Render a layout (canvas + layers: photo, pattern, shape, sticker, text, band) or a named "
                       "template (meme_frame, sticker_frame, brag_collage, popart_cutout, slop_card) with slots.",
                       salt=_salt))
registry.register(Node("compose_video", 4, {"video": "video", "subtitles": "subtitles?", "images": "images?"},
                       {"video": "video"}, ComposeVideoParams, compose_video_fn,
                       "A reel from a layout or style template: timed layers, animated text, word captions "
                       "(chunk / word / karaoke), kinetic stack, scrolling list, insets, optional pause trimming; "
                       "one encode.", salt=_salt))
registry.register(Node("frame_video", 4, {"video": "video", "subtitles": "subtitles?"}, {"video": "video"},
                       FrameVideoParams, frame_video_fn,
                       "Put a video in a layout (meme frame: bands, title) with optional 1-3 word captions; one encode "
                       "(the first form of compose_video).", salt=_salt))
registry.register(Node("grid_plan", 3, {"images": "images"}, {"images": "images"}, GridPlanParams, grid_plan_fn,
                       "Render a 3/6/9 post set as one profile grid: a template, palette and pattern per tile.",
                       salt=_salt))
