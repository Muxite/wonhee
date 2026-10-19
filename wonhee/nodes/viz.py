"""viz nodes: pictures made from data rather than from footage (moved in from vizkit 2026-10-19), plus a subject
cutout for any still.

- viz_depgraph (-> image): a repo's file dependency graph, nodes sized by lines of code, coloured by language.
- viz_diskviz (-> image): a treemap or circle-pack of a path by real file size.
- viz_history (-> image): a commit heatmap, or a lines-of-code-per-language bar.
- viz_wavegen (-> image): a WaveSim shader still from mood knobs, checked by a vision judge.
- mask_cutout (image -> image): keep the subject, replace the background.

Render nodes read a live path the cache key cannot see, so they are not cached; `rev` pins a past commit instead,
and every render flags VIZ_SOURCE with the resolved commit so the picture can be made again. Their output chains
straight into filter_image / look_image / poster, and publish_muksite puts the result on the portfolio.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from wonhee.db import repo
from wonhee.edit import mask as mk
from wonhee.edit import style
from wonhee.edit.viz import depgraph, depgraph_html, diskviz, diskviz_html, history
from wonhee.edit.viz.export import ExportUnavailable, export_png
from wonhee.edit.viz.ignore import globs_spec
from wonhee.edit.viz.slice import neighbourhood
from wonhee.edit.viz.source import SourceError, snapshot
from wonhee.edit.viz.themes import THEMES, parse_size
from wonhee.edit.viz.wavegen import judge as wjudge
from wonhee.edit.viz.wavegen.generate import generate as wavegen_generate
from wonhee.edit.viz.wavegen.heuristics import COLOR_THEMES
from wonhee.media import path_for
from wonhee.nodes import registry
from wonhee.nodes.base import Node
from wonhee.pipeline.context import StageFailed

Theme = Literal["paper", "dark"]
SIZE = r"^\d{2,5}x\d{2,5}$"
assert set(Theme.__args__) == set(THEMES)  # type: ignore[attr-defined]


class SourceParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(description="directory to draw (a repo, or a folder inside one)")
    rev: str | None = Field(default=None, description="draw the path as of this commit/tag instead of the working tree")
    include: list[str] = Field(default=[], description="gitignore-style globs; only matching files are drawn")
    exclude: list[str] = Field(default=[], description="gitignore-style globs to leave out (private data, vendored)")
    tracked: bool = Field(default=True, description="inside a git repo, draw only files git tracks (no ignored data, "
                                                     "venvs or submodules)")
    title: str | None = None
    theme: Theme = "paper"
    size: str = Field(default="1200x600", pattern=SIZE)


def _store_png(ctx, png: Path) -> str:
    return repo.save_file(ctx.conn, ctx.settings.media_dir, png, "image", "image/png")


def _export(html: str, size: str) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="wonhee-viz-"))
    (tmp / "page.html").write_text(html, encoding="utf-8")
    try:
        export_png(tmp / "page.html", tmp / "out.png", *parse_size(size))
    except ExportUnavailable as e:
        raise StageFailed("EXPORT_UNAVAILABLE", str(e)) from e
    return tmp / "out.png"


def _source_note(ctx, p: SourceParams, sha: str | None, counts: str) -> None:
    where = f"{p.path} @ {sha[:12]}" if sha else f"{p.path} (working tree)"
    ctx.flag("VIZ_SOURCE", f"{where}; {counts}")


# --- viz_depgraph ----------------------------------------------------------------------------------------------

class DepgraphParams(SourceParams):
    focus: list[str] = Field(default=[], description="globs: keep only these files and their import neighbours")
    hops: int = Field(default=1, ge=0, le=5, description="how many import steps out from the focus files to keep")
    hint: bool = Field(default=False, description="draw the file/LOC count line in the corner")


def depgraph_fn(ctx, inputs, p: DepgraphParams) -> dict[str, str]:
    try:
        with snapshot(Path(p.path), p.rev, p.tracked) as src:
            graph = depgraph.build_graph(src.root, globs_spec(p.include), globs_spec(p.exclude), src.only)
            title = p.title or Path(p.path).expanduser().resolve().name
            sha = src.sha
    except SourceError as e:
        raise StageFailed("VIZ_EMPTY", str(e)) from e
    if p.focus:
        graph = neighbourhood(graph, globs_spec(p.focus), p.hops)
    s = graph.stats()
    if not graph.nodes:
        raise StageFailed("VIZ_EMPTY", f"no code files in {p.path} after include/exclude/focus")
    _source_note(ctx, p, sha, f"{s['file_count']} files, {s['total_loc']} LOC, {s['edge_count']} imports")
    if not graph.edges:
        ctx.flag("VIZ_NO_EDGES", "no imports resolved; point path at the import root (e.g. backend/)")
    html = depgraph_html.render_html(graph, title=title, theme=p.theme, show_hint=p.hint)
    return {"image": _store_png(ctx, _export(html, p.size))}


# --- viz_diskviz -----------------------------------------------------------------------------------------------

class DiskvizParams(SourceParams):
    layout: Literal["rect", "circle"] = "rect"
    labels: bool = Field(default=False, description="bake file names into big cells (off: safe for private repos)")
    hint: bool = False


def diskviz_fn(ctx, inputs, p: DiskvizParams) -> dict[str, str]:
    try:
        with snapshot(Path(p.path), p.rev, p.tracked) as src:
            tree = diskviz.build_tree(src.root, globs_spec(p.include), globs_spec(p.exclude), src.only)
            title = p.title or Path(p.path).expanduser().resolve().name
            sha = src.sha
    except SourceError as e:
        raise StageFailed("VIZ_EMPTY", str(e)) from e
    s = diskviz.tree_stats(tree)
    if not s["file_count"] or not s["total_bytes"]:
        raise StageFailed("VIZ_EMPTY", f"no files in {p.path} after include/exclude")
    _source_note(ctx, p, sha, f"{s['file_count']} files, {s['total_bytes']} bytes")
    html = diskviz_html.render_html(tree, title=title, layout=p.layout, theme=p.theme, show_hint=p.hint,
                                    labels=p.labels)
    return {"image": _store_png(ctx, _export(html, p.size))}


# --- viz_history -----------------------------------------------------------------------------------------------

class HistoryParams(SourceParams):
    chart: Literal["commits", "languages"] = "commits"
    branch: str = Field(default="HEAD", description="commits: whose history to count (rev is ignored)")


def history_fn(ctx, inputs, p: HistoryParams) -> dict[str, str]:
    size = parse_size(p.size)
    path = Path(p.path).expanduser().resolve()
    title = p.title or ""
    if p.chart == "commits":
        try:
            days = history.commit_days(path, p.branch)
        except ValueError as e:
            raise StageFailed("VIZ_EMPTY", str(e)) from e
        if not days:
            raise StageFailed("VIZ_EMPTY", f"no commits on {p.branch} in {path}")
        ctx.flag("VIZ_SOURCE", f"{path} {p.branch}: {len(days)} commits")
        im = history.commit_heatmap(days, p.theme, size, title)
    else:
        try:
            with snapshot(path, p.rev, p.tracked) as src:
                graph = depgraph.build_graph(src.root, globs_spec(p.include), globs_spec(p.exclude), src.only)
                sha = src.sha
        except SourceError as e:
            raise StageFailed("VIZ_EMPTY", str(e)) from e
        if not graph.nodes:
            raise StageFailed("VIZ_EMPTY", f"no code files in {path}")
        _source_note(ctx, p, sha, f"{len(graph.nodes)} files")
        im = history.language_mix(graph, p.theme, size, title)
    with tempfile.TemporaryDirectory(prefix="wonhee-viz-") as tmp:
        out = Path(tmp) / "out.png"
        im.save(out, optimize=True)
        return {"image": _store_png(ctx, out)}


# --- viz_wavegen -----------------------------------------------------------------------------------------------

class WavegenParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mood: str = Field(default="calm ocean", max_length=120)
    color_theme: Literal[tuple(COLOR_THEMES)] | None = None  # type: ignore[valid-type]
    resolution: str = Field(default="1200x600", pattern=SIZE)
    chaos: float = Field(default=0.4, ge=0, le=1)
    linearity: float = Field(default=0.3, ge=0, le=1)
    seed: int | None = Field(default=None, ge=0)
    judge: bool = Field(default=True, description="let a vision model accept or nudge each render (up to 3 tries)")
    engine: Literal["local", "remote"] = "remote"


def wavegen_fn(ctx, inputs, p: WavegenParams) -> dict[str, str]:
    from wonhee.nodes.vibe import _ask  # the shared local/remote vision call

    def judge(png: bytes, mood: str, chaos: float, linearity: float, theme: str | None) -> wjudge.Verdict:
        with tempfile.TemporaryDirectory(prefix="wonhee-wave-") as tmp:
            f = Path(tmp) / "frame.png"
            f.write_bytes(png)
            return _ask(ctx, p.engine, wjudge.Verdict, wjudge.RUBRIC,
                        wjudge.prompt(mood, chaos, linearity, theme), [f], "wavegen_verdict")

    try:
        png, notes = wavegen_generate(p.mood, p.color_theme, p.resolution, p.chaos, p.linearity,
                                      image=ctx.settings.wavesim_image, seed=p.seed,
                                      judge=judge if p.judge else None)
    except (RuntimeError, FileNotFoundError) as e:
        raise StageFailed("WAVESIM_FAILED", str(e)[:400]) from e
    ctx.flag("VIZ_SOURCE", f"wavesim {ctx.settings.wavesim_image}; " + " | ".join(notes))
    with tempfile.TemporaryDirectory(prefix="wonhee-wave-") as tmp:
        out = Path(tmp) / "out.png"
        out.write_bytes(png)
        return {"image": _store_png(ctx, out)}


# --- mask_cutout -----------------------------------------------------------------------------------------------

BACKDROP = {"paper": (235, 242, 244), "dark": (18, 13, 11)}  # BGR of the viz themes' backgrounds


class CutoutParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: mk.Kind = Field(default="object", description="person, or object (any salient thing)")
    background: Literal["transparent", "paper", "dark", "blur"] = "transparent"
    feather: float = Field(default=0.004, ge=0, le=0.05, description="edge softness, x the long side")


def cutout(im: np.ndarray, m: np.ndarray, background: str) -> np.ndarray:
    """BGR image + 0..1 mask -> BGRA (transparent) or BGR with the background replaced."""
    if background == "transparent":
        return np.dstack([im, (m * 255).round().astype(np.uint8)])
    if background == "blur":
        back = cv2.GaussianBlur(im, (0, 0), max(4.0, 0.02 * max(im.shape[:2])))
    else:
        back = np.empty_like(im)
        back[:] = BACKDROP[background]
    a = m[..., None]
    return (im.astype(np.float32) * a + back.astype(np.float32) * (1 - a)).round().astype(np.uint8)


def cutout_fn(ctx, inputs, p: CutoutParams) -> dict[str, str]:
    im = style.load(path_for(ctx.settings.media_dir, inputs["image"]))
    m = mk.subject(im, ctx.settings.home / "models", p.kind, p.feather)
    out = cutout(im, m, p.background)
    with tempfile.TemporaryDirectory(prefix="wonhee-cutout-") as tmp:
        dest = Path(tmp) / "out.png"  # png either way: it keeps the alpha channel and the cut edge sharp
        cv2.imwrite(str(dest), out)
        return {"image": _store_png(ctx, dest)}


registry.register(Node("viz_depgraph", 1, {}, {"image": "image"}, DepgraphParams, depgraph_fn,
                       "Draw a repo's file dependency graph (size = lines of code, colour = language).",
                       cacheable=False))
registry.register(Node("viz_diskviz", 1, {}, {"image": "image"}, DiskvizParams, diskviz_fn,
                       "Draw a treemap or circle-pack of a folder by real file size.", cacheable=False))
registry.register(Node("viz_history", 1, {}, {"image": "image"}, HistoryParams, history_fn,
                       "Draw a repo's commit heatmap or its lines-of-code-per-language bar.", cacheable=False))
registry.register(Node("viz_wavegen", 1, {}, {"image": "image"}, WavegenParams, wavegen_fn,
                       "Render a WaveSim shader still from mood knobs, judged by a vision model (GPU, docker).",
                       cacheable=False, lane="cuda",
                       salt=lambda s: {"llm": s.llm_model, "wavesim": s.wavesim_image}))
registry.register(Node("mask_cutout", 1, {"image": "image"}, {"image": "image"}, CutoutParams, cutout_fn,
                       "Keep the subject (person or object) and make the background transparent, flat or blurred.",
                       salt=lambda s: {"mask": "u2net-onnx"}))
