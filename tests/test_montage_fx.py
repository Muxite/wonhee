"""Tests for the montage fx registry (ported from slopcut's test_fx.py) and a smoke run of every effect in ffmpeg."""
import json
import shutil
import subprocess

import pytest

from wonhee.edit.montage import fx as fxreg
from wonhee.edit.montage.fx import CATEGORIES, FX, Ctx, by_category, expand, get, names, slot_seed

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg missing")

W, H, FPS, N = 180, 320, 30, 15     # ~0.5 s 9:16 clip

# Effects that fail on the ffmpeg this box has, with the reason. Empty means every effect passed when last run.
KNOWN_BROKEN: dict[str, str] = {}


def _norm(size, n):
    """The renderer's post-stack normalisation (slopcut render._norm): size, fps, pix_fmt and exact frame count."""
    return (f"scale={size[0]}:{size[1]},setsar=1,fps={FPS},format=yuv420p,"
            f"tpad=stop_mode=clone:stop_duration={n / FPS:.3f},trim=end_frame={n},setpts=PTS-STARTPTS,settb=1/{FPS}")


def _run(graph, size, n, out=None):
    """Run `graph` (must end in [out]) over a lavfi testsrc of `size`/`n` frames; returns the CompletedProcess."""
    dur = n / FPS
    cmd = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"testsrc=s={size[0]}x{size[1]}:r={FPS}:d={dur:.3f}",
           "-filter_complex", graph, "-map", "[out]"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", str(out)] if out else ["-f", "framecrc", "-"]
    return subprocess.run(cmd, capture_output=True, text=True)


def _probe(path):
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                        "-show_entries", "stream=width,height,nb_read_frames", "-of", "json", str(path)],
                       capture_output=True, text=True, check=True)
    s = json.loads(p.stdout)["streams"][0]
    return s["width"], s["height"], int(s["nb_read_frames"])


def test_registry_and_expand():
    assert {"negative", "fire", "stutter"} <= set(FX)
    ctx = Ctx(270, 480, 30, 30)
    assert expand("negative", ctx, "a", "b", "p") == "[a]negate[b]"
    g = expand("fire", ctx, "a", "b", "p")
    assert g.count("[a]") == 1 and g.endswith("[b]") and "[p_f]" in g and "[IN]" not in g and "[_" not in g
    with pytest.raises(ValueError):
        expand("nope", ctx, "a", "b", "p")


def test_names_get_and_categories():
    all_names = names()
    assert all_names == sorted(FX) and len(all_names) >= 40
    assert get("fire").kind == "overlay" and get("fire").category == "overlay"
    cats = by_category()
    assert tuple(cats) == CATEGORIES and all(cats[c] for c in CATEGORIES)
    assert sorted(n for v in cats.values() for n in v) == all_names      # a partition: each effect in one category
    assert names("cut") == cats["cut"] and "stutter" in cats["cut"]
    with pytest.raises(ValueError, match="fire"):                         # error lists the known names
        get("nope")
    with pytest.raises(ValueError, match="overlay"):
        names("nope")


def test_overlap_with_edit_fx_is_a_separate_namespace():
    from wonhee.edit import fx as pixel_fx
    assert "bloom" in FX and callable(pixel_fx.bloom)
    assert not hasattr(fxreg, "fisheye")        # the registry never re-exports the pixel looks


def test_slot_seed_is_deterministic_and_distinct():
    assert slot_seed(5, 1, 2) == slot_seed(5, 1, 2)
    assert len({slot_seed(5, i, s) for i in range(4) for s in range(3)}) == 12


def test_fragments_are_deterministic():
    ctx = Ctx(W, H, FPS, N, seed=7)
    for name in names():
        assert get(name).build(ctx) == get(name).build(ctx)


@needs_ffmpeg
@pytest.mark.parametrize("name", names())
def test_each_fx_runs_and_keeps_frame_size(tmp_path, name):
    if name in KNOWN_BROKEN:
        pytest.xfail(KNOWN_BROKEN[name])
    ctx = Ctx(W, H, FPS, N, seed=3)
    graph = expand(name, ctx, "0:v", "f", "p") + f";[f]{_norm((W, H), N)}[out]"
    out = tmp_path / "o.mp4"
    proc = _run(graph, (W, H), N, out)
    assert proc.returncode == 0, proc.stderr[-800:]
    assert _probe(out) == (W, H, N)


@needs_ffmpeg
@pytest.mark.parametrize("name", names())
def test_fx_never_drops_a_short_clip(name):
    # A stage that emits nothing stalls the xfade chain and freezes the rest of the video on one frame
    # (stutter -> rewind_jitter on a 15-frame clip used to emit 0). Every fx must emit >= 8.
    if name in KNOWN_BROKEN:
        pytest.xfail(KNOWN_BROKEN[name])
    n = 15
    first = expand(name, Ctx(64, 64, FPS, n, 5), "0:v", "o", "p") + ";[o]tpad=stop=0,null[out]"
    chained = (expand("stutter", Ctx(64, 64, FPS, n, 5), "0:v", "s", "q") + ";"
               + expand(name, Ctx(64, 64, FPS, 8, 5), "s", "o", "p") + ";[o]null[out]")
    for graph in (first, chained):
        proc = _run(graph, (64, 64), n)
        assert proc.returncode == 0, proc.stderr[-800:]
        frames = sum(1 for line in proc.stdout.splitlines() if line and not line.startswith("#"))
        assert frames >= 8, (name, frames)


@needs_ffmpeg
def test_stacked_fx_keep_frame_count(tmp_path):
    ns = names()
    parts, label = [], "0:v"
    for k in range(4):
        parts.append(expand(ns[k * 7 % len(ns)], Ctx(W, H, FPS, N, slot_seed(1, 0, k)), label, f"f{k}", f"x{k}"))
        parts.append(f"[f{k}]scale={W}:{H},setsar=1[r{k}]")
        label = f"r{k}"
    out = tmp_path / "o.mp4"
    proc = _run(";".join(parts) + f";[{label}]{_norm((W, H), N)}[out]", (W, H), N, out)
    assert proc.returncode == 0, proc.stderr[-800:]
    assert _probe(out) == (W, H, N)


@needs_ffmpeg
def test_size_changing_fx_can_be_followed_by_an_overlay(tmp_path):
    # At 360x640 tile_grid outputs 636 rows (3 x 212); fire's blend then needs its source rescaled to match.
    size, n = (360, 640), 15
    g = (expand("tile_grid", Ctx(*size, FPS, n, 1), "0:v", "a", "p") + f";[a]scale={size[0]}:{size[1]},setsar=1[b];"
         + expand("fire", Ctx(*size, FPS, n, 2), "b", "c", "q") + f";[c]{_norm(size, n)}[out]")
    out = tmp_path / "tg.mp4"
    proc = _run(g, size, n, out)
    assert proc.returncode == 0, proc.stderr[-800:]
    assert _probe(out) == (*size, n)
