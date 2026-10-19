"""viz engines and nodes on a tiny generated repo. No browser: export_png is replaced by a fake that writes a PNG
(the one real Playwright export is in tests/live)."""
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tests.fakes import fake_deps
from tests.test_nodes_subs import flags, out, run_chain
from wonhee import media
from wonhee.edit.viz import depgraph, depgraph_html, diskviz, diskviz_html, history
from wonhee.edit.viz.colors import hash_hue
from wonhee.edit.viz.ignore import globs_spec
from wonhee.edit.viz.slice import neighbourhood
from wonhee.edit.viz.source import snapshot
from wonhee.nodes import viz as vizn


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
                          env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                               "GIT_COMMITTER_EMAIL": "t@t", "PATH": "/usr/bin:/bin",
                               "GIT_AUTHOR_DATE": "2026-03-02T12:00:00", "GIT_COMMITTER_DATE": "2026-03-02T12:00:00"}
                          ).stdout


@pytest.fixture
def repo(tmp_path):
    """backend/app is a package whose modules import each other as `app.x`; one commit adds ledger.py later."""
    r = tmp_path / "proj"
    pkg = r / "backend" / "app"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "core.py").write_text("import os\n\nVALUE = 1\n")
    (pkg / "api.py").write_text("from app.core import VALUE\nfrom app import util\n")
    (pkg / "util.py").write_text("from . import core\n")
    (r / "web").mkdir()
    (r / "web" / "main.ts").write_text("import { x } from './lib'\n")
    (r / "web" / "lib.ts").write_text("export const x = 1\n")
    (r / "private").mkdir()
    (r / "private" / "notes.py").write_text("SECRET = 1\n")
    (r / ".gitignore").write_text("data/\n")
    (r / "data").mkdir()
    (r / "data" / "dump.py").write_text("x = 1\n")
    subprocess.run(["git", "init", "-q", str(r)], check=True)
    _git(r, "add", ".")
    _git(r, "commit", "-qm", "first")
    (pkg / "ledger.py").write_text("from app.core import VALUE\nfrom app.util import core\n")
    _git(r, "add", ".")
    _git(r, "commit", "-qm", "ledger")
    return r


def _edges(g):
    return {(e.from_, e.to) for e in g.edges}


def test_python_imports_resolve_from_the_package_root(repo):
    g = depgraph.build_graph(repo / "backend")
    e = _edges(g)
    assert ("app/api.py", "app/core.py") in e      # from app.core import ...
    assert ("app/api.py", "app/util.py") in e      # from app import util -> a submodule
    assert ("app/util.py", "app/core.py") in e     # from . import core
    # the same files drawn from the repo root still resolve, because backend/ is found as the import root
    assert ("backend/app/api.py", "backend/app/core.py") in _edges(depgraph.build_graph(repo))


def test_scan_root_inside_a_package_keeps_its_dotted_prefix(repo):
    g = depgraph.build_graph(repo / "backend" / "app")
    assert ("api.py", "core.py") in _edges(g)


def test_typescript_relative_imports_and_language_tags(repo):
    g = depgraph.build_graph(repo / "web")
    assert ("main.ts", "lib.ts") in _edges(g)
    assert {n.language for n in g.nodes} == {"typescript"}


def test_include_exclude_globs(repo):
    g = depgraph.build_graph(repo, exclude=globs_spec(["private/", "web/"]))
    paths = {n.path for n in g.nodes}
    assert not any(p.startswith(("private/", "web/")) for p in paths) and "backend/app/core.py" in paths
    g = depgraph.build_graph(repo, include=globs_spec(["web/**"]))
    assert {n.path for n in g.nodes} == {"web/main.ts", "web/lib.ts"}


def test_working_tree_scan_skips_gitignored_files_unless_asked(repo):
    with snapshot(repo, None) as src:
        tracked = {n.path for n in depgraph.build_graph(src.root, only=src.only).nodes}
    with snapshot(repo, None, tracked=False) as src:
        everything = {n.path for n in depgraph.build_graph(src.root, only=src.only).nodes}
    assert "data/dump.py" not in tracked and "data/dump.py" in everything


def test_snapshot_at_a_past_rev_sees_only_that_commit(repo):
    first = _git(repo, "rev-list", "--max-parents=0", "HEAD").strip()
    with snapshot(repo / "backend", first) as src:
        files = {n.path for n in depgraph.build_graph(src.root).nodes}
        assert src.sha == first
    assert "app/ledger.py" not in files and "app/core.py" in files
    assert (repo / "backend" / "app" / "ledger.py").exists()  # the working tree is untouched


def test_neighbourhood_keeps_focus_and_n_hops(repo):
    g = depgraph.build_graph(repo / "backend")
    s0 = neighbourhood(g, globs_spec(["**/ledger.py"]), hops=0)
    assert {n.path for n in s0.nodes} == {"app/ledger.py"} and not s0.edges
    s1 = neighbourhood(g, globs_spec(["**/ledger.py"]), hops=1)
    assert {n.path for n in s1.nodes} == {"app/ledger.py", "app/core.py", "app/util.py"}
    assert neighbourhood(g, globs_spec(["nothing*"])).nodes == []


def test_depgraph_html_carries_every_node_and_the_settle_flag(repo):
    g = depgraph.build_graph(repo)
    html = depgraph_html.render_html(g, title="t", theme="paper", show_hint=False)
    assert all(n.path in html for n in g.nodes)
    assert "window.__vizkitSettled" in html and "#F4F2EB" in html


def test_diskviz_labels_off_bakes_no_names(repo):
    tree = diskviz.build_tree(repo)
    assert diskviz.tree_stats(tree)["file_count"] >= 6
    assert "> Infinity)" in diskviz_html.render_html(tree, "t", labels=False)
    assert "> 2200)" in diskviz_html.render_html(tree, "t", labels=True)


def test_commit_heatmap_and_language_mix_draw_at_the_asked_size(repo):
    days = history.commit_days(repo)
    assert len(days) == 2
    assert history.commit_heatmap(days, "dark", (800, 400), "t").size == (800, 400)
    im = history.language_mix(depgraph.build_graph(repo), "paper", (600, 300), "t")
    assert im.size == (600, 300)
    assert history.commit_heatmap([], "paper", (300, 150)).size == (300, 150)  # empty repo: a note, no crash


def test_hash_hue_matches_the_js_template():
    # the JS mirror in depgraph_html must stay identical (same hash, same modulo)
    assert hash_hue("python") == hash_hue("python") and 0 <= hash_hue("typescript") < 360
    assert "Math.floor((h * 137.508) % 360)" in depgraph_html._TEMPLATE


@pytest.fixture
def fake_export(monkeypatch):
    seen = {}

    def export(html_path, png_path, w, h, **_):
        seen["html"] = Path(html_path).read_text()
        Image.new("RGB", (w, h), "white").save(png_path)
    monkeypatch.setattr(vizn, "export_png", export)
    return seen


def test_viz_depgraph_node_stores_the_png_and_flags_its_source(conn, settings, repo, fake_export):
    first = _git(repo, "rev-list", "--max-parents=0", "HEAD").strip()
    rid, status = run_chain(conn, settings, [{"id": "v", "node": "viz_depgraph", "params": {
        "path": str(repo / "backend"), "rev": first, "size": "640x320"}}], fake_deps())
    assert status == "done"
    png = media.path_for(settings.media_dir, out(conn, rid, "v", "image"))
    assert Image.open(png).size == (640, 320)
    assert "VIZ_SOURCE" in flags(conn, rid, "v") and "ledger.py" not in fake_export["html"]


def test_viz_depgraph_flags_a_graph_with_no_imports(conn, settings, repo, fake_export):
    rid, status = run_chain(conn, settings, [{"id": "v", "node": "viz_depgraph", "params": {
        "path": str(repo / "private")}}], fake_deps())
    assert status in ("done", "weak") and "VIZ_NO_EDGES" in flags(conn, rid, "v")


def test_viz_depgraph_fails_cleanly_when_nothing_matches(conn, settings, repo, fake_export):
    rid, status = run_chain(conn, settings, [{"id": "v", "node": "viz_depgraph", "params": {
        "path": str(repo), "focus": ["nope/*"]}}], fake_deps())
    assert status == "failed" and "VIZ_EMPTY" in flags(conn, rid, "v")


def test_missing_browser_becomes_export_unavailable(conn, settings, repo, monkeypatch):
    from wonhee.edit.viz.export import ExportUnavailable

    def broken(*a, **k):
        raise ExportUnavailable("no chromium")
    monkeypatch.setattr(vizn, "export_png", broken)
    rid, status = run_chain(conn, settings, [{"id": "v", "node": "viz_depgraph", "params": {
        "path": str(repo)}}], fake_deps())
    assert status == "failed" and "EXPORT_UNAVAILABLE" in flags(conn, rid, "v")


def test_viz_history_node_draws_without_a_browser(conn, settings, repo):
    rid, status = run_chain(conn, settings, [{"id": "h", "node": "viz_history", "params": {
        "path": str(repo), "chart": "commits", "size": "600x300"}}], fake_deps())
    assert status == "done"
    assert Image.open(media.path_for(settings.media_dir, out(conn, rid, "h", "image"))).size == (600, 300)


def test_viz_render_chains_into_a_finishing_node(conn, settings, repo, fake_export):
    rid, status = run_chain(conn, settings, [
        {"id": "v", "node": "viz_depgraph", "params": {"path": str(repo), "size": "320x160"}},
        {"id": "l", "node": "look_image", "in": {"image": "v.image"}, "params": {"preset": "faded"}}], fake_deps())
    assert status == "done" and out(conn, rid, "l", "image") != out(conn, rid, "v", "image")


def test_cutout_modes():
    im = np.full((10, 10, 3), 200, np.uint8)
    m = np.zeros((10, 10), np.float32)
    m[2:8, 2:8] = 1
    rgba = vizn.cutout(im, m, "transparent")
    assert rgba.shape == (10, 10, 4) and rgba[0, 0, 3] == 0 and rgba[5, 5, 3] == 255
    flat = vizn.cutout(im, m, "dark")
    assert tuple(flat[0, 0]) == vizn.BACKDROP["dark"] and tuple(flat[5, 5]) == (200, 200, 200)
    assert vizn.cutout(im, m, "blur").shape == (10, 10, 3)


def test_mask_cutout_node_uses_the_subject_mask(conn, settings, tmp_path, monkeypatch):
    from wonhee.db import repo as dbrepo
    src = tmp_path / "a.png"
    Image.new("RGB", (40, 20), (10, 120, 200)).save(src)
    sha = dbrepo.save_file(conn, settings.media_dir, src, "image", "image/png")
    monkeypatch.setattr(vizn.mk, "subject", lambda im, d, kind, feather: np.ones(im.shape[:2], np.float32))
    rid, status = run_chain(conn, settings, [{"id": "c", "node": "mask_cutout", "in": {"image": sha},
                                              "params": {"background": "transparent"}}], fake_deps())
    assert status == "done"
    assert Image.open(media.path_for(settings.media_dir, out(conn, rid, "c", "image"))).mode == "RGBA"


def test_wavegen_judge_nudges_knobs_then_accepts(monkeypatch):
    from wonhee.edit.viz.wavegen import generate as g
    from wonhee.edit.viz.wavegen.judge import Adjust, Verdict
    seen = []
    monkeypatch.setattr(g, "render", lambda params, image: (seen.append(params), b"\x89PNG fake")[1])
    verdicts = iter([Verdict(accept=False, adjust=Adjust(chaos=0.9, color_theme="not-a-theme")),
                     Verdict(accept=True, reasoning="ok")])
    calls = []

    def judge(png, mood, chaos, lin, theme):
        calls.append((chaos, theme))
        return next(verdicts)
    png, notes = g.generate("storm", "ocean", "64x32", 0.2, 0.5, image="img", seed=1, judge=judge)
    assert png.startswith(b"\x89PNG") and len(seen) == 2 and len(notes) == 2
    assert calls == [(0.2, "ocean"), (0.9, "ocean")]  # chaos moved, an unknown theme was ignored
