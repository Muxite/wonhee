"""One real browser export: the depgraph page loads d3 from cdnjs, settles, and Chromium screenshots it at the
asked size. Needs network and `uv sync --extra viz && uv run playwright install chromium-headless-shell`."""
import pytest
from PIL import Image

from wonhee.edit.viz import depgraph, depgraph_html
from wonhee.edit.viz.export import export_png

pytestmark = pytest.mark.live


def test_depgraph_exports_a_settled_png(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "a.py").write_text("from pkg import b\n")
    (tmp_path / "pkg" / "b.py").write_text("X = 1\n")
    html = tmp_path / "g.html"
    html.write_text(depgraph_html.render_html(depgraph.build_graph(tmp_path), "t", "paper", show_hint=False))
    export_png(html, tmp_path / "g.png", 400, 200)
    im = Image.open(tmp_path / "g.png").convert("RGB")
    assert im.size == (400, 200)
    assert len(set(im.getdata())) > 2  # nodes were drawn, not just the background
