import json
import os

import numpy as np
import pytest

from wonhee import art
from wonhee.art import compose, graphkit as gk, metrics

GRAPH_EFFECTS = [n for n in art.names(family="graph")]


def test_graph_effects_registered():
    assert len(GRAPH_EFFECTS) >= 12
    for must in ("graph-dag-flow", "graph-critical-path", "graph-force-glow", "graph-matrix", "graph-btree", "graph-skiplist", "graph-ring-wheel"):
        assert must in GRAPH_EFFECTS


@pytest.mark.parametrize("name", GRAPH_EFFECTS)
def test_effect_shape_dtype_determinism_nonconstant(name):
    kw = dict(size=(300, 200), palette="ember")
    a = compose.render_effect(name, seed=1, **kw)
    b = compose.render_effect(name, seed=1, **kw)
    c = compose.render_effect(name, seed=2, **kw)
    assert a.shape == (200, 300, 3) and a.dtype == np.float32
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.std() > 0.03 and a.min() >= 0 and a.max() <= 1


@pytest.mark.parametrize("name", ["graph-dag-flow", "graph-force-glow", "graph-matrix", "graph-constellation", "graph-skiplist", "graph-ring-wheel"])
def test_no_focal_point_at_card_size(name):
    rep = metrics.analyze(compose.render_effect(name, seed=0, size=(900, 600), palette="ultramarine"))
    assert rep.center_bias <= 1.6 and rep.focal_ratio <= 3.6 and rep.coverage >= 0.75, rep


# --- loaders on tiny graphs ----------------------------------------------------------------------------------------
def test_loader_nodes_edges():
    g = gk.as_graph({"nodes": [{"id": "a", "group": "x", "weight": 3}, {"id": "b", "group": "y"}, "c"],
                     "edges": [{"source": "a", "target": "b", "weight": 2}, {"source": "b", "target": "c"}, {"source": "a", "target": "a"}]})
    assert g.n == 3 and g.m == 2 and g.group_names == ["", "x", "y"] and g.weight[0] == 3 and g.eweight[0] == 2


def test_loader_imports_dict_and_adjacency_and_pairs():
    g = gk.as_graph({"pkg/a.py": ["pkg/b.py", "util/c.py"], "pkg/b.py": ["util/c.py"]})
    assert g.n == 3 and g.m == 3 and set(g.group_names) == {"pkg", "util"}
    h = gk.as_graph({"a": {"b": {"weight": 2}, "c": 1}, "b": {"c": {}}})
    assert h.n == 3 and h.m == 3
    p = gk.as_graph([("x", "y"), ("y", "z")])
    assert p.n == 3 and p.m == 2
    ev = gk.as_graph({"events": [{"type": "job_created", "jobId": "p", "job": {"kind": "plan"}}, {"type": "job_created", "jobId": "s", "parentJobId": "p", "job": {"kind": "search"}}]})
    assert ev.n == 2 and ev.m == 1
    flow = gk.as_graph({"nodes": [{"id": "a", "layer": 0}, {"id": "b", "layer": 2}], "edges": [{"from": "a", "to": "b"}]})
    assert list(flow.layer) == [0, 2]
    with pytest.raises(ValueError):
        gk.as_graph(42)


def test_from_repo_on_a_tiny_tree(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "a.py").write_text("from . import b\nimport pkg.c\n")
    (tmp_path / "pkg" / "b.py").write_text("from pkg.c import x\n")
    (tmp_path / "pkg" / "c.py").write_text("x = 1\n")
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "m.ts").write_text("import { f } from './n'\nimport z from '@/lib/q'\n")
    (tmp_path / "web" / "n.ts").write_text("export const f = 1\n")
    g = gk.from_repo(str(tmp_path))
    ids = set(g.ids)
    assert {"pkg/a.py", "pkg/b.py", "pkg/c.py", "web/m.ts", "web/n.ts"} <= ids
    e = {(g.ids[a], g.ids[b]) for a, b in g.edges}
    assert ("pkg/a.py", "pkg/b.py") in e and ("pkg/b.py", "pkg/c.py") in e and ("web/m.ts", "web/n.ts") in e


def test_from_ts_muksite_fixtures_if_present():
    base = "/home/muk/projects/muksite/src/components/projectViz"
    f = os.path.join(base, "nexusGraphData.ts")
    if not os.path.exists(f):
        pytest.skip("muksite checkout not available")
    g = gk.from_ts(f)
    assert g.n > 30 and g.m > 20
    e = gk.from_json(os.path.join(base, "euglenaTrace/fixtures/02-wide-coverage-replan.json"))
    assert e.n > 5 and e.m >= 5


def test_layout_and_longest_path_on_tiny_cyclic_graph():
    g = gk.as_graph([("a", "b"), ("b", "c"), ("a", "c"), ("c", "a"), ("c", "d")])
    lay = gk.layered(g, np.random.default_rng(0))
    assert lay.n_real == 4 and np.all((lay.y >= 0) & (lay.y <= 1))
    path = gk.longest_path(g)
    assert [g.ids[i] for i in path] == ["a", "b", "c", "d"]
    # the long edge a -> c gets one dummy waypoint
    assert len(lay.layer) >= 5


@pytest.mark.parametrize("name", ["graph-dag-flow", "graph-critical-path", "graph-force-glow", "graph-matrix", "graph-constellation", "graph-dag-bundle"])
def test_effects_accept_real_data(name):
    data = {"nodes": [{"id": c, "group": "g%d" % (i % 3)} for i, c in enumerate("abcdefghij")],
            "edges": [{"source": a, "target": b} for a, b in zip("abcdefghi", "bcdefghij")] + [{"source": "a", "target": "e"}, {"source": "c", "target": "j"}]}
    img = compose.render_effect(name, size=(240, 160), seed=0, data=data)
    assert img.shape == (160, 240, 3) and img.std() > 0.03
    img2 = compose.render_effect(name, size=(240, 160), seed=0, data=data)
    assert np.array_equal(img, img2)


def test_ring_and_structure_data_hooks():
    a = compose.render_effect("graph-ring-lanes", size=(240, 160), seed=0, data={"values": [0.9, 0.1, 0.5], "head": [0.2, 0.5, 0.8]})
    b = compose.render_effect("graph-ring-lanes", size=(240, 160), seed=0)
    assert not np.array_equal(a, b)


def test_proj_dag_arcs_real_shape():
    import numpy as np
    from wonhee.art import compose
    d = {"nodes": [{"id": "p", "group": "plan"}, {"id": "a", "group": "search"}, {"id": "b", "group": "visit", "weight": 0.4}],
         "edges": [{"source": "p", "target": "a"}, {"source": "a", "target": "b"}]}
    a = compose.render_effect("proj-dag-arcs", size=(300, 200), seed=2, palette="ultramarine", data=d, params={"tiles": 2})
    b = compose.render_effect("proj-dag-arcs", size=(300, 200), seed=2, palette="ultramarine", data=d, params={"tiles": 2})
    assert a.shape == (200, 300, 3) and np.allclose(a, b)

def test_proj_nexus_sectors_runs_with_real_style_data():
    data = {"nodes": [{"id": f"n{i}", "group": i % 3} for i in range(9)], "edges": [{"source": "n0", "target": "n1"}]}
    a = compose.render_effect("proj-nexus-sectors", seed=1, size=(300, 200), palette="ember", data=data, params={"dots": 200, "spokes": 40})
    assert a.shape == (200, 300, 3) and a.std() > 0.02

def test_proj_moeka_traffic():
    a = compose.render_effect("proj-moeka-traffic", seed=1, size=(300, 200), palette="ember")
    b = compose.render_effect("proj-moeka-traffic", seed=1, size=(300, 200), palette="ember")
    assert a.shape == (200, 300, 3) and np.array_equal(a, b) and a.std() > 0.03 and a.min() >= 0 and a.max() <= 1
