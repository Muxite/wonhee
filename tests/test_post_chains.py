"""The post-assistant chains (spec §2.8): docs/chains/image_post.json, talking_head.json, mixed_reel.json.

Every chain must load and validate against the real registry, and the first nodes must run on tiny fixtures. The
chain files name placeholder inputs (a local file such as photo_1.jpg); the run tests bind those to stored blobs of
the port's type. Subject masks are switched off by monkeypatch so nothing loads a model or touches the network.
"""
import json
from pathlib import Path

import pytest

from tests.fakes import fake_deps
from tests.test_nodes_subs import flags, out, run_chain
from wonhee import media
from wonhee.db import repo
from wonhee.edit import mask as mk
from wonhee.nodes import registry
from wonhee.nodes.images import Images, Slide, save_images
from wonhee.nodes.chain import ID_RE, WIRE_RE, parse_chain, validate
from wonhee.nodes.types import PORT_TYPES, is_literal, parse_type

CHAINS = Path(__file__).resolve().parents[1] / "docs" / "chains"
POST_CHAINS = ["image_post.json", "talking_head.json", "mixed_reel.json"]
FAKE_SHA = "a" * 64


@pytest.fixture(autouse=True)
def _nodes():
    registry.load_all()


def load(name):
    return json.loads((CHAINS / name).read_text())


def wire_source(value: str, ids: set[str]):
    """(node id, port) when value is a wire from a node in this chain, else None."""
    m = WIRE_RE.match(value)
    return m.groups() if m and m.group(1) in ids else None


def placeholders(nodes):
    """(node index, port, value) for each input that is neither a wire nor a stored ref."""
    ids = {n["id"] for n in nodes}
    return [(i, port, value) for i, n in enumerate(nodes) for port, value in n.get("in", {}).items()
            if not is_literal(value) and wire_source(value, ids) is None]


def bind(nodes, value_for):
    """A copy of the chain with each placeholder replaced by value_for(node name, port)."""
    nodes = json.loads(json.dumps(nodes))
    for i, port, _ in placeholders(nodes):
        nodes[i]["in"][port] = value_for(nodes[i]["node"], port)
    return nodes


def store_for(conn, settings, photo, video):
    def value_for(node, port):
        t = parse_type(registry.NODES[node].inputs[port])[0]
        if t == "images":
            slides = [Slide(sha=repo.save_file(conn, settings.media_dir, photo, "image", "image/jpeg"))] * 2
            return save_images(conn, settings.media_dir, Images(items=slides))
        if t == "image":
            return repo.save_file(conn, settings.media_dir, photo, "image", "image/jpeg")
        if t == "video":
            return repo.save_file(conn, settings.media_dir, video, "video", "video/mp4")
        raise AssertionError(f"no fixture for {node}.{port} ({t})")
    return value_for


@pytest.mark.parametrize("name", POST_CHAINS)
def test_chain_loads_and_validates_against_registry(name):
    obj = bind(load(name), lambda node, port: FAKE_SHA)  # a stored ref has the right shape; no DB needed
    chain = parse_chain(obj)
    order = validate(chain, registry.NODES)
    assert len(order) == len(obj)
    for cn in chain.nodes:
        assert ID_RE.match(cn.id), cn.id
        node = registry.NODES[cn.node]
        for t in list(node.inputs.values()) + list(node.outputs.values()):
            assert parse_type(t)[0] in PORT_TYPES


@pytest.mark.parametrize("name", POST_CHAINS)
def test_chain_wires_only_declared_ports(name):
    nodes = load(name)
    ids = {n["id"]: n for n in nodes}
    assert len(ids) == len(nodes), "duplicate ids"
    for n in nodes:
        node = registry.NODES[n["node"]]
        for port, value in n.get("in", {}).items():
            assert port in node.inputs, f"{n['id']}: {port} is not an input of {n['node']}"
            src = wire_source(value, set(ids))
            if src:
                src_id, src_port = src
                assert src_port in registry.NODES[ids[src_id]["node"]].outputs


def test_image_post_is_one_photo_from_style_to_card():
    assert [n["node"] for n in load("image_post.json")] == ["text_style", "photo_fit", "text_place", "text_image"]


def test_talking_head_stops_after_take_pick_with_plan_and_clean_audio():
    nodes = load("talking_head.json")
    assert [n["node"] for n in nodes] == ["speech_subs", "take_pick", "audio_clean"]
    assert nodes[1]["in"] == {"subtitles": "subs.subtitles"}


def test_known_port_gaps_are_still_gaps():
    """Each wire below is not expressible today. When glue lands these start passing, and the chains should change."""
    gaps = [
        # text_place's placement (text) has no text input on text_image to land in
        [{"id": "p", "node": "text_place", "in": {"image": FAKE_SHA}, "params": {"text": "x"}},
         {"id": "c", "node": "text_image", "in": {"image": FAKE_SHA, "text": "p.placement"}, "params": {"text": "x"}}],
        # take_pick's plan cannot drive compose_video (no text or plan input there)
        [{"id": "t", "node": "speech_subs", "in": {"video": FAKE_SHA}},
         {"id": "p", "node": "take_pick", "in": {"subtitles": "t.subtitles"}},
         {"id": "r", "node": "compose_video", "in": {"video": FAKE_SHA, "subtitles": "p.plans"},
          "params": {"template": "meme_frame"}}],
        # one image cannot feed carousel, which takes a list of images
        [{"id": "c", "node": "text_image", "in": {"image": FAKE_SHA}, "params": {"text": "x"}},
         {"id": "k", "node": "carousel", "in": {"images": "c.image"}, "params": {"texts": ["x"]}}],
    ]
    for g in gaps:
        with pytest.raises(Exception):
            validate(parse_chain(g), registry.NODES)


@pytest.fixture
def no_subject_model(monkeypatch):
    def boom(*a, **k):
        raise OSError("model not available offline")
    monkeypatch.setattr(mk, "subject", boom)


def test_image_post_runs_on_one_photo(conn, settings, tiny_jpg, no_subject_model):
    nodes = bind(load("image_post.json"), store_for(conn, settings, tiny_jpg, None))
    rid, status = run_chain(conn, settings, nodes, fake_deps())
    assert status == "done"
    assert "NO_SUBJECT_MASK" in flags(conn, rid, "fit") and "NO_SUBJECT_MASK" in flags(conn, rid, "place")
    assert out(conn, rid, "card", "image")


def test_image_post_first_two_nodes_run_alone(conn, settings, tiny_jpg, no_subject_model):
    """Stopping early returns what the chain has: the fit report from the first two nodes."""
    nodes = bind(load("image_post.json"), store_for(conn, settings, tiny_jpg, None))
    rid, status = run_chain(conn, settings, nodes[:2], fake_deps())
    assert status == "done"
    fit = json.loads(media.path_for(settings.media_dir, out(conn, rid, "fit", "fit")).read_text())
    assert fit["verdict"] in ("ok", "tight", "reject")


def test_talking_head_runs_to_plan_and_clean_audio(conn, settings, tiny_video):
    nodes = bind(load("talking_head.json"), store_for(conn, settings, None, tiny_video))
    rid, status = run_chain(conn, settings, nodes, fake_deps())
    assert status == "done"
    plan = json.loads(media.path_for(settings.media_dir, out(conn, rid, "plan", "plans")).read_text())
    assert isinstance(plan, dict)
    assert out(conn, rid, "clean", "video") and out(conn, rid, "clean", "report")


def test_talking_head_first_node_runs_alone(conn, settings, tiny_video):
    nodes = bind(load("talking_head.json"), store_for(conn, settings, None, tiny_video))
    rid, status = run_chain(conn, settings, nodes[:1], fake_deps())
    assert status == "done" and out(conn, rid, "subs", "subtitles")


def test_mixed_reel_joins_the_graded_screen_record_to_a_b_roll_montage(conn, settings, tiny_video, tiny_jpg):
    """Cuts and effects live on the bottom (b-roll) video only; the talking head branch is not joined or cut."""
    nodes = load("mixed_reel.json")
    by_id = {n["id"]: n for n in nodes}
    assert by_id["bottom"]["node"] == "join_videos"
    assert by_id["bottom"]["in"] == {"video": "grade.video", "video_b": "broll.video"}
    assert not any(v.startswith(("subs.", "plan.", "clean.")) for v in by_id["bottom"]["in"].values())
    rid, status = run_chain(conn, settings, bind(nodes, store_for(conn, settings, tiny_jpg, tiny_video)), fake_deps())
    assert status == "done"
    assert out(conn, rid, "grade", "video") and out(conn, rid, "broll", "video") and out(conn, rid, "bottom", "video")
