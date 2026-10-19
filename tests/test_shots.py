import json

import numpy as np
import pytest

from tests.fakes import fake_deps
from tests.test_nodes_subs import out, run_chain, store_video
from wonhee import service
from wonhee.analyze import embed
from wonhee.media import path_for
from wonhee.nodes import shots

COLOURS = {"red": [1, 0, 0], "green": [0, 1, 0], "blue": [0, 0, 1]}


def _unit(v):
    v = np.asarray(v, np.float32)
    return (v / np.linalg.norm(v, axis=-1, keepdims=True)).astype(np.float16)


@pytest.fixture
def fake_embed(monkeypatch):
    """Vectors = a frame's mean RGB; text vectors = the colour named in the text (grey otherwise)."""
    calls = []

    def image_vecs(ims, batch=16):
        calls.append(len(ims))
        return _unit([im.reshape(-1, 3).mean(0)[::-1] + 1e-3 for im in ims])

    def text_vecs(texts):
        return _unit([next((c for n, c in COLOURS.items() if n in t), [1, 1, 1]) for t in texts])

    monkeypatch.setattr(embed, "available", lambda: True)
    monkeypatch.setattr(embed, "image_vecs", image_vecs)
    monkeypatch.setattr(embed, "text_vecs", text_vecs)
    monkeypatch.setattr(embed, "MODEL_ID", "fake/rgb")
    monkeypatch.setattr(shots, "_tag_vecs", {})
    return calls


def _index(conn, settings, sha):
    return run_chain(conn, settings, [{"id": "x", "node": "shot_index", "in": {"video": sha}}], fake_deps())


def test_shot_index_stores_one_vector_per_shot(conn, settings, tiny_video, fake_embed):
    sha = store_video(conn, settings, tiny_video)
    rid, status = _index(conn, settings, sha)
    assert status == "done"
    m = json.loads(path_for(settings.media_dir, out(conn, rid, "x", "index")).read_text())
    assert m["model"] == "fake/rgb" and len(m["shots"]) == 3  # red, blue, green segments
    assert all(path_for(settings.media_dir, s["keyframe"]).exists() for s in m["shots"])
    assert all(len(s["tags"]) == 3 for s in m["shots"])
    rows = conn.execute("SELECT ord, vec FROM shot_embedding WHERE media_sha256=? ORDER BY ord", (sha,)).fetchall()
    assert [r["ord"] for r in rows] == [0, 1, 2]
    assert len(rows[0]["vec"]) == 3 * 2  # float16 x 3 dims


def test_search_ranks_by_description(conn, settings, tiny_video, fake_embed):
    sha = store_video(conn, settings, tiny_video)
    _index(conn, settings, sha)
    top = service.shots_search(conn, settings, "something green", k=1).splitlines()[1]
    assert f"{sha[:12]} shot 2" in top  # the third segment is the green one


def test_rerun_is_a_cache_hit(conn, settings, tiny_video, fake_embed):
    sha = store_video(conn, settings, tiny_video)
    _index(conn, settings, sha)
    _index(conn, settings, sha)
    assert fake_embed == [3]  # embedded once


def test_search_with_nothing_indexed(conn, settings, fake_embed):
    assert "no shots indexed" in service.shots_search(conn, settings, "anything")


@pytest.mark.gpu
def test_siglip2_puts_text_and_images_in_one_space():
    red, blue = np.zeros((64, 64, 3), np.uint8), np.zeros((64, 64, 3), np.uint8)
    red[..., 2], blue[..., 0] = 255, 255  # BGR
    iv = embed.image_vecs([red, blue]).astype(np.float32)
    tv = embed.text_vecs(["a plain red square", "a plain blue square"]).astype(np.float32)
    s = iv @ tv.T
    assert s[0, 0] > s[0, 1] and s[1, 1] > s[1, 0]
