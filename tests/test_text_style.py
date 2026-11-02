"""text_style: measurements recover known styles from synthetic frames (text rendered with edit/text.py onto gradient
and noise backgrounds), the node's profile and flags, the text_style table, and an optional run on a local reference."""
import json
import sqlite3

import cv2
import numpy as np
import pytest
from PIL import Image

from tests.fakes import fake_deps
from tests.test_nodes_subs import flags, out, run_chain
from wonhee.db import repo
from wonhee.edit import text as T
from wonhee.edit import textstyle as ts
from wonhee.edit.text import TextSpec

W, H = 540, 960


@pytest.fixture
def node():
    import wonhee.nodes.textstyle  # noqa: F401 - registers text_style
    yield


def background(kind: str, seed: int = 1) -> Image.Image:
    rng = np.random.default_rng(seed)
    y, x = np.linspace(0, 1, H)[:, None, None], np.linspace(0, 1, W)[None, :, None]
    base = {"dark": 40 + 60 * y + 20 * x, "bright": 170 + 50 * y + 0 * x, "mid": 90 + 90 * x + 40 * y}[kind]
    base = np.broadcast_to(base, (H, W, 3)).astype(float) + np.array([5, 0, -5]) + rng.normal(0, 10, (H, W, 1))
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def paste_block(base: Image.Image, spec: TextSpec, box: tuple[int, int, int, int]):
    """Render the text aligned by spec.align inside `box` (x, y, w, h) and return the image and the ink's true box as
    fractions of the frame: (cx, cy, w, h)."""
    x, y, w, h = box
    layer = T.render_block(spec, (w, h), ref_h=H)
    ink = layer.getchannel("A").point(lambda v: 255 if v > 128 else 0).getbbox()
    full = Image.new("RGBA", base.size, (0, 0, 0, 0))
    full.alpha_composite(layer, (x, y))
    out_im = base.convert("RGBA")
    out_im.alpha_composite(full)
    ix0, iy0, ix1, iy1 = ink[0] + x, ink[1] + y, ink[2] + x, ink[3] + y
    return out_im.convert("RGB"), ((ix0 + ix1) / 2 / W, (iy0 + iy1) / 2 / H, (ix1 - ix0) / W, (iy1 - iy0) / H)


def bgr(im: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def dist(a: str, b: str) -> int:
    pa, pb = (tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in (a, b))
    return max(abs(x - y) for x, y in zip(pa, pb))


# (name, spec, background, box, true line count)
CASES = [
    ("white outline centred", TextSpec("still shipping at three am", colours=("#ffffff", "#000000"), size_frac=0.05,
                                       width_frac=0.7, align="center"), "mid", (80, 330, 380, 200), 2),
    ("cream no outline left", TextSpec("the quick brown fox jumps over the lazy dog again", colours=("#f5ecd0", "#000000"),
                                       stroke_em=0.0, size_frac=0.04, align="left"), "dark", (60, 400, 230, 300), 5),
    ("white on dark box", TextSpec("last day of claude max", colours=("#ffffff", "#000000"), box="box",
                                   box_colour="#101010", size_frac=0.04, align="center"), "bright", (90, 540, 360, 120), 2),
    ("right aligned outline", TextSpec("we are so back this is fine really it is okay now", colours=("#ffffff", "#222222"),
                                      size_frac=0.045, align="right"), "mid", (170, 150, 340, 450), 5),
    ("big yellow caps", TextSpec("HELLO WORLD", colours=("#ffd400", "#000000"), size_frac=0.08, align="center"),
     "dark", (60, 220, 420, 220), 2),
]


@pytest.mark.parametrize("name,spec,bg,box,nlines", CASES, ids=[c[0] for c in CASES])
def test_measurements_recover_the_known_style(name, spec, bg, box, nlines):
    im, (cx, cy, bw, bh) = paste_block(background(bg), spec, box)
    s = ts.measure(bgr(im))
    assert s.found
    assert dist(s.fill, spec.colours[0]) <= 12
    assert abs(s.size_frac - spec.size_frac) / spec.size_frac <= 0.15
    assert s.lines == nlines
    assert s.align == spec.align
    assert abs(s.x - cx) <= 0.1 and abs(s.y - cy) <= 0.1
    assert abs(s.w - bw) <= 0.05 and abs(s.h - bh) <= 0.05
    assert bool(s.box) == (spec.box != "none")
    if spec.box != "none":
        assert dist(s.box_colour, spec.box_colour) <= 12 and s.stroke is None
    elif spec.stroke_em == 0.0:
        assert s.stroke is None
    else:
        assert s.stroke is not None and dist(s.stroke, spec.colours[1]) <= 20


def test_single_line_alignment_is_a_guess():
    im, _ = paste_block(background("dark"), TextSpec("one line", colours=("#ffffff", "#000000"), size_frac=0.05),
                        (100, 400, 340, 80))
    s = ts.measure(bgr(im))
    assert s.lines == 1 and s.align == "center" and s.align_certain is False


def test_dimming_follows_the_photo_brightness():
    spec = TextSpec("a line of text here", colours=("#ffffff", "#000000"), size_frac=0.045)
    bright, _ = paste_block(background("bright"), spec, (60, 400, 420, 100))
    dark = Image.fromarray((np.array(bright).astype(float) * 0.45).astype(np.uint8))
    dark, _ = paste_block(dark.copy(), spec, (60, 400, 420, 100))  # text stays white: a dimmed photo, undimmed text
    sb, sd = ts.measure(bgr(bright)), ts.measure(bgr(dark))
    assert 0 <= sb.dim <= 0.01
    assert sd.dim > sb.dim + 0.25 and abs(sd.dim - 0.35) <= 0.15  # 1 - luma/128, the photo's luma being 0.45 * 190


def test_dark_text_on_a_light_photo():
    im, (cx, cy, _, _) = paste_block(background("bright"), TextSpec("black words on bright", colours=("#111111", "#eeeeee"),
                                                                    stroke_em=0.0, size_frac=0.05), (60, 400, 420, 100))
    s = ts.measure(bgr(im))
    assert s.found and s.polarity == "dark" and dist(s.fill, "#111111") <= 30
    assert abs(s.x - cx) <= 0.1 and abs(s.y - cy) <= 0.1


def test_no_text_is_all_none():
    s = ts.measure(bgr(background("mid")))
    assert s.found is False
    p = ts.to_profile(s)
    assert p["colours"] is None and p["size_frac"] is None and p["at"] is None and p["measured"]["lines"] is None


def test_profile_is_textspec_shaped():
    spec, bg, box = CASES[0][1], CASES[0][2], CASES[0][3]
    im, _ = paste_block(background(bg), spec, box)
    p = ts.to_profile(ts.measure(bgr(im)), role="sans")
    built = TextSpec("x", preset=p["preset"], colours=tuple(p["colours"]), size_frac=p["size_frac"], align=p["align"],
                     box=p["box"], stroke_em=p["stroke_em"], at=tuple(p["at"]), width_frac=p["width_frac"],
                     role=p["role"])
    assert built.role == "sans" and built.colours[1] == "#000000"


# -- node ------------------------------------------------------------------------------------------------------------

def stored(conn, settings, tmp_path, im: Image.Image) -> str:
    f = tmp_path / "slide.png"
    im.save(f)
    return repo.save_file(conn, settings.media_dir, f, "image", "image/png")


def test_node_returns_the_profile_and_stores_it(conn, settings, tmp_path, node):
    from wonhee.media import path_for
    spec, bg, box = CASES[0][1], CASES[0][2], CASES[0][3]
    sha = stored(conn, settings, tmp_path, paste_block(background(bg), spec, box)[0])
    rid, status = run_chain(conn, settings, [{"id": "t", "node": "text_style", "in": {"image": sha},
                                              "params": {"role_hint": "sans"}}], fake_deps())
    assert status == "done" and flags(conn, rid, "t") == []
    prof = json.loads(path_for(settings.media_dir, out(conn, rid, "t", "profile")).read_text())
    assert prof["role"] == "sans" and prof["box"] == "none" and dist(prof["colours"][0], "#ffffff") <= 12
    assert json.loads(repo.load_text_style(conn, sha, 1)) == prof


def test_role_is_none_without_a_hint(conn, settings, tmp_path, node):
    from wonhee.media import path_for
    spec, bg, box = CASES[1][1], CASES[1][2], CASES[1][3]
    sha = stored(conn, settings, tmp_path, paste_block(background(bg), spec, box)[0])
    rid, _ = run_chain(conn, settings, [{"id": "t", "node": "text_style", "in": {"image": sha}}], fake_deps())
    assert json.loads(path_for(settings.media_dir, out(conn, rid, "t", "profile")).read_text())["role"] is None


def test_node_flags_when_there_is_no_text(conn, settings, tmp_path, node):
    from wonhee.media import path_for
    sha = stored(conn, settings, tmp_path, background("mid"))
    rid, status = run_chain(conn, settings, [{"id": "t", "node": "text_style", "in": {"image": sha}}], fake_deps())
    assert status == "done" and flags(conn, rid, "t") == ["NO_TEXT_FOUND"]
    prof = json.loads(path_for(settings.media_dir, out(conn, rid, "t", "profile")).read_text())
    assert prof["colours"] is None and prof["size_frac"] is None


def test_node_rejects_unknown_params(conn, settings, tmp_path, node):
    sha = stored(conn, settings, tmp_path, background("mid"))
    _, status = run_chain(conn, settings, [{"id": "t", "node": "text_style", "in": {"image": sha},
                                            "params": {"role_hint": "comic_sans"}}], fake_deps())
    assert status != "done"
    with pytest.raises(Exception, match="bogus"):
        run_chain(conn, settings, [{"id": "t", "node": "text_style", "in": {"image": sha}, "params": {"bogus": 1}}],
                  fake_deps())


def test_text_style_table_is_keyed_by_image_and_version(conn, settings, tmp_path):
    sha = stored(conn, settings, tmp_path, background("mid"))
    assert repo.load_text_style(conn, sha, 1) is None
    repo.save_text_style(conn, sha=sha, analyzer_version=1, profile_json='{"a": 1}')
    repo.save_text_style(conn, sha=sha, analyzer_version=2, profile_json='{"a": 2}')
    repo.save_text_style(conn, sha=sha, analyzer_version=1, profile_json='{"a": 3}')  # same key: replaced
    assert repo.load_text_style(conn, sha, 1) == '{"a": 3}' and repo.load_text_style(conn, sha, 2) == '{"a": 2}'
    with pytest.raises(sqlite3.IntegrityError):
        repo.save_text_style(conn, sha="f" * 64, analyzer_version=1, profile_json="{}")  # no such media


# -- optional: a real reference slide from the local library (never copied into the repo) ---------------------------

@pytest.mark.parametrize("content_id", [66, 67, 70, 71, 72])
def test_real_reference_slide_profile_is_sane(content_id):
    from pathlib import Path

    from wonhee.config import Settings
    from wonhee.db.connect import open_db
    from wonhee.media import path_for
    try:
        st = Settings()
        db = st.home / "wonhee.db"
        if not db.exists():
            pytest.skip("no local database")
        c = open_db(st)
        rows = c.execute("SELECT media_sha256 FROM content_media WHERE content_id=? AND kind='image' ORDER BY ord",
                         (content_id,)).fetchall()
    except pytest.skip.Exception:
        raise
    except Exception as e:  # noqa: BLE001 - a missing or older local DB is a skip, not a failure
        pytest.skip(f"local library not usable: {e}")
    files = [path_for(st.media_dir, r[0]) for r in rows if Path(path_for(st.media_dir, r[0])).exists()]
    if not files:
        pytest.skip(f"no local images for content {content_id}")
    for f in files:
        s = ts.measure(cv2.imread(str(f)))
        if not s.found:
            continue
        assert s.fill and 0 < s.size_frac < 0.3 and s.lines >= 1 and s.align in ("left", "center", "right")
        assert -0.001 <= s.x - s.w / 2 and s.x + s.w / 2 <= 1.001 and -0.001 <= s.y - s.h / 2 and s.y + s.h / 2 <= 1.001
        assert 0 <= s.dim <= 1
    # a slide with no readable text is allowed (found may be 0); it must just not crash
