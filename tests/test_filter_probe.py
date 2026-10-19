import json

import pytest

from tests.fakes import fake_deps
from tests.test_filters import _fake_models
from tests.test_vibe import _write, day, night
from wonhee import filterstore, localllm, service
from wonhee.nodes.filter import CORE, KNOBS

ANSWER = {"knob": "temperature", "direction": "up", "amount": "some", "range": None}
SEES = ["temperature", "exposure", "contrast", "saturation", "tint", "shadows", "highlights", "vignette"]


def _sighted(monkeypatch):
    """A fake reviewer that names the knob right when it is in SEES and otherwise guesses among SEES (in turn), so
    no single answer dominates. It learns the true move by watching apply_moves, since it cannot see pixels."""
    import wonhee.nodes.filter as nf
    truth, real = [], nf.apply_moves
    monkeypatch.setattr(nf, "apply_moves", lambda f, moves: (truth.extend(m.knob for m in moves[:1]), real(f, moves))[1])
    turn = iter(range(10 ** 6))

    def post(s, body):
        k = truth[-1]
        ans = {**ANSWER, "knob": k if k in SEES else SEES[next(turn) % len(SEES)]}
        return {"message": {"content": json.dumps(ans)}, "prompt_eval_count": 10}
    monkeypatch.setattr(localllm, "_post", post)


def _calibrate(conn, settings, tmp_path, monkeypatch, n=len(KNOBS), replies=None):
    photos = [_write(tmp_path, "a.jpg", day()), _write(tmp_path, "b.jpg", night(1))]
    if replies is None:
        _sighted(monkeypatch)
    else:
        _fake_models(monkeypatch, {"Identify": replies})
    text, path = service.filter_calibrate(conn, settings, fake_deps(), photos, n=n)
    return photos, text, path


def test_calibrate_scores_knobs_and_writes_file(conn, settings, tmp_path, monkeypatch):
    _, text, path = _calibrate(conn, settings, tmp_path, monkeypatch, n=2 * len(KNOBS))
    cal = json.loads(path.read_text())
    assert path.name == "_calibration.json" and cal["model"] == settings.local_llm_model
    k = cal["knobs"]
    assert k["temperature"]["trials"] == 2 and k["temperature"]["seen_rate"] == 1.0
    assert all(v["seen_rate"] == (1.0 if n in SEES else 0) for n, v in k.items())
    assert "not asked" not in text or all(x not in SEES for x in text.split("not asked")[1].split("\n")[0].split())
    assert cal["guessing"] is False
    assert "grain" in text and "hidden from the reviewer" in text
    assert filterstore.saved(settings.home) == []  # "_" files are not filters
    assert filterstore.calibration(settings.home, settings.local_llm_model)["n"] == 2 * len(KNOBS)
    assert filterstore.calibration(settings.home, "other-model") is None
    assert filterstore.calibration(settings.home, settings.local_llm_model, "remote") is None


def test_hidden_knobs_reach_the_reviewer_and_are_ignored(conn, settings, tmp_path, monkeypatch):
    photos, *_ = _calibrate(conn, settings, tmp_path, monkeypatch, n=2 * len(KNOBS))
    sent = _fake_models(monkeypatch, {
        "Draft": [{"base": "none", "notes": "n", "moves": []}],
        "Adjust": [{"score": 3, "verdict": "meh", "moves": [{"knob": "grain", "direction": "up", "amount": "a lot"},
                                                          {"knob": "exposure", "direction": "up",
                                                           "amount": "a little"}]},
                   {"score": 4, "verdict": "ok", "moves": []}]})
    text, _ = service.filter_make(conn, settings, fake_deps(), "plain", photos, name="plain", rounds=2)
    prompt = [b for n, b in sent if n == "Adjust"][0]["messages"][1]["content"]
    line = prompt.split("Do not use these knobs: ")[1].split(".\n")[0]
    hidden = line.split(", ")
    assert "grain" in hidden and not set(hidden) & CORE and len(hidden) <= len(KNOBS) // 2
    assert f"hidden knobs" in text
    f, _ = filterstore.load(settings.home, "plain")
    assert f.grain == 0 and f.exposure == pytest.approx(0.1)  # grain move ignored, exposure kept


def test_no_calibration_leaves_the_prompt_alone(conn, settings, tmp_path, monkeypatch):
    photos = [_write(tmp_path, "a.jpg", day())]
    sent = _fake_models(monkeypatch, {"Draft": [{"base": "none", "notes": "n", "moves": []}],
                                      "Adjust": [{"score": 4, "verdict": "ok", "moves": []}]})
    text, _ = service.filter_make(conn, settings, fake_deps(), "plain", photos, name="plain", rounds=1)
    assert "Do not use" not in [b for n, b in sent if n == "Adjust"][0]["messages"][1]["content"]
    assert "hidden knobs" not in text


def test_calibrate_needs_photos(conn, settings):
    text, path = service.filter_calibrate(conn, settings, fake_deps(), [])
    assert path is None and "photo" in text


def test_a_model_that_always_names_one_knob_hides_nothing(conn, settings, tmp_path, monkeypatch):
    n = 2 * len(KNOBS)
    _, text, path = _calibrate(conn, settings, tmp_path, monkeypatch, n=n, replies=[ANSWER] * n)
    assert json.loads(path.read_text())["guessing"] is True and "guessing" in text
    assert service._hidden_knobs(settings, "local") == []
