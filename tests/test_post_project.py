"""Post projects keep named slots that fill in any order and say what still blocks a build.

Design: docs/specs/2026-10-26-post-assistant-design.md, 2.1 Post project (the scratchpad).
These tests use the public creative actions only; no models, files or network are needed.
"""
from __future__ import annotations

import pytest

from wonhee import creative
from wonhee.db.connect import open_db


def _post(conn, settings, fmt, name="Post"):
    return creative.post_create(conn, settings, name=name, format=fmt, goal="One clear point")


def _set(conn, settings, project_id, slot, state, **fields):
    return creative.post_slot_set(conn, settings, project_id=project_id, slot=slot, state=state, **fields)


def test_create_seeds_every_slot_of_the_format_as_empty(conn, settings):
    post = _post(conn, settings, "mixed_reel", name="Last day of Max")
    assert post["id"] and post["format"] == "mixed_reel" and post["goal"] == "One clear point"
    states = {r["name"]: r["state"] for r in conn.execute(
        "SELECT name, state FROM post_slot WHERE project_id=? ORDER BY name", (post["id"],))}
    assert states == {"lines": "empty", "selfie": "empty", "screen_record": "empty",
                      "typing_audio": "empty", "voice": "empty"}


def test_create_is_idempotent_for_the_same_post_and_refuses_a_different_one(conn, settings):
    first = _post(conn, settings, "image_post", name="Two photos")
    again = _post(conn, settings, "image_post", name="Two photos")
    assert again["id"] == first["id"]
    with pytest.raises(ValueError, match="different"):
        _post(conn, settings, "talking_head", name="Two photos")


def test_unknown_format_is_a_clear_error(conn, settings):
    with pytest.raises(ValueError, match="unknown post format 'carousel'.*talking_head"):
        _post(conn, settings, "carousel")


def test_unknown_slot_and_state_are_clear_errors(conn, settings):
    post = _post(conn, settings, "talking_head")
    with pytest.raises(ValueError, match="unknown slot 'photos'"):
        _set(conn, settings, post["id"], "photos", "have", ref="a.jpg")
    with pytest.raises(ValueError, match="state"):
        _set(conn, settings, post["id"], "take", "maybe", ref="a.mov")


def test_have_needs_a_ref_or_text_and_rejected_needs_a_reason(conn, settings):
    post = _post(conn, settings, "talking_head")
    with pytest.raises(ValueError, match="needs a ref"):
        _set(conn, settings, post["id"], "take", "have")
    with pytest.raises(ValueError, match="reason"):
        _set(conn, settings, post["id"], "take", "rejected")


def test_set_have_records_source_and_empty_clears_it(conn, settings):
    post = _post(conn, settings, "talking_head")
    row = _set(conn, settings, post["id"], "take", "have", ref="takes/day1.mov", source="owner")
    assert row["state"] == "have" and row["ref"] == "takes/day1.mov" and row["source"] == "owner"
    assert row["reason"] is None and row["text"] is None
    cleared = _set(conn, settings, post["id"], "take", "empty")
    assert cleared["state"] == "empty" and cleared["ref"] is None and cleared["source"] is None


def test_talking_head_blocks_on_take_and_names_lines_as_skipped(conn, settings):
    post = _post(conn, settings, "talking_head")
    text = creative.post_status(conn, settings, project_id=post["id"])
    assert "Build: blocked" in text
    assert "Missing required: take" in text
    assert "Skipped optional: lines" in text
    assert "Next: fill take" in text


def test_talking_head_allows_build_once_take_is_in(conn, settings):
    post = _post(conn, settings, "talking_head")
    _set(conn, settings, post["id"], "take", "have", ref="takes/day1.mov", source="owner")
    text = creative.post_status(conn, settings, project_id=post["id"])
    assert "Build: allowed" in text
    assert "take (takes/day1.mov, from owner)" in text
    assert "Missing required" not in text


def test_image_post_needs_lines_and_photos(conn, settings):
    post = _post(conn, settings, "image_post")
    text = creative.post_status(conn, settings, project_id=post["id"])
    assert "Build: blocked" in text and "Missing required: lines, photos" in text
    _set(conn, settings, post["id"], "lines", "have", text="Bold line", source="owner")
    text = creative.post_status(conn, settings, project_id=post["id"])
    assert "Build: blocked" in text and "Missing required: photos" in text
    _set(conn, settings, post["id"], "photos", "have", ref="photos/1.jpg", source="owner")
    assert "Build: allowed" in creative.post_status(conn, settings, project_id=post["id"])


def test_mixed_reel_says_a_preview_builds_without_screen_record(conn, settings):
    post = _post(conn, settings, "mixed_reel")
    _set(conn, settings, post["id"], "lines", "have", text="Lines", source="owner")
    _set(conn, settings, post["id"], "selfie", "have", ref="selfie.mov", source="owner")
    _set(conn, settings, post["id"], "typing_audio", "have", ref="typing.wav", source="owner")
    text = creative.post_status(conn, settings, project_id=post["id"])
    assert "Build: allowed" in text
    assert "screen_record" in text and "a preview can build without" in text
    assert "voice" in text


def test_rejected_slot_always_blocks_and_shows_its_reason(conn, settings):
    post = _post(conn, settings, "mixed_reel")
    for slot in ("lines", "typing_audio"):
        _set(conn, settings, post["id"], slot, "have", ref=f"{slot}.wav", source="owner")
    _set(conn, settings, post["id"], "selfie", "have", ref="selfie.mov", source="owner")
    _set(conn, settings, post["id"], "selfie", "rejected", ref="selfie.mov", reason="face is in shadow")
    text = creative.post_status(conn, settings, project_id=post["id"])
    assert "Build: blocked" in text
    assert "Rejected: selfie: face is in shadow" in text
    assert "Next: replace selfie (face is in shadow)" in text


def test_rejected_optional_slot_also_blocks(conn, settings):
    post = _post(conn, settings, "mixed_reel")
    for slot in ("lines", "selfie", "typing_audio"):
        _set(conn, settings, post["id"], slot, "have", ref=f"{slot}.x", source="owner")
    _set(conn, settings, post["id"], "voice", "rejected", ref="voice.wav", reason="room hum")
    text = creative.post_status(conn, settings, project_id=post["id"])
    assert "Build: blocked" in text and "Rejected: voice: room hum" in text


def test_status_of_a_plain_project_is_refused(conn, settings):
    plain = creative.project_create(conn, settings, name="Plain project")
    with pytest.raises(ValueError, match="not a post project"):
        creative.post_status(conn, settings, project_id=plain["id"])


def test_post_state_survives_reopened_connections(conn, settings):
    post = _post(conn, settings, "image_post", name="Reopen check")
    _set(conn, settings, post["id"], "lines", "have", text="Stays", source="owner")
    _set(conn, settings, post["id"], "photos", "rejected", ref="p.jpg", reason="too dark")
    before = creative.post_status(conn, settings, project_id=post["id"])

    other = open_db(settings)
    try:
        assert creative.post_status(other, settings, project_id=post["id"]) == before
    finally:
        other.close()
    assert "Rejected: photos: too dark" in before


def _slot_rows(conn, project_id):
    return {r["name"]: (r["state"], r["ref"], r["source"]) for r in conn.execute(
        "SELECT name, state, ref, source FROM post_slot WHERE project_id=?", (project_id,))}


def _intake_count(conn):
    return conn.execute("SELECT count(*) FROM creative_intake").fetchone()[0]


def test_explicit_slot_fills_that_slot_with_the_intake_ref_and_source(conn, settings, tiny_video):
    post = _post(conn, settings, "mixed_reel", name="Explicit slot")
    info = creative.inbox_add(conn, settings, source=str(tiny_video), project_id=post["id"], slot="selfie")
    slots = _slot_rows(conn, post["id"])
    assert info["slot"] == "selfie" and "slot_question" not in info
    assert slots["selfie"] == ("have", info["ref"], str(tiny_video))
    assert slots["screen_record"] == ("empty", None, None)


def test_explicit_slot_replaces_what_the_slot_held(conn, settings, tiny_video, tiny_jpg):
    post = _post(conn, settings, "image_post", name="Replace slot")
    creative.inbox_add(conn, settings, source=str(tiny_jpg), project_id=post["id"], slot="photos")
    second = creative.inbox_add(conn, settings, source=str(tiny_jpg), project_id=post["id"], slot="photos",
                                source_key="second")
    assert second["slot"] == "photos"
    assert _slot_rows(conn, post["id"])["photos"][1] == second["ref"]


def test_auto_pick_fills_the_only_empty_slot_that_fits_the_kind(conn, settings, tiny_jpg):
    post = _post(conn, settings, "image_post", name="Auto photo")
    info = creative.inbox_add(conn, settings, source=str(tiny_jpg), project_id=post["id"])
    assert info["slot"] == "photos" and "slot_question" not in info
    assert _slot_rows(conn, post["id"])["photos"] == ("have", info["ref"], str(tiny_jpg))


def test_auto_pick_with_no_empty_fitting_slot_asks_and_still_records_the_intake(conn, settings, tiny_jpg, tmp_path):
    from PIL import Image
    other = tmp_path / "other.jpg"
    Image.new("RGB", (160, 200), (30, 30, 30)).save(other, quality=90)
    post = _post(conn, settings, "image_post", name="Full photos")
    creative.inbox_add(conn, settings, source=str(tiny_jpg), project_id=post["id"])
    before = _intake_count(conn)
    info = creative.inbox_add(conn, settings, source=str(other), project_id=post["id"], source_key="again")
    assert "slot" not in info
    assert info["slot_question"]["candidates"] == ["lines"]  # the only empty slot; photos is already held
    assert _intake_count(conn) == before + 1
    assert conn.execute("SELECT status FROM creative_intake WHERE id=?", (info["id"],)).fetchone()[0] == "stored"


def test_tie_between_two_empty_slots_asks_instead_of_guessing(conn, settings, tiny_video):
    post = _post(conn, settings, "mixed_reel", name="Tie")
    before = _intake_count(conn)
    info = creative.inbox_add(conn, settings, source=str(tiny_video), project_id=post["id"])
    assert "slot" not in info
    assert info["slot_question"]["candidates"] == ["selfie", "screen_record"]
    assert "selfie" in info["slot_question"]["text"] and "screen_record" in info["slot_question"]["text"]
    assert _intake_count(conn) == before + 1
    assert all(state == "empty" for state, _, _ in _slot_rows(conn, post["id"]).values())


def test_no_matching_kind_asks_with_the_empty_slots_and_records_the_intake(conn, settings, tiny_video):
    post = _post(conn, settings, "image_post", name="Wrong kind")
    before = _intake_count(conn)
    info = creative.inbox_add(conn, settings, source=str(tiny_video), project_id=post["id"])
    assert "slot" not in info
    assert set(info["slot_question"]["candidates"]) == {"lines", "photos"}
    assert _intake_count(conn) == before + 1
    assert _slot_rows(conn, post["id"])["photos"][0] == "empty"


def test_unknown_slot_raises_and_writes_nothing(conn, settings, tiny_video):
    post = _post(conn, settings, "talking_head", name="Unknown slot")
    before = _intake_count(conn)
    with pytest.raises(ValueError, match="unknown slot 'photos'"):
        creative.inbox_add(conn, settings, source=str(tiny_video), project_id=post["id"], slot="photos")
    assert _intake_count(conn) == before
    assert _slot_rows(conn, post["id"])["take"] == ("empty", None, None)


def test_explicit_slot_without_a_post_project_raises_and_writes_nothing(conn, settings, tiny_video):
    plain = creative.project_create(conn, settings, name="Plain for slot")
    before = _intake_count(conn)
    with pytest.raises(ValueError, match="slot needs a post project"):
        creative.inbox_add(conn, settings, source=str(tiny_video), project_id=plain["id"], slot="take")
    assert _intake_count(conn) == before


def test_plain_project_intake_is_unchanged(conn, settings, tiny_video):
    plain = creative.project_create(conn, settings, name="Plain intake")
    info = creative.inbox_add(conn, settings, source=str(tiny_video), project_id=plain["id"])
    assert "slot" not in info and "slot_question" not in info
    assert info["project_id"] == plain["id"] and info["status"] == "stored"
    assert conn.execute("SELECT count(*) FROM post_slot").fetchone()[0] == 0


def test_replay_of_a_slotted_intake_reports_its_slot_without_asking_again(conn, settings, tiny_jpg):
    post = _post(conn, settings, "image_post", name="Replay slot")
    first = creative.inbox_add(conn, settings, source=str(tiny_jpg), project_id=post["id"], source_key="telegram:1")
    again = creative.inbox_add(conn, settings, source=str(tiny_jpg), project_id=post["id"], source_key="telegram:1")
    assert again["id"] == first["id"] and again["slot"] == "photos" and "slot_question" not in again
    assert _slot_rows(conn, post["id"])["photos"][1] == first["ref"]
