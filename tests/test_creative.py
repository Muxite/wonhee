"""Creative records retain source history and remain useful without semantic dependencies.

Design: docs/specs/2026-10-24-moeka-integration-design.md, Creative projects and memory / Intake.
These tests exercise public service behavior; no GPU, embeddings or remote calls are needed.
"""
from __future__ import annotations

import json
from collections.abc import Mapping

import pytest

from wonhee import creative
from wonhee.db.connect import open_db


@pytest.fixture(autouse=True)
def lexical_memory(monkeypatch):
    monkeypatch.setattr(creative, "_docstore", lambda *a, **k: None)


def _ids(result):
    return {row["id"] for row in result["results"]}


def _rows(result):
    """List actions may use a domain name; the records themselves keep stable ids."""
    assert isinstance(result, Mapping)
    groups = [value for value in result.values() if isinstance(value, list)]
    assert len(groups) == 1, result
    return groups[0]


def test_project_selection_survives_connection_reopen(conn, settings):
    """design: Conversation and ownership, durable project selection per chat/topic."""
    campus = creative.project_create(conn, settings, name="Campus reel")
    midnight = creative.project_create(conn, settings, name="Midnight water")
    route = "telegram:owner:topic:42"
    creative.project_select(conn, settings, project_id=campus["id"], route_key=route)
    creative.project_select(conn, settings, project_id=midnight["id"], route_key="telegram:owner:topic:43")

    other = open_db(settings)
    try:
        selected = creative.project_show(other, settings, route_key=route)
        assert selected["id"] == campus["id"]
        assert creative.project_show(other, settings, project_id=midnight["id"])["name"] == "Midnight water"
    finally:
        other.close()


def test_memory_query_combines_project_and_creator_without_cross_project_leakage(conn, settings):
    """design: Creative projects and memory, creator constraints plus the active project only."""
    a = creative.project_create(conn, settings, name="Campus reel")
    b = creative.project_create(conn, settings, name="Midnight water")
    shared = creative.memory_remember(conn, settings, content="Keep captions understated")
    campus = creative.memory_remember(
        conn, settings, content="Campus captions use the blue band", project_id=a["id"],
    )
    midnight = creative.memory_remember(
        conn, settings, content="Midnight captions use warm orange", project_id=b["id"],
    )
    result = creative.memory_query(conn, settings, query="captions", project_id=a["id"], scope="auto")
    assert _ids(result) == {shared["id"], campus["id"]}
    assert midnight["id"] not in _ids(result)
    assert result["backend"]


def test_memory_correction_is_immediate_source_linked_and_persistent(conn, settings):
    """design: Creative projects and memory, correction works before Dream and after restart."""
    original = creative.memory_remember(
        conn, settings, content="Captions should be aggressive", source={"message_id": "100"},
    )
    corrected = creative.memory_correct(
        conn, settings, memory_id=original["id"], content="Captions should be understated",
        source={"message_id": "101"},
    )
    assert corrected["id"] != original["id"]
    assert corrected["source"]["message_id"] == "101"

    other = open_db(settings)
    try:
        result = creative.memory_query(other, settings, query="captions")
        assert _ids(result) == {corrected["id"]}
        assert result["results"][0]["content"] == "Captions should be understated"
    finally:
        other.close()


def test_forget_excludes_memory_after_reindex(conn, settings):
    """design: Creative projects and memory, rebuilding an index cannot revive forgotten facts."""
    forgotten = creative.memory_remember(conn, settings, content="Use magenta captions")
    retained = creative.memory_remember(conn, settings, content="Use readable captions")
    creative.memory_forget(conn, settings, memory_id=forgotten["id"])
    creative.memory_reindex(conn, settings)
    result = creative.memory_query(conn, settings, query="captions")
    assert _ids(result) == {retained["id"]}


def test_unconfirmed_suggestion_is_searchable_only_when_requested(conn, settings):
    """design: Creative projects and memory, model suggestions are not confirmed constraints."""
    confirmed = creative.memory_remember(conn, settings, content="Keep captions sparse")
    suggested = creative.memory_remember(
        conn, settings, content="Try neon captions", status="proposed",
    )
    assert _ids(creative.memory_query(conn, settings, query="captions")) == {confirmed["id"]}
    proposals = creative.memory_query(conn, settings, query="captions", status="proposed")
    assert _ids(proposals) == {suggested["id"]}


def test_memory_count_and_timeline_use_structured_filters(conn, settings):
    """design: Approved implementation details, counts and timelines are exact structured queries."""
    for content in ("Sparse captions", "Readable captions", "Short captions"):
        creative.memory_remember(conn, settings, content=content, kind="preference")
    creative.memory_remember(conn, settings, content="The shot order is settled", kind="decision")
    result = creative.memory_query(
        conn, settings, kind="preference", operation="count", limit=1,
    )
    assert result["count"] == 3
    timeline = creative.memory_query(conn, settings, kind="decision", operation="timeline")
    assert [row["content"] for row in timeline["results"]] == ["The shot order is settled"]


@pytest.mark.parametrize("query", ['captions" OR *', "' OR 1=1; DROP TABLE creative_memory; --", "(((("])
def test_ordinary_word_queries_treat_search_syntax_as_data(conn, settings, query):
    """design: Approved implementation details, ordinary words never become SQL or FTS syntax."""
    saved = creative.memory_remember(conn, settings, content="Keep captions readable")
    result = creative.memory_query(conn, settings, query=query)
    assert isinstance(result["results"], list)
    assert _ids(creative.memory_query(conn, settings, operation="list")) == {saved["id"]}


def test_memory_export_is_human_readable_and_preserves_correction_and_forget_history(conn, settings):
    """design: Approved implementation details, text exports retain ids, revisions and tombstones."""
    first = creative.memory_remember(conn, settings, content="Caption preference before correction")
    second = creative.memory_correct(conn, settings, memory_id=first["id"], content="Caption preference after correction")
    creative.memory_forget(conn, settings, memory_id=second["id"])
    creative.memory_export(conn, settings)

    markdown = list(settings.home.rglob("*.md"))
    assert markdown
    assert any("Caption preference" in path.read_text() for path in markdown)
    machine = list(settings.home.rglob("*.json")) + list(settings.home.rglob("*.jsonl"))
    assert machine
    exported = "\n".join(path.read_text() for path in machine)
    assert str(first["id"]) in exported and str(second["id"]) in exported
    assert "superseded" in exported and "forgotten" in exported


def test_intake_replayed_source_is_idempotent_but_distinct_sources_are_preserved(conn, settings, tiny_video):
    """design: Intake, repeated receipts reuse assets while identical bytes retain distinct sources."""
    project = creative.project_create(conn, settings, name="Raw takes")
    kwargs = {"source": str(tiny_video), "project_id": project["id"], "source_key": "telegram:owner:100:0"}
    first = creative.inbox_add(conn, settings, **kwargs)
    replay = creative.inbox_add(conn, settings, **kwargs)
    distinct = creative.inbox_add(
        conn, settings, source=str(tiny_video), project_id=project["id"], source_key="telegram:owner:101:0",
    )
    assert replay["id"] == first["id"]
    assert distinct["id"] != first["id"]
    assets = _rows(creative.inbox_list(conn, settings, project_id=project["id"]))
    assert {asset["id"] for asset in assets} == {first["id"], distinct["id"]}
    assert conn.execute("SELECT COUNT(*) FROM blob").fetchone()[0] == 1
    info = creative.media_info(conn, settings, asset_id=first["id"])
    assert info["id"] == first["id"]


@pytest.mark.parametrize("url", [
    "https://drive.google.com/drive/folders/example",
    "https://docs.google.com/document/d/example/edit",
    "https://example.com/not-a-drive-file.mp4",
])
def test_unsupported_drive_sources_are_rejected_before_network(conn, settings, monkeypatch, url):
    """design: Public Google Drive links, v1 accepts public binary files and rejects other link types."""
    import httpx

    def no_network(*a, **k):
        pytest.fail("unsupported link attempted network access")

    monkeypatch.setattr(httpx.Client, "send", no_network)
    with pytest.raises(ValueError):
        creative.inbox_add(conn, settings, source=url)


def test_memory_source_round_trips_as_data(conn, settings):
    """design: Creative projects and memory, every remembered decision retains its source event."""
    source = {"channel": "telegram", "message_id": "9", "caption": "quoted \" text\nwith lines"}
    saved = creative.memory_remember(conn, settings, content="Keep the original audio", kind="decision", source=source)
    result = creative.memory_query(conn, settings, query="original audio")
    row = next(row for row in result["results"] if row["id"] == saved["id"])
    assert row["source"] == source
    json.dumps(result)


def test_cross_project_search_is_explicit(conn, settings):
    """design: Creative projects and memory, cross-project recall is explicit and remains attributable."""
    a = creative.project_create(conn, settings, name="Campus reel")
    b = creative.project_create(conn, settings, name="Midnight water")
    memories = [
        creative.memory_remember(conn, settings, content="Campus footage needs a title", project_id=a["id"]),
        creative.memory_remember(conn, settings, content="Midnight footage needs a title", project_id=b["id"]),
    ]
    result = creative.memory_query(conn, settings, query="footage", scope="all")
    assert _ids(result) == {row["id"] for row in memories}
    assert {row["project_id"] for row in result["results"]} == {a["id"], b["id"]}


def test_ideas_remain_project_scoped_and_track_acceptance(conn, settings):
    """design: Creative projects and memory, tentative ideas have an attributable lifecycle."""
    a = creative.project_create(conn, settings, name="Campus reel")
    b = creative.project_create(conn, settings, name="Midnight water")
    idea = creative.idea_save(
        conn, settings, content="Deadpan campus founder skit", project_id=a["id"], source={"message_id": "12"},
    )
    creative.idea_save(conn, settings, content="Midnight kettle interview", project_id=b["id"])
    assert idea["status"] == "draft"
    accepted = creative.idea_update(conn, settings, idea_id=idea["id"], status="accepted")
    assert accepted["status"] == "accepted"
    rows = _rows(creative.idea_list(conn, settings, project_id=a["id"], status="accepted"))
    assert [row["id"] for row in rows] == [idea["id"]]
    assert rows[0]["source"]["message_id"] == "12"


def _submission(conn, settings, tiny_video, *, request_key="telegram:owner:make:1", brief="Make a campus reel"):
    project = creative.project_create(conn, settings, name="Submission project")
    asset = creative.inbox_add(conn, settings, source=str(tiny_video), project_id=project["id"], source_key="take:1")
    return creative.make_submit(
        conn, settings, project_id=project["id"], asset_id=asset["id"], request_key=request_key,
        brief=brief, route={"channel": "telegram", "chat_id": "owner"},
    )


def test_submission_replay_reuses_brief_revision_and_execution_request(conn, settings, tiny_video):
    """design: Execution and delivery, a repeated logical request cannot create another submission."""
    first = _submission(conn, settings, tiny_video)
    replay = _submission(conn, settings, tiny_video)
    assert replay["id"] == first["id"]
    assert replay["brief_revision"] == first["brief_revision"]
    assert conn.execute("SELECT count(*) FROM creative_submission").fetchone()[0] == 1
    assert conn.execute("SELECT count(*) FROM creative_brief").fetchone()[0] == 1
    assert creative.make_show(conn, settings, request_key=first["request_key"])["id"] == first["id"]


def test_submission_replay_cannot_replace_an_accepted_brief(conn, settings, tiny_video):
    """design: Execution and delivery, request ids bind the accepted brief revision."""
    _submission(conn, settings, tiny_video)
    with pytest.raises(ValueError):
        _submission(conn, settings, tiny_video, brief="Use a different title and shot order")


def test_submission_rejects_unknown_options_before_queueing(conn, settings, tiny_video):
    """design: Execution and delivery, creation inputs are validated before recording work."""
    project = creative.project_create(conn, settings, name="Submission project")
    asset = creative.inbox_add(conn, settings, source=str(tiny_video), project_id=project["id"])
    with pytest.raises(ValueError):
        creative.make_submit(conn, settings, project_id=project["id"], asset_id=asset["id"], request_key="bad",
                             params={"arbitrary_shell_command": "do something"})
    assert conn.execute("SELECT count(*) FROM creative_submission").fetchone()[0] == 0


def test_phase_records_run_before_execution_and_reuses_it_after_interruption(conn, settings, tiny_video, monkeypatch):
    """design: Execution and delivery, phase correlation commits before work can have side effects."""
    from wonhee import creative_worker, service
    from wonhee.db import repo

    submission = _submission(conn, settings, tiny_video)
    asset = creative.media_info(conn, settings, asset_id=submission["asset_id"])
    runs = []

    def interrupted(c, s, deps, run_id):
        other = open_db(settings)
        try:
            linked = other.execute(
                "SELECT run_id FROM creative_run WHERE submission_id=? AND phase='speech'", (submission["id"],),
            ).fetchone()
            assert linked["run_id"] == run_id
            assert other.execute("SELECT id FROM run WHERE id=?", (run_id,)).fetchone()
        finally:
            other.close()
        runs.append(run_id)
        if len(runs) == 1:
            raise RuntimeError("interrupted before local node work")
        repo.finish_run(c, run_id, "done")

    monkeypatch.setattr(service, "chain_exec", interrupted)
    monkeypatch.setattr(service, "_step_out", lambda *a, **k: {"subtitles": "a" * 64})
    nodes = [{"id": "s", "node": "speech_subs", "in": {"video": asset["ref"]}}]
    with pytest.raises(RuntimeError, match="interrupted"):
        creative_worker._phase(conn, settings, None, submission, "speech", nodes, 1)
    creative_worker._phase(conn, settings, None, submission, "speech", nodes, 1)
    assert runs[0] == runs[1]
    assert conn.execute("SELECT count(*) FROM creative_run").fetchone()[0] == 1


def _output(conn, settings):
    from wonhee.db import repo

    ref = repo.save_bytes(conn, settings.media_dir, b"0123456789", "video/mp4")
    return creative.outbox_add(
        conn, settings, ref=ref, filename="a reel.mp4", route={"channel": "telegram", "chat_id": "owner"},
    )


def test_delivery_requires_actual_transport_ack_and_explicit_failure_retry(conn, settings):
    """design: Execution and delivery, queueing a message is distinct from sending it."""
    delivery = _output(conn, settings)
    assert delivery["status"] == "queued"
    with pytest.raises(ValueError):
        creative.delivery_show(conn, settings, delivery_id=delivery["id"], transition="sent", message_ids=[10])
    creative.delivery_show(conn, settings, delivery_id=delivery["id"], transition="sending")
    with pytest.raises(ValueError):
        creative.delivery_show(conn, settings, delivery_id=delivery["id"], transition="sent")
    creative.delivery_show(conn, settings, delivery_id=delivery["id"], transition="failed", error="temporary outage")
    assert creative.delivery_show(conn, settings)["deliveries"] == []
    retry = creative.delivery_retry(conn, settings, delivery_id=delivery["id"])
    assert retry["status"] == "queued"
    creative.delivery_show(conn, settings, delivery_id=delivery["id"], transition="sending")
    sent = creative.delivery_show(conn, settings, delivery_id=delivery["id"], transition="sent", message_ids=[11])
    assert sent["status"] == "sent" and sent["message_ids"] == [11]
    with pytest.raises(ValueError):
        creative.delivery_retry(conn, settings, delivery_id=delivery["id"])


def test_recovery_marks_inflight_delivery_uncertain_without_resending(conn, settings):
    """design: Execution and delivery, an interrupted send is not assumed successful or safely repeatable."""
    from wonhee.db.repo import creative as writes

    delivery = _output(conn, settings)
    creative.delivery_show(conn, settings, delivery_id=delivery["id"], transition="sending")
    writes.recover_deliveries(conn)
    recovered = creative.delivery_show(conn, settings, delivery_id=delivery["id"])
    assert recovered["status"] == "uncertain"
    assert creative.delivery_show(conn, settings)["deliveries"] == []
    assert creative.delivery_retry(conn, settings, delivery_id=delivery["id"])["status"] == "queued"


def test_host_delivery_read_surfaces_failed_submission_without_an_output(conn, settings, tiny_video):
    """design: Execution and delivery, a failed render remains reportable even without an artifact."""
    from wonhee.db.repo import creative as writes

    failed = _submission(conn, settings, tiny_video)
    _submission(conn, settings, tiny_video, request_key="telegram:owner:make:2")
    writes.update(conn, "creative_submission", failed["id"], status="failed", error="render input could not decode")

    result = creative.delivery_show(conn, settings)
    assert result["deliveries"] == []
    assert result["failures"] == [{
        "id": failed["id"], "project_id": failed["project_id"],
        "route": {"channel": "telegram", "chat_id": "owner"}, "error": "render input could not decode",
    }]
    assert creative.delivery_show(conn, settings)["failures"] == result["failures"]
    assert creative.make_show(conn, settings, submission_id=failed["id"])["status"] == "failed"


def test_private_outbox_serves_only_registered_tokens_and_supports_byte_ranges(conn, settings):
    """design: Execution and delivery, private links resolve registered artifacts with range support."""
    import http.client
    import threading
    from http.server import ThreadingHTTPServer

    from wonhee import creative_worker

    output = _output(conn, settings)
    server = ThreadingHTTPServer(("127.0.0.1", 0), creative_worker._handler(settings))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = http.client.HTTPConnection(*server.server_address, timeout=5)
    try:
        client.request("GET", f"/files/{output['token']}", headers={"Range": "bytes=2-5"})
        response = client.getresponse()
        assert response.status == 206
        assert response.getheader("Content-Range") == "bytes 2-5/10"
        assert response.getheader("Accept-Ranges") == "bytes"
        assert "attachment" in response.getheader("Content-Disposition")
        assert response.read() == b"2345"

        client.request("GET", f"/files/{output['token']}", headers={"Range": "bytes=-3"})
        response = client.getresponse()
        assert response.status == 206 and response.read() == b"789"

        client.request("GET", f"/files/{output['token']}", headers={"Range": "bytes=20-30"})
        response = client.getresponse()
        assert response.status == 416
        response.read()

        for path in ("/files/../../wonhee.db", f"/files/{output['ref']}", "/files/" + "f" * 32):
            client.request("GET", path)
            response = client.getresponse()
            assert response.status == 404
            response.read()

        from wonhee import media
        media.path_for(settings.media_dir, output["ref"]).unlink()
        client.request("GET", f"/files/{output['token']}")
        response = client.getresponse()
        assert response.status == 404
        response.read()
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_execute_renders_a_real_cpu_video_and_reuses_completed_phases(conn, settings, tiny_video):
    """design: Execution and delivery, recorded local phases produce a real artifact and survive replay.

    Speech recognition is fake; video composition and encoding use real CPU ffmpeg. This is not a Telegram test.
    """
    from pathlib import Path

    from tests.fakes import fake_deps
    from tests.test_nodes_subs import CountingTranscriber
    from wonhee import creative_worker, media

    submission = _submission(conn, settings, tiny_video)
    transcriber = CountingTranscriber()
    deps = fake_deps(transcriber=transcriber)
    result = creative_worker.execute(conn, settings, deps, submission)
    assert result["status"] == "done"
    assert result["full_check"] is None
    assert {run["phase"] for run in result["runs"]} == {"speech", "render"}
    assert len(result["deliveries"]) == 1
    delivery = result["deliveries"][0]
    assert delivery["status"] == "queued"
    path = Path(delivery["path"])
    assert path.is_file() and path.suffix == ".mp4"
    probe = media.probe(path)
    assert probe.width == 1080 and probe.height == 1920
    assert 2900 <= probe.duration_ms <= 3100

    again = creative_worker.execute(conn, settings, deps, submission)
    assert again["result_ref"] == result["result_ref"]
    assert [run["run_id"] for run in again["runs"]] == [run["run_id"] for run in result["runs"]]
    assert len(again["deliveries"]) == 1
    assert transcriber.n == 1


def test_hybrid_search_partitions_project_scopes_before_semantic_retrieval(conn, settings, monkeypatch):
    """design: Creative projects and memory, unrelated projects cannot crowd out scoped semantic recall.

    Fake semantic results verify wiring and canonical filtering, not the embedding model's retrieval quality.
    """
    from types import SimpleNamespace

    project = creative.project_create(conn, settings, name="Subtle campus captions")
    other = creative.project_create(conn, settings, name="Other creative work")
    creator = creative.memory_remember(conn, settings, content="Keep overlays restrained")
    scoped = creative.memory_remember(conn, settings, content="Campus subtitles should be understated", project_id=project["id"])
    unrelated = creative.memory_remember(conn, settings, content="Orange typography", project_id=other["id"])
    selected_scopes = []

    class Store:
        available = True

        def clear(self, **kwargs):
            pass

        def add(self, *args, **kwargs):
            return 1

        def search(self, query, *, tags, **kwargs):
            selected_scopes.append(tags)
            assert "status:confirmed" in tags
            # Even a stale or malformed index result must pass canonical scope checks.
            return [SimpleNamespace(source=f"memory:{record['id']}") for record in (creator, scoped, unrelated)]

    monkeypatch.setattr(creative, "_docstore", lambda *a, **k: Store())
    result = creative.memory_query(conn, settings, query="minimal lettering", project_id=project["id"])
    assert result["backend"] == "hybrid"
    assert _ids(result) == {creator["id"], scoped["id"]}
    assert {tag for tags in selected_scopes for tag in tags if tag.startswith("project:")} == {
        "project:None", f"project:{project['id']}",
    }
