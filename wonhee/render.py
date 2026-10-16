"""Plain-text, phone-readable views of a shot list and of a run."""
from __future__ import annotations

import textwrap
from datetime import datetime
from pathlib import Path

from wonhee.db import queries
from wonhee.media import path_for

WIDTH = 68


def _wrap(text: str, indent: str = "  ") -> list[str]:
    return textwrap.wrap(text, WIDTH, initial_indent=indent, subsequent_indent=indent + "  ") or [indent]


def _cited_trend_name(conn, shotlist_id: int, project_id: int) -> str | None:
    """The trend this shot list actually cites (a later pick may have changed the project's primary)."""
    primary = queries.primary_trend(conn, project_id)
    row = conn.execute(
        """SELECT t.name FROM shot_citation sc
           JOIN trend_example te ON te.content_id = sc.content_id JOIN trend t ON t.id = te.trend_id
           WHERE sc.shotlist_id = ? GROUP BY t.id
           ORDER BY count(*) DESC, t.id = ? DESC, t.id DESC LIMIT 1""",
        (shotlist_id, primary["trend_id"] if primary else -1)).fetchone()
    return row[0] if row else (primary["name"] if primary else None)


def render_shotlist(conn, project_id: int) -> str:
    sl = queries.latest_shotlist(conn, project_id)
    if sl is None:
        return f"No shot list for project {project_id} yet. Run the scout first."
    run_id = conn.execute("SELECT run_id FROM step WHERE id=?", (sl["created_by_step"],)).fetchone()[0]
    project = conn.execute("SELECT title FROM project WHERE id=?", (project_id,)).fetchone()
    out = [f"Shot list v{sl['version']}: {project['title']}"]
    trend_name = _cited_trend_name(conn, sl["id"], project_id)
    if trend_name:
        out.append(f"Trend: {trend_name}")

    for st in conn.execute("SELECT * FROM v_current_steps WHERE run_id=? ORDER BY id", (run_id,)):
        for f in queries.step_flags(conn, st["id"]):
            line = f"⚠ {st['stage']}: {queries.flag_meaning(conn, f['code'])}"
            out.append(line + (f" ({f['detail']})" if f["detail"] else ""))

    cites = queries.citations(conn, sl["id"])
    refs: list[str] = []  # url, numbered by first appearance
    by_shot: dict[int, list[int]] = {}
    for c in cites:
        url = c["content_url"] or c["source_url"]
        if url is None:
            continue
        if url not in refs:
            refs.append(url)
        n = refs.index(url) + 1
        if n not in by_shot.setdefault(c["shot_ord"], []):
            by_shot[c["shot_ord"]].append(n)

    out.append("")
    for s in queries.shots(conn, sl["id"]):
        parts = [f"{s['ord']}. [{s['duration_ms'] / 1000:.1f}s] {s['framing']} — {s['action']}"]
        if s["on_screen_text"]:
            parts.append(f'text: "{s["on_screen_text"]}"')
        if s["audio_cue"]:
            parts.append(f"audio: {s['audio_cue']}")
        line = " | ".join(parts)
        if s["ord"] in by_shot:
            line += f"  (refs {','.join(map(str, sorted(by_shot[s['ord']])))})"
        out.append(line)  # one line per shot, as specified; the phone wraps it

    out += ["", "References:"]
    out += [f"{i}. {u}" for i, u in enumerate(refs, 1)] or ["(none)"]
    out += ["", f"run {run_id} · ${queries.run_cost(conn, run_id):.2f} · wonhee show run {run_id}"]
    return "\n".join(out)


def _secs(step) -> str:
    if not step["ended_at"]:
        return "-"
    try:
        d = datetime.fromisoformat(step["ended_at"]) - datetime.fromisoformat(step["started_at"])
    except ValueError:
        return "-"
    return f"{d.total_seconds():.0f}"


def render_run(conn, run_id: int, media_dir: Path) -> str:
    run = queries.run_row(conn, run_id)
    steps = queries.run_steps(conn, run_id)
    superseded = {s["supersedes_step"] for s in steps if s["supersedes_step"] is not None}
    out = [f"run {run_id}: {run['status']} · ${queries.run_cost(conn, run_id):.2f} of ${run['cost_cap_usd']:.2f}",
           f"{'stage':<9}{'try':>3} {'status':<7}{'cost':>7}{'sec':>5}  flags"]
    for s in steps:
        flags = queries.step_flags(conn, s["id"])
        mark = "  (superseded)" if s["id"] in superseded else ""
        out.append(f"{s['stage']:<9}{s['attempt']:>3} {s['status']:<7}${s['cost_usd']:>5.2f}{_secs(s):>5}  "
                   + ",".join(f["code"] for f in flags) + mark)
        for f in flags:
            if f["detail"]:
                out.extend(_wrap(f"{f['code']}: {f['detail']}", "      "))
        if s["status"] == "failed":
            for a in queries.step_artifacts(conn, s["id"]):
                if a["role"] == "error":
                    out.append(f"      error file: {path_for(media_dir, a['blob_sha256'])}")
    return "\n".join(out)


def _as_media_dir(x) -> Path:
    return x if isinstance(x, Path) else x.media_dir


def _preview(conn, media_dir: Path, type_: str, ref: str) -> list[str]:
    try:
        return _preview_unsafe(conn, media_dir, type_, ref)
    except Exception:  # noqa: BLE001 - a preview must never break the render
        return [f"    {path_for(media_dir, ref) if type_ != 'content' else ref}"]


def _preview_unsafe(conn, media_dir: Path, type_: str, ref: str) -> list[str]:
    ind = "    "
    if type_ == "subtitles":
        from wonhee.nodes.subtitles import load_subs
        subs = load_subs(media_dir, ref)
        out = [f"{ind}language {subs.language or '-'}, {len(subs.lines)} line(s)"]
        out += [f"{ind}[{ln.start_ms / 1000:.1f}s] {ln.text}" for ln in subs.lines[:3]]
        return out
    path = path_for(media_dir, ref)
    if type_ == "video":
        m = conn.execute("SELECT * FROM media WHERE blob_sha256=?", (ref,)).fetchone()
        try:
            from wonhee.media import has_audio
            audio = "yes" if has_audio(path) else "no"
        except Exception:  # noqa: BLE001 - a preview must never break the render
            audio = "?"
        dur = f"{m['duration_ms'] / 1000:.1f}s" if m and m["duration_ms"] is not None else "?s"
        size = f"{m['width']}x{m['height']}" if m and m["width"] else "?x?"
        return [f"{ind}{path}", f"{ind}{size}, {dur}, audio {audio}"]
    if type_ == "content":
        cid = int(ref.split(":", 1)[1])
        r = conn.execute("SELECT handle, url FROM v_content_latest WHERE content_id=?", (cid,)).fetchone()
        return [f"{ind}{r['url']}  @{r['handle']}"] if r else [f"{ind}content {cid}"]
    out = [f"{ind}{path}"]
    if type_ == "text":
        try:
            out += _wrap(path.read_text(encoding="utf-8")[:300], ind)
        except OSError:
            pass
    return out


def render_chain(conn, settings_or_media_dir, run_id: int) -> str:
    """render_run's step table, then the chain's final outputs (output ports nothing else consumes), with previews."""
    from wonhee.nodes.chain import WIRE_RE
    from wonhee.nodes.types import is_literal
    from wonhee.pipeline.orchestrator import load_request

    media_dir = _as_media_dir(settings_or_media_dir)
    _, chain = load_request(conn, run_id)
    out = [render_run(conn, run_id, media_dir)]
    consumed = {WIRE_RE.match(v).groups() for cn in chain.nodes for v in cn.in_.values()
                if not is_literal(v) and WIRE_RE.match(v)}  # (node id, port) pairs something reads
    lines: list[str] = []
    for cn in chain.nodes:
        step = queries.current_step(conn, run_id, cn.id)
        if step is None or step["status"] not in ("ok", "weak"):
            continue
        for port, o in queries.step_outputs(conn, step["id"]).items():
            if (cn.id, port) in consumed:
                continue
            lines.append(f"  {cn.id}.{port} {o['type']}")
            lines += _preview(conn, media_dir, o["type"], o["ref"])
    if lines:
        out += ["", "outputs:", *lines]
    return "\n".join(out)


def render_library(conn, project_id: int, media_dir: Path) -> str:
    """What a grab stored: one block per reel (metrics, file, cuts, speech, on-screen text)."""
    import json

    from wonhee.scout.stages.analyze import ANALYZER_VERSION

    rows = conn.execute(
        """SELECT l.*, c.caption, c.media_sha256, a.title AS audio_title, a.artist AS audio_artist
           FROM (SELECT content_id FROM project_output WHERE project_id=:p
                 UNION SELECT content_id FROM project_reference WHERE project_id=:p) x
           JOIN v_content_latest l ON l.content_id = x.content_id
           JOIN content_item c ON c.id = x.content_id
           LEFT JOIN audio_track a ON a.id = c.audio_id
           ORDER BY x.content_id""", {"p": project_id}).fetchall()
    out: list[str] = []
    for r in rows:
        out.append(f"[content {r['content_id']}] @{r['handle']} ({r['relation']})  {r['url']}")
        n = lambda v: "-" if v is None else f"{v:,}"  # noqa: E731 - None = hidden or not reported
        out.append(f"  posted {r['posted_at'] or '?'} · plays {n(r['plays'])} · likes {n(r['likes'])} · "
                   f"comments {n(r['comments'])} · shares {n(r['shares'])}  (as of {r['observed_at']})")
        if r["caption"]:
            out.append("  caption:")
            out.extend(_wrap(r["caption"], "    "))
        out.append(f"  audio: {r['audio_title'] or '-'} / {r['audio_artist'] or '-'}")
        sha = r["media_sha256"]
        if not sha:
            out.append("  video: not downloaded")
            continue
        m = conn.execute("SELECT * FROM media WHERE blob_sha256=?", (sha,)).fetchone()
        out.append(f"  video: {m['width']}x{m['height']} {round(m['fps'] or 0, 2)}fps {m['duration_ms']}ms  "
                   f"{path_for(media_dir, sha)}")
        st = conn.execute("SELECT * FROM v_media_style WHERE media_sha256=? AND analyzer_version=?",
                          (sha, ANALYZER_VERSION)).fetchone()
        if st is None:
            out.append("  analysis: none yet")
            continue
        cps = f"{st['cuts_per_sec']:.2f}" if st["cuts_per_sec"] is not None else "-"
        out.append(f"  cuts: {st['shot_count']} shot(s), {cps} cuts/s · first word: "
                   f"{str(st['first_word_ms']) + ' ms' if st['first_word_ms'] is not None else 'no speech'} · "
                   f"language: {st['language'] or '-'}")
        ma = conn.execute("SELECT transcript_sha256 FROM media_analysis WHERE media_sha256=? AND analyzer_version=?",
                          (sha, ANALYZER_VERSION)).fetchone()
        if ma and ma["transcript_sha256"]:
            t = json.loads(path_for(media_dir, ma["transcript_sha256"]).read_text())
            text = " ".join(s["text"].strip() for s in t["segments"] if s["text"].strip())
            if text:
                out.append("  speech:")
                out.extend(_wrap(text[:1200], "    "))
        for s in conn.execute("SELECT * FROM media_shot WHERE media_sha256=? AND analyzer_version=? ORDER BY ord",
                              (sha, ANALYZER_VERSION)):
            bits = [f"  shot {s['ord']} {s['start_ms']}-{s['end_ms']}ms"]
            if s["on_screen_text"]:
                bits.append(f'text "{s["on_screen_text"]}"')
            if s["visual"]:
                bits.append(s["visual"])
            if s["keyframe_sha256"]:
                bits.append(f"keyframe {path_for(media_dir, s['keyframe_sha256'])}")
            out.append(" | ".join(bits))
    return "\n".join(out) if out else "(nothing stored)"
