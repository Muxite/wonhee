"""Everything wonhee knows about one stored post, as text for a cheap (text-only) teardown call.

Built from rows grab/analyze already paid for: caption, metrics, audio, transcript, per-shot or per-slide
on-screen text and visual descriptions. No images are sent again.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from wonhee.db import queries
from wonhee.llm import data_block
from wonhee.media import path_for


@dataclass
class Dossier:
    content_id: int
    text: str
    sources: list[str] = field(default_factory=list)  # caption / on-screen / speech, for the no-copy guard
    foreign: list[str] = field(default_factory=list)  # texts that look non-English (for local translation)
    language: str | None = None


def _non_latin_share(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    return sum(1 for c in letters if ord(c) > 0x024F) / len(letters) if letters else 0.0


def looks_foreign(s: str | None, whisper_lang: str | None = None) -> bool:
    if not s or not s.strip():
        return False
    return _non_latin_share(s) > 0.3 or (whisper_lang is not None and whisper_lang not in ("en", ""))


def build(conn, media_dir, content_id: int) -> Dossier:
    from wonhee.library.grab import IMAGE_ANALYZER_VERSION
    from wonhee.scout.stages.analyze import ANALYZER_VERSION

    r = conn.execute(
        """SELECT l.*, c.caption, c.media_sha256, a.title AS audio_title, a.artist AS audio_artist
           FROM v_content_latest l JOIN content_item c ON c.id=l.content_id
           LEFT JOIN audio_track a ON a.id=c.audio_id WHERE l.content_id=?""", (content_id,)).fetchone()
    if r is None:
        raise ValueError(f"content {content_id} is not stored")
    d = Dossier(content_id, "")
    n = lambda v: "hidden" if v is None else f"{v:,}"  # noqa: E731
    lines = [f"post content_id={content_id} @{r['handle']} ({r['relation']}) posted {r['posted_at'] or '?'}",
             f"metrics: plays {n(r['plays'])}, likes {n(r['likes'])}, comments {n(r['comments'])}",
             f"audio: {r['audio_title'] or '-'} / {r['audio_artist'] or '-'}",
             data_block("caption", r["caption"] or "")]
    if r["caption"]:
        d.sources.append(r["caption"])
        if looks_foreign(r["caption"]):
            d.foreign.append(r["caption"])
    sha = r["media_sha256"]
    ana = queries.media_analysis_row(conn, sha) if sha else None
    if sha and ana is not None:
        m = conn.execute("SELECT duration_ms, width, height FROM media WHERE blob_sha256=?", (sha,)).fetchone()
        d.language = ana["language"]
        lines.append(f"video: {m['width']}x{m['height']}, {m['duration_ms'] / 1000:.1f}s; speech language "
                     f"{ana['language'] or '-'}; first word {ana['first_word_ms']} ms")
        speech = ""
        if ana["transcript_sha256"]:
            t = json.loads(path_for(media_dir, ana["transcript_sha256"]).read_text())
            speech = " ".join(s["text"].strip() for s in t["segments"] if s["text"].strip())[:2000]
        lines.append(data_block("speech transcript (whisper; may be song lyrics)", speech))
        if speech:
            d.sources.append(speech)
            if looks_foreign(speech, ana["language"]):
                d.foreign.append(speech)
        for s in queries.media_shots(conn, sha, ANALYZER_VERSION):
            lines.append(f"shot {s['ord']} {s['start_ms']}-{s['end_ms']} ms: "
                         + data_block("on-screen text", s["on_screen_text"] or "") + " "
                         + data_block("visual", s["visual"] or ""))
            if s["on_screen_text"]:
                d.sources.append(s["on_screen_text"])
                if looks_foreign(s["on_screen_text"]) and s["on_screen_text"] not in d.foreign:
                    d.foreign.append(s["on_screen_text"])
    slides = conn.execute(
        """SELECT m.ord, m.kind, a.on_screen_text, a.visual FROM content_media m
           LEFT JOIN image_analysis a ON a.media_sha256=m.media_sha256 AND a.analyzer_version=?
           WHERE m.content_id=? ORDER BY m.ord""", (IMAGE_ANALYZER_VERSION, content_id)).fetchall()
    if not sha and slides:
        lines.append(f"photo post: {len(slides)} slide(s)")
        for s in slides:
            lines.append(f"slide {s['ord'] + 1} ({s['kind']}): " + data_block("on-screen text", s["on_screen_text"] or "")
                         + " " + data_block("visual", s["visual"] or "(not analysed)"))
            if s["on_screen_text"]:
                d.sources.append(s["on_screen_text"])
                if looks_foreign(s["on_screen_text"]) and s["on_screen_text"] not in d.foreign:
                    d.foreign.append(s["on_screen_text"])
    if not sha and not slides:
        lines.append("media: not downloaded (judge from caption and metrics only; confidence low)")
    d.text = "\n".join(lines)
    return d


def _words(s: str) -> list[str]:
    return "".join(c.lower() if c.isalnum() else " " for c in s).split()


def copies_source(template: str, sources: list[str], n: int = 5) -> str | None:
    """The first n-word run the template shares with a source text (case-insensitive), or None.
    [slots] break runs, so only the template's fixed wording is compared."""
    grams = set()
    for part in re.split(r"\[[^\]]*\]", template):
        w = _words(part)
        grams |= {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}
    for src in sources:
        w = _words(src)
        for i in range(len(w) - n + 1):
            g = " ".join(w[i:i + n])
            if g in grams:
                return g
    return None
