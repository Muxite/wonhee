"""translate node: timed subtitles in any language -> natural target-language subtitles, ids and timings kept."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from wonhee.llm import data_block
from wonhee.nodes import registry
from wonhee.nodes.base import Node
from wonhee.nodes.subtitles import Subtitles, load_subs, save_subs, save_text
from wonhee.scout.prompts import SYSTEM_TRANSLATE


class TLine(BaseModel):
    id: int
    text: str


class Term(BaseModel):
    source: str
    target: str


class TranslateReply(BaseModel):
    lines: list[TLine]
    glossary: list[Term] = []
    caption: str | None = None
    gist: str | None = None
    notes: list[str] = []


class TranslateParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    to: str = Field(default="en", pattern=r"^[A-Za-z][A-Za-z -]{0,31}$")  # a code or name: en, English, pt-BR
    style: Literal["natural", "literal"] = "natural"
    batch: int = 80
    overlap: int = 10
    engine: Literal["remote", "local"] = "remote"  # local = the on-box model (settings.local_llm_model), $0


def _listing(lines) -> str:
    return "\n".join(f"[{ln.id}] {ln.start_ms / 1000:.1f}s: {ln.text}" for ln in lines)


def _prompt(p: TranslateParams, lines, *, caption: str | None, context, glossary: dict[str, str],
            first: bool = False) -> str:
    how = ("idiomatic subtitles, how a native speaker would write them, not word for word"
           if p.style == "natural" else "close to the source wording, staying faithful rather than idiomatic")
    parts = [f"Translate these subtitle lines into {p.to} as {how}. Return one line per numbered id, same ids."]
    if first:
        parts.append("This is the start of the video: also give a one or two sentence gist of the whole piece"
                     + (" and translate the caption." if caption else "."))
    if caption:
        parts.append(data_block("caption", caption))
    if context:
        parts.append(data_block("previous lines (context only, already translated, do not translate these again)",
                                "\n".join(f"[{ln.id}] {ln.text} => {t}" for ln, t in context)))
    if glossary:
        parts.append(data_block("glossary so far", "\n".join(f"{s} => {t}" for s, t in glossary.items())))
    parts.append(data_block("subtitle lines", _listing(lines)))
    return "\n\n".join(parts)


def _caption_text(ctx, ref: str | None) -> str | None:
    if not ref:
        return None
    row = ctx.conn.execute("SELECT caption FROM content_item WHERE id=?", (int(ref.split(":", 1)[1]),)).fetchone()
    return (row[0] or None) if row else None


def translate_fn(ctx, inputs, p: TranslateParams) -> dict[str, str]:
    d = ctx.settings.media_dir
    subs = load_subs(d, inputs["subs"])
    if not subs.lines:
        ctx.flag("NO_TEXT", "no subtitle lines to translate")
        return {"subs": save_subs(ctx.conn, d, Subtitles(language=p.to)), "notes": save_text(ctx.conn, d, "")}
    caption = _caption_text(ctx, inputs.get("caption"))
    size, overlap = max(p.batch, 1), max(p.overlap, 0)
    done: dict[int, str] = {}
    glossary: dict[str, str] = {}
    gist: str | None = None
    caption_tr: str | None = None
    notes: list[str] = []
    gaps: list[int] = []

    def ask(lines, first: bool, context):
        prompt = _prompt(p, lines, caption=caption if first else None, context=context, glossary=glossary,
                         first=first)
        if p.engine == "local":
            from wonhee import localllm
            return localllm.chat_json(ctx.settings, TranslateReply, system=SYSTEM_TRANSLATE, prompt=prompt)
        return ctx.llm(TranslateReply, schema_name="TranslateReply", system=SYSTEM_TRANSLATE, prompt=prompt)

    for start in range(0, len(subs.lines), size):
        batch = subs.lines[start:start + size]
        ids = {ln.id for ln in batch}
        ctx_lines = [(ln, done[ln.id]) for ln in subs.lines[max(start - overlap, 0):start]] if overlap else []
        reply = ask(batch, start == 0, ctx_lines)
        got = {t.id: t.text for t in reply.lines if t.id in ids and t.text.strip()}
        for term in reply.glossary:
            glossary.setdefault(term.source, term.target)
        gist = gist or reply.gist
        caption_tr = caption_tr or reply.caption
        notes += [n for n in reply.notes if n not in notes]
        missing = [ln for ln in batch if ln.id not in got]
        if missing:
            miss_ids = {ln.id for ln in missing}
            near = {i for k, ln in enumerate(batch) if ln.id in miss_ids
                    for i in range(max(k - 2, 0), min(k + 3, len(batch)))}
            neighbours = [(ln, got[ln.id]) for k, ln in enumerate(batch) if k in near and ln.id in got]
            retry = ask(missing, False, ctx_lines + neighbours)
            for t in retry.lines:
                if t.id in miss_ids and t.text.strip():
                    got.setdefault(t.id, t.text)
            for term in retry.glossary:
                glossary.setdefault(term.source, term.target)
            notes += [n for n in retry.notes if n not in notes]
        for ln in batch:
            if ln.id in got:
                done[ln.id] = got[ln.id]
            else:
                done[ln.id] = ln.text
                gaps.append(ln.id)

    if gaps:
        ctx.flag("TRANSLATION_GAP", "ids: " + ", ".join(map(str, gaps)))
        ctx.weak = True
    # word order changes in translation: lines keep their timings and drop `words` (reel-styles spec §1)
    out = Subtitles(language=p.to, lines=[ln.model_copy(update={"text": done[ln.id], "words": None}) for ln in subs.lines])
    sections = []
    if gist:
        sections.append(gist)
    if caption_tr and caption:
        sections.append(f"caption: {caption_tr}")
    if glossary:
        sections.append("glossary:\n" + "\n".join(f"- {s} => {t}" for s, t in glossary.items()))
    if notes:
        sections.append("notes:\n" + "\n".join(f"- {n}" for n in notes))
    return {"subs": save_subs(ctx.conn, d, out), "notes": save_text(ctx.conn, d, "\n\n".join(sections))}


registry.register(Node("translate", 2, {"subs": "subtitles", "caption": "content?"},
                       {"subs": "subtitles", "notes": "text"}, TranslateParams, translate_fn,
                       "Translate subtitles into natural target-language subtitles with a name glossary and notes.",
                       salt=lambda s: {"llm_model": s.llm_model, "local_llm_model": s.local_llm_model}))
