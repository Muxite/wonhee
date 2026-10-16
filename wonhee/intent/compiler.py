from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from wonhee import llm
from wonhee.db import queries, repo
from wonhee.intent.creator import load_creator
from wonhee.intent.gaps import IG_URL, apply_defaults, derive_mode, find_gaps
from wonhee.intent.schemas import SCHEMA_NAME, SCHEMA_VERSION, Constraints, Extracted, ScoutRequest
from wonhee.intent.window import resolve_window

GO_ALWAYS = {"go", "run", "run it", "do it", "👍"}
CONFIRM_ONLY = {"yes", "y", "ok", "okay"}


def _norm(text: str) -> str:
    return text.strip().lower().rstrip(".!? \t\n")

SYSTEM = (
    "You turn a creator's message into fields for an Instagram trend scout. "
    "Fill only what the message actually says; leave everything else null. "
    "window_phrase: 'now', '6mo', '1y' or '2y' when the user means roughly that; use window_start/window_end "
    "only for explicit dates. reference_urls: only URLs present in the message. "
    "keep_as_is: when the user explicitly agrees to keep a flagged setting (e.g. 'yes, this once' to filming "
    "on camera, or 'keep it' for a long length) list that field (on_camera or max_length_s). "
    "drop_bad_urls: true when the user says to drop/remove the invalid link. Never invent values."
)


@dataclass
class CompileResult:
    intent_id: int
    status: Literal["questions", "confirm", "confirmed"]
    message: str
    request: ScoutRequest


def _questions(gaps) -> str:
    lines = ["Before I start:"]
    for i, g in enumerate(gaps, 1):
        opts = " / ".join(g.options)
        dflt = f" (default: {g.default})" if g.default else ""
        lines.append(f"{i}. {g.question} [{opts}]{dflt}")
    if all(g.default for g in gaps):
        lines.append("Answer, or reply go to accept the defaults.")
    return "\n".join(lines)


def _merge(req: ScoutRequest, ex: Extracted, today: date) -> ScoutRequest:
    r = req.model_copy(deep=True)
    if ex.niche: r.niche = ex.niche
    if ex.trend_name: r.trend_name = ex.trend_name
    if ex.reference_urls: r.reference_urls = ex.reference_urls
    if ex.drop_bad_urls: r.reference_urls = [u for u in r.reference_urls if IG_URL.match(u)]
    r.acknowledged = list(dict.fromkeys([*r.acknowledged, *ex.keep_as_is]))
    w = resolve_window(ex.window_phrase, ex.window_start, ex.window_end, today)
    if w: r.window = w
    c = r.constraints
    r.constraints = Constraints(
        on_camera=ex.on_camera if ex.on_camera is not None else c.on_camera,
        max_length_s=ex.max_length_s or c.max_length_s,
        tone=ex.tone or c.tone)
    return derive_mode(r)


def compile_message(conn, settings, text: str, *, intent_id: int | None = None, today: date | None = None) -> CompileResult:
    today = today or date.today()
    creator = load_creator(settings.creator_path)
    if intent_id is None:
        intent_id = repo.create_intent(conn, text, SCHEMA_NAME, SCHEMA_VERSION, ScoutRequest().model_dump(mode="json"))
    row = queries.intent_row(conn, intent_id)
    if row["confirmed_at"]:
        req = ScoutRequest.model_validate_json(row["compiled_json"])
        return CompileResult(intent_id, "confirmed", "Already confirmed.", req)
    req = ScoutRequest.model_validate_json(row["compiled_json"])
    repo.add_intent_turn(conn, intent_id, "user", text, None)
    turns = queries.intent_turns(conn, intent_id)
    last_compiler = next((t for t in reversed(turns) if t["role"] == "compiler"), None)

    t = _norm(text)
    no_match = False
    is_go = last_compiler is not None and (
        t in GO_ALWAYS or (t in CONFIRM_ONLY and last_compiler["gaps_json"] == "[]"))
    if is_go:
        gaps = find_gaps(req, creator)
        if not gaps and last_compiler["gaps_json"] == "[]":
            repo.confirm_intent(conn, intent_id)
            msg = f"Confirmed. Run it with: wonhee go {intent_id}"
            repo.add_intent_turn(conn, intent_id, "compiler", msg, None)
            return CompileResult(intent_id, "confirmed", msg, req)
        if all(g.default for g in gaps):
            req = apply_defaults(req, gaps, creator, today)
    else:
        asked = ""
        if last_compiler is not None:
            asked = f"{llm.data_block('questions asked', last_compiler['text'])}\n\n"
        prompt = (f"Current request so far:\n{llm.data_block('current request', req.model_dump_json())}\n\n"
                  f"{asked}{llm.data_block('user message', text)}")
        before = req.model_dump()
        ex = llm.call_json(conn, settings, Extracted, schema_name="ScoutExtract", system=SYSTEM,
                           prompt=prompt, intent_id=intent_id).value
        req = _merge(req, ex, today)
        no_match = last_compiler is not None and req.model_dump() == before

    repo.update_intent(conn, intent_id, req.model_dump(mode="json"))
    gaps = find_gaps(req, creator)
    if gaps:
        msg = _questions(gaps)
        if no_match:
            msg = "I couldn't match that to a question — " + msg
        repo.add_intent_turn(conn, intent_id, "compiler", msg, [g.as_dict() for g in gaps])
        return CompileResult(intent_id, "questions", msg, req)
    msg = f"{req.describe()} Reply go to run."
    repo.add_intent_turn(conn, intent_id, "compiler", msg, [])
    return CompileResult(intent_id, "confirm", msg, req)
