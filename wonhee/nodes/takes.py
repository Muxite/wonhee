"""take_pick: rank candidate cuts of the owner talking (post-assistant spec §2.2). It returns a plan as JSON text; the
render is the existing compose_video, so this node never touches pixels and never needs a GPU."""
from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, Field

from wonhee.analyze.speech import SpokenWord
from wonhee.edit import takes as T
from wonhee.nodes import registry
from wonhee.nodes.base import Node
from wonhee.nodes.subtitles import load_subs, line_words, save_text

TAKE_PORTS = ("subtitles", "subtitles_b", "subtitles_c", "subtitles_d")  # one per take, in take order


class PickParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    top_n: int = Field(default=3, ge=1, le=20)
    # Loudness spread in dB per take (loud minus quiet parts), measured by the caller; None = not measured and the
    # loudness term is skipped. Order follows the take ports.
    loudness_db: list[float | None] | None = None


def pick_fn(ctx, inputs, p: PickParams) -> dict[str, str]:
    d = ctx.settings.media_dir
    takes, estimated, ports = [], False, []
    for port in TAKE_PORTS:
        if not inputs.get(port):
            continue
        words = []
        for ln in load_subs(d, inputs[port]).lines:
            ws, est = line_words(ln)
            estimated = estimated or est
            words += [SpokenWord(w.start_ms, w.end_ms, w.text) for w in ws]
        takes.append(words)
        ports.append(port)
    if estimated:
        ctx.flag("WORDS_ESTIMATED", "subtitles carry line timing only; words timed evenly within each line, "
                                    "so stumble and dead-air timings are approximate")
    plans = T.rank_takes(takes, p.loudness_db, p.top_n)
    if not plans:
        ctx.flag("NO_TEXT", "no spoken words in any take; nothing to rank")
    body = {"takes": ports, "plans": [pl.to_dict() for pl in plans]}
    return {"plans": save_text(ctx.conn, d, json.dumps(body, indent=1, sort_keys=True))}


registry.register(Node("take_pick", 1, {"subtitles": "subtitles", "subtitles_b": "subtitles?",
                                        "subtitles_c": "subtitles?", "subtitles_d": "subtitles?"},
                       {"plans": "text"}, PickParams, pick_fn,
                       "Rank candidate cuts of one or more takes of someone talking, by stumbles, dead air and how soon "
                       "the point lands (plan as JSON, not a video)."))
