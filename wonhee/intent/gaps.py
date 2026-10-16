"""Deterministic gap rules. The compiler may only ask about a field whose rule fires here."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date

from wonhee.intent.creator import Creator
from wonhee.intent.schemas import ScoutRequest
from wonhee.intent.window import resolve_window

IG_URL = re.compile(r"^https?://(www\.)?instagram\.com/(reel|reels|p)/[\w-]+/?")


@dataclass
class Gap:
    field: str
    question: str
    options: list[str]
    default: str | None

    def as_dict(self) -> dict:
        return asdict(self)


def find_gaps(req: ScoutRequest, creator: Creator) -> list[Gap]:
    out: list[Gap] = []
    if not (req.niche or req.reference_urls or req.trend_name):
        out.append(Gap("niche", "What should I scout?",
                       ["a niche (e.g. 'gym humor')", "paste an Instagram reel link", "name a trend"], None))
    if req.mode == "discover" and req.window is None:
        out.append(Gap("window", "Which time?", ["now", "6 months ago", "1 year ago", "2 years ago"], "now"))
    bad = [u for u in req.reference_urls if not IG_URL.match(u)]
    if bad:
        out.append(Gap("reference_urls", f"Not an Instagram reel/post link: {bad[0]}. Replace it or drop it?",
                       ["paste another link", "drop it"], "drop it"))
    if req.constraints.on_camera is True and "on_camera" not in req.acknowledged and creator.on_camera is False:
        out.append(Gap("on_camera", "Your creator profile says no face-cam. Film on camera this time?",
                       ["yes, this once", "no, keep off-camera"], "no, keep off-camera"))
    if (req.constraints.max_length_s and req.constraints.max_length_s > 90
            and "max_length_s" not in req.acknowledged):
        out.append(Gap("max_length_s", f"{req.constraints.max_length_s}s is long for a trend reel. Cap it?",
                       ["cap at 30s", "cap at 60s", "keep it"], "cap at 30s"))
    return out


def apply_defaults(req: ScoutRequest, gaps: list[Gap], creator: Creator, today: date) -> ScoutRequest:
    r = req.model_copy(deep=True)
    for g in gaps:
        if g.field == "window":
            r.window = resolve_window("now", None, None, today)
        elif g.field == "reference_urls":
            r.reference_urls = [u for u in r.reference_urls if IG_URL.match(u)]
        elif g.field == "on_camera":
            r.constraints.on_camera = creator.on_camera
        elif g.field == "max_length_s":
            r.constraints.max_length_s = 30
    return derive_mode(r)


def derive_mode(r: ScoutRequest) -> ScoutRequest:
    r = r.model_copy(deep=True)
    r.mode = "from_reference" if (r.reference_urls or r.trend_name) else ("discover" if r.niche else None)
    return r
