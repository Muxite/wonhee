"""Ideas from two separate sources: inspiration teardowns (`c<id>`) and the creator's own brief (`b-xxxxx`).

The model cites references; the ORIGIN of each idea is computed here from what it cites, never chosen by the model:
inspiration only -> "inspiration", brief only -> "own", both -> "blend", neither -> "new". Footage is a shared
shoot list, so a plate that serves many lines (decoupled b-roll) ranks first.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field

from wonhee.edit.looks import Preset as LookPreset
from wonhee.edit.text import Position
from wonhee.edit.text import Preset as TextPreset
from wonhee.inspire.schema import Camera, Format

Origin = Literal["inspiration", "own", "blend", "new"]
ORIGINS: tuple[str, ...] = ("blend", "inspiration", "own", "new")
_CREF = re.compile(r"^c(\d+)$")


class Plate(BaseModel):
    subject: str
    setting: str
    camera: Camera
    framing: str
    light: str
    duration_s: float = Field(ge=0, le=180)
    person_on_camera: bool


class Idea(BaseModel):
    title: str
    format: Format
    inspiration: list[str] = Field(default_factory=list, description="post refs it borrows from, e.g. c12")
    brief: list[str] = Field(default_factory=list, description="brief item refs it builds on, e.g. b-94c4b")
    how: str = Field(description="one sentence: what it takes from each reference")
    line: str | None = Field(default=None, max_length=90, description="on-screen text, or null")
    caption: str
    plates: list[int] = Field(default_factory=list, description="indices into shoot_list")
    text_preset: TextPreset = "outline"
    text_position: Position = "upper"
    look: LookPreset | None = None
    effort_minutes: int = Field(ge=0, le=2000)
    cost_usd: float = Field(ge=0, le=500)


class IdeaSet(BaseModel):
    ideas: list[Idea]
    shoot_list: list[Plate]


class IdeaReply(BaseModel):
    """What the model fills: one array per intended source, so a requested mix is a slot to fill, not a hope.
    The origin reported is still computed from citations, whatever array an idea sits in."""
    blends: list[Idea] = Field(default_factory=list, description="a saved post's format carrying a brief idea")
    from_inspiration: list[Idea] = Field(default_factory=list, description="cite posts only, no brief items")
    from_brief: list[Idea] = Field(default_factory=list, description="cite brief items only, no posts")
    new: list[Idea] = Field(default_factory=list, max_length=2, description="no source; at most two")
    shoot_list: list[Plate]

    def to_set(self) -> IdeaSet:
        return IdeaSet(ideas=[*self.blends, *self.from_inspiration, *self.from_brief, *self.new],
                       shoot_list=self.shoot_list)


def split(n: int) -> dict[str, int]:
    """How many ideas to ask for per array: about 40% blends, 30% inspiration only, 30% brief only."""
    b, i = round(n * 0.4), round(n * 0.3)
    return {"blends": b, "from_inspiration": i, "from_brief": n - b - i}


def digest(teardowns: list[dict], metrics: dict[int, dict]) -> str:
    """One compact block per post: what the post does (creator-blind teardown fields only)."""
    out = []
    for t in teardowns:
        m = metrics.get(t["content_id"]) or {}
        o = t["text_overlay"]
        ovl = f'"{o["english"]}" ({o["style"]}@{o["position"]})' if o["present"] and o.get("english") else "none"
        foot = "; ".join(f"{f['subject']} [{f['camera']}]" for f in t.get("footage_to_record", [])[:2]) or "-"
        out.append("\n".join([
            f"c{t['content_id']}: {t['format']}, text/footage {t['coupling']['level']}, replicate {t['replicate']['score_1to5']}/5, "
            f"plays {m.get('plays') or '-'}, likes {m.get('likes') or '-'}",
            f"  what: {t['summary']}",
            f"  hook: {t['hook']['mechanism']}",
            f"  overlay: {ovl}; look {t['look'] or 'none'}; audio {t['audio']['kind']}",
            f"  template: {t.get('text_template') or '-'}",
            f"  footage: {foot}",
            f"  learn: {' | '.join(t.get('learn', [])[:3]) or '-'}"]))
    return "\n".join(out)


def origin_of(idea: Idea) -> Origin:
    if idea.inspiration and idea.brief:
        return "blend"
    return "inspiration" if idea.inspiration else "own" if idea.brief else "new"


def _brief_ref(r: str, brief_ids: set[str]) -> str | None:
    """The brief id a reference means: exact, or the one id it is a unique prefix of (models clip hex ids)."""
    r = r.strip()
    if r in brief_ids:
        return r
    hits = [b for b in brief_ids if len(r) >= 5 and b.startswith(r)]
    return hits[0] if len(hits) == 1 else None


def clean(s: IdeaSet, post_ids: set[int], brief_ids: set[str]) -> list[str]:
    """Drop references that were not in the input (in place); repair clipped brief ids. Returns one note per
    dropped reference."""
    notes = []
    for i, idea in enumerate(s.ideas):
        keep_c = [r for r in dict.fromkeys(idea.inspiration) if (m := _CREF.match(r.strip())) and int(m.group(1)) in post_ids]
        fixed = {r: _brief_ref(r, brief_ids) for r in idea.brief}
        idea.brief = [fixed[r] or r for r in idea.brief]
        keep_b = list(dict.fromkeys(fixed[r] for r in fixed if fixed[r]))
        keep_p = [p for p in dict.fromkeys(idea.plates) if 0 <= p < len(s.shoot_list)]
        for bad in [*set(idea.inspiration) - set(keep_c), *set(idea.brief) - set(keep_b)]:
            notes.append(f"idea {i}: unknown reference {bad!r}")
        for bad in set(idea.plates) - set(keep_p):
            notes.append(f"idea {i}: no plate {bad}")
        idea.inspiration, idea.brief, idea.plates = keep_c, keep_b, keep_p
        if idea.line:
            idea.line = idea.line.strip().strip('"“”').strip() or None
    return notes


def to_json(s: IdeaSet) -> dict:
    ideas = [{"origin": origin_of(i), **i.model_dump()} for i in s.ideas]
    serves = Counter(p for i in s.ideas for p in i.plates)
    plates = [{"serves": [n for n, i in enumerate(s.ideas) if k in i.plates], **p.model_dump()}
              for k, p in enumerate(s.shoot_list)]
    return {"ideas": ideas, "shoot_list": plates, "plate_reuse": dict(serves)}


def report(data: dict, posts: dict[int, str], brief: dict[str, str], header: str = "") -> str:
    """Markdown: ideas split by origin (blend / inspiration / own / new), then the shoot list ranked by reuse."""
    ideas, plates = data["ideas"], data["shoot_list"]
    by = Counter(i["origin"] for i in ideas)
    used_c = {r for i in ideas for r in i["inspiration"]}
    used_b = {r for i in ideas for r in i["brief"]}
    out = [f"# Ideas: {len(ideas)} posts to make, {len(plates)} plates to film", "", header,
           f"Origins: {', '.join(f'{o} {by[o]}' for o in ORIGINS if by[o])}. "
           f"Uses {len(used_c)} of {len(posts)} inspiration posts and {len(used_b)} of {len(brief)} brief items."
           + (f" Posts not used this batch: {', '.join(f'c{c}' for c in sorted(posts) if f'c{c}' not in used_c)}."
              if len(used_c) < len(posts) else ""), "",
           "Origin is computed from what each idea cites: **inspiration** = only saved posts, **own** = only your "
           "brief, **blend** = both, **new** = neither.", ""]
    titles = {"blend": "Blends (a saved post's format × your idea)", "inspiration": "From inspiration only",
              "own": "From your brief only", "new": "New (no source)"}
    for o in ORIGINS:
        group = [(n, i) for n, i in enumerate(ideas) if i["origin"] == o]
        if not group:
            continue
        out += [f"## {titles[o]}", ""]
        for n, i in group:
            render = [f"preset={i['text_preset']}", f"position={i['text_position']}"] if i["line"] else []
            if i["look"]:
                render.append(f"look={i['look']}")
            out.append(f"**{n}. {i['title']}** ({i['format']}, ~{i['effort_minutes']} min, ${i['cost_usd']:g})")
            if i["line"]:
                out.append(f"- line: “{i['line']}”")
            out.append(f"- caption: {i['caption']}")
            refs = [f"c{r[1:]} {posts.get(int(r[1:]), '')[:70]}" for r in i["inspiration"]]
            refs += [f"{r} “{brief.get(r, '')[:70]}”" for r in i["brief"]]
            if refs:
                out.append("- from: " + "; ".join(refs))
            out.append(f"- how: {i['how']}")
            if i["plates"]:
                out.append(f"- film: {', '.join(f'P{p}' for p in i['plates'])}")
            if render:
                out.append(f"- render: `{' '.join(render)}`")
            out.append("")
    out += ["## Shoot list (b-roll), most reused first", "",
            "| plate | serves ideas | subject | setting | camera, framing | light | s | you on camera |",
            "|---|---|---|---|---|---|---|---|"]
    for k, p in sorted(enumerate(plates), key=lambda kp: (-len(kp[1]["serves"]), kp[0])):
        out.append(f"| P{k} | {', '.join(map(str, p['serves'])) or '-'} | {p['subject']} | {p['setting']} | "
                   f"{p['camera']}, {p['framing']} | {p['light']} | {p['duration_s']:g} | "
                   f"{'yes' if p['person_on_camera'] else 'no'} |")
    out.append("")
    return "\n".join(out)
