"""Find trend candidates: from the user's reference reels, from live hashtags (now), or from the web."""
from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta

from pydantic import BaseModel, Field

from wonhee.clock import now_iso
from wonhee.db import queries, repo
from wonhee.intent.gaps import IG_URL
from wonhee.llm import data_block
from wonhee.pipeline.context import Ctx, StageFailed
from wonhee.scout import prompts
from wonhee.sources.apify import (
    ACTOR_HASHTAG, ACTOR_REELS, ACTOR_SCRAPER, ReelRecord, hashtag_input, parse_item, profiles_input, urls_input,
)
from wonhee.sources.ingest import ingest_records

NOW_DAYS = 7
MAX_CLUSTERS = 8
WIDEN_DAYS = 60
MAX_CANDIDATES, MAX_HANDLES, MAX_URLS = 5, 3, 5


class HashtagPlan(BaseModel):
    tags: list[str]


class ClusterName(BaseModel):
    key: str
    name: str
    summary: str
    is_trend: bool


class ClusterVerdict(BaseModel):
    items: list[ClusterName]


class WebCandidate(BaseModel):
    name: str
    summary: str
    creator_handles: list[str] = Field(default_factory=list)
    example_urls: list[str] = Field(default_factory=list)
    hashtags: list[str] = Field(default_factory=list)
    source_ids: list[int] = Field(default_factory=list)


class WebCandidates(BaseModel):
    items: list[WebCandidate]


def _apify(ctx, actor, run_input):
    items = ctx.apify(actor, run_input)
    ctx.save_json({"actor": actor, "input": run_input, "items": items}, "api_response")
    recs = [r for r in map(parse_item, items) if r]
    ids = ingest_records(ctx.conn, ctx.step_id, recs, now_iso())
    return recs, ids


def _in_window(rec, start: date, end: date) -> bool:
    return bool(rec.posted_at) and start <= date.fromisoformat(rec.posted_at[:10]) <= end


def _dedupe(recs: list[ReelRecord]) -> list[ReelRecord]:
    seen: dict[str, ReelRecord] = {}
    for r in recs:
        seen.setdefault(r.external_id, r)
    return list(seen.values())


def _window(ctx: Ctx) -> tuple[date, date]:
    w = ctx.request.window
    return (w.start, w.end) if w else (ctx.today - timedelta(days=30), ctx.today)


def _cluster_line(key: str, recs: list[ReelRecord]) -> str:
    plays = sum(r.plays or 0 for r in recs)
    caps = "\n".join(f"- {r.caption or ''}" for r in recs[:6])
    return f"{data_block('key', key)} n={len(recs)} plays={plays} captions: {data_block('captions', caps)}"


def _name_clusters(ctx: Ctx, clusters: dict[str, list[ReelRecord]], niche_hint: str) -> list[ClusterName]:
    if not clusters:
        return []
    prompt = f"Niche: {niche_hint}\n\nClusters:\n" + "\n".join(_cluster_line(k, v) for k, v in clusters.items())
    verdict = ctx.llm(ClusterVerdict, schema_name="ClusterVerdict", system=prompts.SYSTEM_CLUSTERS, prompt=prompt)
    return [i for i in verdict.items if i.key in clusters]


def _reference_path(ctx: Ctx, refs, start: date, end: date) -> list[dict]:
    ref_ids = [r["content_id"] for r in refs]
    tags = queries.top_hashtags(ctx.conn, ref_ids, 3)
    ref_audio = {r["audio_id"] for r in refs if r["audio_id"]}
    example_ids = list(ref_ids)
    members: list[ReelRecord] = []
    if tags:
        recs, ids = _apify(ctx, ACTOR_HASHTAG, hashtag_input(tags, ctx.settings.hashtag_results))
        for r in _dedupe(recs):
            if r.audio_id in ref_audio and ids[r.external_id] not in example_ids:
                example_ids.append(ids[r.external_id])
                members.append(r)
    ref_recs = [ReelRecord(external_id=r["external_id"], url="", owner_id="", owner_handle="",
                           posted_at=r["posted_at"], caption=r["caption"], plays=r["plays"]) for r in refs]
    verdicts = _name_clusters(ctx, {"reference": ref_recs + members}, f"reference reels {len(refs)}")
    named = verdicts[0] if verdicts else None
    return [{"name": named.name if named else f"Trend from {refs[0]['external_id']}",
             "summary": named.summary if named else None,
             "examples": example_ids, "sources": [], "window": (start, end)}]


def _now_path(ctx: Ctx, start: date, end: date) -> list[dict]:
    plan = ctx.llm(HashtagPlan, schema_name="HashtagPlan", system=prompts.SYSTEM_HASHTAGS,
                   prompt=f"Niche: {ctx.request.niche}\nPeriod: {start} to {end}")
    tags = [t.strip().lstrip("#").strip() for t in plan.tags]
    tags = [t for t in tags if t and " " not in t][:3]
    if not tags:
        raise StageFailed("NO_CANDIDATES", f"no usable hashtags for niche {ctx.request.niche!r}; "
                                           "try a more specific niche")
    recs, ids = _apify(ctx, ACTOR_HASHTAG, hashtag_input(tags, ctx.settings.hashtag_results))
    by_audio: dict[str, list[ReelRecord]] = defaultdict(list)
    for r in _dedupe(recs):
        if r.audio_id and _in_window(r, start, end):
            by_audio[r.audio_id].append(r)
    clusters = {k: v for k, v in by_audio.items() if len(v) >= 2}
    top = sorted(clusters, key=lambda k: -sum(r.plays or 0 for r in clusters[k]))[:MAX_CLUSTERS]
    named = _name_clusters(ctx, {k: clusters[k] for k in top}, str(ctx.request.niche))
    return [{"name": n.name, "summary": n.summary, "examples": [ids[r.external_id] for r in clusters[n.key]],
             "sources": [], "window": (start, end)} for n in named if n.is_trend]


def _queries(ctx: Ctx, start: date, end: date) -> list[str]:
    month = (start + (end - start) / 2).strftime("%B %Y")
    if ctx.request.trend_name:
        t = ctx.request.trend_name
        return [f"{t} instagram trend", f"{t} trend meaning", f"site:instagram.com/reel {t} {month}"]
    niche = ctx.request.niche
    return [f"{niche} instagram reels trend {month}", f"trending instagram reels {month}",
            f"{niche} tiktok trend {month}", f"site:instagram.com/reel {niche} {month}"]


def _web_path(ctx: Ctx, start: date, end: date) -> list[dict]:
    sources: dict[int, tuple[str, str, str]] = {}
    for q in _queries(ctx, start, end):
        for res in ctx.deps.search.search(q, 10):
            blob = {"url": res.url, "title": res.title, "snippet": res.snippet, "age": res.age, "query": q}
            sha = ctx.save_json(blob, "search_result")
            sid = repo.upsert_web_source(ctx.conn, res.url, res.title, now_iso(), sha)
            sources.setdefault(sid, (res.title, res.url, res.snippet))
    if not sources:
        return []
    listing = "\n".join(f"[{sid}] {title} — {url} — {snip}" for sid, (title, url, snip) in sources.items())
    subject = ctx.request.trend_name or ctx.request.niche
    prompt = (f"Subject: {subject}\nPeriod: {start} to {end}\n\nSources:\n"
              + data_block("search results", listing))
    cands = ctx.llm(WebCandidates, schema_name="WebCandidates", system=prompts.SYSTEM_WEB, prompt=prompt).items

    found: list[tuple[WebCandidate, list[ReelRecord], dict[str, int]]] = []
    for c in cands[:MAX_CANDIDATES]:
        handles = [h.strip().lstrip("@").strip() for h in c.creator_handles]
        handles = [h for h in handles if h and not any(ch.isspace() for ch in h)][:MAX_HANDLES]
        urls = [u.strip() for u in c.example_urls if IG_URL.match(u.strip())][:MAX_URLS]
        recs: list[ReelRecord] = []
        ids: dict[str, int] = {}
        if urls:
            r, i = _apify(ctx, ACTOR_SCRAPER, urls_input(urls))
            recs += r
            ids.update(i)
        if handles:
            r, i = _apify(ctx, ACTOR_REELS, profiles_input(handles, (start - timedelta(days=WIDEN_DAYS)).isoformat(), 60))
            recs += r
            ids.update(i)
        found.append((c, _dedupe(recs), ids))

    def pick(s: date, e: date):
        return [[ids[r.external_id] for r in recs if _in_window(r, s, e)] for _, recs, ids in found]

    examples = pick(start, end)
    if found and not any(examples):
        start, end = start - timedelta(days=WIDEN_DAYS), end + timedelta(days=WIDEN_DAYS)
        ctx.flag("WINDOW_WIDENED", f"no examples in the window; widened to {start}..{end}")
        examples = pick(start, end)
    out = []
    for (c, _, _), ex in zip(found, examples):
        if ex:
            out.append({"name": c.name, "summary": c.summary, "examples": ex, "window": (start, end),
                        "sources": sorted({s for s in c.source_ids if s in sources})})
    return out


def run(ctx: Ctx) -> str:
    start, end = _window(ctx)
    req = ctx.request
    refs = queries.project_references(ctx.conn, ctx.project_id) if req.mode == "from_reference" else []
    if refs:
        path, found = "reference", _reference_path(ctx, refs, start, end)
    elif req.trend_name:
        path, found = "web", _web_path(ctx, start, end)
    elif req.mode == "discover" and end >= ctx.today - timedelta(days=NOW_DAYS):
        path, found = "now", _now_path(ctx, start, end)
    else:
        path, found = "web", _web_path(ctx, start, end)

    if not found:
        raise StageFailed("NO_CANDIDATES",
                          f"{path}: no candidates in {start}..{end}; try a wider window or a different niche")
    for t in found:
        repo.create_trend(ctx.conn, name=t["name"], summary=t["summary"], window_start=t["window"][0].isoformat(),
                          window_end=t["window"][1].isoformat(), step_id=ctx.step_id,
                          example_ids=t["examples"], source_ids=t["sources"])
    weak = [t for t in found if len(set(t["examples"])) < 2]
    if weak:
        ctx.flag("LOW_SAMPLE", "; ".join(f"{t['name']}: {len(set(t['examples']))} example(s)" for t in weak))
        return "weak"
    if any(f["code"] == "WINDOW_WIDENED" for f in queries.step_flags(ctx.conn, ctx.step_id)):
        return "weak"
    return "ok"
