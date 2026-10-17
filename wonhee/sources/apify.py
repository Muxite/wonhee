from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Protocol

ACTOR_HASHTAG = "apify/instagram-hashtag-scraper"
ACTOR_SCRAPER = "apify/instagram-scraper"
ACTOR_REELS = "apify/instagram-reel-scraper"


WAIT_MINUTES = 10


class ApifyError(RuntimeError):
    """An actor run that did not succeed. cost_usd is what it still cost, so callers can record it."""

    def __init__(self, message: str, cost_usd: float = 0.0):
        super().__init__(message)
        self.cost_usd = cost_usd


class ApifyRunner(Protocol):
    def run(self, actor: str, run_input: dict, *, max_total_charge_usd: float | None = None,
            max_items: int | None = None) -> tuple[list[dict], float]: ...


class LiveApify:
    def __init__(self, token: str, price_per_1k: float, client=None):
        if client is None:
            from apify_client import ApifyClient

            client = ApifyClient(token)
        self.client, self.price_per_1k = client, price_per_1k

    def run(self, actor: str, run_input: dict, *, max_total_charge_usd: float | None = None,
            max_items: int | None = None) -> tuple[list[dict], float]:
        kw: dict = {"wait_duration": timedelta(minutes=WAIT_MINUTES), "logger": None}  # actor log stays on Apify
        if max_total_charge_usd is not None:
            kw["max_total_charge_usd"] = Decimal(str(round(max(max_total_charge_usd, 0.01), 4)))
        if max_items is not None:
            kw["max_items"] = max_items
        run = self.client.actor(actor).call(run_input=run_input, **kw)
        if run is None:
            raise ApifyError(f"apify actor {actor} returned no run")
        status = str(getattr(run.status, "value", run.status))
        if status != "SUCCEEDED":
            partial = float(getattr(run, "usage_total_usd", None) or 0.0)
            if status in ("READY", "RUNNING"):  # our wait ran out: stop paying for it
                try:
                    self.client.run(run.id).abort()
                except Exception:  # noqa: BLE001 - best effort
                    pass
                raise ApifyError(f"apify actor {actor} still {status} after {WAIT_MINUTES} min; "
                                 f"aborted (run {run.id})", partial)
            raise ApifyError(f"apify actor {actor} finished {status} (run {run.id})", partial)
        items = list(self.client.dataset(run.default_dataset_id).iterate_items())
        cost = None
        for _ in range(3):  # usage totals settle a few seconds after the run
            fresh = self.client.run(run.id).get()
            cost = getattr(fresh, "usage_total_usd", None)
            if cost:
                break
            time.sleep(4)
        if not cost:
            cost = len(items) / 1000 * self.price_per_1k
        return items, float(cost)


@dataclass
class ReelRecord:
    external_id: str
    url: str
    owner_id: str
    owner_handle: str
    posted_at: str | None
    caption: str | None
    hashtags: list[str] = field(default_factory=list)
    audio_id: str | None = None
    audio_title: str | None = None
    audio_artist: str | None = None
    plays: int | None = None
    likes: int | None = None
    comments: int | None = None
    shares: int | None = None
    video_url: str | None = None
    post_type: str = "Video"  # Apify 'type': Video | Image | Sidecar
    children: list[tuple[str, str]] = field(default_factory=list)  # (kind 'image'|'video', url) for photo posts


def _children(item: dict) -> list[tuple[str, str]]:
    """The files of a photo post: one image, or each carousel slide (a slide can be a video)."""
    kids = item.get("childPosts") or []
    if kids:
        out = []
        for k in kids:
            if k.get("type") == "Video" and k.get("videoUrl"):
                out.append(("video", k["videoUrl"]))
            elif k.get("displayUrl"):
                out.append(("image", k["displayUrl"]))
        return out
    if item.get("images"):
        return [("image", u) for u in item["images"] if u]
    return [("image", item["displayUrl"])] if item.get("displayUrl") else []


def _count(v):
    return None if v is None or (isinstance(v, (int, float)) and v < 0) else int(v)


def _iso(ts: str | None) -> str | None:
    if not ts:
        return None
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).isoformat(timespec="seconds")


def parse_item(item: dict) -> ReelRecord | None:
    code = item.get("shortCode")
    if not code or item.get("error"):
        return None
    music = item.get("musicInfo") or {}
    plays = next((c for c in (_count(item.get(k)) for k in ("videoPlayCount", "igPlayCount", "videoViewCount"))
                  if c is not None), None)
    post_type = item.get("type") or ("Video" if item.get("videoUrl") else "Image")
    photo = post_type in ("Image", "Sidecar")
    return ReelRecord(
        external_id=code,
        url=item.get("url") or f"https://www.instagram.com/{'p' if photo else 'reel'}/{code}/",
        owner_id=str(item.get("ownerId") or item.get("ownerUsername") or "unknown"),
        owner_handle=item.get("ownerUsername") or "unknown",
        posted_at=_iso(item.get("timestamp")),
        caption=item.get("caption"),
        hashtags=list(item.get("hashtags") or []),
        audio_id=str(music["audio_id"]) if str(music.get("audio_id") or "0") != "0" else None,  # "0" = muted/unavailable
        audio_title=music.get("song_name"),
        audio_artist=music.get("artist_name"),
        plays=plays, likes=_count(item.get("likesCount")),
        comments=_count(item.get("commentsCount")), shares=_count(item.get("sharesCount")),
        video_url=item.get("videoUrl") if not photo else None,
        post_type=post_type,
        children=_children(item) if photo else [],
    )


def hashtag_input(tags: list[str], limit: int) -> dict:
    return {"hashtags": [t.lstrip("#") for t in tags], "resultsType": "reels", "resultsLimit": limit}


def urls_input(urls: list[str]) -> dict:
    return {"directUrls": urls, "resultsType": "posts", "resultsLimit": 1, "addParentData": False}


def profiles_input(handles: list[str], newer_than: str, limit: int) -> dict:
    return {"username": [h.lstrip("@") for h in handles], "resultsLimit": limit,
            "onlyPostsNewerThan": newer_than, "skipPinnedPosts": True}
