"""The grab node: pull one reel by URL into the library and hand on its content ref and video."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from wonhee.intent.gaps import IG_URL
from wonhee.library.grab import grab_urls
from wonhee.nodes import registry
from wonhee.nodes.base import Node
from wonhee.pipeline.context import StageFailed


class GrabParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str
    relation: str = "external"
    refetch: bool = False


def grab_fn(ctx, inputs, p: GrabParams) -> dict[str, str]:
    url = p.url.strip().split("?")[0]
    if not IG_URL.match(url):
        raise StageFailed("NO_CANDIDATES", f"not an Instagram reel/post URL: {p.url[:200]}")
    ids = grab_urls(ctx, [url], p.relation, p.refetch)
    cid = next(iter(ids.values()))
    sha = ctx.conn.execute("SELECT media_sha256 FROM content_item WHERE id=?", (cid,)).fetchone()[0]
    if not sha:
        raise StageFailed("DOWNLOAD_FAILED", f"no video stored for {url}")
    return {"content": f"content:{cid}", "video": sha}


registry.register(Node("grab", 1, {}, {"content": "content", "video": "video"}, GrabParams, grab_fn,
                       "Pull a reel by URL into the library (metadata, metrics, video).", cacheable=False))
