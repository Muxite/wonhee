"""Grab reels by URL into the library: metadata, a metric snapshot, the video file, and its analysis.

Used for our own posts (relation 'ours'), friends' posts and anyone else's. The URLs travel in the
intent's ScoutRequest.reference_urls; the relation travels as the run parameter 'relation'."""
from __future__ import annotations

import tempfile
from pathlib import Path

from pydantic import BaseModel

from wonhee import llm
from wonhee.clock import now_iso
from wonhee.db import queries, repo
from wonhee.db.connect import tx
from wonhee.media import path_for
from wonhee.pipeline.context import Ctx, StageFailed
from wonhee.scout import prompts
from wonhee.scout.stages.analyze import ANALYZER_VERSION, analyze_video
from wonhee.sources.apify import ACTOR_SCRAPER, parse_item, urls_input
from wonhee.sources.fetch import DownloadError
from wonhee.sources.ingest import ingest_records


def project_content_ids(conn, project_id: int) -> list[int]:
    """Content a library project holds: our posts (project_output) and others' (project_reference)."""
    return [r[0] for r in conn.execute(
        """SELECT content_id FROM project_output WHERE project_id=:p
           UNION SELECT content_id FROM project_reference WHERE project_id=:p ORDER BY 1""", {"p": project_id})]


def grab(ctx: Ctx) -> str:
    urls = list(ctx.request.reference_urls)
    ids = grab_urls(ctx, urls, ctx.param("relation", "external"), ctx.param("refetch", False))
    return "weak" if ids.failed else "ok"


class GrabResult(dict):
    """external_id -> content_id, plus how many downloads failed in this call."""
    failed: int = 0


def grab_urls(ctx: Ctx, urls: list[str], relation: str, refetch: bool) -> GrabResult:
    """Fetch, ingest and download the given reels under ctx.project_id; returns external_id -> content_id."""
    if not urls:
        raise StageFailed("NO_CANDIDATES", "no URLs to grab")
    run_input = urls_input(urls)
    items = ctx.apify(ACTOR_SCRAPER, run_input)
    ctx.save_json({"actor": ACTOR_SCRAPER, "input": run_input, "items": items}, "api_response")
    recs = [r for r in map(parse_item, items) if r]
    if not recs:
        raise StageFailed("NO_CANDIDATES", f"reel(s) not found: {', '.join(urls)}")
    ids = ingest_records(ctx.conn, ctx.step_id, recs, now_iso())
    with tx(ctx.conn):
        for r in recs:
            if relation != "external":  # never downgrade an account already marked ours/friend
                repo.upsert_account(ctx.conn, "instagram", r.owner_id, r.owner_handle, relation)
            cid = ids[r.external_id]
            if relation == "ours":
                repo.add_output(ctx.conn, ctx.project_id, cid)
            else:
                repo.add_reference(ctx.conn, ctx.project_id, cid, "inspiration")

    # Download now: Apify's media URLs are signed links that expire.
    got = 0
    for r in recs:
        cid = ids[r.external_id]
        if r.children:
            got += _download_photo_post(ctx, cid, r, refetch)
            continue
        row = ctx.conn.execute("SELECT media_sha256 FROM content_item WHERE id=?", (cid,)).fetchone()
        if row["media_sha256"] and not refetch:
            got += 1
            continue
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / f"{r.external_id}.mp4"
            try:
                ctx.deps.downloader.download(r.video_url, r.url, dest)
                sha = repo.save_file(ctx.conn, ctx.settings.media_dir, dest, "video", "video/mp4")
            except (DownloadError, RuntimeError) as e:
                ctx.flag("DOWNLOAD_FAILED", f"{r.url}: {e}"[:300])
                continue
        repo.set_content_media(ctx.conn, cid, sha)
        repo.set_content_files(ctx.conn, cid, [(sha, "video")])
        got += 1
    if got == 0:
        raise StageFailed("DOWNLOAD_FAILED", "nothing could be downloaded (see DOWNLOAD_FAILED flags)")
    result = GrabResult(ids)
    result.failed = len(recs) - got
    return result


def _download_photo_post(ctx: Ctx, cid: int, r, refetch: bool) -> int:
    """Store every image (and video slide) of a photo/carousel post in order; 1 if at least one file landed."""
    have = ctx.conn.execute("SELECT count(*) FROM content_media WHERE content_id=?", (cid,)).fetchone()[0]
    if have and not refetch:
        return 1
    files: list[tuple[str, str]] = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (kind, url) in enumerate(r.children):
            dest = Path(tmp) / f"{r.external_id}_{i:02d}.{'mp4' if kind == 'video' else 'jpg'}"
            try:
                if kind == "video":
                    ctx.deps.downloader.download(url, r.url, dest)
                else:
                    ctx.deps.downloader.fetch(url, dest)
                sha = repo.save_file(ctx.conn, ctx.settings.media_dir, dest, kind,
                                     "video/mp4" if kind == "video" else "image/jpeg")
            except (DownloadError, RuntimeError) as e:
                ctx.flag("IMAGE_FAILED", f"{r.url} slide {i + 1}: {e}"[:300])
                continue
            files.append((sha, kind))
    if not files:
        ctx.flag("DOWNLOAD_FAILED", f"{r.url}: no slide could be downloaded")
        return 0
    repo.set_content_files(ctx.conn, cid, files)
    return 1


IMAGE_ANALYZER_VERSION = 1
MAX_POST_IMAGES = 10


class ImageText(BaseModel):
    n: int
    on_screen_text: str | None = None
    visual: str


class ImageTexts(BaseModel):
    images: list[ImageText]


THUMB_PX = 768  # long side sent to the vision model; full-size originals stay in the media store


def _thumb(src: Path, dest: Path) -> Path:
    from PIL import Image, ImageOps
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((THUMB_PX, THUMB_PX))
        im.save(dest, quality=82)
    return dest


def analyze_images(ctx: Ctx, content_id: int) -> int:
    """One vision call for a photo/carousel post's not-yet-analysed images; returns how many were analysed."""
    shas = [r[0] for r in ctx.conn.execute(
        """SELECT m.media_sha256 FROM content_media m WHERE m.content_id=? AND m.kind='image'
           AND NOT EXISTS (SELECT 1 FROM image_analysis a WHERE a.media_sha256=m.media_sha256
                           AND a.analyzer_version=?) ORDER BY m.ord""", (content_id, IMAGE_ANALYZER_VERSION))]
    shas = list(dict.fromkeys(shas))[:MAX_POST_IMAGES]
    if not shas:
        return 0
    with tempfile.TemporaryDirectory(prefix="wonhee-thumbs-") as tmp:
        thumbs = [_thumb(path_for(ctx.settings.media_dir, s), Path(tmp) / f"{i}.jpg") for i, s in enumerate(shas)]
        reply = ctx.llm(ImageTexts, schema_name="ImageTexts", system=prompts.SYSTEM_IMAGE_TEXT,
                        prompt=f"There are {len(shas)} images, numbered 1 to {len(shas)} in the order attached.",
                        images=thumbs)
    by_n = {t.n: t for t in reply.images if 1 <= t.n <= len(shas)}
    repo.save_image_analyses(
        ctx.conn, analyzer_version=IMAGE_ANALYZER_VERSION, step_id=ctx.step_id,
        rows=[(sha, by_n[n].on_screen_text if n in by_n else None, by_n[n].visual if n in by_n else None)
              for n, sha in enumerate(shas, 1)])
    return len(shas)


def analyze(ctx: Ctx) -> str:
    cids = project_content_ids(ctx.conn, ctx.project_id)
    shas = [r[0] for r in ctx.conn.execute(
        f"SELECT DISTINCT media_sha256 FROM content_item WHERE media_sha256 IS NOT NULL AND id IN "
        f"({','.join('?' * len(cids))})", cids)] if cids else []
    todo = [s for s in shas if not queries.has_analysis(ctx.conn, s, ANALYZER_VERSION)]
    if todo:
        ensure = getattr(ctx.deps.transcriber, "ensure_ready", None)
        if ensure:
            ensure()  # a broken GPU setup fails once, up front
    failed = 0
    for sha in todo:
        try:
            analyze_video(ctx, sha)
        except llm.LLMError:
            raise
        except Exception as e:  # noqa: BLE001 - one bad video must not sink the others
            if isinstance(e, RuntimeError) and "CUDA" in str(e):
                raise
            ctx.flag("ANALYZE_FAILED", f"{sha[:12]}: {e}"[:300])
            failed += 1
    photo_posts = [r[0] for r in ctx.conn.execute(
        f"SELECT DISTINCT content_id FROM content_media WHERE kind='image' AND content_id IN "
        f"({','.join('?' * len(cids))})", cids)] if cids else []
    for cid in photo_posts:
        try:
            analyze_images(ctx, cid)
        except llm.LLMError:
            raise
        except Exception as e:  # noqa: BLE001 - one bad post must not sink the others
            ctx.flag("ANALYZE_FAILED", f"content {cid}: {e}"[:300])
            failed += 1
    if (shas or photo_posts) and failed == len(shas) + len(photo_posts):
        raise StageFailed("ANALYZE_FAILED", "every post failed analysis (see flags)")
    return "weak" if failed else "ok"
