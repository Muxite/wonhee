# Post-game analysis (PGA)

Owner rule 2026-11-09: **every thing we make gets a post-game analysis, written within 1 week of its result.** A
"thing" is a published reel or post, and also a project that ends in a deliverable. Result means the numbers have
mostly settled (growth flat), or day 7, whichever comes first. Later than 7 days is overdue (backfill it anyway, and
say it is late). Each PGA lives in this folder as `YYYY-MM-DD-<slug>.md`, from [TEMPLATE.md](TEMPLATE.md).

## Why

Numbers without the reasoning behind the post teach nothing, and the reasoning without the numbers is a guess. Two
things go in every PGA: what the post did, and what we believed when we made it. Lessons that apply to future work
go to the memory/playbook the same day, and to [TOOLS.md](../TOOLS.md) when a tool was found or built.

## Getting the numbers

1. `uv run wonhee grab --mine "<clean url>"` adds a metric snapshot (paid Apify call, so one per sitting). It
   gives plays, likes, comments (shares are often empty from the scraper).
2. Owner-reported Instagram insights (views, viewers, watch time, follows, shares, saves, reposts, the views-over-time
   curve) are the better source. Paste them in; record what time after posting they were taken.
3. Our posts and their snapshot history:

```bash
uv run wonhee db sql "SELECT v.content_id id, v.url, v.posted_at, v.plays, v.likes, v.comments, v.observed_at
  FROM v_content_latest v WHERE v.relation='ours' ORDER BY v.posted_at DESC"
uv run wonhee db sql "SELECT * FROM v_content_growth"      # how each post moved between snapshots
uv run wonhee inspire <content_id>                         # the stored analysis (speech, shots, text, look)
```

## Registry

The registry of posts (what shipped, its numbers, PGA status) and the per-post analyses are kept private, in
`REGISTRY.md` and the dated files in this folder (gitignored). Add a row to `REGISTRY.md` when a post ships.
