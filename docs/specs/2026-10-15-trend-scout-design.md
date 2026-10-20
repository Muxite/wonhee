# wonhee — trend scout (sub-project 1) — design

Status: approved · 2026-10-15 · built. Amended 2026-10-20 to match later rulings (drift D5–D7 in [DESIGN.md](../DESIGN.md#drift)).
Where this spec and DESIGN.md disagree, DESIGN.md wins.

wonhee is the flagship for influencer-side automation: trend scouting, editing,
skit props, and an agent that ties them together. This spec covers only the
first sub-project, the **trend scout**, plus the two foundations every later
sub-project reuses: the **intent compiler** and the **relational store**.

Sub-project order: trend scout → editor → prop builder → director/inbox.

## 1. Goals

- **G1** Freeform phone message → a shot list you can film without further thought,
  from either a niche + time window (now / 6mo / 1y / 2y ago) or a specific reel/trend.
- **G2** Ask before guessing: clarification up front beats delivering the wrong thing.
- **G3** Every output is traceable to the step, run, and source content that produced it.
- **G4** Off-the-shelf components do the work; wonhee is glue, schemas, prompts, and SQL.
- **G5** The fixed pipeline (B) is built so an agent loop (C) can drive it later
  without changing stage code.
- **G6** Accumulate data: every observation of a content item is kept, never overwritten.

Non-goals here: editing footage, building props, posting, scheduling, a web UI.
Not a production system — correctness and clarity of structure beat speed.

## 2. Repository

The former `wonhee` (job-application form filler) moved out to its own repo,
`../awork-fill`, with its never-submit invariant unchanged. `wonhee` restarts as:

```
wonhee/
  (moeka)                pinned pip dependency (ruling 2026-10-15, project-scoped keys) — LLM access in-process
  creator.yaml           what you can film: on_camera, locations, props, niches, language
  wonhee/
    db/                  connection, migration runner, repository (the ONLY write path)
      migrations/        0001_init.sql, 0002_… — plain numbered SQL, applied in order (package data)
    media.py             content-addressed blob store
    intent/              intent schemas + clarify loop (reused by later tools)
    pipeline/            stage registry, fixed orchestrator, resume
    scout/stages/        one module per stage
    sources/             apify.py, websearch.py (direct Brave client), fetch.py (direct GET, yt-dlp fallback)
    analyze/             scenes.py (PySceneDetect), speech.py (faster-whisper),
                         frames.py (ffmpeg), phash.py (imagehash)
    llm.py               the ONLY module that calls a model
    cli.py               `wonhee scout …`, `wonhee resume …`, `wonhee db …`
  data/                  gitignored: wonhee.db, media/
```

## 3. Architecture

```
phone ─Telegram─▶ moeka agent ─▶ intent compiler ─(confirmed intent row)─▶ orchestrator
                       ▲              │ clarifying Qs                          │
                       └──────────────┘                     resolve → discover → pick → fetch →
                                                             analyze → describe → shotlist
                                                                               │
                                         SQLite (wonhee.db) + media/<sha> ◀────┘
                                                                               │
                         shot list ◀── moeka replies on Telegram ◀─────────────┘
```

moeka is a client of wonhee: its agent gets one tool per entry point
(`scout`, `resume`, `show`), not access to stages, APIs, or SQL.

## 4. Intent compiler

Required:

- **R1** Each tool defines an intent schema. `ScoutRequest`:
  `mode` (discover | from_reference), `niche`, `window` (absolute date range,
  "1y ago" → centre ±1 month), `reference_urls`, `trend_name`, `platform`
  (default instagram), `constraints` (on_camera, max_length_s, tone) defaulted
  from `creator.yaml`.
- **R2** Loop: model extracts fields → deterministic gap check → all gaps asked in
  ONE message as numbered multiple choice with a recommended default → repeat
  until no gaps.
- **R3** Always end with a one-line confirmation; nothing runs before "go". "go" accepts every proposed default;
  a gap with no sensible default (the niche) must be answered first (amended 2026-10-20).
  "go" accepts all proposed defaults.
- **R4** Gap rules are written per field (missing / ambiguous / contradicts
  `creator.yaml`). The compiler may only ask about fields that have a firing rule.
- **R5** The dialogue is stored as `intent_turn` rows; the confirmed result is the
  `intent` row every run points to.

Optimizations (later): learn habitual answers and stop asking; "just go" mode.

## 5. Stages

Uniform contract: `run(ctx) -> "ok" | "weak"`; a hard failure raises `StageFailed(code, detail)` (amended
2026-10-20; the same information as the original `StageResult`). A stage reads prior results from the
DB, writes its outputs to tables through the repository, records raw payloads
as `step_artifact` blobs, and returns `status` (ok | weak | failed) plus
`step_flag` codes.

| # | stage | does | tools | model |
|---|---|---|---|---|
| 01 | resolve | normalize intent; for URLs fetch metadata (audio, hashtags, caption) | Apify | no |
| 02 | discover | **now**: Apify hashtag/audio search in window. **throwback**: web search for trend recaps from that period → model extracts candidate trends → Apify confirms example reels exist, filtered by `posted_at` | Apify, moeka web search | yes |
| 03 | pick | rank candidates by popularity × filmability (`creator.yaml`), choose one, record reasons | — | yes |
| 04 | fetch | download 3–5 example reels + metadata; snapshot metrics | yt-dlp, Apify media URL fallback | no |
| 05 | analyze | per reel: duration, cuts/sec, time-to-first-word, transcript, keyframe per shot, keyframe pHash, on-screen text | PySceneDetect, faster-whisper, ffmpeg, imagehash, vision model | partly |
| 06 | describe | trend row + beats (hook→turn→payoff), audio, invariants vs variations | — | yes |
| 07 | shotlist | numbered shots adapted to `creator.yaml`, each with citations | — | yes |

Required:

- **R6** All model calls go through `llm.py`: JSON-schema output, validated,
  one retry, else stage `failed` with `SCHEMA_RETRY`/`SCHEMA_FAIL` flag. No free text
  crosses a stage boundary.
- **R7** Every shot has ≥1 `shot_citation`; uncited claims raise `UNCITED_CLAIM`.
- **R8** External text (search results, captions, transcripts) is passed to models
  as delimited data, never as instructions.
- **R9** Per-run cost cap (default $1: Apify results + model tokens). Hitting it
  stops the run with `COST_CAP`.
- **R10** Stages declare inputs/outputs; a run is an ordered list; the fixed
  orchestrator walks it.
- **R11** `wonhee resume <run> --from <stage> --set k=v` re-runs from any stage.
  New steps get `attempt+1` and `supersedes_step`; old rows stay.
- **R12** `weak` + flags (e.g. `LOW_SAMPLE`, `WINDOW_WIDENED`) are delivered to the
  user in B, and are the signal C acts on.
- **R13** All writes go through `wonhee/db/repository.py`. Stages and agents never
  execute raw SQL.
- **R14** Every content item seen in any API response gets a `metric_snapshot`,
  even if it is not selected (G6).

Path to C: a moeka agent gets `run_stage`, `read_step`, `resume_from` tools over the
same registry, recorded with `run.orchestrator = 'agent'`. No stage changes.

## 6. Storage

SQLite, `STRICT` tables, `PRAGMA foreign_keys=ON`, WAL. Schema kept
Postgres-portable (ISO-8601 TEXT timestamps, INTEGER ids, no SQLite-only types)
so it can move if multi-host moeka instances need to write. Normalized to BCNF;
many-to-many relations always get a join table; no polymorphic FKs (a link that
can target two tables uses two nullable FKs + `CHECK` exactly one).

### Media

Files are stored at `data/media/<sha[:2]>/<sha256>`. The path is derived, never
stored, so moving files can't leave dangling pointers; inserting the same bytes
twice is a no-op.

### Tables

Media
```
blob            sha256 PK, bytes, mime, stored_at
media           blob_sha256 PK→blob, kind(video|audio|image|transcript),
                duration_ms, width, height, fps
media_phash     blob_sha256→media, frame_ms, phash          PK(blob_sha256, frame_ms)
```

World
```
account         id PK, platform, external_id, handle,
                relation(ours|friend|external)               UNIQUE(platform, external_id)
content_item    id PK, platform, external_id, url, account_id→account, posted_at,
                caption, audio_id→audio_track NULL,
                media_sha256→media NULL                      UNIQUE(platform, external_id)
audio_track     id PK, platform, external_id, title, artist UNIQUE(platform, external_id)
hashtag         id PK, tag UNIQUE
content_hashtag content_id→content_item, hashtag_id→hashtag  PK(both)
metric_snapshot content_id→content_item, observed_at, plays, likes, comments, shares,
                step_id→step                                 PK(content_id, observed_at)
web_source      id PK, url, title, retrieved_at, snapshot_sha256→blob
```
Duration lives only on `media` (measured), not on `content_item`, to avoid two
sources of truth. Account handles can change; identity is `external_id`.

Trends
```
trend           id PK, name, summary, window_start, window_end, created_by_step→step
trend_beat      trend_id→trend, ord, role(hook|setup|turn|payoff|outro), description
                                                             PK(trend_id, ord)
trend_example   trend_id→trend, content_id→content_item      PK(both)
trend_source    trend_id→trend, web_source_id→web_source     PK(both)
```

Our work
```
intent          id PK, raw_text, schema_name, schema_version, compiled_json, confirmed_at
intent_turn     intent_id→intent, ord, role(user|compiler), text, gaps_json
                                                             PK(intent_id, ord)
project         id PK, title, kind(scout|skit|edit),
                status(open|filming|editing|posted|dropped), intent_id→intent, created_at
project_trend      project_id→project, trend_id→trend, role(primary|secondary)    PK(project_id, trend_id)
project_reference  project_id→project, content_id→content_item,
                   role(inspiration|counterexample|style_ref)                     PK(project_id, content_id)
project_output     project_id→project, content_id→content_item                    PK(both)
asset           id PK, project_id→project, media_sha256→media,
                role(raw_take|edit|prop_capture|thumbnail), created_by_step→step NULL
shotlist        id PK, project_id→project, version, created_by_step→step
                                                             UNIQUE(project_id, version)
shot            shotlist_id→shotlist, ord, duration_ms, framing, action,
                on_screen_text, audio_cue                    PK(shotlist_id, ord)
shot_citation   id PK, shotlist_id, shot_ord, content_id→content_item NULL,
                web_source_id→web_source NULL,
                FK(shotlist_id, shot_ord)→shot, CHECK(exactly one target)
```
`project_output` is many-to-many: one post can serve two projects, and a project
can produce several posts, with no duplicated rows.

Provenance
```
run             id PK, project_id→project, intent_id→intent, orchestrator(fixed|agent),
                status, cost_cap_usd, started_at, ended_at
step            id PK, run_id→run, stage, attempt, params_json,
                status(ok|weak|failed), supersedes_step→step NULL,
                started_at, ended_at, cost_usd
step_input      step_id→step, input_step_id→step             PK(both)
step_flag       step_id→step, code, detail                   PK(step_id, code)
flag_code       code PK, meaning                             -- seeded enum table
step_artifact   step_id→step, blob_sha256→blob,
                role(api_response|llm_prompt|llm_reply|frame) PK(step_id, blob_sha256, role)
llm_call        id PK, step_id→step, model, schema_name, tokens_in, tokens_out,
                cost_usd, valid
```
Run cost is derived (sum of `step.cost_usd`), not stored.

### Views

| view | answers |
|---|---|
| `v_content_latest` | each content item + its newest snapshot |
| `v_content_growth` | per-interval deltas between consecutive snapshots |
| `v_project_overview` | project → trends, reference count, outputs, summed latest metrics |
| `v_post_vs_trend` | our/friend post plays vs median of the trend's examples |
| `v_trend_popularity` | trend → example count, median plays, window |
| `v_shot_lineage` | shot → citations → content → producing step/run |
| `v_near_duplicates` | media pairs whose keyframe pHashes are within threshold |
| `v_run_cost` | cost per run and per stage, LLM validity rate |
| `v_current_steps` | latest non-superseded step per (run, stage) |

## 7. Errors and delivery

- Hard failure: run stops; Telegram names the stage, the flag, and a suggested fix
  ("widen window to ±2mo?"). The reply goes back through the intent compiler.
- Weak: delivered, with the flags as a one-line warning above the shot list.
- Success: shot list as Telegram text + run id for `resume`.

## 8. Testing

- Stages run against recorded Apify/search payloads and short fixture videos; no
  network in the default suite.
- LLM stubbed via env-var command (same pattern as slopcut's `SLOPCUT_LLM_CMD`).
- Schema tests: migrations apply on an empty DB; FK and CHECK violations rejected;
  every view runs.
- Intent compiler: table of freeform messages → expected gaps → expected compiled intent.
- One opt-in live smoke run (single niche, capped cost).

## 9. Optimizations (not in first build)

- Scheduled re-snapshots of `ours`/`friend` content (and tracked examples) to grow
  the metric time series — candidate for the eujin poller.
- Whole-video perceptual hashing beyond keyframes.
- Agent orchestrator (C).
- Intent compiler habit learning.

## 10. Defaults to confirm during planning

- Web search: a direct Brave client (`sources/websearch.py`), key from the project env (amended 2026-10-20).
- Apify actors: Instagram hashtag, reel, and audio scrapers — exact actors and
  their date-filter behaviour verified in the plan's first task.
- Reference reels per trend: 5. Cost cap per run: $1.

## 11. Adjustments during planning

Made in Task 1 so spec and code agree.

1. `run.intent_id` dropped: it is derivable via `project.intent_id` (BCNF).
2. New `media_analysis` and `media_shot` tables hold per-video analysis. Cut count, cuts/sec and similar values are derived in `v_media_style`, not stored.
3. New `trend_rule(trend_id, ord, kind invariant|variation, text)` holds the describe stage's invariants and variations.
4. `llm_call` gets `step_id NULL, intent_id NULL, CHECK exactly one`, because intent-compiler model calls happen before any run exists.
5. Migrations live in `wonhee/db/migrations/` (package data), not top-level `migrations/`.
6. `content_item(account_id, platform)` has a composite FK to `account(id, platform)`, so the duplicated platform column can't disagree.
7. `v_near_duplicates` uses a `hamming()` SQL function registered by `wonhee.db.connect`. Query it with `wonhee db sql`, not the bare `sqlite3` CLI.
8. `project_trend` is the *current* choice: re-running pick replaces it. History stays in step artifacts.
9. `step_artifact.role` adds `error` and `search_result`. `step.status` adds `running`. `run.status` values are `running|done|weak|failed|stopped`.
10. Flag codes are seeded in `flag_code`: LOW_SAMPLE, WINDOW_WIDENED, SCHEMA_RETRY, SCHEMA_FAIL, UNCITED_CLAIM, COST_CAP, NO_CANDIDATES, DOWNLOAD_FAILED, LENGTH_OVER, EXCEPTION, MISSING_KEY.
