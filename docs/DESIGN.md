# wonhee: system design

Status: approved 2026-10-20. Living document: the current truth about what wonhee is built to do. The
[vision](VISION.md) says why. [PROCESS.md](PROCESS.md) says how this page changes and how to decide, when code and
design disagree, which one is wrong.

Each area below names its stage, its contract (what the code must do), where the details live, and a status
(`proposed` / `approved` / `built` / `superseded`). An area spec linked here is part of the design. Where an area
spec and this page disagree, this page wins and the spec gets fixed.

## Cross-cutting contracts (all areas)

These hold everywhere; breaking one is a defect.

1. **Everything is a node.** A capability is a `Node(name, version, inputs, outputs, params, fn, doc, cacheable,
   salt, lane)` in `wonhee/nodes/`, registered in `registry.MODULES`. Params are pydantic with `extra="forbid"`.
   Values move between nodes as refs (`content:<id>`, a media sha, a blob sha), never as bytes.
2. **Port types:** `content`, `video`, `image`, `images` (an ordered manifest), `subtitles`, `text`, `file`.
   A new type is an M-size design change.
3. **The cache key** is sha256 of `{node, version, inputs, params, salt}`, where `salt` holds model names. Any change
   to a node's output bumps its `version`. A cache hit copies refs, costs $0 and flags `CACHED`. A hit whose files
   are gone is a miss. Weak steps are never cached.
4. **One surface, two doors.** Each user-facing action in stages 1–6 is one `service` function, called by both a
   CLI command and an MCP tool. Operator commands (`worker serve`, `profile`, `db sql`, `account mark`,
   `chain exec`, `doctor`) are CLI-only by design.
5. **Provenance.** Work runs as project → run → step, with flags (`step_flag`), artifacts (raw API replies, prompts,
   errors) and outputs (`step_output`). Free-text notes never stand in for a table.
6. **Store.** SQLite STRICT, foreign keys on, WAL. Migrations are numbered plain SQL and are never edited once
   applied. Media is content-addressed under `data/media/<sha[:2]>/<sha>`. Writes go through `wonhee/db/repo/`.
   Reads may use `db/queries.py` or SQL in `service`/`render`. Two narrow exceptions: `worker.py` owns the `job` and
   `worker` tables, and `nodes/chain.py` and `nodes/runner.py` own `node_cache` and `step_output`.
7. **Model calls** go through `llm.py` (remote, via the pinned moeka kernel) or `localllm.py` (Ollama on the box).
   Every call has a JSON schema, is validated, and gets one retry with the validation error as feedback.
   External text is passed as delimited data, never as instructions.
8. **Money.** Each run has `cost_cap_usd` (default $1). Paid calls, Apify and remote LLM alike, are checked against
   the remaining budget **before** they're made, and are refused with `COST_CAP` when it's spent. Intent-compiler
   turns happen before a run exists, so they sit outside any cap; they are logged in `llm_call` against the intent.
9. **Gates.** Publishing, deleting media and spending over the cap need a person (`--yes`, or a `resume --set`).
10. **Keys** come from the process env or the repo `.env` (or `--env-file`). Subprocesses get `procenv.clean_env()`.
    No user-level config is read.
11. **The GPU is shared.** GPU nodes declare `lane="cuda"`. People and agents hold `gpu-lock` for GPU work outside
    a worker.
12. **Unknown is `None`**, never 0, in reports and measurements.
13. **Pixels are real.** No image generation anywhere. Edits move the pixels you shot.

## Areas

### Stage 0: platform

| Area | Contract | Detail | Status |
|---|---|---|---|
| Nodes and chains | Chains are JSON (a list, or `{nodes, cost_cap_usd, no_cache}`). Ids match `^[a-z][a-z0-9_]{0,31}$`. Validation happens before anything is recorded and reports every problem in one error (exit 2). Nodes run in topological order with ties in file order. A failed node stops its downstream. Detached runs hand off to a live worker of the right lane, else spawn `wonhee chain exec` | [toolkit R0](specs/2026-10-15-toolkit-design.md) (as amended by contracts 1–3 above) | built |
| Agent discovery | MCP `nodes_list()` returns name, ports and doc for every node. `nodes_list(name)` also returns that node's param JSON schema, because 40 schemas in one reply wastes context | this page | built |
| Warm workers | One process per lane. It claims jobs atomically from `job` and heartbeats in `worker`. A dead worker's job is requeued, up to 3 attempts | [workers.md › Worker contract](production/workers.md#worker-contract) | built |
| Production rules | Touch pixels once, lanes and limits, batching, failure, human gates | [policies.md](production/policies.md) | approved (rules 1, 5 and §2 wait on `render`) |
| Profiling (gauge) | Measures any job in a container. Never imported by normal wonhee use | [profiling/GUIDE.md](profiling/GUIDE.md) | built |

### Stage 1: discover

| Area | Contract | Detail | Status |
|---|---|---|---|
| Intent compiler | Freeform text → `ScoutRequest`. Gaps come from deterministic rules, and all questions go in one numbered message. Nothing runs before "go". "go" accepts the proposed defaults; a gap with no sensible default (niche) must be answered | [scout spec §4](specs/2026-10-15-trend-scout-design.md) | built |
| Trend scout | resolve → discover → pick → fetch → analyze → describe → shotlist. Pick records the decision and the model's reason as a step artifact. Every shot cites ≥1 source, or it is flagged `UNCITED_CLAIM` and the run is weak | [scout spec §5](specs/2026-10-15-trend-scout-design.md) | built (see drift) |
| Library grab | `grab URL` stores a post (reel, photo, carousel) with metrics, media and analysis. Every item seen gets a metric snapshot | toolkit §1, scout R14 | built |
| Teardown and inspire | Creator-blind teardown of each stored post (closed vocabularies), cached per post. `inspire` aggregates them into a report: what to build, what to shoot, a line bank, looks | [workers.md](production/workers.md) row `teardown` | built |
| Shot search | `shots index` cuts footage into shots and embeds the keyframes (SigLIP2). `shots search` finds shots by description | workers.md row 6 | built |
| GitHub slop machine | Endless feed of public repos, each turned into a credited 4:5 card set (`gh_pick` → `slop_script` → `slop_cards`); voice lines from an AI, facts from code; local infinite-scroll page; an agent skill writes scripts | [github slop spec](specs/2026-10-20-github-slop-design.md) | approved |

### Stage 2: plan

| Area | Contract | Detail | Status |
|---|---|---|---|
| Ideate | Ideas come from two separate sources: inspiration (`c<n>`) and the brief (`b-xxxxx`). Each idea's origin is computed from its citations, never chosen by the model. Unknown refs are dropped (`UNKNOWN_REF`) and copied overlays are dropped (`IDEA_COPY`) | workers.md row `ideate` | built |
| Larp | A mundane task → professional-moment framings (line, caption, hashtags, look, shot), optionally rendered | workers.md row `larp` | built |
| Shot guide | How to shoot a look, grounded on measured numbers | [photo-looks.md](production/photo-looks.md) | built |
| Style memory | Voice and rhythm styles from chosen posts, with a no-copy guard | [toolkit R6](specs/2026-10-15-toolkit-design.md) | approved, not built |

### Stage 3: capture

| Area | Contract | Detail | Status |
|---|---|---|---|
| Phone inbox | Upload raw takes from the phone over the tailnet only | [toolkit R4](specs/2026-10-15-toolkit-design.md) | approved, not built |

### Stage 4: make

| Area | Contract | Detail | Status |
|---|---|---|---|
| Video timeline + `render` | Edit ops are timeline transforms (`timeline` → `timeline`). Only `render` encodes, once per output, plus one preview. Grade, text, trim, speed, reframe and audio mix compile into one ffmpeg graph | [policies.md §2](production/policies.md), [workers.md › Build next](production/workers.md) 1–4 | approved, not built |
| Per-op encodes (old R1) | Each op re-encodes | toolkit R1 | **superseded** by the timeline design: measured 3–11× slower, 2026-10-17 |
| Translation | grab → speech_subs → screen_text → merge_subs → translate → to_srt / burn_subtitles. Empty output flags `NO_TEXT`. Translation gaps flag `TRANSLATION_GAP` and keep the source line | [toolkit R7](specs/2026-10-15-toolkit-design.md) | built |
| Looks and text (video and photo) | Named looks are one filter string each. Text layers in outline / band / script presets inside IG safe zones, with colour emoji via Pillow (never libass emoji). **Safe zones** (measured 2026-10-20 on a 1080×2181 phone screenshot; a 9:16 video may sit low on a 20:9 screen, so the worst case is used): text stays below 12% and above 76% of the frame height. On tall frames (h/w ≥ 1.7), a block that reaches the button strip (35–88% of the height) wraps within the middle 72% of the width, because the like/comment/share buttons cover the right 13%. A top title above the strip may use the full wrap width | workers.md rows `look`, `text_*`, `carousel` | built |
| Photo: filters | A parametric filter applies correct-first (`auto`), point ops with 8 colour ranges, effects, then skin undertone. `filter make` drafts, renders, measures, guards, reviews and saves. Reviewer calibration hides knobs the model can't see, unless the model is guessing. Usage logs set the default intensity | [photo-looks.md › Filters](production/photo-looks.md#filters-a-vibe-of-your-own-no-reference-needed) | built |
| Photo: vibe | Look profile from references → per-photo grade along the ladder global → gentler → range → soft mask → mask. The first step that keeps skin readable wins. If none does, the least-bad step wins and is flagged unreadable | photo-looks.md, workers.md row `vibe` | built |
| Photo: formats | `poster`, `carousel`, `mask_cutout` | workers.md | built |
| Meme frame, sticker, pop-art formats | `meme_frame`, `word_captions`, sticker text preset, `sticker`, `pattern_bg`, `shape_frame`, `grid_plan` (from the 2026-10-20 references) | [formats spec](specs/2026-10-20-meme-and-popart-formats-design.md) | **proposed** |
| Audio | `audio_check` measures loudness, peak, range and speech-to-noise. `audio_clean` brings a clip to -14 LUFS and -1.5 dBTP, denoises, and can optionally isolate the voice | [photo-looks.md › Audio defaults](production/photo-looks.md#audio-defaults-why) | built |
| Data stills (from vizkit) | See [Data stills](#data-stills-viz) below | this page | built |

### Stage 5: check

| Area | Contract | Detail | Status |
|---|---|---|---|
| File check | `check` on the rendered file: decode, streams, loudness, black runs, A/V drift, 9:16, resolution, codecs, length, VFR, text in safe zones. Stored per (sha, checker version) | [toolkit R3](specs/2026-10-15-toolkit-design.md) | approved, not built |
| Look readability | The vibe and filter checks above | photo | built |

### Stage 6: deliver

| Area | Contract | Detail | Status |
|---|---|---|---|
| Export | `export <ref> <dir>` copies outputs out of the store. `dest` must be a folder | README | built |
| Phone outbox | Download finished edits on the phone, with check badges | toolkit R4 | approved, not built |
| Publish to muksite | Uploads a light/dark pair to muksite's Supabase Storage and prints the YAML entry. It refuses to overwrite without `--overwrite` and needs `--yes` (public) | [Data stills](#data-stills-viz) | built |
| Social posting | Never | [VISION › Never](VISION.md#never) | n/a |

## Data stills (viz)

Absorbed from vizkit on 2026-10-19. Pictures made from data are content like any other: they chain into
`look_image`, `filter_image`, `poster` and `mask_cutout`, and end at `publish_muksite`.

**Intent:** presentation first. A striking picture of a real structure; it's "truthful-ish", not an audit. Within
that:
- **The picture must be traceable.** Every render flags `VIZ_SOURCE` with the path, the resolved commit and counts.
- **A render reads only what it says.** A git working tree means tracked files only (`tracked=false` reads all of
  disk). `--rev` reads a commit through `git archive` and never touches the working tree.
- **Nothing private leaks by default.** `--exclude` takes gitignore patterns; the repo's `.vizkitignore` (the legacy
  name, kept so existing repos work) adds more. `labels=false` hides file names.

| Node | In → out | Contract |
|---|---|---|
| `viz_depgraph` | path → image | file import graph (Py, JS/TS, C/C++/CUDA, Rust). Imports resolve from each package's import root. Node size = lines of code, colour = language (`hash_hue`, matching the JS template). Deterministic layout (seeded). `--focus GLOB --hops N` slices. `VIZ_NO_EDGES` if nothing resolves |
| `viz_diskviz` | path → image | treemap (`rect`) or circle-pack by real file size |
| `viz_history` | repo → image | commit heatmap or lines-of-code-per-language bar |
| `viz_wavegen` | mood knobs → image | WaveSim shader still in docker on the GPU (`lane="cuda"`), up to 3 tries, judged by wonhee's vision path |
| `publish_muksite` | image (+ dark) → entry | see Stage 6 |

Themes: `paper` (default), `dark`, `both`. Size `WxH` (default 1200×600). PNG export via Playwright Chromium.
`export_unavailable` is the clear error when the browser is missing.

**Dropped from vizkit on purpose:** the separate `panel.png` legend image, the raw `graph.json`/`tree.json` and
standalone interactive HTML outputs, and `--skip-export`. A still is the product. Bring one back as an M-size change
if a use appears.
**WaveSim source:** `github.com/Muxite/WaveSim`, pinned by `ARG WAVESIM_REV` in `wonhee/edit/viz/vendor/wavesim.Dockerfile`
and patched at build time. Bump the pin on purpose, never float it.

## Drift

Every known mismatch between this design and the code, with the verdict from [PROCESS.md §2](PROCESS.md). Audited
2026-10-20 (four read-only agents: toolkit spec, scout spec, production docs, vizkit parity).

| # | Where | Mismatch | Verdict | Action |
|---|---|---|---|---|
| D1 | scout pick | Pick discards its reasons (`PickChoice.reason`) | code wrong (scout §5 row 03, G3) | **fixed**: decision + reason saved as a step artifact (`pick.py`). My first DESIGN row also demanded "why it passed", which the spec never said: design wrong (overreach), row corrected |
| D2 | cost cap | LLM calls aren't checked against the remaining budget before they're made, so one stage can overshoot | code wrong (contract 8, scout R9) | **fixed**: `Ctx._require_budget()` before Apify and LLM calls |
| D3 | `library/grab.py` | A raw `INSERT INTO image_analysis` bypasses the repo | code wrong (contract 6) | **fixed**: `repo.save_image_analyses` |
| D4 | MCP `nodes_list` | No param schemas (toolkit R0 wanted them) | design wrong (detail: 40 schemas per call is too much context) | **fixed**: `nodes_list(name)` |
| D5 | scout spec §2/§10 | Says moeka is a submodule and search goes via moeka | design wrong (ruling 2026-10-15: pinned pip dep; direct Brave client) | spec amended |
| D6 | scout R3 | "go accepts all defaults", but niche has none | design wrong (underspecified) | spec amended: a gap with no default must be answered |
| D7 | scout §5 contract | `StageResult` type vs `"ok"`/`"weak"` + `StageFailed` | design wrong (detail; same information) | spec amended |
| D8 | toolkit R0 cache | Spec: note `cached`. Code: `CACHED` flag, salt in the key, missing files = miss, weak never cached | design wrong (code is more explainable) | contract 3 above |
| D9 | toolkit R7 `speech_subs` | Spec says flag-free when there's no audio; code flags `NO_TEXT` | design wrong (explainability) | toolkit R7 amended |
| D10 | toolkit R1 | Per-op encodes | design superseded by the timeline (measured) | marked superseded |
| D11 | toolkit R2–R6 | Not built | n/a: approved and not built | roadmap ([VISION › Where it's going](VISION.md#where-its-going-ordered)) |
| D12 | about 27 nodes, worker, viz, publish | Code with no design | unowned → owned | designed on this page |
| D13 | scout §7 | A hard-failure reply goes back through the intent compiler: not built (needs the moeka/Telegram side) | approved, not built | later, with the director |
| D14 | viz | WaveSim source not pinned | false finding: the Dockerfile already clones `Muxite/WaveSim` at `1d78d63` (vizkit's submodule pin, still upstream `master` on 2026-10-20). The audit missed the `ARG` | closed |
| D15 | MCP tools | Fewer capabilities than their CLI twins: `audio` (isolate, music_db), `vibe` (set, guide_engine), `larp` (register), `node_run`/`chain_run` (no_cache). No MCP `filter show`/`filter calibrate` | code wrong (contract 4) | **fixed**: args added; new `filter_show` (cube written under `<home>/filters/cube/`) and `filter_calibrate` |
| D16 | MCP `publish_muksite` | Publishes with no `yes`. The only gate is the MCP client's tool-approval prompt plus the docstring ("only when the user asked") | **owner decision**: is a client approval prompt "a human yes" (contract 9)? | open |
| D17 | text safe zones | watchouts.md said top 14% / bottom 20–25%; the code used top 12% / bottom 20% and nothing reserved the right-hand button strip | design wrong on evidence (phone screenshot 2026-10-20: header 7.5%, caption from 79% of the screen, buttons on the right 13% from 37–86%) | **fixed**: Stage 4 contract amended; `edit/text.py` bottom 76% plus the narrower wrap in the button strip |
| D18 | worker health | workers.md promised node versions in the health report; the `worker` table has none, so a worker started before a code change runs old code | design wrong (detail), real risk | doc amended; restart workers after deploys; versions in health = M change later |
| D19 | production docs | README, policies.md and production/README stated `render`/timeline as live | doc stale | amended: marked approved, not built |
| D21 | intent compiler | Its LLM calls aren't under a run's cost cap | design underspecified (no run exists yet) | contract 8 amended: outside the cap, logged per intent |
| D20 | fonts | "never depend on host fonts": emoji, CJK and the date stamp use host fonts | doc stale (doctor warns) | watchouts.md amended |
