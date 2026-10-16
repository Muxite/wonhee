# wonhee toolkit design

Status: approved in chat 2026-10-15; R0 nodes/chains and R7 translation added 2026-10-15 (awaiting review) (sub-project 2, before the editor proper). Builds on the trend scout
(`2026-10-15-trend-scout-design.md`) and reuses its store, provenance and pipeline machinery.

## 1. Goal

Small shared tools that people and agents both use, so wonhee replaces basic editing chores and keeps a
memory of how we post:

- **Edit ops**: trim silent ends, normalise loudness, captions, speed, cut, concat, 9:16 reframe, transcode.
- **Integrity checks**: "is this file OK to post?" before anything goes out.
- **Phone link**: upload raw takes from the phone; download finished edits to the phone.
- **Grab**: pull any reel by URL into the library (built 2026-10-15: `wonhee grab URL [--mine|--friend]`).
- **Style memory**: keep the voice and edit rhythm of chosen posts (ours or anyone's) in a form that
  generalises, without copying their specifics.
- **Nodes and chains**: every tool is a typed node; an agent (or a person) wires several into a chain once and
  the results pipe from one node into the next with nobody watching. Translation is the first node set.

Editing stays "basic and not noticeable", like most of the platform. Corny transitions come later from slopcut.

## 2. Principles

- Reuse, don't parallel: edits and style rips are projects → runs → steps, so `wonhee show run`, flags,
  error artifacts, cost caps and `resume` work unchanged. Every file goes into the content-addressed store.
- Off-the-shelf first: ffmpeg filters (`silencedetect`, `loudnorm`, `ebur128`, `blackdetect`, `ass`), PySceneDetect,
  faster-whisper, Starlette. No hand-written DSP.
- One surface: the CLI and the MCP server call the same `service` functions; agents can do anything a person can.
- Keys project-scoped (see memory rule): process env or the repo `.env`; subprocesses get `procenv.clean_env()`.
- Organisation via structure: lineage lives in tables (`asset.created_by_step`, `style_rule_source`), not notes.

## 3. Requirements

### R0 Nodes and chains (`wonhee/nodes/`)
The way every tool below is exposed, so an agent can decide "these steps must happen" once and let wonhee pipe
each result into the next without supervising.

**Node.** `Node(name, version, inputs: dict[port, type], outputs: dict[port, type], params: type[BaseModel], fn)`;
`fn(ctx, inputs: dict[port, Ref], params) -> dict[port, Ref]`. Registered in `wonhee/nodes/registry.py`.
Optional inputs are written `"type?"`. Port naming: a node with a single port of a type names it after the type
(`video`, `subtitles`); otherwise the table below gives the names (`translate`: `subs`, `caption` → `subs`, `notes`). Bumping `version` invalidates that node's cache.

**Port types** and what a `Ref` holds:

| type | ref | stored as |
|---|---|---|
| `content` | content_item id | row |
| `video`, `image` | media sha | file in the store |
| `subtitles` | blob sha | JSON `{"language", "lines": [{"id","start_ms","end_ms","text","source": "speech"\|"screen"}]}` |
| `text` | blob sha | UTF-8 text |
| `file` | blob sha | any file (e.g. `.srt`) |

Values move between nodes as refs only, never bytes.

**Chain.** JSON list of `{"id", "node", "in": {port: "<id>.<port>" | literal ref}, "params": {...}}`.
Literal refs: a sha, `content:<id>`, or a local path (stored first). Validated before anything runs or is spent:
unknown node, unknown port, type mismatch, missing required input, duplicate id, cycle, bad params (pydantic)
→ one error listing every problem, exit 2, nothing recorded.

**Execution.** A chain is stored as a confirmed intent (`schema_name='chain'`) and runs as a project of kind `chain`;
each node invocation is one step (stage name = the chain id, `params_json` = its params). The orchestrator's
stage list is built from the chain in topological order (ties keep file order), so `show run`, flags, cost cap,
`resume RUN --from <id>` and superseding work unchanged. A failed node stops the run; downstream nodes don't run.
Outputs are recorded in `step_output(step_id, port, type, ref)`; downstream nodes read their inputs from the
*current* step of the upstream id (as `resume` already guarantees for stages).

**Cache.** Key = sha256 of canonical JSON `{node, version, inputs (refs), params}`. Before running a node,
`node_cache` is checked; a hit copies the refs into `step_output`, marks the step `ok` with note `cached`, costs $0.
`--no-cache` (per chain) / `"cache": false` (per node) skips the lookup.

**Detached runs.** `chain_run(chain, wait=False)` validates, records the run, spawns `wonhee chain exec <run>`
as a detached process (`start_new_session`), and returns the run id at once. The child gets `clean_env()` plus the
project keys the parent loaded (it is wonhee itself, not a third-party tool) and logs to `data/logs/run-<id>.log`.
`chain_show(run)` reports status per node and the final outputs; a run whose process died shows `running` with a
stale `started_at` and `wonhee doctor` lists it.

**Surface.** CLI: `wonhee node list`, `wonhee node show <name>`, `wonhee node run <name> --in port=ref … [--set k=v]`
(one-node chain), `wonhee chain run <file.json|-> [--detach]`, `wonhee chain show <run>`.
MCP: `nodes_list` (names, ports, param JSON schemas, one-line docs), `node_run`, `chain_run(chain, wait=True)`,
`chain_show`. Every node result prints a short human preview (first lines of subtitles, video probe, check badges).

### R7 Translation nodes (`wonhee/nodes/translate.py`, `wonhee/nodes/subs.py`)
Proven by hand on reel DcSv-shxIDR (108 s Mandarin skit, 63 lines, $0.003); these make it a feature.

| node | in → out | params (defaults) | behaviour |
|---|---|---|---|
| `grab` | → `content`, `video` | `url`, `relation="external"`, `refetch=false` | wraps the library grab (scrape, download with the yt-dlp audio fallback) |
| `speech_subs` | `video` → `subtitles` | — | GPU whisper; reuses the stored transcript when `media_analysis` already has one for the sha; no audio → empty lines, flag-free |
| `screen_text` | `video`, `context: subtitles?` → `subtitles` | `every_s=2.0`, `max_frames=90`, `dedupe_hamming=4` | sample a frame every `every_s` (cap `max_frames`, widen the step if over); skip frames within `dedupe_hamming` of the last *read* frame (extend its line instead); vision model reads text in batches of 8 frames; the context subtitles go in as a data block so names match the speech; identical consecutive text merges into one line |
| `merge_subs` | `a: subtitles`, `b: subtitles` → `subtitles` | `prefer="a"` | union by time; a `b` line is dropped when contiguous text blocks shared with the `a` lines within ±`slack_ms` (2000) cover ≥ `cover` (0.8) of its normalised text. Only blocks of ≥ 3 characters count (≥ 2 when the `b` text has CJK), so scattered letters never match. A `b` line of ≥ `long_chars` (20) characters is compared against `a` lines within ±`long_slack_ms` (15000), for end cards that replay the last speech |
| `translate` | `subs: subtitles`, `caption: content?` → `subs: subtitles`, `notes: text` | `to="en"`, `style="natural"` (or `literal`) | one English line per input line, same ids and timings; batches of 80 lines with the previous 10 as context; a glossary of names/terms from batch 1 is carried into later batches; missing ids → one retry for those ids, then `TRANSLATION_GAP` flag listing them; notes = gist + names/wordplay/cultural references |
| `to_srt` | `subtitles` → `file` | `max_chars=42` | standard SRT, long lines wrapped at word (or CJK character) boundaries |
| `burn_subtitles` | `video`, `subtitles` → `video` | the R1 `captions` style | `.ass` burn-in, same plain lower-third look; ≤ 2 lines on screen |

Example: translate a foreign reel and burn English subtitles, unattended:
```json
[{"id":"g","node":"grab","params":{"url":"https://www.instagram.com/reel/DcSv-shxIDR/"}},
 {"id":"s","node":"speech_subs","in":{"video":"g.video"}},
 {"id":"o","node":"screen_text","in":{"video":"g.video","context":"s.subtitles"}},
 {"id":"m","node":"merge_subs","in":{"a":"s.subtitles","b":"o.subtitles"}},
 {"id":"t","node":"translate","in":{"subs":"m.subtitles","caption":"g.content"},"params":{"to":"en"}},
 {"id":"srt","node":"to_srt","in":{"subtitles":"t.subs"}},
 {"id":"b","node":"burn_subtitles","in":{"video":"g.video","subtitles":"t.subs"}}]
```
The edit ops (R1) and `check` (R3) are registered as nodes with the same signatures (`video` → `video`, and
`video` → `text` findings), so they chain with these.

### R1 Edit ops (`wonhee/edit/ops.py`)
Each op is a pure function `op(src: Path, dst: Path, **params) -> OpReport` around ffmpeg, run with
`clean_env()`. `OpReport(params_used: dict, notes: list[str])`.

| op | what | key params (defaults) |
|---|---|---|
| `normalize` | h264 High / yuv420p / AAC 48k stereo, CFR 30 fps, faststart; adds a silent AAC track if there is no audio | `fps=30`, `crf=20` |
| `trim_silence` | cut leading/trailing silence only (never the middle) via `silencedetect` | `threshold_db=-40`, `min_silence_s=0.35`, `pad_s=0.15` |
| `loudnorm` | two-pass EBU R128 | `target_lufs=-14`, `true_peak=-1.5`, `lra=11` |
| `speed` | `setpts` + chained `atempo` | `factor` (0.5–2.0) |
| `cut` | remove time ranges | `remove=[(start_s,end_s),…]` |
| `concat` | join clips (all normalised first) | `inputs=[…]` |
| `reframe_9x16` | landscape/square → 1080x1920, blurred background pad; no-op if already 9:16 | `width=1080` |
| `captions` | whisper word timestamps → `.ass` → burn in | style: white, black outline, lower third, ≤ 3 words per line, `font="DejaVu Sans"` |

Captions reuse `analyze.speech` (GPU, word timestamps on). A video with no audio stream gets no captions and a note.

### R2 Saved chains (replaces "recipes")
A recipe is just a saved chain (R0). Built-in chain `basic` = normalize → trim_silence → loudnorm → captions → check,
stored as `wonhee/chains/basic.json`. Edit outputs are also recorded as `asset(role='edit', created_by_step=<step>)`;
the input raw take is `asset(role='raw_take')`. `--set node.param=value` overrides a param (`captions.enabled=false`,
`loudnorm.target_lufs=-16`). The final video of `basic` is added to the outbox automatically.
CLI: `wonhee edit <upload-id|sha|path> [--chain basic] [--set k=v …]` (sugar for `chain run` with the input bound),
`wonhee node run <op> --in video=<sha|path> [--set k=v …]` for one-off ops.

### R3 Integrity checks (`wonhee/check/`)
`check(path) -> list[Finding(code, severity: 'error'|'warning', detail)]`. Results are stored per
`(media_sha256, checker_version)` in `media_check` so a file is checked once.

| code | severity | rule |
|---|---|---|
| `DECODE_ERROR` | error | `ffmpeg -v error -i f -f null -` prints anything |
| `NO_VIDEO` | error | no video stream |
| `NO_AUDIO` | warning | no audio stream |
| `SILENT_AUDIO` | warning | integrated loudness < -50 LUFS |
| `LOUDNESS_OFF` | warning | integrated loudness outside -14 ± 2 LUFS |
| `BLACK_RUN` | warning | `blackdetect` run ≥ 0.5 s |
| `AV_DURATION_MISMATCH` | warning | audio and video stream durations differ by > 0.2 s |
| `NOT_9x16` | warning | aspect ratio not 9:16 within 1 % |
| `LOW_RES` | warning | short side < 720 |
| `CODEC_NOT_IG` | warning | video not h264/hevc or audio not aac |
| `TOO_LONG` | warning | longer than 180 s (Instagram reel limit) |
| `VFR` | warning | variable frame rate (avg ≠ r_frame_rate) |

CLI `wonhee check <path|sha>`; MCP `check`. Final stage of every recipe; outbox shows the badges.

### R4 Inbox, outbox and the phone page (`wonhee/serve/`)
- `upload(id, media_sha256, original_name, uploaded_at, note)`; ingest = store → probe → check → transcript and
  keyframes (reusing `analyze_video` in a `library` project, so agents can see the clip).
- `outbox(id, media_sha256, label, added_at, from_step NULL)`.
- `wonhee serve`: Starlette app bound **only to the tailscale IP** (from `tailscale ip -4`; refuses to start
  without it unless `--host` is given). Pages: inbox list + upload form (streamed to disk, no size cap beyond
  disk), outbox list with download buttons and check badges. Tailnet membership is the only auth.
  Android Chrome is the target browser.
- A user systemd unit file is provided in `deploy/` but never enabled by wonhee.
- CLI: `wonhee inbox list`, `wonhee outbox add <sha> [--label]`, `wonhee outbox list`.

### R5 What agents see (`media_info`)
`wonhee media <sha|upload-id|content-id>` / MCP `media_info`: path, probe, check findings, transcript, shots with
on-screen text and **keyframe file paths** so a vision-capable agent can look at the clip. MCP tools:
`inbox_list`, `media_info`, `nodes_list`, `node_run`, `chain_run`, `chain_show`, `outbox_add`,
`library_grab` (exists), `style_rip`, `style_show`, `style_write`. (Edit ops and `check` are reached through
`node_run`/`chain_run`, not separate tools.)

### R6 Style memory (`wonhee/style/`)
Two kinds: `voice` (captions and on-screen text) and `rhythm` (length, cuts/s, first-word time, speech vs music).

Tables: `style(id, name, kind, version, created_at, created_by_step)` with `UNIQUE(name, kind, version)`;
`style_source(style_id, content_id)`; `style_trait(style_id, key, value_num, value_text)`;
`style_rule(style_id, ord, text)`; `style_rule_source(style_id, rule_ord, content_id)`;
`style_template(style_id, ord, text)`.

- Sources are `content_item`s: `wonhee style rip --from-handle @x [--limit 12]` (Apify profile scrape → grab
  pipeline stages) or `--from-content 1,2,3` or `--mine` (all `relation='ours'` content). Project kind `style`.
- **Traits** are measured, deterministic: median caption words, emoji per caption, hashtags per caption,
  lowercase ratio, line breaks per caption, question/exclamation rate; rhythm: median duration, cuts/s,
  first-word ms, share with speech.
- **Rules** come from the model (`StyleRules` schema), each citing ≥ 2 source content ids; a rule backed by fewer
  is dropped (with 1 source total, rules are kept but the step is `weak` with `LOW_SAMPLE`).
- **Templates** are short examples with slots (`[topic]`, `[relatable situation]`).
- **No-copy guard (deterministic, after the model):** reject a rule or template that shares any 4-word run with a
  source caption/on-screen text, contains a number, @handle or hashtag that appears in the sources, or contains a
  word the model listed in its required `removed_specifics` field. Rejections are flagged `STYLE_LEAK` with the
  offending n-gram; the model gets one retry with the list.
- Versions: re-ripping the same name creates v+1; older versions stay.
- Use: `wonhee style show <name>`, `wonhee style write <name> "<idea>"` (draft 3 captions in that voice; output also
  passes the leak guard), `wonhee style list`. Rhythm styles feed recipe params (`target_duration`, `speed`) via
  `wonhee edit … --style <name>`.

## 4. Optimisations (nice, not required)
- Cache whisper word timestamps per sha so captions after a trim re-use them (offset), instead of re-transcribing.
- Parallel ffmpeg for independent recipe stages: not needed while stages are sequential.

## 5. Out of scope
Telegram; slopcut transitions; auto-applying styles inside scout shot lists; posting to Instagram (never).

## 6. Testing
- Generated ffmpeg fixtures: silence at both ends, a black run, no-audio, VFR, landscape, HEVC portrait.
- Each op asserts measurable effect (duration shrinks by the silent ends; loudness within ±1 LU of target;
  9:16 dims; caption burn changes frames in the lower third).
- Checks: one fixture per finding code.
- Serve: Starlette TestClient (upload, list, download); binding refuses non-tailscale host by default.
- Nodes/chains: validation rejects each bad-chain class with all problems listed; topological order; failure stops
  downstream; resume from a node id supersedes it and everything after; cache hit costs $0 and is marked `cached`;
  version bump misses the cache; detached run returns at once and finishes (subprocess test with fake deps via env).
- Translation: fake LLM; `translate` keeps every id and timing, retries missing ids then flags `TRANSLATION_GAP`,
  carries the glossary across batches (3 batches in the test); `screen_text` on a generated video with text changing
  every 3 s finds each change and skips duplicate frames (counted model calls); `merge_subs` drops overlapping
  duplicates; `to_srt` round-trips timings and wraps CJK.
- Style: traits on fixed captions; leak guard unit tests (4-gram, numbers, handles, hashtags, removed_specifics);
  model calls through the fake LLM.

## 7. Data changes
Migration 0006: `step_output(step_id, port, type, ref, PK(step_id, port))`, `node_cache(key_sha256, port, type, ref,
node, node_version, created_by_step, PK(key_sha256, port))`, `upload`, `outbox`, `media_check`, `style*` tables;
`project.kind` gains `chain` (FK-off rebuild, as 0005); new flag codes `STYLE_LEAK`, `CHECK_FAILED`,
`TRANSLATION_GAP`. (Migration 0005, already applied, widened `project.kind` with `library` and `style`.)
