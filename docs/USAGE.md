# Using wonhee

Setup, commands and troubleshooting for the current implementation. Run the commands below from the
repository root. For the product overview, see the [README](../README.md); for contribution rules, see
[AGENTS.md](../AGENTS.md).

## Setup
    uv sync --all-extras          # moeka is a pinned pip dependency; there is no submodule
    cp creator.example.yaml creator.yaml   # what you can film
    cp .env.example .env          # then fill in OPENROUTER_API_KEY, BRAVE_API_KEY, APIFY_API_KEY
    uv run wonhee doctor

Keys come from the process environment only. `wonhee` loads a `KEY=VALUE` file at startup
(`--env-file PATH`, else `$WONHEE_ENV_FILE`, else `.env` in the repo), and anything you already
exported in your shell wins over the file. `.env` is gitignored. `wonhee doctor` reports, by name
only, whether each key came from the shell or the file. wonhee never reads `~/.nanobot` or any
other user-level moeka state; the LLM runs in-process through the pinned moeka kernel, and
subprocesses (ffmpeg, yt-dlp) get a minimal environment with no keys.

## Scout and manage runs
    uv run wonhee scout "gym humor, what was trending a year ago"
    # asks, confirms, runs (progress on stderr), prints the shot list
    uv run wonhee ask "gym humor now" [--intent 4]   # one compile turn
    uv run wonhee go 4                               # run a confirmed intent
    uv run wonhee show run 3                         # steps: status, flags, cost
    uv run wonhee show project 2                     # the shot list
    uv run wonhee resume 3 --from fetch --set cost_cap_usd=3
    uv run wonhee db sql "SELECT * FROM v_trend_popularity"
    uv run wonhee account mark some_friend friend

Live smoke test (spends real money, cap $0.75): `uv run pytest -m live -s`.

## Debugging a bad result
1. `wonhee show run <id>`: which stage is weak/failed, and why (flag codes).
2. `wonhee db sql "SELECT role, blob_sha256 FROM step_artifact WHERE step_id=<n>"`:
   raw API replies and model prompts/replies are files under
   `data/media/<sha[:2]>/<sha>`.
3. Fix the input and `wonhee resume <id> --from <stage>`; earlier attempts stay
   in the DB. If a stage found nothing (NO_CANDIDATES), resuming will not help:
   start a new scout with a wider window or a different niche.
4. Run stopped on the cost cap? Continue with
   `wonhee resume <id> --from <stage> --set cost_cap_usd=3`.

Errors: model-call failures print one line plus the log path; every error's
traceback is appended to `data/logs/wonhee.log` (under `WONHEE_HOME`). Apify
spend is capped per call by the run's remaining budget, and each actor run
waits at most 10 minutes before being aborted.

## Known limitations
- LLM cost uses the provider-reported cost when moeka exposes one, else `WONHEE_LLM_PRICE_IN_PER_MTOK` /
  `WONHEE_LLM_PRICE_OUT_PER_MTOK` (defaults 0.25 / 0.75); `wonhee doctor` warns when one is set to 0.
- Whisper runs on CUDA by default. To use CPU on purpose, set
  `WONHEE_WHISPER_DEVICE=cpu` and `WONHEE_WHISPER_COMPUTE_TYPE=int8`.
- A run left `running` after a hard kill is not picked up automatically, and
  `resume` refuses a running run: take it over with
  `wonhee resume <id> --from <stage> --set force=true`. Ctrl-C is handled: the
  run is marked stopped and resumes normally.

## Nodes and chains
Every tool is a typed node; a chain wires nodes together and wonhee pipes each result into the next.

    uv run wonhee node list                                   # what exists: ports in -> out, one-line doc
    uv run wonhee node show translate                         # params (JSON schema)
    uv run wonhee node run to_srt --in subtitles=subs.json    # one node; inputs are refs, content:<id>, or local files
    uv run wonhee chain run docs/chains/translate_reel.json --detach   # returns a run id at once
    uv run wonhee chain show 7                                # per-node status, cost, final outputs with file paths
    uv run wonhee resume 7 --from t --set no_cache=true       # redo from a node id (cache skipped)

`docs/chains/translate_reel.json` grabs a reel, reads speech and on-screen text, merges them, translates to English,
and writes an `.srt` plus a subtitled video. Change the `url` param to use it on another reel. Results are cached per
(node, version, inputs, params): rerunning an unchanged node costs nothing and shows `CACHED`.
Agents use the same through MCP: `nodes_list`, `node_run`, `chain_run(chain, wait=False)`, `chain_show`.

## Layouts (meme frame, sticker frame, collage, pop-art)

One engine, `compose`, draws a canvas plus ordered layers (photo, pattern, shape, sticker, text, band) the same way
every time; each format is a named template with slots (`uv run wonhee node show compose` lists them). The spec is
`docs/specs/2026-10-20-meme-and-popart-formats-design.md`. Photos must be yours: no third-party imagery or marks.

    uv run wonhee node run compose --in images=data/photos/desk.jpg --set template=brag_collage     # a desk, piled with stickers
    uv run wonhee node run compose --in images=me.jpg --set template=popart_cutout --set 'slots={"palette":"mint"}'
    uv run wonhee chain run docs/chains/sticker_frame.json                                          # heart-shaped photo + speech bubble
    uv run wonhee node run frame_video --in video=clip.mp4 --set 'slots={"title":"my title"}'       # 9:16 frame, one encode
    uv run wonhee node run grid_plan --in images=data/photos/nine/ --set cutout=true                # a 3x3 profile grid

- Templates: `meme_frame`, `sticker_frame`, `brag_collage`, `popart_cutout`, `slop_card`. A one-off layout goes in the
  `layout` param (see the schema); a new format is one JSON file in `wonhee/edit/compose_templates/`.
- `docs/chains/sticker_frame.json` uses `content:12` as a placeholder: put a stored post's id there.
- Text and stickers on tall canvases are moved out of the reel UI zones (12%-76% of the height, the centred 72% where the
  buttons are, rotated extent included) and flagged `LAYOUT_ADJUSTED`.
- `compose` takes an optional `refs` text (an inspiration digest or plain lines): a text or sticker line that copies it
  is dropped and flagged `IDEA_COPY`.
- `cutout` and `focus: subject` use the CPU ONNX mask models (downloaded on first use, as for `mask_cutout`).
- `compose_video` (video, subtitles?, images? -> video) is the timed form: layers take `t` {in_ms, out_ms} and `anim`
  (fade, pop, type, rise); layer types `captions` (chunk, word, karaoke), `stack` and `list` read the subtitles' words;
  photo layers from `images` are insets; `tighten: {}` cuts pauses and re-times everything (`PAUSES_TRIMMED` reports the
  seconds removed). It is one ffmpeg encode (a take with no subtitles adds one silencedetect analysis pass).
  `frame_video` stays as its first form. Spec: `docs/specs/2026-10-20-reel-styles-design.md` §2-§3.
- `frame_video` captions come from `subtitles`; with line timing only, each line's time is split evenly across its words
  (`WORDS_ESTIMATED`).

## Reels

One command turns a talking clip into a styled reel (speech to words, then one encode with timed text, captions and insets):

```bash
uv run wonhee reel styles                                   # the six styles, one line each
uv run wonhee reel clip.mp4 --style explainer --title "a trick i wish i had known"
uv run wonhee reel clip.mp4 --style scrapbook --items "the problem, the fix, the catch" --inset a.png --inset b.png
uv run wonhee reel clip.mp4 --style lecture            # the larp lecture register writes the title, theses, footnotes
```

Output lands in `data/reels/<date>/<name>/` (the mp4 and a `report.md` with the style, seconds of pauses trimmed and
flags). Speech uses whisper on the GPU (take the lock first). Each style, its slots and what it copies from the
references: [docs/production/reels.md](production/reels.md). MCP: `reel` and `reel_styles`.

## Pictures from data (viz) and publishing to muksite

wonhee also makes stills from data instead of footage (this absorbed `~/projects/vizkit` on 2026-10-19): a repo's
file dependency graph, a disk-usage treemap, a commit heatmap, a lines-of-code-per-language bar, and WaveSim shader
stills. They are ordinary nodes (`viz_depgraph`, `viz_diskviz`, `viz_history`, `viz_wavegen`), so a render chains
into `look_image`, `filter_image`, `poster` or the new `mask_cutout`, and `publish_muksite` puts the result on the
portfolio.

    uv sync --extra viz && uv run playwright install chromium-headless-shell   # graphs and treemaps render in Chromium
    uv run wonhee viz depgraph ~/projects/awork --exclude profiles/ share/ --theme both
    uv run wonhee viz depgraph ~/projects/webRAG --rev 7c91f087 --theme both        # the repo as it was in March 2026
    uv run wonhee viz depgraph ~/projects/webRAG/agent/app --focus '**/ledger_*' --hops 1   # one module + neighbours
    uv run wonhee viz history ~/projects/webRAG --chart commits --title "webRAG commits"
    uv run wonhee viz depgraph ~/projects/muksite/src --then look=faded              # finish with a look/filter/cutout
    uv run wonhee publish muksite data/viz/.../x-paper.png --dark data/viz/.../x-dark.png \
        --slug awork --name dependency-graph --caption "..." --alt "..." --yes

- A working-tree render of a git repo draws only the files git tracks (no ignored data, venvs or submodules);
  `--set tracked=false` draws everything on disk. `--rev` reads a past commit through `git archive` and never touches
  the working tree. Every render flags `VIZ_SOURCE` with the path, resolved commit and counts.
- Python imports resolve from each package's import root, so pointing at a repo root still links `from app.x import`
  inside `backend/app/`. `VIZ_NO_EDGES` means nothing resolved.
- `publish muksite` uploads to muksite's Supabase Storage (`portfolio-images/projects/<slug>/<name>[-dark].png`) and
  prints the YAML `photos:` entry. It needs `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` in `.env`, refuses to replace an
  existing image without `--overwrite`, and asks for `--yes` because the image becomes public. muksite's
  `scripts/build_image_variants.py` then makes the WebP variants.
- `viz_wavegen` needs the WaveSim docker image (`docker build -f wonhee/edit/viz/vendor/wavesim.Dockerfile -t
  wonhee-wavesim:dev wonhee/edit/viz/vendor`) and the GPU; take the GPU lock (AGENTS.md) before running it.
- Agents: MCP tools `viz` and `publish_muksite`.

## GitHub slop

Public GitHub repos become short, credited 4:5 card sets (1080x1350) for a person to post by hand. Facts on the
cards (stars, languages, dates, licence) come from GitHub through code; a model writes only the voice lines.

```bash
uv run wonhee slop next -n 5                  # pick, script (local model) and render 5 repos; --loop for endless
uv run wonhee slop pick [--repo owner/name]   # agent path: print a dossier, write script.json, then:
uv run wonhee slop render owner/name --script script.json
uv run wonhee slop serve                      # (alias: slop feed) the endless scroll page, 127.0.0.1 only
```

Sets land in `data/slop/<date>/<owner>__<name>/` (`01.png` to `04.png`, `caption.txt`, `script.json`) and in
`data/slop/feed.json`. `GITHUB_TOKEN` in the project `.env` is optional and raises the API rate limit. Slide 2 draws
the repo's dependency graph and needs the viz browser (see below); without it slide 2 is text only. The local model
needs Ollama (hold the GPU lock). Design: `docs/specs/2026-10-20-github-slop-design.md`.

## Workload profiling (optional)

The repo also carries **gauge**, a profiler that runs one job per Docker container and measures it from the host:
CPU and RAM from cgroup v2, VRAM per process from NVML, and, optionally, kernel-level GPU traces from Nsight Systems.
Day-to-day wonhee use never touches it. Use it to find out what a node (or any GPU task) actually costs, alone or next to another job.

```bash
uv run wonhee profile doctor                          # same as: uv run gauge doctor
uv run wonhee profile build                           # workload images in profiling/images
uv run wonhee profile run wonhee.speech_subs clip_s=30
```

How it works, and the results so far: `docs/profiling/GUIDE.md` and `profiling/report/shared-gpu-node.html`.
Data goes to `data/gauge/` (gitignored).

## Producing reels fast

`docs/production/` has the rules for agent-planned video DAGs. The target design is that edit nodes write a timeline and one `render` encodes it (approved, not built yet: see DESIGN.md › Stage 4). It also covers lanes and concurrency limits, which workers to build next from what's already on this machine, and the pitfalls found so far. All numbers are measured.

Photos and audio, all on the box except the optional shooting guide:

- `wonhee filter list | apply <name> <photos...> | make "<a vibe>" --name <name> --on <photos...> [--from <filter>] | show <name> --cube out.cube | calibrate --on <photos...>`: filters that give a vibe without copying anyone: a 13-filter pack, or a filter a local model drafts and tunes on your photos and saves for later; exports as a .cube LUT for phone and video editors.
- `wonhee vibe <photos...> --like <ref folder | content:id> --name <look>`: learn a look from references, get a shooting guide, and edit your photos toward it (measured grade, subject mask, lens effects; no image generation). See `docs/production/photo-looks.md`.
- `wonhee audio <clip> [--preset voice|mixed|music] [--check] [--isolate [--music-db -15]]`: denoise, tame peaks, -14 LUFS; `--isolate` splits voice from music/noise on the GPU.
- GPU extras (SAM masks from a text prompt, Demucs voice isolation, shot search): `uv sync --group gpu`.
- `wonhee shots index <video>` / `wonhee shots search "bus at sunset"`: cut footage into shots, embed and tag each keyframe (SigLIP2), then find shots by description across everything indexed.

## Warm GPU worker

A cold GPU node reloads its model on every call (whisper ~5 s, masks ~10 s, demucs ~5 s, SigLIP2 ~7 s). A worker keeps them loaded:

    uv run wonhee worker serve --lane cuda --preload whisper,mask   # foreground; SIGTERM finishes the current job
    uv run wonhee node run speech_subs --in video=clip.mp4 --detach # goes to the worker while it is alive
    uv run wonhee worker status                                     # workers, heartbeats, queue

Detached chains with a `cuda` node go to a live worker. Without one they start a background process as before. A job whose worker dies is requeued (up to 3 attempts). See `wonhee/worker.py`.

## Telegram creative workspace

The existing Telegram moeka now hands creative work to its paired wonny instance. Select a project with
`/wonhee project <name>`, send a take or public Drive file link, and ask for an edit. Videos return in Telegram;
oversized originals use a private Tailscale download page. Wonhee memory supports ordinary-word hybrid search,
project-scoped decisions, corrections, ideas and feedback, with readable exports.

See [Telegram creative workspace](production/telegram-creative.md) for setup, recovery and current
validation. Implementation is deployed; one Drive video has passed original-file Telegram delivery and phone playback.
Raw Telegram intake, live reel rendering and full file checking remain separate acceptance steps. [Integration design](specs/2026-10-24-moeka-integration-design.md).

### Download from Google Drive

Use a public file or folder shared with “Anyone with the link”:

```bash
uv run wonhee download 'https://drive.google.com/drive/folders/FOLDER_ID' data/takes
uv run wonhee creative inbox_add --data '{"source":"https://drive.google.com/drive/folders/FOLDER_ID","project_id":1}'
```

Downloads preserve subfolders and refuse to overwrite existing files. The first command also accepts an
individual file link. MCP exposes `drive_download(url, dest)` and `inbox_add(source, project_id, dest)`; MCP
output folders must be under wonhee's data directory. Folder intake registers each file as an asset without
starting analysis. Private links and native Google Docs/Sheets/Slides are unsupported.

## Export a result

Copy a stored result into a local folder. The destination must be a directory:

```bash
uv run wonhee export <ref> data/exports
```

Use a stored media reference or `content:<id>`. Agents have the matching `export(ref, dest)` MCP tool.

### Stagecraft desktop footage and a simple caption

Stagecraft builds the desktop display only by default: the large graph and profiler
share the viewport. Wonhee handles the separate caption and portrait composition.
Use stagecraft_build with profiler_asset_id pointing to project-owned plot-data.json
and caption containing the finishing text. The caption stays out of the authored
app; omitted caption uses the title, and an empty caption explicitly keeps raw
footage. Fresh desktop captures default to 1280x800. For legacy revisions supply
landscape desktop dimensions and caption_above_dashboard=false.

A fixed offline script in sibling Stagecraft scripts/paragraph_echo.py produces
seeded paragraph corpus CSV, verified per-message HTTP/stdio samples, paired CSV,
full-data histograms, bounded replay JSON and hashes. Import plot-data.json with
inbox_add, then use its asset ID as profiler_asset_id. Supplied timing values cannot
be rewritten by the coding builder. Scope is single-worker local echo, not agent
or mesh latency; the mesh and replay clock remain cinematic.

To caption already stored footage, make_submit accepts style=desktop_caption with
params={"title":"Your simple caption"}. The existing CPU compositor contains the
whole desktop clip on a dark 1000x1600 canvas with the caption above. This style
skips speech recognition and model/GPU work. Ordinary speech reel styles retain
their speech pipeline. Final completion retains the original capture and separate
Wonhee output hashes; retries do not stage the raw video or duplicate deliveries.

### Ask Moeka; no scripts or JSON import needed

For a finished demo, say: “Make a 20-second desktop demo with a huge cyan mesh next
to a profiler comparing HTTP and stdio on fresh random paragraphs. Add a simple
caption above the video.” Moeka uses stagecraft_build; the queue measures data,
registers evidence, dispatches coding, checks the desktop, captions it in Wonhee
and stages one final delivery. Its default benchmark=auto measures data when needed;
refresh requests a new run and off disables measurement. An existing profiler_asset_id
is optional reuse, never something the creator must locate or invent.

For data only, say “Make a random-paragraph HTTP/stdio dataset.” Moeka uses
stagecraft_dataset and the worker delivers one ZIP with CSV, plot JSON, source and
method/hash manifest. No coding agent is dispatched for data-only work.

For status or revisions, Moeka finds the project job with stagecraft_list, reads
stagecraft_show and inspects returned preview/review asset IDs with media_info.
Failures expose bounded compiler/layout/transport diagnostics and recovery guidance.
The assistant can perform a corrective revision in the same request without an
identity collision; captured-output recovery does not redo measurement or coding.
Arbitrary benchmark servers/programs remain outside the supported fixed workflow.

For desktop caption corrections use `make_submit(style="desktop_caption", caption="...")`
or `caption_lines=["HOOK", "5.5× lower median RTT vs HTTP", "Local echo · 512 paragraphs · 5 trials", "Workload-specific; not live mesh latency"]`.
The optional four lines become hook, metric, proof and caveat with distinct typography,
visible throughout the video. The brief stores context; put the actual copy in these
fields (or params.title / params.slots). Blank or unreadably long copy is rejected.
A finished Wonhee asset can be supplied: recorded lineage recovers its original
desktop, so corrections do not shrink an already framed portrait. Inspect make_show
for the effective caption and resolved source before describing the result.

Use `stagecraft_benchmarks` to find existing real paragraph HTTP/stdio datasets in
this project before creating another dataset. It returns actual median RTTs, ratios,
example paired observations and the profiler/evidence asset IDs. Feed its
`profiler_asset_id` to stagecraft_build when a video is requested. The graph can be
theatrical while the profiler replays real samples; describe the ratio as measured
local echo RTT, rather than a speedup measured across thousands of agents. Discovery
never queues a benchmark or video.

For two cuts of one dashboard, build Stagecraft with presentation="mesh_story",
primitive="scene", a saved profiler_asset_id and prepare_only=True. Then reuse its desktop
asset in make_submit(style="desktop_reel"). For a static cut supply caption; for a
reveal cut supply caption_cues with start_seconds, end_seconds and text. Ordered
cues must cover the source continuously. The result is 1080x1920 (9:16). The original
raw Stagecraft source is recovered for revisions; source pixels remain contained.

Supply variant="viral-discovery" and variant="slow-unravel" to distinguish two
make_submit calls from one message. Keep each label stable on retries. Shared-source
prepare_only imports the verified raw asset without staging a delivery; it rejects
nonempty caption text. Then use that asset in both final finishing jobs when asked.

## Post assistant

A post is a small project with named slots (`post_create`, `post_slot_set`, `post_status` under `wonhee creative`). The
nodes `take_pick`, `text_style`, `text_place` and `photo_fit` pick takes, read a reference's text style, place a line
on a photo and check that a photo can carry it. The numbers behind them are starting values that have not been tuned
on the owner's takes or reference slides: see [Post assistant](production/post-assistant.md).

## Chroma key (no real green screen needed)

```bash
uv run wonhee node run chroma_key --in video=take.mp4                        # drawn plane cabin behind you
uv run wonhee node run chroma_key --in video=take.mp4 --in background=cabin.mp4 --param guard=person
uv run wonhee node run chroma_key --in video=take.mp4 --param fill=matte     # white subject on black: check the key first
```

It finds the backdrop colour from the top and side edges, so a whitish wall works. Removal only happens where the
backdrop touches the frame edge, and the person mask (CPU, same as `mask_cutout`) protects a white shirt on a white
wall. Read the `report` output: `BACKDROP_UNEVEN` means add light or even out the wall. Light the wall evenly, keep it
brighter than you, and stay a hand's width off it. Spec: [chroma key](specs/2026-11-01-chroma-key-design.md).
