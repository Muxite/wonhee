# wonhee

A content creation pipeline for one creator and the agents working for them: find what is working in the niche,
plan the post and the footage, guide the shoot, make the post (video, photo, audio, and stills drawn from data),
check it, and deliver it. It supersedes vizkit (absorbed 2026-10-19).

wonhee is agent-centric (moeka over Telegram, Claude over MCP) but the work runs as typed nodes over a relational
store, so every output traces back to the run, step, and source content that produced it.

**Design first.** [docs/VISION.md](docs/VISION.md) says what wonhee is for, [docs/DESIGN.md](docs/DESIGN.md) says
what each part must do (with status and known drift), and [docs/PROCESS.md](docs/PROCESS.md) says how a change goes
from design to code, and how to tell whether the code or the design is wrong when they disagree.

The job-application form filler that used to live here is now `../awork-fill`.

Changing the code? Start with [AGENTS.md](AGENTS.md): code map, add-a-node checklist, tests, GPU etiquette.

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

## Use
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
- `frame_video` captions come from `subtitles`; with line timing only, each line's time is split evenly across its words
  (`WORDS_ESTIMATED`).

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
