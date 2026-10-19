# wonhee

The flagship for influencer-side automation: find what is trending in a niche
(now, or 6 months to 2 years back for throwbacks), turn it into a shot list you
can just film, edit raw recordings the way most of the platform currently does,
and build small disposable prop projects for skits.

wonhee is agent-centric (moeka drives it over Telegram) but the work runs as a
fixed pipeline over a relational store, so every output traces back to the run,
step, and source content that produced it.

Sub-projects, in build order:

1. **Trend scout**: [design](docs/specs/2026-10-15-trend-scout-design.md)
2. Editor
3. Prop builder
4. Director / idea inbox

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
- LLM cost uses the provider-reported cost when moeka exposes one, else `WONHEE_LLM_PRICE_*`;
  `wonhee doctor` warns when prices are unset.
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

`docs/production/` has the rules for agent-planned video DAGs. Edit nodes write a timeline and one `render` encodes it. It also covers lanes and concurrency limits, which workers to build next from what's already on this machine, and the pitfalls found so far. All numbers are measured.

Photos and audio, all on the box except the optional shooting guide:

- `wonhee filter list | apply <name> <photos...> | make "<a vibe>" --name <name> --on <photos...> [--from <filter>] | show <name> --cube out.cube`: filters that give a vibe without copying anyone: a 13-filter pack, or a filter a local model drafts and tunes on your photos and saves for later; exports as a .cube LUT for phone and video editors.
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
