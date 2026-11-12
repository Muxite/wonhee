# Find the tool (index of what wonhee already does)

Read this before building anything. Most things you want already exist: a CLI command, an MCP tool, a node or an
engine function. This page says which, and where the code lives. It points at the deeper docs and does not repeat them.
To list live capability: `uv run wonhee --help`, `uv run wonhee node list`, `uv run python -m wonhee.art list`.

## Get a video or post from the internet, and analyse it

| You want | Run | Code | Notes |
|---|---|---|---|
| Store an Instagram reel/post by URL: metadata, metric snapshot, video file, analysis (speech, shots, on-screen text, keyframe descriptions) | `uv run wonhee grab [--mine\|--friend] <url...>` · MCP `library_grab` · node `grab` | `wonhee/library/grab.py`, `wonhee/sources/apify.py`, `wonhee/sources/fetch.py` | Metadata through Apify (`apify/instagram-scraper`, needs `APIFY_API_KEY` in `.env`, paid per call). The video is fetched from the signed Instagram CDN link, with `yt-dlp` as fallback. Strip `?obrf=...` tracking from URLs. `--mine` marks the account ours. `--refetch` re-downloads. Stored under `data/media/<hash>`, row in `content_item`; run `uv run wonhee show run <id>` to see the output again. |
| Tear down stored posts into a "what to film / what to build" report | `uv run wonhee inspire [ids] [--relation ours\|friend\|external] [--since DATE]` · MCP `inspire` · node `teardown` | `wonhee/inspire/`, `wonhee/nodes/teardown.py` | Reports land in `data/inspire/<date>-run<N>/`. |
| Ideas for posts plus the b-roll to film, from saved posts and `data/brief/*.md` | `uv run wonhee ideate [-n 20] [--focus ...]` | `wonhee/inspire/ideas.py` | Each idea cites its sources. |
| Subtitles, on-screen text, translation of a stored reel | chain `docs/chains/translate_reel.json`; nodes `speech_subs`, `screen_text`, `translate`, `burn_subtitles` | `wonhee/nodes/subs.py`, `analyze/speech.py` | See USAGE › Nodes and chains. |
| Look at footage | node `contact_sheet` (timestamped real frames, CPU) | `wonhee/analyze/contact.py` | Do this before trusting any event time you guessed. |
| Search your own footage by description | `uv run wonhee shots index <video>` then `shots search "bus at sunset"` | `wonhee/analyze/shots.py` (SigLIP2, GPU) | Take the GPU lock. |
| Download a public Google Drive file or folder | `uv run wonhee download <drive-url> <dest>` · node `drive_download` | `wonhee/drive.py` | Drive only, not Instagram. |
| Scout trends from a request | `uv run wonhee scout` / `ask` / `go` | `wonhee/scout/` | Discovery by hashtag/Brave search, then fetch, analyse, shot list. |

Reference sets already grabbed and what they showed: [production/references.md](production/references.md),
[production/reel-study-2026-10-20.md](production/reel-study-2026-10-20.md). Our own results:
[production/projects/](production/projects/).

## Make or change media

| You want | Run | Code / doc |
|---|---|---|
| A styled reel from a talking clip | `uv run wonhee reel clip.mp4 --style <name>` (`reel styles` lists them) | [production/reels.md](production/reels.md) |
| Key out a plain wall, put a backdrop behind a person | node `chroma_key` | `wonhee/edit/chroma.py`, [USAGE › Chroma key](USAGE.md#chroma-key-no-real-green-screen-needed) |
| Colour grade video or photo | nodes `look`, `look_image`; `wonhee filter list\|apply\|make`; `wonhee vibe <photos> --like <refs>` | `wonhee/nodes/look.py`, `wonhee/filterstore.py`, [production/photo-looks.md](production/photo-looks.md). Filters export to .cube. |
| Clean or measure audio (-14 LUFS, denoise, voice isolation) | `uv run wonhee audio <clip> [--check] [--isolate]` | `wonhee/edit/` audio, nodes `audio_check`, `audio_clean` |
| Text/captions on photo or video | nodes `text_image`, `text_video`, `compose_video`, `burn_subtitles` | `wonhee/nodes/text.py`, `compose.py` |
| Layouts, collage, carousel, poster | nodes `compose`, `carousel`, `poster`, `grid_plan` | [USAGE › Layouts](USAGE.md#layouts-meme-frame-sticker-frame-collage-pop-art) |
| Transitions, montage, sound effects | nodes `join_videos`, `video_fx`, `montage`, `sfx` | `wonhee/nodes/montage.py` |
| Image effects, backdrops, covers (200+) | `uv run python -m wonhee.art list` | [ART.md](ART.md) |
| Pictures from data (dependency graph, treemap, commit heatmap) | `uv run wonhee viz ...`, then `publish muksite` | [USAGE › viz](USAGE.md#pictures-from-data-viz-and-publishing-to-muksite) |
| GitHub repo card set | `wonhee slop pick` / `slop render` | `.claude/skills/github-slop/SKILL.md` |
| Frame a mundane task as a professional moment | `uv run wonhee larp "<task>" [--video PATH]` | `wonhee/nodes/larp.py` |
| Any single node or a chain | `uv run wonhee node run <name>` / `chain run` · MCP `node_run` | [USAGE › Nodes and chains](USAGE.md#nodes-and-chains) |
| Copy a result out | `uv run wonhee export <ref> <dir>` | |

## Not in git yet (find it before you rebuild it)

The plane skit tooling lives under `data/plane/`, which is gitignored, so it is easy to lose:

- `data/plane/tools/render_skit.py`: spec-driven short-form renderer (per-segment speed, boomerang, word-timed captions,
  `fx` list, `ig` caption style, loop-friendly output). Usage in `data/plane/RESUME.md`.
- `data/plane/tools/check_timing.py`: caption vs voice timing check.
- `data/plane/voice/` (Kokoro voice build, whisper word timing, `treat.sh`), `data/plane/PLAYBOOK.md` and `REPORT.md`
  (process lessons), `data/plane/cells/BRIEF.md` + `slot.sh` (parallel cell agents with a render cap).

Moving the renderer into `wonhee/edit/` as a proper node (with tests) needs an owner decision (M lane): see the drift
row in DESIGN.md.

## Rules that save time

- **Search before building**: `uv run wonhee --help`, `uv run wonhee node list`, `grep -ri <word> docs wonhee`.
- **Grabbed once means stored**: ask the DB (`wonhee inspire <id>`, `wonhee show run <id>`) before grabbing again;
  each grab is a paid Apify call.
- **GPU work holds the lock** (AGENTS.md).
- **Write the beat table first** for any short video, and ship the simplest cut first
  ([plane skit note](production/projects/2026-11-09-plane-skit.md)).
