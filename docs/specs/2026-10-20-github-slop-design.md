# GitHub slop machine: design + plan

Status: **approved** 2026-10-20 (the owner asked for design, plan and build in one go; kept deliberately small).
Stages: 1 discover (repos) → 4 make (card sets) → 6 deliver (a person posts). Idea borrowed from Stockgen: an AI
writes a small declarative script, and a deterministic renderer turns it into media.

## Goal

Doomscroll GitHub. An endless feed of public repos, each turned into a short, punchy, scrollable card set
(Instagram carousel slides). Two audiences scroll the same output:
- the owner, in a local infinite-scroll page;
- Instagram followers, after the owner posts a set by hand.

## Pipeline

```
gh_pick ──dossier──▶ slop_script ──script──▶ slop_cards ──images──▶ feed page / export / IG (by hand)
   (GitHub API +        (local LLM, or an       (Pillow + the viz depgraph
    shallow clone)       agent via the skill)     engine; 4:5 slides)
```

| Node | In → out | Contract |
|---|---|---|
| `gh_pick` | → `dossier: text` | Picks one unseen public repo from GitHub search: a random query from a topic rotation, `stars:20..20000`, pushed in the last year, not a fork, not archived, has a description, size ≤ 50 MB. `params.repo="owner/name"` picks that repo instead. Shallow-clones into `data/slop/repos/` and writes a JSON dossier: name, owner, url, description, stars, forks, language breakdown, licence, topics, created/pushed dates, a README excerpt (≤ 3,000 chars) and the clone path. Marks the repo seen (`slop_seen`). Not cacheable. Uses `GITHUB_TOKEN` from the project env when set (higher rate limit), anonymous otherwise |
| `slop_script` | `dossier` → `script: text` | The local LLM (`engine=local`, default) or the remote one writes a `SlopScript`. Facts on cards (stars, languages, dates, licence) are **filled in by code from the dossier**, never by the model. The model writes only the voice lines |
| `slop_cards` | `dossier`, `script` → `images` | 4 slides, 1080×1350 (4:5): 1 hook (big text over a dark gradient, repo name small), 2 "what it is" plus the depgraph still of the clone (skipped with a `SLOP_NO_GRAPH` flag if the export browser is missing or nothing resolves), 3 stats (stars, top languages bar, age, licence), 4 roast or verdict plus "github.com/owner/name". Uses the sticker-ish outline text preset available today. Every slide carries `owner/name` (credit) |

`SlopScript` (pydantic, `extra="forbid"`): `hook` (≤ 9 words), `what` (≤ 24 words), `why` (≤ 24 words), `roast`
(≤ 16 words, affectionate, about the code, never the author), `caption` (≤ 300 chars), `hashtags` (≤ 6, no `#` in
the values).

## Rules

- **No fabrication.** Voice lines may exaggerate the tone ("this repo is a lifestyle"). They may not invent
  facts (users, benchmarks, funding). The prompt gives the dossier as delimited data (contract 7). Numbers on a
  card come from the dossier only.
- **Credit and licence** appear on the slides (`owner/name` on every slide, the licence on slide 3). No secrets:
  only the README and the file tree are read, never `.env`-like files.
- **Real pixels only:** text, gradients, charts and the repo's own graph; nothing generated.
- **Posting is by hand** (VISION never-list). wonhee writes the caption and hashtags next to the slides.

## Surfaces

- CLI:
  - `wonhee slop next [-n 5] [--engine local|remote] [--loop]` picks, scripts and renders N repos (forever with
    `--loop`, pausing 20 s between).
  - `wonhee slop pick [--repo owner/name]` prints the dossier (the agent skill path).
  - `wonhee slop render <dossier-ref> --script script.json` renders an agent-written script.
  - `wonhee slop feed` / `wonhee slop serve [--port 8765]` serves the feed page on 127.0.0.1.
- MCP: `slop_next`, `slop_pick`, `slop_render`.
- Output: each set goes to `data/slop/<date>/<owner>__<name>/` (`01.png` … `04.png`, `caption.txt`, `script.json`).
  `data/slop/feed.json` is appended, newest first.
- **Feed page** (`data/slop/index.html`, static): vertical scroll-snap, one set per screen. Slides swipe
  sideways. It loads `feed.json` in pages of 10 as you near the end and re-polls every 30 s, so a running
  `slop next --loop` keeps it endless.
- **Agent skill** `.claude/skills/github-slop/SKILL.md` (Claude Code). Codex is pointed at the same file from
  AGENTS.md. The steps are `slop pick`, write `script.json` to the `SlopScript` schema from the dossier, then
  `slop render`. The skill states the no-fabrication rule.

## Store

Migration `0016_slop.sql`: `slop_seen(repo TEXT PRIMARY KEY, first_seen TEXT, run_id INTEGER)`, plus flags
`SLOP_NO_GRAPH` and `SLOP_RATE_LIMIT`. Runs are chains (`docs/chains/github_slop.json`), so provenance and caching
come for free.

## Tests (fakes; no network)

- `gh_pick` with a fake GitHub client: skips seen repos, forks and repos over the size cap, and writes the dossier
  fields.
- `slop_script` with a fake LLM: lines longer than their limit are refused and retried. The stars and licence on the
  cards come from the dossier even when the script says otherwise.
- `slop_cards`: 4 PNGs at 1080×1350. `owner/name` is drawn on each (non-empty bbox in the footer band). A missing
  browser gives 4 slides with a text slide 2 plus the `SLOP_NO_GRAPH` flag.
- The feed page: `feed.json` is valid and newest first, and `index.html` references it.

## Plan (one Sonnet agent, one wave)

1. Migration, `SlopScript`, and a GitHub client (`wonhee/sources/github.py`, httpx, a fake in tests).
2. Nodes in `wonhee/nodes/slop.py`; engine drawing in `wonhee/edit/slop.py`.
3. `service.slop_*`, CLI and MCP twins, the chain JSON, the feed page template.
4. Skill file and the AGENTS.md pointer.
5. Tests, then one live run: `wonhee slop next -n 2` (anonymous GitHub, local LLM under the gpu-lock).
