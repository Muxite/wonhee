# Port slopcut's transitions, effects and synthesized sound into wonhee

Status: **approved** 2026-10-28 (the owner said "port over slopcut capabilities"). New contracts, so large lane; the owner's
instruction stands in for the spec approval. Source: `slopcut` (the owner's own project, not a git repo,
newest file 2026-10-15, now dormant). It is not deleted or edited by this work.

## Why

The last-day-of-CapCut-premium parody and any "slop" reel need transitions, motion effects, glitch/boom overlays and
sound effects timed to them. wonhee's `edit/fx.py`, `edit/looks.py`, `edit/filters.py` and `edit/viz/` cover colour looks,
grain and data pictures only: no `xfade` transitions, no motion presets, no overlay/warp/cut effects, no sound effects,
no node that joins two clips. The post-assistant chains also stop short for lack of a join/concat node
([post-assistant gaps](../production/post-assistant.md)).

## What is ported (additive; nothing existing is replaced)

| slopcut | wonhee | Notes |
|---|---|---|
| `video/transitions.py`, `transitions_custom.py` | `wonhee/edit/montage/transitions.py` | ~55 built-in xfade names in style pools (calm, hype, glitch, slop) plus the expression-based custom ones |
| `video/effects.py` | `edit/montage/effects.py` | zoom/pan/punch/shake/glitch/flash presets for stills |
| `video/fx/*.py` | `edit/montage/fx/` | 43 named ffmpeg effects (overlay, boom, color, warp, cut) |
| `video/render.py`, `plan.py` | `edit/montage/render.py`, `plan.py` | seeded plan, hand-editable, re-rendered exactly; segment cache |
| `audio/dsp.py`, `sounds.py`, `cues.py` | `edit/sfx/` | sounds synthesized in code (no samples, no licensing); cues map each effect/transition to a sound on the visual hit |
| `video/review.py` | later | vision-model critique of a plan; needs wonhee's `ctx.llm` path and cost caps, so a follow-up |
| `carousel.py` | not now | pixel-seam splitting of a wide image; different from wonhee's carousel; a follow-up |

Nodes (names avoid the existing `slop`): `montage` (stills and/or one video → one video with chosen style, seed, length,
optional audio), `join_videos` (two videos + a named transition → one video; closes the post-assistant join gap),
`video_fx` (apply one named effect to a video), `sfx` (a plan or a named sound → audio). Each takes its params with
`extra="forbid"`, is seeded and deterministic, and appears in `wonhee node list`.

## Rules for the port

- Adapt, do not copy blindly: use wonhee's ffmpeg/ffprobe helpers (`wonhee/media.py` with the `LOCAL_ONLY` protocol
  whitelist), its settings and its test fixtures. No new dependencies (slopcut needs only Pillow and numpy, which wonhee has).
- Engines in `edit/` know nothing about the DB; nodes do the DB work.
- Keep slopcut's tests, ported, as the starting test set.
- Unknowns are `None`, never 0. New flag codes need a migration (`ctx.flag` on an unknown code fails the step).
- The owner's real footage gets few cuts; transitions are for the parody/b-roll parts, which the chains must respect.

## Anti-goals

Randomness that cannot be replayed (same seed, same video); a transition that reads wrong on a 9:16 frame; shipping any
third-party sound sample.
