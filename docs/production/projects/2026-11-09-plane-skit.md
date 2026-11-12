# Plane skit: "reverse a linked list" (2026-11-08 to 11-09)

Result: the simple static cut worked. The elaborate Two Sum variants (boomerang replays, turn-back, fx) were not needed.
Stored in the library as content 79 (`wonhee grab --mine`, run 548): `wonhee inspire 79` or `wonhee node run teardown`
on it reads its analysis.

## Result and analysis

Numbers, the three-step rationale and the lessons live in the post-game analysis:
the private post-game analysis.

Clip: `data/plane/cells/final_two/1c_standup_linkedlist_static_loop_nocaption.mp4` (spec `cells/su_ll3/su_ll3.json`),
6.7 s, 720x1280, one shot, no on-screen caption (caption written in Instagram), built to loop. Source of truth for
state: `data/plane/RESUME.md`.

## What went wrong in the process

Details and the seven root causes are in `data/plane/REPORT.md`. Short form: no written beat table before cutting,
guessed footage event times, voice chosen after specs were timed, effects added then removed, ambiguous requests
guessed instead of confirmed. Rule from the result: for short comedy reels, ship the simplest cut first.

## Tools used (all reusable, see [docs/TOOLS.md](../../TOOLS.md))

- `wonhee grab` (Apify scrape + CDN download + analysis) for the post and its metrics.
- `chroma_key` / `wonhee/edit/chroma.py` for the plane-window backdrop (face-mask guard).
- `audio` (loudness check), whisper via `speech_subs` for caption timing checks.
- Skit renderer `data/plane/tools/render_skit.py` and voice build scripts in `data/plane/voice/`: **not in git**
  (`data/` is ignored). See TOOLS.md for the plan to move it into the tree.
