# wonhee: vision

Status: approved 2026-10-20. Changing this page needs the owner's say-so (see [PROCESS.md](PROCESS.md)).

## What wonhee is

**A content creation pipeline for one creator and the agents working for them.** It takes you from "what should I
post?" to a finished, checked file ready to post, and for the portfolio it takes you all the way to a published
image. Every step is a typed node that a person or an agent can run alone or chain, and every output traces back to
the run and the sources that made it.

wonhee absorbed vizkit on 2026-10-19 and supersedes it. Pictures made from data (repo graphs, treemaps, commit
charts, shader stills) are one more source of content in the same pipeline. They aren't a separate tool.
`~/projects/vizkit` is frozen.

## The pipeline

| Stage | Question it answers | What does it today (nodes / commands) |
|---|---|---|
| 1. Discover | What's working in my niche, now or back then? | `scout`, `grab`, `teardown`, `inspire`, `content_images` |
| 2. Plan | What do I post, and what do I film or say? | `ideate`, `larp`, `shot_guide`, idea bank |
| 3. Capture | How do I shoot it? How does the take get in? | `shot_guide` (guidance). Getting takes from the phone is not built (toolkit R4) |
| 4. Make | Turn takes or data into the post | video: `look`, `text_video`, `burn_subtitles`, translation chain, `audio_clean`; photo: `look_image`, `filter_*`, `vibe_image`, `text_image`, `carousel`, `poster`, `mask_cutout`; data: `viz_*` |
| 5. Check | Is it OK to post? | `audio_check`, the vibe/filter readability checks. The full file check (toolkit R3) is not built |
| 6. Deliver | Get it out | `export`, `publish_muksite` (gated). The phone outbox is not built |

Stage 0 runs under all of them: the store (SQLite, content-addressed media), provenance (run → step → flag), nodes
and chains with caching, warm GPU workers, and the gauge profiler for measuring cost.

## What "good" means

1. **Raw take to postable file fast.** Measure it in wall time and in human touches, not in features shipped.
2. **Edits are basic and unnoticeable,** like most of the platform. Looks give a vibe; they never copy a specific
   person's look.
3. **Agents can do anything a person can.** Every action has a CLI command and an MCP tool, and both call the same
   service function.
4. **Every output is explainable.** You can see which run, step, inputs, params and model made it, and what was
   flagged.
5. **Local first, cheap by default.** Work that the box can do runs on the box. Paid calls have a cap. Keys are
   project-scoped.
6. **Numbers are measured.** Any number in a doc cites a run or a bench.

## Never

- Posting to Instagram or any social account. A person posts.
- Generated imagery passed off as a photo. Edits move real pixels; there is no image generation.
- Faking real events. A staged skit labelled as a skit is fine.
- Copying a creator: no lifted captions, handles, or a look cloned from one person.
- Anything irreversible or public (publishing, deleting, spending over the cap) without a human yes.
- Committing third-party media.

## Who it's for

The creator is @frictionlessmuk. The niche (set 2026-10-17): low-budget Asian engineering student, gorpcore, K-pop,
and performative AI-founder humour. The creator profile lives in `creator.yaml`. Agents (moeka over Telegram, Claude
over MCP) are first-class users.

## Where it's going (ordered)

1. **Close the Make → Check → Deliver loop for reels:** `render` (one encode per chain), timeline ops, a full
   file check, and the phone inbox/outbox.
2. **Style memory** (voice and rhythm) so captions and cuts sound like us.
3. **Prop builder** (small throwaway projects for skits) and **director** (idea inbox that turns into plans).

The detailed designs are linked from [DESIGN.md](DESIGN.md).
