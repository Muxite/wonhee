# Post assistant: a scratchpad that turns a reference, loose footage and an idea into a finished post

Status: **approved** 2026-10-27 (the owner approved the build plan; the two open questions in §6 stay on their
defaults: wording help only when asked, render the top take plan and show it). New contracts (a project that holds
unfinished chunks, a take picker, a text-style reader and a placement step), so this was **large lane** (PROCESS §2).
Evidence: the owner's own reels `content:1`, `14`, `73`–`75`; the reference image posts `content:66`, `67`, `70`–`72`
and the reels `68`, `69`, `76`, `77` (all third-party, reference only, never committed). Builds on
[reel styles](2026-10-20-reel-styles-design.md), [meme reel styles](2026-10-22-meme-reel-styles-design.md) and the
[direct moeka](2026-10-25-direct-moeka-design.md) conversation path.

Marks: **(owner)** is something the owner said. **(guess)** is my reading; the owner has not confirmed it.

## 0. Why

The owner makes short posts that borrow a polished format (an "elegant" tech or engineering student, a LinkedIn
comment pile, a "last day of CapCut premium" gag) and push it into hyperbole. The ideas and the wording are the owner's
(owner). The costs are, in order: **editing and trimming, recording, then writing and tightening** (owner). Nobody
finishes a post in one sitting: b-roll, voice and screen recordings are made on different days (owner).

wonhee already trims pauses (`compose_video` `tighten`), cleans voice (`audio_clean`), writes readable caption lines
(`speech_subs` + `compose_video`), tears a reference down (`teardown`) and places styled text inside safe zones
(`edit/text.py`). The owner has run that edit through a Claude Code agent (`2026-10-22-midnight-waterloo.md`). What is
missing is the glue: choosing the take, knowing what a post still needs, and matching a reference's look on photos.
This spec adds that glue and nothing else.

## 1. Goals, non-goals, anti-goals

Goals:
1. The owner records freely and unscripted (owner). wonhee finds the good lines and builds the cut.
2. Fewer cuts, not more (owner): on selfie and real-world footage, prefer one clean take over spliced fragments.
3. A post is a small project with named slots (lines, photos, voice, b-roll, screen recording, audio track). It can sit
   half-filled for days and builds from whatever has arrived (owner: "coroutines in life").
4. wonhee says what is still missing and what to change in a photo or clip so the build works (owner).
5. Image posts and photo slides come out in the reference's text style, close but not pixel-perfect (owner).
6. If a full build is not possible, wonhee returns a partial edit with the text positions worked out, so the owner can
   paste text and finish (owner).

Non-goals: rotated text that follows lines in a photo (owner: "later"); writing the jokes (the owner supplies ideas
and phrases; wonhee may tighten wording on request (guess)); trending-audio licensing (policies.md).

Anti-goals (a failure even if it runs): a take picked by length or loudness alone; a cut that sounds spliced; text
over a face; copying a reference's footage or exact wording (policies.md: references are reference only).

## 2. Parts

**2.1 Post project (the scratchpad).** A project of kind `post` with a goal line, a format (a stored reference, or
none), and slots. Each slot has a state: `empty`, `have` (a file or text, with its source), `rejected` (with the reason).
`wonhee post status <id>` lists what is filled, what blocks the build, and the next thing the owner can do. Slots are
filled from the CLI, MCP and moeka (a photo or clip sent in Telegram lands in the selected project's matching slot). A
build is allowed when its required slots are `have`; optional slots are skipped and named in the report. (guess: the
build is one chain per format, so a slot list belongs to the format.)

**2.2 Take picker (new node `take_pick`).** Input: one or more takes of the owner talking (video). It reads the
transcript and word timings and returns ranked candidate cuts: a start and end, why it scored, and what was dropped.
It scores from measured things: stumbles and false starts (repeated words, fillers, trailing-off), dead air, reaching
the point early, and closeness of the audio to its own loudness range; **delivery and fewest stumbles first**
(owner). It never scores "energy" higher than clean delivery, because some videos are deadpan on purpose (owner). It
prefers one contiguous span; it joins spans only if the join is under a pause the existing `tighten` would keep (guess:
this keeps cuts rare). Output is a plan the owner can accept, not a rendered file; the render is the existing
`compose_video`.

*Deadpan versus dead air (my call).* The picker does not try to read intent. A pause is only penalised when it is
longer than the format's allowance (default: the `tighten` threshold, 250 ms, plus the length of a normal sentence
break), so a deliberate beat inside a line survives and a long gap does not. Rule-based first; a model judge is added
only if the owner's takes show rules failing. The owner never marks takes by hand.

*Ground truth (my call).* The accepted cuts are the owner's published reels: `content:14`, `73`, `74`, `75`, whose
final video and transcript are stored. The raw takes are on the owner's drive or flash drive; the owner supplies a
few with the final cut they chose, and `take_pick` is judged by whether its top plan lands on the same lines.

**2.3 B-roll and footage picker (extends `shot_index` / `shots search`).** Given a line or beat, rank stored shots and
new clips by fit, sharpness and framing, and show a contact sheet (`contact_sheet`). Recommends; the owner picks.

**2.4 Style reader (new node `text_style`).** From a reference image post, measure per slide: colour pair, relative
size, line count, alignment, which bundled font role is closest, box or outline or none, how much the photo was
darkened, and where the text block sits (as fractions of the frame). Stored next to `image_analysis` in a new table
with its own analyzer version. Reference images already go to the hosted vision model during `grab` (`analyze_images`),
so the font-role read follows the same path and the same cost caps; a local model is an optimisation, not a
requirement. Measured where possible (colour, size, darkening, box) and model-read where not
(font role). Output is a `TextSpec`-shaped profile, so the renderer needs no new vocabulary. From the evidence:
horizontal, bold tightly set sans, warm white or cream, no outline, some slides dimmed, mixed alignment.

**2.5 Placement (new node `text_place`).** Input: a photo, the text, the style profile. Output: the position (and
size) that puts the block in the quietest area, clear of the face and subject, inside the safe zones, with a score.
It uses the subject mask (`mask_cutout`/`edit/mask.subject`) and local contrast and brightness, and tries the reference's
own habit first (centred over a dark wall, right-aligned above the subject, a short line in empty space). It
returns top candidates, not one answer. Placement is deterministic and testable without a model.

*The bar for "close" (my call, numbers are a starting point for the builder to measure).* Per reference slide:
text colour within a small colour distance of the original, relative size within 15%, block centre within 10% of the
frame height and width, same alignment, no overlap with the face or subject. The nine slides we hold are the test
set, and the owner accepts or rejects the rebuilt versions side by side; the target is at least seven of nine
accepted. The measured numbers go to `docs/production/`.

**2.6 Fit check (new node `photo_fit`).** For each photo and its line: does a quiet enough area exist, is the photo too
busy, too bright or too tight? If not, return a concrete fix ("leave more empty space above the head", "the text will
sit over the face; use a looser shot") and mark the slot `rejected` with that reason. Pairs with `shot_guide`.

**2.7 Compose and deliver.** Existing nodes: `text_image` / `carousel` / `compose_video` render. They need the missing
knobs (font role, colour, box, tracking, exact x and y) exposed at node level; `compose` already has them.

**2.8 Chains.** `docs/chains/` gets one chain per owner format: the talking-head cut (`take_pick` → `speech_subs` →
`audio_clean` → `compose_video`), the image post (`text_style` → per photo `text_place` + `photo_fit` → `carousel`),
and the mixed reel (talking head over a held screen recording or b-roll, cut and effects only on the b-roll part,
owner). Every chain can stop after any node and return what it has.

## 3. The "last day of Claude Max" reel (the first thing to build to)

(owner) The visual is one large tmux window with six agents, each with subagents on different projects, the owner
flipping between panes; the screen recording comes later. The top part is the owner pretending to type, with a loud
continuous typing track recorded separately. The cuts and effects parody a CapCut subscription, not Claude. Real
footage of the owner gets few cuts (owner). Slots: `lines`, `selfie` (pretend typing), `typing_audio`, `screen_record`
(later), optional `voice`. It can build a preview without `screen_record` and says so. (guess: the existing storyboard
artifact is a good source of the shot list.)

## 4. Order of build (guess)

1. Post project and slot status (2.1), CLI and MCP, no model. Smallest piece, unblocks the rest.
2. `take_pick` (2.2), tested on the owner's takes with their known good cuts as ground truth.
3. `text_style`, `text_place`, `photo_fit` (2.4–2.6) on the nine reference slides, judged by eye against the originals.
4. Node-level knobs (2.7) and the chains (2.8).
5. moeka intake into slots (2.1).

## 5. Left to the builder

Scoring weights and thresholds for `take_pick` and `text_place` (measure on the owner's own takes and the reference
slides; record the numbers in `docs/production/`); table layouts; whether the post project reuses `project`/`intent` or
adds a table; model choice for the font-role read (cost first, local where possible); CLI wording.

## 6. Open for the owner

1. How far should wonhee go on wording: only tighten what the owner wrote, or offer alternatives in the reference's
   voice when asked? (guess: only when asked.)
2. Should the owner approve a take plan before the render, or should wonhee render the top plan and show it? (guess:
   render the top plan; the plan is cheap to redo.)

Decided as my call: "ours or external" is set per post, as `grab` does today.
