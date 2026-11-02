# Post assistant

The post assistant is the glue for the owner's short posts: a post project with named slots, a take picker, a
style reader for reference image posts, a placement step and a fit check. Design: [post assistant spec](../specs/2026-10-26-post-assistant-design.md).

**Every number on this page is a starting number.** The weights and thresholds below are copied from the source
constants and have **not yet been tuned** on the owner's own takes or on the reference slides. The only measurement
here is the synthetic text-style check in §4, and it covers five generated frames, not real posts. Tune before
trusting a verdict, and record the change in this file.

## 1. What exists

| Piece | Where | Runs as |
|---|---|---|
| Post project and slots | `wonhee/creative.py` (`post_create`, `post_slot_set`, `post_status`, `POST_FORMATS`) | `uv run wonhee creative <action> --data '{...}'` |
| Take picker | `wonhee/edit/takes.py`, node `take_pick` | `uv run wonhee node run take_pick` |
| Style reader | `wonhee/edit/textstyle.py`, node `text_style` | `uv run wonhee node run text_style` |
| Placement | `wonhee/edit/place.py`, node `text_place` | `uv run wonhee node run text_place` |
| Fit check | `wonhee/edit/fit.py`, node `photo_fit` | `uv run wonhee node run photo_fit` |
| Render knobs | node `text_image` (`TextParams`), node `carousel` (`text_style`) | `uv run wonhee node run ...` |

No node needs a GPU. `text_place` and `photo_fit` load the person mask model when present (§3, flag
`NO_SUBJECT_MASK`).

## 2. Using a post

### 2.1 Create the post and fill slots

`post_create` makes the project and seeds every slot for the format as `empty`. It returns the `id` you pass to
the other actions. Calling it again with the same name, format and goal returns the same project; a different
format or goal for the same name is refused.

```bash
uv run wonhee creative post_create --data '{"name":"last-day-of-claude-max","format":"mixed_reel","goal":"<one line: what the post says>"}'
# -> {"id": 1, "name": "last-day-of-claude-max", "format": "mixed_reel", "goal": "..."}

uv run wonhee creative post_slot_set --data '{"project_id":1,"slot":"lines","state":"have","text":"last day of claude max","source":"owner"}'
uv run wonhee creative post_slot_set --data '{"project_id":1,"slot":"selfie","state":"have","ref":"data/takes/selfie.mp4","source":"owner"}'
uv run wonhee creative post_slot_set --data '{"project_id":1,"slot":"selfie","state":"rejected","reason":"the face is under the top bar; reshoot with more headroom"}'
uv run wonhee creative post_slot_set --data '{"project_id":1,"slot":"selfie","state":"empty"}'
```

Rules the code enforces:

- `state` is `empty`, `have` or `rejected`.
- `have` needs a `ref` (a file or stored reference) or `text`. The code stores the string as given and does not check
  that the file exists.
- `rejected` needs a `reason`. `empty` clears `ref`, `text` and `source`.
- Slot names come from the format. Any other name is an error.
- Field limits: `ref` and `source` 2000 and 500 characters, `text` 20000, `reason` 2000, `name` 200, `goal` 2000.

Formats and their slots (`POST_FORMATS`; `True` = required for a build, `False` = optional):

| Format | Slots |
|---|---|
| `talking_head` | `take` (required), `lines` (optional) |
| `image_post` | `lines` (required), `photos` (required) |
| `mixed_reel` | `lines` (required), `selfie` (required), `typing_audio` (required), `screen_record` (optional), `voice` (optional) |

### 2.2 Check what is still missing

```bash
uv run wonhee creative post_status --data '{"project_id":1}'
```

The output is one **JSON string** (the CLI prints the returned text JSON-encoded), so it appears with `\n` escapes
and quotes. Decode it to read it, for example `... | python3 -c 'import json,sys; print(json.load(sys.stdin))'`.
Its lines are:

- `Have:` each filled slot and where it came from.
- `Rejected:` each rejected slot with its reason. Any rejected slot blocks the build, required or not.
- `Missing required:` required slots still `empty`. These block the build.
- `Skipped optional:` optional slots still `empty`. A preview can build without them.
- `Build: allowed` or `Build: blocked`, then one `Next:` line.

For a `mixed_reel`, a preview without `screen_record` is allowed (spec §3); `screen_record` is optional here for that reason.

### 2.3 Run the nodes

Nodes run one at a time with `--in port=ref|path` (repeatable) and `--set k=v` (repeatable). A `--set` value is
parsed as an int, then a float, then `true`/`false`, then a JSON list or object, and otherwise kept as a string.
Nested knobs take a JSON object. `uv run wonhee node list` shows every node's ports, and `node show <name>` shows its
parameter schema.

```bash
# take_pick: subtitles of each take (a speech_subs output, one port per take, in order)
uv run wonhee node run take_pick --in subtitles=data/takes/take1.subs.json --in subtitles_b=data/takes/take2.subs.json \
  --set top_n=3 --set 'loudness_db=[4.2, null]'

# text_style: a reference slide, with the font role read by eye (role_hint must be a key of edit/text.py ROLES)
uv run wonhee node run text_style --in image=data/refs/slide1.png --set role_hint=sans

# text_place: where a line goes on a photo; mask=off skips the person mask
uv run wonhee node run text_place --in image=data/photos/desk.jpg --set text="still shipping at three am" --set preset=outline --set n=3

# photo_fit: verdict and fixes for one photo and line
uv run wonhee node run photo_fit --in image=data/photos/desk.jpg --set text="still shipping at three am" --set slot=slide-2
```

`take_pick` returns JSON text, not a video (§5). `text_place` and `photo_fit` return JSON text too. `photo_fit` does
not touch the post. When its verdict is `reject`, the caller sets the slot:

```bash
uv run wonhee creative post_slot_set --data '{"project_id":1,"slot":"slide-2","state":"rejected","reason":"<the fix from photo_fit>"}'
```

`text_style` also writes a row to the `text_style` table (key: image sha, analyzer version 1). Its profile is
`TextSpec`-shaped, so you can copy the fields into a render; §5 lists which ones have no knob yet.

For a render, pass the knobs as a JSON object on the `carousel` node:

```bash
uv run wonhee node run carousel --in images=<images ref> --set text_preset=outline \
  --set 'text_style={"align":"left","box":"none","stroke_em":0.0,"at":[0.5,0.7],"colours":["#f5ecd0"]}'
```

`text_image` takes the same `TextStyle` fields as top-level `--set` knobs (it inherits them), plus `preset`, `position`,
`size_frac` and `width_frac`. Only `carousel` nests them under `text_style`.

## 3. The numbers

All values are from the source. Nothing here is tuned on the owner's takes or the reference slides.

### 3.1 `take_pick` (`wonhee/edit/takes.py`)

Score is points (higher is better). Each term is signed; stumbles dominate, so one stumble costs more than a second of
dead air.

| Constant | Value | Meaning |
|---|---|---|
| `STUMBLE_WEIGHT` | repeat 10, false_start 10, trail_off 10, filler 8, like 4 | per stumble word kept |
| `DEAD_AIR_PER_SEC` | 8.0 | per second of silence past the allowance, inside a kept run |
| `DEAD_AIR_ALLOWANCE_MS` | 650 (= `TIGHTEN_MIN_PAUSE_MS` 250 + `SENTENCE_BREAK_MS` 400) | a pause up to this is not penalised |
| `PREAMBLE_WEIGHT` | 2.0 | per throat-clearing word before the content ("so", "okay", "um") |
| `LEAD_IN_PER_SEC` / `LEAD_IN_GRACE_MS` | 2.0 / 500 | per second of lead-in beyond 500 ms |
| `LOUDNESS_PER_DB` / `LOUDNESS_GRACE_DB` | 0.3 / 6.0 | per dB of loudness spread beyond 6 dB; skipped when no spread is given |
| `SPLICE_WEIGHT` | 12.0 | per join; a splice must beat a clean span by a wide margin |
| `COVERAGE_WEIGHT` | 20.0 | times the share of the take's words kept |
| `MIN_SPAN_MS` | 1500 | shorter plans are not usable unless the whole take is shorter |
| Join room | 40 ms (`TIGHTEN_KEEP_MS` 80 / 2) of silence on each side of a cut | a splice is only offered when both sides have this room |
| Edge padding | 40 ms (`TIGHTEN_KEEP_MS` / 2) on each side of a span | |
| `top_n` (node) | default 3, range 1–20 | plans returned |

Filler words: um, uh, uhm, umm, er, erm, ah, hmm, mm, mmm; plus "know" after "you". Preamble words: so, okay, ok,
alright, anyway, well, hey, hi, right, basically. A trailing "..." or "…" marks trail-off; a trailing dash marks a false
start. The `TIGHTEN_*` values are copies of `nodes.compose.Tighten`'s defaults (a test pins them).

`loudness_db` is an input the caller must supply (§5). Nothing in the codebase produces it.

### 3.2 `text_place` (`wonhee/edit/place.py`, node `text_place`)

Position score = 1 − cost, clipped to 0..1. Cost is the sum of:

| Constant | Value | Term |
|---|---|---|
| `W_BUSY` | 1.0 | busyness of the box: 0.5 × local luma std (full at 50) + 0.5 × edge density (full at 0.15) |
| `W_SUBJECT` | 3.0 | fraction of the box covered by the subject mask |
| `W_VIOLATION` | 4.0 | how far the box leaves the safe band (`SAFE_TOP` 0.12, `SAFE_BOTTOM` 0.76) or the side margin (`MARGIN` 0.04) |

Habit bonuses, only when the box is inside the safe zone and clear of the subject: `centred_dark` 0.08 (centred within
0.03 of the middle, mean luma below 0.38, busyness below 0.25), `right_above_subject` 0.08, `short_in_empty` 0.06
(one line, busyness below 0.08, width below 0.6).

Search: grid step `STEP_X` 0.025, `STEP_Y` 0.02; analysis at 256 px on the long side; candidates must overlap at most
`MAX_IOU` 0.2 with a better one. Tall frames (h/w ≥ `TALL` 1.7) avoid the like/comment strip (`BUTTONS_Y` 0.35–0.88,
edge `BUTTON_EDGE` 0.87) and re-wrap to `BUTTONS_WRAP` 0.72 where a row reaches it. The node defaults are preset
`outline`, `width_frac` 0.84, `n` 3, `mask` auto. `LOW_SCORE` 0.5 raises `NO_QUIET_AREA`.

### 3.3 `photo_fit` (`wonhee/edit/fit.py`, node `photo_fit`)

The verdict uses the best position `place` finds.

| Constant | Value | Use |
|---|---|---|
| `OK_SCORE` | 0.5 | best score at or above this is `ok` |
| `TIGHT_SCORE` | 0.3 | best score from here to 0.5 is `tight`; below it is `reject` |
| `SUBJECT_OVERLAP` | 0.02 | best spot covering more of the subject than this is a hard failure |
| `SUBJECT_AREA` | 0.55 | a subject covering more of the frame than this is a hard failure |
| `HEAD_HIGH` | 0.35 | subject top above this (fraction of height) gets the "more space above the head" fix |
| `BUSY_FIX` | 0.5 | best-box busyness above this is named as a failure (soft) |
| `LIGHT_LUMA` | 0.5 | text fill luma above this is "light text" |
| `BRIGHT_MAX` | 0.7 | light text on a frame brighter than this is a hard failure |
| `DARK_MIN` | 0.3 | dark text on a frame darker than this is a hard failure |

A `reject` verdict comes from any hard failure or a score below 0.3. A `tight` verdict is a score from 0.3 to 0.5 with no
hard failure. Soft failures (busy, block leaves the safe area) add a fix but do not reject by themselves. Fixes are
returned only when the verdict is not `ok`.

### 3.4 `text_style` (`wonhee/edit/textstyle.py`, node `text_style`)

| Constant | Value | Use |
|---|---|---|
| `LONG_SIDE` | 1000 | analysis size; outputs are fractions of the frame |
| `TOPHAT_MIN` | 50 | stroke contrast after a 5×5 blur (0..255) |
| `LIGHT_MIN` / `DARK_MAX` | 150 / 105 | absolute luma for a light / dark stroke |
| `HEIGHT_PER_EM` | 0.74 | p90 glyph height ÷ font size |
| `MIN_COMPONENTS` | 3 | glyph-sized strokes needed to call it text |

Method: top-hat (light text) or black-hat (dark text) on the grey image removes anything wider than a stroke. Glyph-sized
components are kept (height from 0.8% of the frame, width at most 4× height, area fill at least 12%). Components of the
commonest height (0.7–1.45× that height) are clustered by proximity. The heavier cluster wins; when both polarities fire,
light glyphs sitting inside dark boxes mean light text. Size is the p90 glyph height ÷ 0.74. Lines are a 1-D split of
glyph centres at 0.6 em. Fill is the median of the brightest (or darkest) interior strokes. A box is a flat modal colour
in the padded block; an outline is a uniform thin ring that differs from the photo and the fill. Dimming is
`1 − median luma / 128` for the frame, and a local term for the area around the block (when there is no box).

Alignment: with two or more lines, the edge (left, right or centre) with the smallest spread wins if it spreads at most
0.12 of the block width; otherwise the block is called by where it sits. With one line the alignment is a guess
(`align_certain` false), which is recorded but **not flagged**.

**Measured error on synthetic frames.** The five cases in `tests/test_text_style.py` (text drawn with `edit/text.py` on
gradient and noise backgrounds) were measured on 2026-10-27 with a scratch script (not committed). The test
tolerances are the ones in parentheses.

| Case | Fill (max channel) | Size error | Centre dx / dy | Width / height error | Lines and alignment |
|---|---|---|---|---|---|
| white outline centred | 0 | 9.0% | 0.0019 / 0.0010 | 0.0148 / 0.0083 | match |
| cream no outline left | 0 | 5.5% | 0.0018 / 0.0010 | 0.0000 / 0.0000 | match |
| white on dark box | 0 | 5.0% | 0.0019 / 0.0005 | 0.0370 / 0.0323 | match |
| right aligned outline | 0 | 3.3% | 0.0009 / 0.0010 | 0.0167 / 0.0083 | match |
| big yellow caps | 0 | 10.2% | 0.0027 / 0.0016 | 0.0241 / 0.0135 | match |
| **Worst** | **0** (≤ 12) | **10.2%** (≤ 15%) | **0.0027** (≤ 0.10) | **0.037 / 0.032** (≤ 0.05) | all match |

This is a check on clean generated frames only. A real reference slide with a busy background, a photo filter or
a font other than the bundled roles is not measured. The font role is never measured (only `role_hint`).

### 3.5 Flag codes

| Code | Raised by | When | Migration |
|---|---|---|---|
| `NO_SUBJECT_MASK` | `text_place`, `photo_fit` | `mask=off`, or the person model fails to load; the placement or verdict was not checked against the subject | `0026_place_flags.sql` |
| `NO_QUIET_AREA` | `text_place` (score < 0.5), `photo_fit` (score < 0.5 or none) | the best spot is still busy or crowds the subject | `0026_place_flags.sql` |
| `NO_TEXT_FOUND` | `text_style` | no text block; every profile field is null | `0027_text_style.sql` |
| `TEXT_STYLE_UNSURE` | `text_style` | the measured line count differs from the stored on-screen text's line count (the only case that raises it) | `0027_text_style.sql` |
| `WORDS_ESTIMATED` | `take_pick` | subtitles carry line timing only; words are timed evenly within each line, so stumble and dead-air timings are approximate | `0018_words_estimated.sql` |
| `NO_TEXT` | `take_pick` | no spoken words in any take | `0006_nodes.sql` (generic) |

## 4. Known gaps

- **`align` has no visible effect on `text_image` or `carousel`.** `draw_on` uses `render_layer`, which always centres
  each line (or, with `at` set, centres the block on `at[0]`). `align` only reaches `render_block` (the compose path).
  `TextStyle.align` says the same: "text_image lines are centred".
- **`take_pick` cannot feed `compose_video` yet.** Its output is a JSON text plan. `compose_video` takes `video`,
  `subtitles` and `images` only. Someone must read the plan and pass the spans to `compose_video` by hand (the spans are
  on the take's own timeline).
- **No `loudness_db` producer.** `take_pick` takes a per-take loudness spread as a parameter. Nothing in the code
  measures it, so the loudness term is always skipped by default.
- **Font role only via `role_hint`.** `text_style` never measures the font; the caller must supply one of the `ROLES`
  keys (for example `sans`, `editorial`, `headline`). `role_hint` is read by eye or by a vision model; no code path
  does the vision read yet.
- **Brightness limits are unmeasured.** `BRIGHT_MAX` 0.7 and `DARK_MIN` 0.3 are constants in `edit/fit.py`. No measurement
  is cited for them in the code or in this file.
- **`post_status` prints JSON-quoted on the CLI.** The action returns plain text, and the CLI prints it through
  `json.dumps` (§2.2). MCP and Python callers get the text itself.
- **No face detector.** The "face" check is the person segmentation mask (`u2net_human_seg`, `mask.subject(...,
  "person")`) with its overlap rules. A mask can miss a face or cover a shoulder that is not the face. It is a proxy.
- **Profile sizes do not map one to one.** `text_style` returns `size_frac` and `width_frac`, but the `TextStyle` knobs on
  `carousel` have no `size_frac`, `width_frac` or `preset` field, so those fields must be passed through `text_image`
  (`size_frac`, `width_frac`) or set by hand. `at` is a `TextStyle` field and works on `carousel`.
- **Slot files are not checked.** `post_slot_set` stores the `ref` string. A missing file is found only when a node reads it.
- **Moeka intake into slots** (spec §2.1, §4 step 5) is not built in the files read for this page; not verified.
- **No post chains.** Spec §2.8 names chains under `docs/chains/`; none for the post formats was found in the files read.
- **The spec's ground-truth check (§2.2, §4 step 2) is not done.** `take_pick` has not been run on the owner's takes, so
  no claim is made about how often its top plan matches a known good cut.
