# Meme reel styles: `meme`, `cutaway`, `hack`, and the layers they need

Status: **proposed** 2026-10-22. This is a new contract (new styles, two new layer kinds, a second video input), so it
is **large lane**: the owner approves this spec before any code (PROCESS §2). Stage 4 (make).
Evidence: the inspo library, `content:2`–`65` in the database (third-party, reference only). The cards for each reel
are in the inspo-library artifact; the ids below point at them. Builds on the approved
[reel styles spec](2026-10-20-reel-styles-design.md) and the [layouts spec](2026-10-20-meme-and-popart-formats-design.md).

## 0. Why

The first study (five reels, `content:48`–`52`) covered voice-led explainers and lists. The 13 reels grabbed
2026-10-21/22 and the re-read older set are a different class: **text-and-visual jokes, 6–14 s, no voice**, and they
have the highest reach in the library. Our six styles all assume a talking clip. This spec adds the three formats
that the evidence shows we cannot make today, and nothing else.

What the evidence says, in the terms the styles need:

1. **The text is the joke.** One line, in the top third or centre, readable in under 1.5 s, present from frame 0.
   Seven of the 13 new reels are a single held line (`54`, `55`, `56`, `57`, `58`, `59`, `60`).
2. **Over-specific sentence.** The length and the real-sounding technical detail are the punchline (`54`, `65`).
3. **Cutaway to a borrowed clip** on a matched gesture: self, then a meme clip, then self (`55`, `56`, `58`, `59`; `29`
   with no caption at all).
4. **A fake UI card that changes at each beat** (`45`, a PR card whose diff size climbs), and **a red circle or arrow**
   on a detail (`40`, `63`).
5. **Format-swap parody:** a known script with one noun changed (`23`, `45`, `65`). That is a writing concern (§6),
   not a rendering one.
6. **Short and silent.** The audio on all of these is trending in-app audio or nothing; none carries speech.

## 1. New layer kinds

### `callout`

`{type: "callout", kind: "circle" | "arrow" | "both", target: {x, y, r}, from_deg, colour, stroke_em, t, anim}`

A red annotation drawn over the footage (`40`: circle plus arrow on a small object; `63`: arrow at an inset). `target`
is a centre and radius as fractions of the canvas (`r` of the short side). `from_deg` is the direction the arrow
comes from (default 225, upper left). Defaults: colour `#e5251f`, stroke 0.7% of the short side, round caps, an
anti-aliased Pillow render at 3× supersampling (as shapes do). `t` and `anim` are the existing timed-layer fields;
`anim` may be `none` or `pop`. Safe zones apply to text only, so a callout may sit anywhere on the footage.

### `ui_card`

`{type: "ui_card", kind: "pr" | "prompt" | "lines", box, steps: [{at_ms, ...}], font_role, t}`

A fake UI inset that changes at beat times (`45`, `58`). The card is generic: a dark panel, a status dot row, mono
text. It carries **no brand, no logo and no real repository or company name** (VISION: no faking real things); a
named product is refused with `UI_CARD_BRAND`.

| kind | Fields per step | Shows |
|---|---|---|
| `pr` | `title`, `added`, `deleted` | a pull-request row: title, `+added −deleted`, five status dots |
| `prompt` | `text`, `choices` (default `Y/n`) | a CLI prompt line with the choices, a blinking-cursor state |
| `lines` | `lines[]` | up to 6 mono lines, one per step if `reveal: line` |

States change only at `at_ms` times (the word-driven-layer rule), so a 12 s card with 5 steps is 5 PNGs. The card is
drawn by Pillow and obeys the 12–76 % vertical text margin, inside the centred 72 % where text is.

### No new `stamp` kind

The clock jump in `53` (`00:49`, `00:50`) is two timed `text` layers in `mono`. It needs no new kind. A `pov` style
that wraps it is **deferred** (§8).

## 2. A second video input and a cut list

`compose_video` has one `video` input and an `images` list for insets. `cutaway` needs a second clip as a first-class
source.

- New optional input port `clip2: video?`. A `photo` layer may take `src: "clip2"`.
- New template-level param `cuts: [{src: "clip" | "clip2", in_ms, out_ms}]`. The output video track is the
  concatenation of these segments, each scaled to cover the canvas window. The output duration is the sum of
  `out_ms − in_ms`. Timed layers use the **output timeline**.
- New param `audio: "clip" | "clip2" | "none"`. **Default `none` for `meme` and `cutaway`** (the reference reels carry
  in-app audio, added when posting); `clip` for `hack`. A render with `none` says so in the report, so the missing
  sound is a decision and not an accident.
- It stays **one encode** (policies rule 1): segments are `trim`/`concat` inside the same filter graph, as `tighten`
  already does.
- A segment longer than its source loops that source; the report lists `LOOPED`.

## 3. The styles

All three skip `speech_subs` (a template flag `reel: {speech: false}`), since there is no speech to caption. Text is
`compose`'s existing typography, drawn by Pillow, inside the safe zones.

### `meme` (from `54`, `60`, `65`, `15`, `37`; the format-swap lines of `23`)

One image or clip, one line.

- Input: one image or clip, `--title "the line"` (required, ≤ 400 characters).
- `variant: pill | centred | claim`:
  - `pill` (`60`): a white rounded box with black `ui` text in the upper third (y 0.11–0.30).
  - `centred` (`54`, `65`): white `ui` text with a soft dark stroke, centred, y 0.36–0.62, no box.
  - `claim` (`37`): for the first `claim_ms` (default 1500) a flat colour card carries the line, then the media is
    revealed with the line gone. A second `--reveal` line is optional.
- Length: a clip is trimmed or looped to `dur_s` (default 8, max 12). An image holds for `dur_s` with an optional 4 %
  slow zoom (`zoom: true`).
- Fit: the text shrinks to fit its box down to `size_frac` 0.028; below that the render is refused with
  `TEXT_TOO_LONG` (the line is cut by the owner, not by us).
- The text is on screen from frame 0 (hook rule from the first study).

### `cutaway` (from `55`, `56`, `58`, `59`; `29` with `--title` omitted)

Self, then a borrowed reaction clip, then self.

- Inputs: the self clip, `--with <clip>` (the borrowed clip), `--title "the line"` (optional; omit for `29`).
- The line uses the `meme` variants, held over **every** segment.
- `--cuts "self:0-1.5, with:1.5-4.5, self:4.5-8"` (seconds, `self` = `clip`, `with` = `clip2`). The default is
  `self` 0–35 %, `with` 35–75 %, `self` 75–100 % of `dur_s`. **The gesture match is the owner's job**: the cut points
  are given, never guessed. The evidence is that the match is what makes the cut work, and we can't see a gesture.
- **Third-party clip rule:** the borrowed clip is third-party media. It stays under `data/`, never in git, and the
  report carries `THIRD_PARTY_CLIP`. We make no claim about rights; the owner supplies and owns the choice.
- Audio: `none` by default. `--set audio=clip` keeps the self clip's audio over the whole timeline.

### `hack` (from `64`: a deadpan "life hack" parody)

A 20–60 s reel in the satirical engagement-bait format.

- Layout: the `series` split. A face clip on top, the process footage below (`clip2`, or an `images` list), a
  `headline` all-caps title on a band between them, `caption` chunks low, one encode, `tighten` on.
- Inputs: `--title "HOW TO ..."`, the face clip, `--with` the process clip, `--cta "comment your ..."`.
- The closing card is the comment question (`64` still pulls 566 comments). It is part of the joke and is in the
  template as `cta`.
- Audio: `clip` (the face clip's voice).
- Since this reuses the `series` split geometry, the template extends it and does not copy it. If the owner prefers,
  `hack` can be `series --set variant=hack` instead (open question 4).

## 4. The command

`wonhee reel <media> --style meme|cutaway|hack [--title "..."] [--with CLIP] [--cuts "..."] [--callout "x,y,r[@a-b]"]
[--card "pr:title,+12,-3@2s"] [--cta "..."] [--set k=v]`.

- `--title` is **the line**. One flag for the thing everyone calls a caption, a title or a line.
- `--callout` adds a `callout` layer (`kind: both`) to **any** style, over the time `a`–`b` seconds (default the whole
  reel). Several may be given.
- `--card` adds a `ui_card` step. Repeating it adds steps at their times.
- `reel styles` lists nine styles. MCP: `reel(...)` gains `title`, `with`, `cuts`, `callout`, `card`; `reel_styles()`
  lists nine. No new tools.

## 5. Looks

None new. The existing `bw_grain` (`62`) and `lofi_cam` apply to the clip as before, inside the one encode.

## 6. The writing side (optional, last)

The over-specific sentence (`54`, `65`) and the format swap (`23`, `45`) are writing, not rendering, and the `larp`
node already frames the mundane as a professional moment. Proposal: a `larp` register `caption` with `format:
overspecific | swap | pov`, writing one `line` for `meme`. Rules, as for `lecture`: the form is exaggerated, the
subject is trivial or technical, and no line claims that a real person, company or event did a thing (VISION: no
faking real events). Naming real tools and model names is fine; asserting a real person's actions is not. A code
check refuses a real-person claim, with one retry. This is a separate, later step and ships only with the owner's
yes (open question 5).

## 7. Tests (from this text)

- `callout`: the rendered state has red pixels on the circle's ring within ±1 px of `r`, and none outside the stroke;
  a callout with `t` is absent outside its window. Anti-aliased edges (alpha between 0 and 255 exists).
- `ui_card` `pr`: states change only at `at_ms` times; the number of PNG states equals the number of steps; the text
  inside the safe margin; a brand name (`GitHub`) in a title is refused with `UI_CARD_BRAND`.
- `cuts`: a three-segment cut list gives an output duration equal to the sum, within one frame; every timed layer is
  on the output timeline; exactly one ffmpeg encode (count subprocess calls) for `meme`, `cutaway` and `hack`.
- `audio: none` gives an output with no audio stream and a report line; `clip` keeps the audio.
- `meme` refuses a 500-character line with `TEXT_TOO_LONG`; a 240-character line renders with every text box inside
  12–76 % height and the centred 72 % width.
- `claim`: the first `claim_ms` show the flat card and no media; afterwards the media and no claim text.
- `wonhee reel tiny.mp4 --style X` works for all nine styles on the `tiny_video` fixture (`--with` the same fixture
  for `cutaway` and `hack`), and `reel styles` lists nine.
- A `cutaway` render with a second clip puts `THIRD_PARTY_CLIP` in the report, and the repo's `.gitignore` check still
  passes (no media under tracked paths).

## 8. Deferred (not in this spec)

- `pov` timelapse (`53`): timed `text` layers already do the clock stamps. A style wrapper comes when an example is
  filmed.
- The callout-list (`62`): a `list` variant over a `bw_grain` clip, with a counter. Needs a counter layer; later.
- Kinetic-lecture upgrade (`63`): one-word-per-beat `stack` plus a portrait inset and a `callout` arrow. Revisit once
  `callout` exists, to see whether the `lecture` style should gain them.
- Automatic cut-point or gesture matching for `cutaway`.
- Event-edit pacing (`61`): needs real event footage.

## 9. Order

1. `callout` layer (smallest, used everywhere).
2. `ui_card` layer.
3. `clip2` input, `cuts`, `audio` in `compose_video`.
4. The `meme` style and `reel` flags (`--callout`, `--card`).
5. The `cutaway` style.
6. The `hack` style.
7. `larp` register `caption` (optional, with the owner's yes).

Each step is followed by its alignment check by a different agent (PROCESS §3).

## 10. Open questions for the owner

1. **Second video input and `audio: none` by default** for `meme` and `cutaway`. The reference reels have in-app audio.
   Do you want it silent (add the sound when posting) or the self clip's audio kept?
2. **Borrowed clips.** Is a locally stored, never-committed third-party clip fine for `cutaway`, with only a
   `THIRD_PARTY_CLIP` flag?
3. **`ui_card` scope.** Three kinds (`pr`, `prompt`, `lines`) enough, or only `pr` and `prompt` first?
4. **`hack`** as its own style, or `series --set variant=hack`?
5. **The `larp` `caption` register** (§6): build it with this, later, or not at all?
6. **No-footage fallback.** The owner has no talking-head video in the database yet. All three styles accept any
   image or clip; should the first live check use a still of you with `meme` (the photo you just sent works)?
