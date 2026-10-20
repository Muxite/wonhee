# Reel styles: timed layouts, word-driven text, fonts, and one command

Status: **approved** 2026-10-20 (the owner asked for the study to be turned into easy-to-use capabilities on main,
design first). Stage 4 (make). Evidence: [reel-study-2026-10-20.md](../production/reel-study-2026-10-20.md).
Builds on the approved layout engine ([layouts spec](2026-10-20-meme-and-popart-formats-design.md)): this spec
adds **time** to it, so one engine makes stills and reels.

## 1. Word timings in `subtitles`

Each subtitle line may carry `words: [{start_ms, end_ms, text}]`. `speech_subs` fills them from whisper's word
timestamps (`word_timestamps=True`; same model, so the node gets a version bump). Lines without `words` (older cache,
`screen_text`, translations) are given evenly split timings by the consumer, which flags `WORDS_ESTIMATED`.
Translation keeps the line timings and drops `words` (word order changes). The analyze stage saves word timings with each
stored transcript segment from now on. Transcripts stored before this have none, and consumers estimate them;
nothing is re-analysed.

## 2. Timed layers (the compose engine, extended)

Any `Layout` layer may have `t: {in_ms, out_ms}` (absent means the whole duration) and `anim`:

| `anim` | What |
|---|---|
| `none` | hard in and out (the default) |
| `fade` | opacity ramp, `dur_ms` (default 250) |
| `pop` | scale 0.85 → 1.06 → 1.0 over 300 ms (c48's headline) |
| `type` | characters appear at a steady rate across `dur_ms`, with an optional cursor (c50) |
| `rise` | slides up 4% of the height while it fades in |

Three **word-driven** layer kinds read the `subtitles` input:

| Kind | What | Params |
|---|---|---|
| `captions` | 1–`max_words` (default 3) words at a time, chunked on pauses and punctuation. Modes: `chunk` (the whole chunk shows), `word` (words appear one by one), `karaoke` (the whole chunk shows, the active word sits on a coloured pill) | font role, size, colours, box (none / box / pill), highlight colour, case |
| `stack` | kinetic text: one word per line, building up then clearing on a phrase gap (c51) | font role, size, tracking, max lines |
| `list` | a scrolling list: one entry per line, the active entry bright and its neighbours at 25% and 15%, scrolling one line per entry (c52). Entries come from a `items` param with optional `at_ms`, else from the words | font role, size, window lines, keyword colours |

All text obeys the safe zones (DESIGN Stage 4) and is drawn by Pillow (colour emoji, CJK).

**Rendering** (`frame_video`, generalized; the node is renamed `compose_video` and keeps `frame_video` as an alias
template call): the canvas is the video's duration. The clip goes into its window (`photo` layer with `src: video`).
Timed layers are pre-rendered as PNG states, which change only at their own event times, so a 60 s reel with 120
caption changes is about 120 small PNGs, not 1,800 frames. They're overlaid with `enable='between(t,a,b)'` in **one**
ffmpeg filter graph and **one encode** (policies rule 1). Animations with motion (`pop`, `rise`, `type`) render a
few intermediate states (≤ 10 per animation).

**Insets** (c50, c51): a timed `photo` layer whose `src` is an image or a clip from the node's `images` input, with
shape, outlines, `anim`. A **split** layout (c51) is a template with a top window for insets and the talking head
below.

**Amendments from the wave-2 alignment check (2026-10-20):**
- The text background on layers is `bg_box: none | box | pill | paper` with `bg_colour`, `bg_pad_em` and
  `bg_radius_em`, because `box` is a layer's geometry. `paper` on captions is the same ragged strip as in `text.py`.
  A backing taller than its layer grows the layer image rather than clipping.
- `anim` values are entrances. A layer's exit is a hard cut at `out_ms`. On the word-driven layers (`captions`,
  `stack`, `list`), `t` clips their states and `anim` must be `none`, or the layout is refused.
- `list` scrolls smoothly: `scroll_ms` (default 470) renders up to 8 intermediate offset states per entry. These are
  the only states allowed between entry times.
- The clip (`src: "clip"`, alias `"video"`) is the bottom of the stack. Every other layer, static or timed, draws
  over it in list order. With no clip layer the clip fills the canvas (cover) and `canvas.bg` is unused. `shape`,
  `rotate` and `outlines` on the clip layer are refused.
- Insets are images from `images`. Clip insets need a second video input and come later.
- A reel may have up to 1,500 overlay states. That's measured: about 0.03 s per state, with the encode dominating.

## 3. Pause trimming inside a take

`compose_video` param `tighten: {min_pause_ms: 250, keep_ms: 80}`: pauses in the speech are cut out (from the word
timings, else from `silencedetect`), inside the same filter graph (`trim`/`atrim` + `concat`), so there's still one
encode. Every timed layer and caption is re-timed to the cut timeline. Off by default; on in the talking-head styles.
The report lists the seconds removed.

Amendments (wave-2 check):
- Only pauses *between* words are cut. Leading and trailing silence are left alone.
- Chunking and phrase gaps are decided on the source timeline, before cutting, and then mapped through the cuts.
- Cut points snap to the video frame grid, and audio and video are cut together, so there's no A/V drift.
- Trimmed audio is re-encoded (AAC). Untrimmed audio is copied.
- `tighten` expects speech word timings (`speech_subs`). Without words, cuts come from a decode-only
  `silencedetect` pass (`-vn`, not an encode), so the one-encode rule still holds. A clip with no audio is left
  untrimmed and flagged.
- Cuts never go past the clip's duration.

## 4. Fonts and typography

Bundle these OFL fonts under `wonhee/assets/fonts/` with their licence files: Barlow Condensed SemiBold, Tinos Bold,
Instrument Serif (Regular, Italic), Inter, Inter Tight, Figtree, Poppins SemiBold, Fraunces, Caveat Brush,
Gochi Hand, Mansalva, Arimo, Press Start 2P. Variable fonts are used at named instances.

Text gets **font roles** so templates don't name files: `headline` (Barlow Condensed), `transcript` (Tinos Bold),
`editorial` / `editorial_italic` (Instrument Serif), `grotesk` (Inter Tight), `ui` (Inter), `caption` (Figtree
600), `caption_round` (Poppins), `soft_serif` (Fraunces SOFT 100), `marker` (Caveat Brush), `marker_alt` (Gochi
Hand), `hand` (Mansalva), `autocaption` (Arimo), `pixel` (Press Start 2P), plus the existing `sans`, `mono` and
`script`. Text layers gain `tracking_em` (−0.1..0.2), `line_height` (0.8..1.6) and a background `box: none | box |
pill | paper` with colour, padding and radius. The existing presets (`outline`, `band`, `script`, `sticker`) keep
working.

Amendments from the wave-1 alignment check:
- Text has `stroke_em` (0..0.15). When absent, the preset's own stroke is used. Any `box` other than `none` defaults
  it to 0. Clean titles (`headline`, `editorial`, `ui`, the lecture captions) set it to 0.
- Weights are roles: `ui` is Inter Bold and `ui_medium` is Inter Medium. `caption` is Figtree SemiBold and
  `caption_bold` is Figtree ExtraBold. `numeral` is Fraunces Black (SOFT 0, opsz 144) for chapter tiles.
- The new typography params live on `compose` layers. The older `text_image`/`text_video` nodes keep their schema.
- A missing bundled font is a doctor warning that says the package files are missing (not an apt command).
- Licence files open with the font's own copyright line (name ID 0).

## 5. Looks

New look preset `lofi_cam`: muted (saturation about 0.55), soft (a slight blur), a strong vignette, a magenta-tinted
shadow lift, and a little grain. The look applies to the clip only, never to text layers.

## 6. Styles (templates) and the one command

A **reel style** is a JSON template for `compose_video` (the same folder as the still templates) plus default
params. Six ship:

| Style | From | Layout |
|---|---|---|
| `legend` | c48 | black canvas, clip window y 0.22–0.74, `headline` top band (pops in at 1.6 s), `karaoke` captions in `transcript` serif on a dark box with a lilac pill |
| `explainer` | c49 | full-bleed clip, `editorial` headline for the first 4.7 s, `caption` lowercase chunks at y 0.72, `tighten` on |
| `scrapbook` | c50 | full-bleed clip, `soft_serif` headline on a `paper` box, `word` captions in `caption_round`, numbered `marker` chapter tiles, insets |
| `series` | c51 | split screen (top inset window, speaker below), `grotesk` kinetic `stack` hook, `autocaption` subtitles in a box, a series title card with an `editorial` + `marker_alt` mix and a sticker burst |
| `list` | c52 | near-black canvas, `ui` title pinned at the top, `list` scrolling in the middle, inset cards crossfading above it |
| `lecture` | the pseudo-intellectual style | see §7 |

The command: `wonhee reel <video> --style <name> [--title "..."] [--cta "..."] [--items "a, b, c"] [--inset img...]
[--set k=v]`. It runs `speech_subs → compose_video(template)` as a chain (provenance, cache), then prints the output
path. `wonhee reel styles` lists the styles with one line each. MCP twin: `reel(video, style, title, cta, items,
insets, set)` and `reel_styles()`.

Amendments from the wave-3 build (2026-10-20):
- A template may carry `reel: {summary, words}` (marks it as a style for `wonhee reel`) and `params` (default
  `compose_video` params; today `tighten`). `compose_video`'s `tighten` is `null` (the template's default) | `false` | `true` |
  `{min_pause_ms, keep_ms}`. The talking-head styles (`explainer`, `scrapbook`, `series`, `lecture`) default it on.
- A `photo` layer with `src: "none"` draws nothing (an empty inset slot). Timed slots are `{in_ms, out_ms}` dicts
  (`thesis_1_t`, `inset1_t`, `card_t`, ...); a default of `{in_ms: 9999000}` means never.
- `captions` has `marks` (word -> footnote marker, set after the word's first use); the `reel` command anchors the lecture's
  footnotes on long words and shows each from that word.
- `reel` runs `speech_subs` once as its own chain (it needs the words to place chapters, insets and theses), then
  `speech_subs` (cached) -> `compose_video` as the recorded chain. `--set` also takes `tighten`, `encoder`, `quality`, `look`
  (a look preset applied to the clip inside the same graph; still one encode, policies rule 1), `engine` (lecture writer) and `name`.
- The `explainer` headline/closing line and the `series` closing lines sit on a translucent dark box (the live check showed bare
  thin white type vanishing on bright footage); `legend`'s captions sit at 70.5-75.5% (inside the 76% safe line), not 75-80%.
- All text boxes inside 35-88% of the height are within the centred 72% width, so no style triggers `LAYOUT_ADJUSTED`.

Amendments from the wave-3 alignment check (2026-10-20). These correct the build amendments above where they
conflict:
- **Looks are part of the one encode.** `compose_video` takes `look` (a preset name) and applies its filter string to
  the clip after the cuts and before scaling, never to text or bands. `reel` never runs a separate `look` node.
- **Backings are deliberate.** The `explainer` and `series` text backings depart from c49's clean type, because the
  study itself notes c49's type losing contrast on bright footage. A backing is one block per text layer (never one
  box per line), with line height ≥ 1.05 when backed.
- **`series`** has no pinned header. The title card owns the brand while it is up. When no inset is showing, the top
  window is a flat panel.
- **`list`:** title at y ≈ .11, list at y ≈ .28–.40, cards and clip *below* the list (c52: .39–.73).
- **`legend`:** the window ends at y ≈ .70, so the captions (y .705–.755, moved up from c48's .745–.81 to stay above
  the 76% line) sit fully in the black band.
- `report.md` lists every timed slot with its in and out time, the effective params (tighten, encoder, look), and
  marks truncation with "…".
- Tests: one encode for every style, with and without `look`. `series` card and header never on screen together.
  Two-line backed titles have no overlapping boxes. A live-style render with footnotes shows the marker and the
  footnote line.

## 7. The `lecture` style (pseudo-intellectual, on purpose)

Form: a paper ground (warm off-white), or a dark slide; the clip in a framed figure window labelled `Fig. 1` in
`mono`; a title in `editorial` with an italic subtitle; captions in `transcript` serif as quoted propositions (no
box, ink colour, a hanging open quote); numbered theses (`I.`, `II.`, …) appearing as `fade` lines at phrase gaps;
footnote markers on chosen words with a footnote line in small `mono` at the bottom of the safe area; a closing
"Q.E.D." or "(Muk, 2026)" CTA card. Slots: `title`, `subtitle`, `theses[]`, `footnotes[]`, `citation`. It also works
as a still (`compose` with template `lecture_card`) for carousels.

The writing side (deadpan theses about a mundane act) is a new `larp` register, `lecture`, in the existing `larp`
node; its output fills the slots. Rule: the form is formal, the subject is trivial, and nothing claims a real
credential, publication or institution (VISION: no faking real events). Built 2026-10-20 (aligned): the register runs only when asked for
(`--register lecture`) and writes `lecture: {title, subtitle, abstract, theses[2-5], footnotes[0-3], citation}`
into larp's JSON. The citation must be `(Muk[ et al.], YYYY)`. A code check refuses institutions, credentials,
publication identifiers and non-Muk author citations, with one retry. A bare real name outside citation form isn't
caught. Footnote markers are anchored to caption words on the compose side.

## 8. Fixes found during the study

- Parallel remote-LLM calls fail with moeka `InstanceLockedError` when two processes share the moeka state
  directory (two teardowns failed). Design: each moeka kernel gets its own instance directory `<home>/moeka/<pid>/k<n>/` (moeka has no lock
  wait), swept when its process is dead and removed at exit. Directories are named `<host>-<pid>`, and a sweep
  only touches its own host's, because containers sharing one home can reuse pids (the moeka-multi-instance rule).

## 9. Tests (from this text)

- `speech_subs` puts `words` on lines (fake whisper). A consumer without words splits evenly and flags
  `WORDS_ESTIMATED`.
- `captions`: `karaoke` gives one pill state per word at that word's time ±1 frame. No state has more than
  `max_words`. Everything stays inside the safe zone.
- `stack` and `list` change state only at word or entry times. The list window shows ≤ N lines, with the active
  line brightest.
- `compose_video` runs exactly one ffmpeg encode (count subprocess calls) whatever the number of layers, and the
  output duration equals the input (or the tightened length).
- `tighten` removes the pauses in a fixture with known gaps (within ±1 frame), and the captions move with them.
- Every bundled font has a licence file next to it and loads. Each role resolves to a file.
- `wonhee reel tiny.mp4 --style X` works for every style on the `tiny_video` fixture, and `reel styles` lists 6.
- Two concurrent remote-LLM fake calls in two processes don't raise `InstanceLockedError`.

## 10. Order

1. Fonts and roles, typography params.
2. Word timings.
3. Timed layers, `captions`, `stack`, `list`, and `compose_video` with one encode.
4. `tighten`.
5. `lofi_cam`.
6. The six styles, the `larp` `lecture` register, `reel` CLI and MCP.
7. The moeka lock fix.

Each step is followed by its alignment check (PROCESS §3).
