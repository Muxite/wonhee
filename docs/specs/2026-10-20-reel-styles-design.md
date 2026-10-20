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

## 3. Pause trimming inside a take

`compose_video` param `tighten: {min_pause_ms: 250, keep_ms: 80}`: pauses in the speech are cut out (from the word
timings, else from `silencedetect`), inside the same filter graph (`trim`/`atrim` + `concat`), so there's still one
encode. Every timed layer and caption is re-timed to the cut timeline. Off by default; on in the talking-head styles.
The report lists the seconds removed.

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

## 7. The `lecture` style (pseudo-intellectual, on purpose)

Form: a paper ground (warm off-white), or a dark slide; the clip in a framed figure window labelled `Fig. 1` in
`mono`; a title in `editorial` with an italic subtitle; captions in `transcript` serif as quoted propositions (no
box, ink colour, a hanging open quote); numbered theses (`I.`, `II.`, …) appearing as `fade` lines at phrase gaps;
footnote markers on chosen words with a footnote line in small `mono` at the bottom of the safe area; a closing
"Q.E.D." or "(Muk, 2026)" CTA card. Slots: `title`, `subtitle`, `theses[]`, `footnotes[]`, `citation`. It also works
as a still (`compose` with template `lecture_card`) for carousels.

The writing side (deadpan theses about a mundane act) is a new `larp` register, `lecture`, in the existing `larp`
node; its output fills the slots. Rule: the form is formal, the subject is trivial, and nothing claims a real
credential, publication or institution (VISION: no faking real events).

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
