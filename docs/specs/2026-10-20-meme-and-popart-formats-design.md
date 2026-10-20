# Layouts: meme frame, sticker, pop-art and collage formats, as one engine

Status: **approved** 2026-10-20 by the owner, who asked for it to be generalized further. The generalization
(§ "Generalized design") replaces the per-format node list (§ "Proposed building blocks", kept as the feature
checklist). Stage 4 (make).

## Why

The references saved 2026-10-20 (in `data/refs/2026-10-20/`, gitignored; the reel is `content:48`) show four formats
that fit the niche and that wonhee can't make yet:

| Ref | Format | What makes it |
|---|---|---|
| A. "most legendary video" reel (`content:48`) | **Meme frame** | 9:16 canvas with black bands. The clip sits in the middle. A bold condensed white title (1–2 lines, full width) fills the top band. Short captions are highlighted word by word in a coloured box (here a lilac box with a black serif), placed in the lower band |
| B. "SO SORRY, UNNIE!" | **Sticker frame** | The subject photo cropped to a shape (a heart) with 2–3 offset outlines in a palette (purple, pink, yellow). A comic speech-bubble sticker holds slanted, heavy-outlined yellow text with a halftone fill. White or transparent background |
| C. "ON PREM COMPUTE" | **Maximalist brag collage** | Dozens of stickers (badges, checklists, price bursts, chrome headline) piled over a desk scene. This is the performative-founder joke at full volume. The reference is AI-generated imagery, which is on our never-list. Our version must use real photos: the creator's desk, PC and props, cut out and piled |
| D. ivestarship grid | **Pop-art cutout** | The subject is cut out over a flat colour with a halftone-dot or star pattern. Tiles form a set on the profile grid: a colour per row, one photo split across tiles |

## What already exists

- `mask_cutout`: subject cutout with a transparent, flat or blurred background.
- `vibe_image print=halftone|duotone`: a halftone treatment of the whole image.
- `text_image` and `text_video`: outline, band and script presets inside safe zones, with colour emoji and CJK.
- `carousel`: splits a panorama across slides.
- `poster`: a fixed layout with boxes and a mono paragraph.
- `speech_subs`: whisper word timings (`analyze.speech`).

## Proposed building blocks

Each is a node with ports, params and flags, following DESIGN contracts 1–13.

1. **`text` sticker preset** (an extension of `edit/text.py`): heavy multi-stroke outline (2 colours), a drop
   offset, an optional slant (≤ 12°), and fill colour from a palette. Used by B, C and D. Output change → bump
   `text_*` and `carousel`.
2. **`shape_frame`** (`image` → `image`): crop to `heart | circle | rounded | burst | star`, add 1–3 offset outlines
   in palette colours, on a transparent or flat background. Its subject placement uses the `mask_cutout` box when
   one exists, so the face stays inside the shape.
3. **`pattern_bg`** (`image` (cutout) → `image`): put a cutout on a generated *pattern*, not generated imagery:
   flat colour, halftone dot gradient, stars, or a tiled glyph. Seeded. Palettes are named (`popart-yellow`,
   `popart-pink`, `lilac`). Not a look copied from one group: palettes are generic and the user chooses them.
4. **`sticker`** (`image` → `image`): place one sticker by kind: `bubble` (speech bubble + text), `badge`, `checklist`,
   `burst` (price or claim), `note` (sticky note). Takes position and rotation. Text goes through the sticker preset.
   Many stickers make a collage, by chaining or by a `stickers: [...]` list param.
5. **`meme_frame`** (`video`/`image` → `video`/`image`): fit the clip into 9:16 with bands (`black` / `blur` /
   `white`), and put a title in the top band at full width (the safe-zone top rule). Video: one encode (policies
   rule 1).
6. **`word_captions`** (`video`, `subtitles` → `video`): one to three words at a time from the word timings, in a
   coloured box style (`box`, `highlight`), placed in the lower band above the 76% line. Pillow layers, one encode.
   This is also the R1 `captions` item, rebuilt the right way.
7. **`grid_plan`** (`images` → `images`): order and colour-assign 3, 6 or 9 posts so the profile grid reads as a
   set (a palette per row, optional split across tiles). It reuses the `carousel` split.

## Generalized design (approved form)

The seven building blocks above are one idea: **a picture is a canvas plus an ordered list of layers.** So there is
one engine and one node, and each format is data: a named template plus a few slots. This is the Stockgen idea
again: an agent or a person writes a small declarative layout, and the code renders it the same way every time.

**`compose`** (`images?` → `image`), params `layout` (a `Layout` dict) or `template` (name) + `slots` (dict). Engine:
`wonhee/edit/compose.py`, no DB.

```
Layout = {canvas: {w, h, bg: colour | pattern},
          palette: [colours] | name,
          seed: int,
          layers: [Layer, ...]}            # painted in order

Layer kinds (each has box: {x, y, w, h} as fractions of the canvas, rotate (deg), opacity):
  photo    src: input index | "slot:<name>"; fit: cover | contain; shape: rect | rounded | circle | heart | star | burst;
           outlines: [{width_frac, colour}] (offset strokes, outermost last); cutout: bool (subject only, via mask_cutout's engine)
  pattern  kind: flat | halftone | stars | glyph | stripes; colours; density; angle   (fills its box, or the canvas)
  shape    kind: as photo.shape + bubble (with tail: {x, y}); fill: colour | pattern; outlines
  sticker  kind: bubble | badge | checklist | burst | note | label; text or items; colours   (a shape + text, laid out)
  text     content; preset: outline | band | script | sticker; size_frac; align; colour; stroke colours; slant
  band     side: top | bottom; h_frac; colour          (letterbox bands)
```

- **Templates** are layouts with `"slot:<name>"` holes, shipped as JSON in `wonhee/edit/compose_templates/`:
  `meme_frame` (A, still form), `sticker_frame` (B), `brag_collage` (C), `popart_cutout` (D), and `slop_card`. The
  slop machine's cards can move onto this later; they are not changed now. A template is listed with its slots and
  defaults through `nodes_list(name="compose")` and `wonhee node show compose`.
- **Safe zones** (DESIGN Stage 4) apply to `text` and `sticker` layers on tall canvases. A layout that puts text in
  the caption or button zone is moved inside, and flagged `LAYOUT_ADJUSTED`.
- **Determinism:** the same layout, inputs and seed give identical bytes. Fonts are bundled or reported by doctor.
- **Video formats reuse it:** `meme_frame` for video renders the composed overlay (bands + title) as one PNG with a
  transparent window, and ffmpeg scales the clip into the window and overlays the PNG, in **one encode**.
  `word_captions` adds timed Pillow layers into the same graph. The video form is the `frame_video` node
  (`video`, `subtitles?` → `video`), params `template`, `slots`, `captions` (style `box` | `highlight` | `none`).
- **`grid_plan`** stays its own node (`images` → `images`): it assigns a template and palette per tile so a 3, 6 or
  9 set reads as one grid.

What generalizing buys: a new format is a JSON template (the M lane becomes a data change with a test), and agents
can write one-off layouts directly in a chain without new code.

## Rules

- **Real pixels only.** Patterns, outlines and stickers are drawn, and photos are ours. Format C must be built
  from a real desk shoot, and the shot guide says what to capture (PC glow, cans, books, plush, the Jetson).
- **No lifted text or marks.** No group names, logos or slogans from references in our outputs. The copy-guard idea
  from `ideate` (`IDEA_COPY`) applies to sticker text: a sticker line must not copy a reference overlay.
- **Third-party clips** (format A's source) are reference only. The meme frame wraps our footage.

## Acceptance (tests to write from this text)

- Sticker text stays within its sticker box at every slant. Two strokes are visible (pixel colours at the edge).
- `shape_frame heart`: the face box from the mask lies inside the heart. Outlines appear in the order given.
- `pattern_bg` with the same seed gives identical bytes, and the cutout's alpha is kept.
- `meme_frame`: the clip keeps its aspect, the title lies in the top band, and there is exactly one encode (count
  ffmpeg calls).
- `word_captions`: every word appears at its whisper time ±1 frame, inside the safe zone, and never more than 3
  words on screen.
- `grid_plan`: 9 inputs give 9 outputs. Row palettes follow the plan.

## Order

1. `compose` engine with `text` (+ the sticker preset), `pattern`, `shape`, `photo` (shapes, outlines, cutout) and
   `band` layers, plus the `compose` node and the `sticker_frame` and `popart_cutout` templates.
2. `sticker` layer kinds and the `brag_collage` template.
3. `frame_video` with `meme_frame` + `word_captions` (one encode).
4. `grid_plan`.

The older ordering, kept for reference:

1 sticker preset → 4 sticker → 3 pattern_bg → 2 shape_frame → 6 word_captions → 5 meme_frame → 7 grid_plan. The
first four make formats B, C and D on photos. 5 and 6 make format A and are also the first real piece of the video
timeline work.
