# Meme frame, sticker and pop-art formats: design

Status: **proposed** 2026-10-20, waiting on the owner. Stage 4 (make), photo and video formats. Size: M per format
(new nodes and presets inside an approved area). Written as one spec because the formats share their building
blocks.

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

## Order (if approved)

1 sticker preset → 4 sticker → 3 pattern_bg → 2 shape_frame → 6 word_captions → 5 meme_frame → 7 grid_plan. The
first four make formats B, C and D on photos. 5 and 6 make format A and are also the first real piece of the video
timeline work.
