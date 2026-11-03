# Chroma key without a real green screen: `chroma_key`

Status: approved 2026-11-01 (M lane: a new node in the Stage 4 video family; no new dependency, no model download
beyond the person mask `mask_cutout` already uses).

## Why

Takes for the plane skit ("is there anyone on this plane who can solve two sum") are shot against a whitish wall,
not a real green screen. A plain `ffmpeg chromakey` needs a saturated backdrop of known colour. This node finds the
backdrop itself and keeps the person safe where the backdrop colour appears on them (a white shirt on a white wall).

## Contract

`chroma_key` (video, background? -> video, report). CPU. Deterministic for the same input and params.

| Param | Default | Meaning |
|---|---|---|
| `backdrop` | `auto` | `auto` detects it; `green`, `blue`, `white` force a kind; `#rrggbb` forces a colour |
| `tolerance` | 0.5 (0..1) | how far from the backdrop colour still counts as backdrop |
| `softness` | 0.5 (0..1) | width of the soft edge between backdrop and subject |
| `guard` | `auto` | `person`: the person mask protects the subject and trims specks outside it. `none`: colour only. `auto` = `person` unless the backdrop is a saturated green or blue |
| `fill` | `cabin` | used when no `background` input is given: `cabin` (a drawn plane cabin), `window` (a drawn plane window with sky, clouds and wing, for a window-seat take), `#rrggbb`, or `matte` (white subject on black, for checking the key) |
| `spill` | 0.7 (0..1) | edge colour clean-up: removes backdrop colour from semi-transparent edge pixels |

### Detection

The backdrop colour is the median Lab colour of a ring of pixels along the top and the two sides (never the bottom,
where the subject stands) over up to 9 frames spread through the clip. The ring's spread (median distance to that
colour) says how even the backdrop is. Kind: `green` / `blue` when the colour is saturated in that hue, `white` when
it is bright and nearly neutral, else `other`.

### Matte

1. Distance of each pixel to the backdrop colour in Lab. Saturated backdrops weight chroma over lightness, so
   shadows on the screen still key. White backdrops weight lightness in full.
2. Alpha is a smoothstep of that distance. The threshold widens with the ring's spread, so a dim, uneven wall still keys.
3. Only backdrop-coloured regions connected to the top or side edges are removed. A white shirt inside the frame
   stays. Enclosed background (the gap between an arm and the body) is removed when it is larger than 0.3% of the
   frame and matches the backdrop.
4. With the person guard: the model mask (run every 6th frame, held between) is dilated and multiplies the alpha, so specks
   outside the person go, and its eroded core is forced to 1, so a white shirt can not be keyed out.
5. Alpha is smoothed over time only inside the uncertain band (0.05..0.95), so edges do not shimmer and motion does not ghost.

### Output and honesty

- Composite: subject over `background` (an image or video input, looped and cover-fitted to the clip size) or `fill`.
  Audio is copied from the source. Size, fps and duration of the clip are kept.
- `report` (text, JSON): `backdrop` kind and colour, `spread`, `ring_coverage`, `guard_used`, `frames`, and
  `mean_subject_fraction`. Anything unknown is `null`, never 0.
- Flags: `BACKDROP_UNEVEN` (ring spread above 14 in Lab: the wall is not one colour, the key will be rough, add
  light or use `guard: person`), `SUBJECT_TOO_SMALL` (subject under 3% of the frame: the key probably ate the
  person), `SUBJECT_TOO_BIG` (over 85%: the key probably missed the wall).

## Tests

Synthetic clips: a moving coloured block on a flat green, a flat white, and a noisy off-white wall; a white-shirted
block on a white wall (must survive with the guard stubbed in); detection kinds; the bottom is never sampled;
connected-only removal keeps an inner white block; flags; `report` has `null` for unknowns; audio is carried.
Live check on real media is recorded in `docs/production/` after the owner's first take.

## Cabin fill (`edit/cabin.py`)

A drawn, deterministic one-point-perspective plane cabin (aisle, seat rows, overhead bins, windows with daylight) for
use as the fill and for storyboards. It is a stand-in until real cabin footage exists. `render_window` draws one big plane window with sky, clouds and a wing, static, for a window-seat take.
`render_seat` draws the close window-seat set (window, headrest,
overhead bin) for a seated 3/4-angle take; `render` is the wide aisle view.
