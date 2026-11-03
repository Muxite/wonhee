# wonhee.art: the effect library

Status: built and merged into `main` 2026-11-03 (201 effects, 21 families; `tests/` has per-family tests plus a smoke test over every
registered effect). Owner direction, 2026-11-03. It is a library, not a node family: effects are not yet
wired into chains (see [DESIGN › Art library](DESIGN.md#stage-4-make)).

**What it is.** A growing library of deterministic image effects ("art") plus
the tools to understand what you want and generate it from parameters:
physics simulations, light painting, lightning, dot-matrix systems, graphs and
DAGs, and filters/edits of real project images. No generative AI.

**Scope broadened (owner, 2026-11-03).** The library is not only procedural
art. Photos are welcome (people included), any media is fine as long as it is
not *completely* AI-generated, and we want a huge range of styles, moods,
filters, adjustments and colour grades for the site now and for video later.
Priority for now: what the SITE needs (project cards, the photography page,
blog covers/OG images, heroes and backgrounds, dark/light variants, small
sizes and LQIP placeholders). Images may be shown large but must still work at
small size. The no-focal-point rule below applies to procedural/data card art;
a photo or a stylised single subject that fills the frame is fine.

**What a card image is.** Either (a) filters and edits on some *feature of the
real project* (an original image, a screenshot, real data), (b) a visualization
tightly tied to the subject (WaveSim = wave physics, Euglena = its score
matrix), or (c) both. Generic "glitch / wave / graph" effects are only a
starting point.

**Quality references:** the euglena score heatmap (magma, 38 x 9 cells) and the
WaveSim interference image. Both are all-over: detail everywhere, no hero spot.

## Rules for every effect

1. **No focal point.** Detail is equal or evenly spread across the frame
   (`python -m wonhee.art check` reports coverage >= 0.90, center_bias <= 1.6,
   focal_ratio <= 3.0). Do not centre anything. (NEXUS's radial image is the
   known counter-example: a bright centre.)
2. Readable at 288x192 (the card thumbnail): big structure, no tiny text.
3. Deterministic in (params, seed, size, palette, source, data).
4. Palette-driven: use `ctx.palette` roles (canvas, structure, structure2,
   accent, ramp). Changing the palette hue must give a good image.
5. Colour discipline (OKLCH): contrast comes from lightness; chroma bounded
   (rich, not neon); no pure #000/#fff areas; no muddy mid-lightness
   multi-hue full-field gradients. A single dominant hue family is a good
   default, but SEVERAL hue families in one image are allowed (owner,
   2026-11-03: the sunset was one example): spectra, split complements, 3-4 hue
   heatmaps, as long as the colours are arranged on purpose.
6. Hooks for real material: `ctx.source` (an image), `ctx.data` (a graph,
   matrix, series, point set). Effects document what they accept.
7. <= 8 s at 1500x1000 on CPU (numpy/scipy/cv2).

## Families (see docs/art/INDEX.md for the generated catalog)

physics, light, dots, graph, feature (edits of real images), grade (colour looks, LUT and web
export), atmos and optics (light on real photos), micro and arch (fractured plates, architecture),
region, layers and collage, print, swiss and dither, data and multichan (matrices, composites),
nature, motion and contour. Project-specific effects live under `proj_*` (one set per project).
Renders can carry provenance (`render --out a.png` writes `a.png.meta.json`: `synthetic`,
`has_source`).

## Using it

```
python -m wonhee.art list [--family physics] [--tag wave]
python -m wonhee.art render wave-ripple --out a.png --seed 3 --palette ember
python -m wonhee.art sheet  wave-ripple --out sheet.png --seeds 0,1,2,3 --palettes ember,ultramarine
python -m wonhee.art check  a.png
python -m wonhee.art recipe recipe.json --out a.png     # layers + post ops
```
Run with `uv run python -m wonhee.art ...` from the repo root.

## Adding an effect

Add `wonhee/art/effects/<family>_<name>.py`, decorate with `@effect(...)`
(see `wave_ripple.py`), add a test. It is auto-discovered.
