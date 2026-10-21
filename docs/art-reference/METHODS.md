# Project art: methods reference

Status: **reference, not built.** The shipped design is `docs/DESIGN.md › Project art`. This folder keeps what the prototypes proved, so other viz work can reuse it. `prototype/` runs as is (`cd prototype && uv run python art6.py` writes the pieces into the current directory); it is scratch code, not part of the `wonhee` package. `recipes.json` is every piece as data; `reference-sheet.jpg` shows them.

## What the owner liked and ruled out

Liked: layered DAG, ring buffer, adjacency matrix; WaveSim glow; trainnet-v2 frame ring; the wave-looking digital effect (wave lines, stepped waveforms); **pixel waves** (large squares that quarter into dense clusters in some areas); circles in moderation; blue-violet, red-yellow, white-on-dark, all hue families, 2 or 3 colours; low contrast as well as high.

Ruled out: a central focal point (centred shapes, symmetric spines, centred vignettes, glows in the middle); stretching (ellipses from a non-uniform scale); many circles; multi-hue full-field gradients (muddy); grey ghost blobs.

## Colour

Built in OKLCH (`wonhee/edit/viz/palette.py`): fix lightness per role first, then pick any hue, so contrast does not depend on hue. Roles: canvas, structure, optional structure2 (3-colour palettes), accent, ink; edge = structure pulled 30% toward the canvas; band = canvas 10% toward structure.

- Dark canvas L .13-.16, structure L .47-.62, accent L .74-.90. Light canvas L .95-.97, structure L .40-.50, accent L .15-.58. Poster canvas is saturated (L .34-.89, C up to .19) with a pale or dark structure and a hot accent.
- Measured gaps all passed: structure ΔL ≥ .25, accent ΔL ≥ .30 (`prototype/art4.py` prints them).
- 3-colour: structure2 shades DAG layers top to bottom, ring cells tail to head, pixel squares by field value. The accent stays on the hot path, ring head and field peak.
- 29 palettes are in `art4.PALETTES`: dark (Blue violet, Red yellow, Ember, White dark, Black cherry, Teal coral, Green gold, Neon trio, Indigo lime, Sunset trio, Ocean trio, Rose gold, Forest peach, Blue orange, Magenta mint, Violet dusk), light (Ink on paper, Blue violet light, Red on cream, Teal coral light, Plum olive, Cobalt cream, Mint pink) and poster (Red, Cobalt, Violet, Yellow, Forest, Pink violet).

## Contrast control

Per layer `d` pulls structure and edge toward the canvas by `d`. The accent first moves toward the structure hue (`d*1.1`) and then toward the canvas (`d*.55`), so a ghost accent never greys into a muddy block. `d` 0 is full contrast, about .3 is soft, .5-.8 is a whisper. Several layers at different `d` give depth without extra colours.

## Layer stack (a piece = background, layers bottom to top, post filters)

1. **Background:** flat, corner glow (structure at 16%, centred on a corner), dot grid, halftone field (smooth noise through `fx.halftone`; its output is BGR, flip it), soft top-to-bottom gradient. Never a mid-lightness multi-hue gradient.
2. **Layers**, each drawn full box then moved uniformly (`scale`, `dx`, `dy`; never non-uniform), optionally blurred (depth) or given an `echo` (a copy in another role colour shifted a few px under the layer: the riso misregistration look).
3. **Post:** `fx.bloom` (glow on the hot path), `fx.grain`. No centred vignette.

## Drawers (all in `prototype/art4.py` and `art6.py`)

- **Pixel waves:** quadtree of squares from 16u; a wave field (three sines times an off-corner envelope) decides subdivision depth, so quiet areas stay as large sparse squares and busy areas split into small dense ones. Fill probability and alpha rise with the field; the top few percent take the accent. Variant: outlines only.
- **Wave lines:** about 46 parallel rows displaced by three sines under an envelope centred in a corner; the row with the biggest crest is the accent. **Digital waves:** the same, quantised into steps.
- **Layered DAG:** layers of random width, each node links to its nearest in the next layer (plus extras), every node gets a parent, a random walk makes the hot path. Node styles: box, pill, dot, ring, square; edges: S-curve, line, bundled through a gap midpoint. Layouts: flow, pills, lattice, diamond, bundle, swimlane, tiered, mirror. Prefer asymmetric ones.
- **Ring buffer:** a run of cells ending at a head; head = accent, body fades or shifts hue tail to head, free cells are outlines. Layouts: snake grid, wheel (concentric), lanes, spiral, gauge wall, trace path. Circles must be uniform and placed off-centre, bleeding off an edge.
- **Adjacency matrix:** block-structured dots, density near the diagonal, one accent block.
- Combos: DAG feeding a ring, chord ring, DAG in a ring frame (the frame stretches a circle; avoid).

## Rules that made them better

- Off-centre everything: energy in a corner, shapes bleeding off an edge, DAG scaled to about 80-88% and shifted sideways.
- At most one circular layer per piece; ghost it.
- Density should vary (pixel waves) instead of being uniform.
- Detail follows display size (`detail(css_w)` 42%, 72%, 100% below 320, below 700, above); the 9:16 pieces assume phone width.
- Check by looking: render a contact sheet, view it, then change. The halftone channel flip, grey ghost accents and centred vignettes were all found that way.

## Using it for other viz

Pick a palette (or build one with `art4.make`), choose layers from the drawers, set a `d` per layer, pick a background and post filters, and write the result as a recipe in `recipes.json` shape. The scene engine in DESIGN.md will take these recipes as presets.
