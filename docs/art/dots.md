# Dot-matrix system (`wonhee/art/dotkit.py` + `dots-*` effects)

Owner direction 2026-11-03: "a stronger dot matrix system". Not a set of one-off
effects: one primitive, `render_dots(field, ...)`, that draws ANY field as dots,
and a family of effects built on it. All math (numpy/scipy/PIL), seeded; no AI.
Every effect passes the no-focal-point check on 5 seeds (coverage >= 0.9,
center_bias <= 1.6, focal_ratio <= 3.0) and renders in 0.1-1.3 s at 1500x1000.

## dotkit API

- Fields: `norm01`, `luminance`, `smooth_noise(shape, scale, rng)`, `wave_field(shape, rng, n_sources, wavelength)` (spread sources, never one centre), `resize_field`, `source_field(ctx)`.
- Lattices (return `Dots(x, y, cell[, row, col])`): `square(angle)`, `hexagonal`, `grid(rows, cols)` (matrix/bitmap, indexable), `polar(centre, ring_pitch)`, `poisson(min_dist)` (blue noise), `stipple(density, n)` (weighted Voronoi, Lloyd), `quadtree(field, max_cell, min_cell)` (busy = small squares); string front door `make_lattice`. `bsp_panels` splits the frame into rectangles.
- `render_dots(field, size, lattice, shape=, size_by=, color_by=, alpha_by=, angle_by=, ramp=, off=, glow_sigma=, core=, size_const=, threshold=, canvas=)`. Shapes: round, square, diamond, line, plus, ring (analytic anti-aliased, vectorised in size buckets). Two-field control: `size_by` one field, `color_by` another (any shape; 1-D per-dot arrays also accepted). LED look: `off=` dim dot at every site, `glow_sigma`, `core`, `size_const`.
- `draw_dots(canvas, x, y, r, rgb, alpha, shape, angle)` low-level; `glow`.
- Halftones: `halftone_mask(field, size, mode)` with mode `round` (AM screen at an angle), `line` (optional `warp`), `hex`, `bayer` (2/4/8/16 matrix), `fs`, `atkinson` (error diffusion); `mix_mask` colours it; `error_diffuse` standalone.
- Text/bitmap: `text_matrix(text, font="5x7"|"ttf", ttf=path, rows=)` -> bool matrix (A-Z, 0-9, punctuation, a few symbols; any TTF via PIL area-averaged to `rows`); `tile_text` makes a marquee band; `FONT_5X7` dict. Feed a matrix to `grid` + `render_dots`.
- `palette_ramp(P)`: dim structure -> structure -> accent (never touches the canvas, so low values stay visible).

## Effects (all family "dots"; best palettes ember, green-gold, ultramarine; also fine on light palettes)

| effect | what | key params | ctx.data / ctx.source |
|---|---|---|---|
| dots-led-wall | LED wall, off dim dots + glow, bands of bars / hex ticker / sparkline / bit runs | cols, glow, ticker, bands | data: list of numeric series (bar/spark bands) and/or str (ticker) |
| dots-pixel-waves | owner's pixel waves: quadtree squares quartered where a wave field is busy, outlines in troughs | cell, depth, sources, wavelength, outline | data 2D = field; source = luminance field (quadtree mosaic of a picture) |
| dots-hex-field | hex lattice, size = wave amplitude, colour = signed phase | pitch, sources, wavelength, shape, angle | data 2D / source = size field |
| dots-stipple | Lloyd-relaxed stipple of an image or noise+ripples | dots, size, contrast, invert | source = picture; data 2D = density |
| dots-halftone-lines | engraving line screen, bent and swelling | pitch, angle, warp, mode (line/round/hex), gamma | source / data 2D = width field |
| dots-bayer | BSP panels, Bayer/error-diffusion dithered gradients | panels, pixel, cycles, diffuse | source dithered in every 3rd panel |
| dots-rings | polar dot ring systems, Voronoi-tiled, several centres | centers, pitch, pulse, shape | data: (n,2) centres in 0..1 |
| dots-bitmatrix | a matrix as dots (size+colour = value, zero = dim off dot), tiled edge to edge | cell, layout (repeat/fit), gap, shape, max_cells | data: 2D array or {"matrix","mask"}; default = block-structured adjacency matrix |
| dots-word | dot-matrix lettering marquee, staggered bands over off dots | text, cols, gap, shape, bands | data: str or list of str (per band) |
| dots-circuit | BGA ring-pad arrays, jogged dotted buses, chips | cols, blocks, hot | data 2D: read at pads, high = accent |
| dots-gradient-sweeps | patchwork of dot gradients, lattice and shape per panel | panels, pitch, cycles | source modulates alternate panels |
| dots-dash-flow | needle/hatch field oriented by a wave gradient | pitch, sources, wavelength, shape | data 2D / source = potential field |

Tie-in recipes: euglena -> `dots-bitmatrix` with `data=score_matrix_38x9` (tiled; use `layout: "fit"` for one copy); NEXUS / a project name -> `dots-word` `data="NEXUS"`; a photo -> `dots-stipple` / `dots-pixel-waves` / `dots-halftone-lines` with `source`; wave/physics projects -> `dots-hex-field`, `dots-pixel-waves`; telemetry/benchmarks -> `dots-led-wall` with `data=[series, ...]`; hardware/infra -> `dots-circuit`.

## Strongest and weakest

Strongest at 288x192: dots-word, dots-rings, dots-hex-field, dots-dash-flow, dots-led-wall. Solid: pixel-waves, circuit, gradient-sweeps, bayer. Weakest: dots-halftone-lines (the warped lines read as soft orange cloud at thumb size, moire risk on source images), dots-stipple (fine texture, lowest contrast at thumb), dots-bitmatrix with sparse real data.

## Failure modes

- Complement palettes (ultramarine) put the accent in yellow; mid ramp stops go olive/tan in bayer and gradient-sweeps. Prefer ember / green-gold / mono palettes there.
- Halftone lines on a small canvas alias: keep `pitch` >= 10 px at the output size.
- Pixel waves is the closest to the focal limit (focal_ratio 2.4-3.0): keep `depth` at 2 and `sources` >= 12.
- A tiled small matrix (38x9) repeats visibly; use more rows of context or `layout: "fit"`.
- `error_diffuse` is a Python loop: keep the grid <= ~150k cells (cell >= 4 px at 1500x1000).
- Text: only the 5x7 glyph set; lower case is mapped to upper. Use `font="ttf"` for others.

## Note for a future skill: choosing a dot system for a project

Ask what the project IS: a grid of discrete things (scores, adjacency, benchmark) -> bitmatrix; a name or number -> word; a signal / waves / physics -> hex-field, pixel-waves, dash-flow; a photograph or image feature -> stipple, halftone-lines, pixel-waves with `source`; infrastructure/hardware/telemetry -> led-wall, circuit; rhythmic or circular (rings, buffers, orbits) -> rings; abstract/graphic -> bayer, gradient-sweeps. Then: pick a palette whose hue matches the project, render a 4-seed sheet, check at 288x192, pass real `data`/`source` so the dots carry the project's own numbers, and reject anything with a hero spot (run `python -m wonhee.art check`).
