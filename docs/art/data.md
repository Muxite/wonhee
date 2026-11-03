# Data matrix textures and multichannel composites (`data-*`, `multichan-*`)

Helpers: `wonhee/art/datakit.py` (scales, normalise, cluster order, aggregate, cell renderer, overlay text/legend, stand-in fields, `provenance`) and `wonhee/art/mckit.py` (channel intake, tints, blend, hillshade, contours, transforms). Tests: `tests/test_art_data.py`. Sheets: scratchpad `art-data/` (`data-overview.png`, `multichan-overview.png`, `*-288x192.png`, `overlap-*.png`).

## Overlap table (existing effects rendered on the same data, 1500x1000)

| exists | what it gives | real gap |
|---|---|---|
| dots-bitmatrix | matrix as dots, tiled, one palette ramp | no cell proportions, no ordering/clustering, no light scales, no legend/numbers, not flat bleed cells |
| graph-matrix | block adjacency of a graph, rounded cells | graphs only; no value scales |
| proj-euglena-bench (p-euglena) | flat bleeding heat / runs / skyline / scatter, fixed magma/ember/cyan | euglena-only data shape, fixed ramps, no light, no clustering, no card/full modes, no overlays |
| dots-led-wall, graph-ring-* | LED/ticker, ring tiles | not matrix heatmaps |
| physics-*, grade-thermal-map, feature-thermal | single-field colouring | no multi-channel normalise+blend with explicit ranges, no hillshade/diff/labelled aesthetic |

## Scales (datakit)
Sequential (lightness-monotone, in gamut, dark and light variants): magma, ember, ocean, kelp, ice, rose, cream-red, ink, mono; `auto` keys a scale on the palette hue; `palette`. Diverging (symmetric lightness): blue-red, teal-rose, violet-gold, green-plum. `categorical(n)` alternates two lightness bands. `check_scale` is unit tested. Light scales start slightly darker than the canvas so cells stay visible.

## Data contract and provenance
`ctx.data`: 2D array, or `{"matrix"|"scores": 2D/3D, "rows", "cols", "mask"}`, or dict of 1D series (terminal panels). No data = SEEDED SYNTHETIC values: `datakit.provenance(name, data)["synthetic"]` is True, `mode=full` stamps "synthetic data - decorative, not results". Multichannel: `channel_meta(name, data, source)` returns synthetic/standin/aesthetic flags. All data effects carry tag `synthetic-default`. Modes: `card` (default: no text, always legible at 288x192) and `full` (numbers where readable, real legend with data units).

## data effects (family `data`)
Shared controls: colour = scale, lo, hi, invert, palette; structure = cols/rows/col_var/row_var/growth/order/fit; source = data/synthetic; surface = gap, surface (flat|glass|bevel|paper), surf, jitter.
- **data-heat-bleed** (scientific, analytical, dark-light): the gold-standard flat heatmap bleeding off every edge; crop/all/aggregate; cell proportion variation. Best: any 30x10 score matrix, magma. Failure: all-equal data = flat field. Card: ~6x10 big cells.
- **data-annotated-grow** (annotated, analytical): crisp small block in a corner whose cells grow geometrically to huge fields; outlined call-outs and axis ticks. Failure: check metric flags focal (detail sits in one corner by design); lower `growth`/raise `w0` to flatten.
- **data-cluster-matrix** (clustered, correlation, scientific): hierarchical order with optimal leaf ordering, diverging scale, real dendrograms and categorical cluster strips. Best: correlation/confusion matrices. Failure: very large N makes cells tiny; aggregate first.
- **data-pixel-field** (pixels, print): quantised value field as hard pixels, equal 2x2/4x4 blocks merged, a few accent blocks; cream-red on light palettes. Failure: smooth synthetic field can form one central blob.
- **data-stripe-matrix** (stripes, benchmark): configuration stripes of ragged segments (optionally split into repeat runs), per-stripe mean sidebar. Takes `{"scores": [test][config][run]}`. Failure: many tiny segments = barcode at 288 wide; use `runs=1`.
- **data-terminal-panels** (terminal, dashboard, small-multiples): BSP panels with mini heatmaps, bars, tables, log lines, sparklines; greeked text carries data in lengths/colours; full mode prints numbers. Strongest card result.

## multichan effects (family `multichan`)
Channels: dict of 2D arrays with explicit `ranges` (else 1-99 percentile, reported); RGB `source` gives AESTHETIC derived channels (red green blue luma saturation edges shadows highlights warmth detail), never infrared/satellite; stand-ins are smoothed luminance of physics fields (flagged). Controls: colour = mood (scientific|alien|luminous|analytical), hues, blend, opacity, gamma, floor; structure = transforms (raw|ridge|blobs|haze|dots|edges), contrast; source = data/which/ranges/standin_smooth; surface = glow, grain, halation.
- **multichan-composite**: tinted channels blended (screen/add dark, multiply light), broad fields plus ridge lines. Best: 2-3 real fields. Failure: >4 channels bleach to white; raise `floor`.
- **multichan-fluorescence** (dark): label-style channels (nuclei blobs, filaments, speckle, haze) additive on dark. Weakest: speckle reads as confetti at full size; best at 2-3 channels, fewer dots.
- **multichan-relief**: height channel to hillshade + contours, value channel as colour scale. Failure: noisy height = dense contours (raise smoothing).
- **multichan-diff**: A-B diverging, lightness from A+B, zero line. Cleanest scientific look; needs same units.
- **multichan-channel-mix**: photo to aesthetic multi-plate false colour with misregistration and posterise. Strongest on photos (ducks, foil-hat portrait, sunset all work; people stay recognisable). Light variant is garish.

## Card behaviour and checks
All render <= 2.5 s at 1500x1000. Metrics (`python -m wonhee.art check`): most pass; flagged for coverage/focal are annotated-grow, stripe-matrix and cream-red heat-bleed because large FLAT cells have no gradient energy (metric artefact) except annotated-grow's corner detail (real).

## Note for a future skill
Pick by question: have a matrix of scores -> heat-bleed (cards) / stripe-matrix (benchmarks) / cluster-matrix (correlations); want atmosphere -> pixel-field, terminal-panels; have several gridded fields -> composite / relief / diff; have a photo -> channel-mix. Always pass real `data` when the card is about results, state ranges explicitly, keep `mode=card` for thumbnails, and label synthetic or aesthetic output.
