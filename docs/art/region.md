# Region / material interpreter, layered images, collage (`region-*`, `layers-*`, `collage-*`)

Library round 2, agent `region`. Ten effects, three helper modules, one test file
(`tests/test_art_region.py`, 45 tests). All deterministic in (params, seed, size, palette, source, data),
palette-driven, <= 6 s at 1500x1000 (measured: region 0.5-2 s, layers 2.4-3.9 s, collage 0.7-4.1 s, subject sticker 2.7 s).

## Overlap table (what already existed, what was the gap)

Rendered on the same crops (water/rock, sunset, ducks, clouds, soil): `feature-regions`,
`feature-sunset-grade`, `feature-contour-relief`, `feature-halftone-screen`, `feature-flow-hatch`,
`feature-ripple-refraction`, `grade-teal-orange` (sheet `overlap_existing.png`).

| need | exists | real gap |
|---|---|---|
| partition a photo, edit each part | `feature-regions` (+ `features.region_weights`): hard/soft seams, but regions are quantiles / Voronoi / superpixel k-means, and each region gets a *stylised* feature effect (halftone, contour) | regions that follow real material boundaries, are simplified and snapped to edges, can be hand-corrected, and keep the *physical texture* |
| per-region colour | `feature-sunset-grade` re-keys the whole image to ONE hue; the 36 `grade-*` are global | hue families arranged per region, lightness bands per region, chroma shaped by lightness |
| texture control | `local` equalisation (global) | texture retention by scale (shading / structure / fine band) per region; edge-aware smoothing per region |
| layered / exploded view | nothing | oblique / isometric projection of planes with spacing, thickness, shadows |
| collage | `feature-regions` mixes effects but cannot move or overlap content; `compose.render_recipe` layers full frames | cut-out pieces with overlaps, scale, rotation, paper edges, generated surfaces from other effects, silhouettes |

Built on, not duplicated: `features.superpixels / label_means / _kmeans / fit_cover / feature_crop / detail_crops /
saliency / oklab`. `regionkit.segment(by="superpixel")` reuses the superpixels; `feature-regions` stays the place for
"different feature effect per region".

## Composable controls (every effect)

Parameter descriptions start with their group so colour, structure, source and surface stay independent:
**SOURCE** `crop zoom hue_from_source local`; **STRUCTURE** `regions by smooth merge min_area snap` (or a mask / polygons
in `ctx.data`), plus view / piece params; **COLOUR** `colour hues spread chroma contrast hue_keep unify tint`; **SURFACE**
`texture tex_scale flat calm grain edge paper shadow thick mode`.

## Hand-correctable masks (`ctx.data`)

Any region / layers / collage effect accepts: a mask image path or array (grey levels or flat colours: <= 8 distinct values = one
region each, ordered by brightness; otherwise quantised into `n`), a `.json` path, or
`{"mask": ..., "polygons": [{"region": 1, "points": [[x, y], ...]}, ...], "n": 3}`. Coordinates are in SOURCE image
space (polygons 0..1 fractions or pixels), they follow the same feature crop as the source, so a correction made on
the original stays aligned. Hand masks are polished lightly (`snap`, `smooth`); set both to 0 to keep them exact.
`region-mask-overlay` shows what an effect will use. `layers-terrain-slabs` and `layers-exploded-planes split=depth` take a
height / depth array in `ctx.data` (`{"field": a}` / `{"depth": a}` or the array itself).

## Region family (`region-*`)

Region finding: `by` = `colour` (OKLab k-means, chroma weighted, lightness weighted less so a lit gradient stays one region),
`hue`, `luma`, `detail`, `superpixel`. Close regions merge (`merge`), thin rings and islands are removed (opening +
`min_area`), the one-hot maps are smoothed (`smooth`), upsampled and pulled onto real edges with a guided filter (`snap`).
Regions are ordered dark -> light.

| effect | what | best sources | failure modes | mood | small card |
|---|---|---|---|---|---|
| `region-material-poster` | few designed colour shapes (hue families, lightness bands) with the real surface on top; big shading flattened (`flat`), fine texture kept (`texture`, `tex_scale`) | water vs rock, clouds/sea, soil, wavesim, sunsets | gradients band (lower `regions`, raise `smooth`/`merge`); busy photos give colour confetti; portraits need a mask | bold, calm, tactile | excellent: shapes first |
| `region-tonal-families` | keeps all the photographic light, arranges colour per region, accent `rim` light along boundaries | night cities, water/land, fabric | busy photos give small colour islands (raise `min_area`) | lush, dreamy, cool/warm | good |
| `region-paper-map` | screen-print: flat fills, 0-3 tone steps/region, inked boundaries out of register, paper fibre | landscapes, aerial, terrain, simulation fields | portraits become cartoon masks; grain is heavy under 400 px (`paper` 0.5) | editorial, playful, calm | good |
| `region-scale-retain` | per-region blend between edge-aware smoothed (glass, mist) and photographed texture (`keep` list dark->light) | water + rock, fog, cloud tops | tiny regions flicker smooth/rough | calm, dreamy, serene | good |
| `region-mask-overlay` | tool: regions as flat colours over the dimmed photo | any; check hand masks | - | technical | n/a |

## Layers family (`layers-*`)

Depth inferred from an ordinary photo (`split=depth`) is an APPROXIMATION: a heuristic from vertical position, focus, haze
and tone (`layerkit.depth_proxy`). It is stylised layering, not geometry. Pass a real height map or a hand mask when it matters.

| effect | what | best sources | failure modes | mood | small card |
|---|---|---|---|---|---|
| `layers-exploded-planes` | 3-4 cumulative planes (split = tone, colour, depth, region, band, channel) floated apart: iso / oblique / top view, `spacing`, `thick`, `edge`, soft shadows, hue-family tint, atmospheric dim | landscapes, aerial photos, simulation fields, materials | tone split of a flat landscape gives thin streaks (use colour); portraits get cut by tone; busy photos give dark confetti under `depth` | technical, editorial, playful | 3 planes, `thick` >= 0.015, `edge` >= 2 |
| `layers-terrain-slabs` | scalar field (height array in data / source luminance / procedural) as 4-8 terraced slabs, palette ramp, walls, contours, shadows; fills the frame | simulation fields, photos as height, real elevation grids | noisy fields give islands (raise `smooth`, lower `planes`) | calm, technical, editorial | excellent |

## Collage family (`collage-*`)

Modes: `paper` (torn edges, fibre, soft shadow) and `clean` (crisp edges, hard offset shadows).
Surfaces in `collage-digital-panels` / `-subject-sticker` are other library effects rendered with the library's own deterministic code (no AI).

| effect | what | best sources | failure modes | mood | small card |
|---|---|---|---|---|---|
| `collage-paper-cutouts` | photo recut into 5-8 region-shaped pieces whose content is moved/rotated/enlarged from elsewhere, hue-family unified, shadows and rims | textured photos, water, fabric, city | faces get sliced (shuffle 0 or a mask); ducks-type busy photos look messy | playful, editorial, tactile | keep `regions` <= 7 |
| `collage-digital-panels` | golden-ratio panel grid with even gutters filled with detail crops, `fills` list (any effect name, `crop`, `flat`), per-panel hue families, optional overlapping panels | hero / blog covers mixing photo and project visuals | a fill needing `ctx.data` renders flat; many `flat` fills look dead | technical, editorial, cool | `pieces` 5-6 |
| `collage-subject-sticker` | salient region (or masked subject) lifted as outlined sticker with offset accent echo and shadow, placed on thirds over a generated ground (`surface`) | photos with one clear subject, objects | the auto subject is a guess (sunset gives the island, toronto the sign): give a mask for people | playful, bold | excellent: 2 layers + ground |

## Failure modes worth knowing
- Segmentation is colour-based, not semantic: it cannot find "the person"; use a mask.
- Hue families are rearrangements of the palette hue, so always pick a palette whose hue you want (`hue_from_source` re-keys it to the photo).
- Light palettes: layers/slabs/paper show white canvas corners by design; `ink-on-paper` slabs are strongly saturated blue.
- `region-paper-map` and `collage-paper-cutouts` add grain: fine at 1500, busy at 288 (set `paper` / `grain` lower).

## Note for a future skill
To turn a photo into a card image: (1) `region-mask-overlay` to see regions, tune `regions`/`merge`/`smooth` (or pass a polygon/mask);
(2) pick a look: poster (bold), tonal-families (keeps light), paper-map (print), scale-retain (calm);
(3) set colour with `colour` + palette or `hues`; (4) check at 288x192; (5) for a heroic variant use `layers-terrain-slabs` with the
simulation field as source, or `collage-digital-panels` with project visuals as `fills`.
Mood tags are `mood:*` in the effect tags (`python -m wonhee.art list --tag mood:calm`).
