# Fractured microstructure and architectural systems (`micro-*`, `arch-*`)

Library round 2, worker `micro`. Helpers: `wonhee/art/microkit.py` (hash, fbm, hue families, role colours, shared surface pass).
Effects: `effects/micro_plates.py`, `effects/arch_city.py`, `effects/arch_frame.py`. Tests: `tests/test_art_micro.py`.
All render in 0.5-2 s at 1500x1000, deterministic in (params, seed, size, palette, source, data), dark and light polarity
(light = `ink-on-paper`, `red-on-cream`, or any `hue:H:light:...`). Sheets: scratchpad `art-micro/` (overview_micro, overview_arch, overview_288).

## Overlap table (same Toronto crop, `art-micro/baseline_overlap.png`)
| existing | what it gives | gap |
|---|---|---|
| feature-circuit-traces | tone contours as 45-degree wire, pads; thin line on calm field | no large areas, no per-region grid, no plates |
| dots-circuit | BGA pad arrays, dotted buses in BSP blocks on a dot grid | dots not fills; blocks are rectangles with no fracture, no hierarchy of line weights |
| graph-btree/hash/... | rounded node-link diagrams | diagrams, not surfaces |
| feature-superpixel-mosaic, quadtree-mosaic | flat cells / squares of a photo | cells carry colour only: no oriented fill |
| feature-blueprint-notes | drafting grid and edges over a photo | one grid for the whole frame |
| (architecture) feature-* / dots-* | tone re-drawn as lines / dots | no perspective, no window grids, no structural members |

Real gaps built: (1) large jagged plates, each with its own coherent oriented fine grid, phase offset, hierarchy and substrate;
(2) perspective facades / frames with repeated window grids, piers, grouped windows and a lit-window distribution.

## Controls (independent groups, same names on every effect)
- colour: `palette` (render arg), `families` 1-3, `hue_spread` deg, `chroma`, `contrast`.
- structure: effect specific (below).
- source: `source_mix`, `hue_from_source` (micro), `lit`, `lit_shape` (arch).
- surface: `grain`, `wear` (dust/mottle), `glow` (bloom of brightest traces / lit windows); micro also `crack`, `bevel`, `tilt`; arch `haze`.
Mood tags (in `tags`): engineered, tactile, precise, urban, monumental, nocturnal, plus `light-dark`, `small-card`, `source`, `data`.

## micro-* (one engine, `render_plates`)
Plate boundaries: Voronoi / strata / macro-blocks, or from a source or data, roughened by a multi-scale warp (smooth + ridged jag).
Per plate: angle (global + 0/90/45/tilts by `orient`), lane pitch, bundles with a gap lane, long segments with pads, phase offset
(`offset`), up to 3 crossing layers, sub-blocks turned 90 degrees (`blocks`), coarse trunk layer with halo, vias and hot pads
(`hierarchy`), tilt shading and grain, cracks and bevels.
Params: `boundary plates rough angle orient pitch hierarchy layers blocks offset density crack bevel tilt` (+ groups).

| effect | what | best use | failure modes |
|---|---|---|---|
| micro-plates | Voronoi plates, 3 hue families (teal/coral, violet/rose, blue/amber by palette) | hero, cover, OG | 3 families can read busy: set `families=2` |
| micro-plates-card | 5 plates, coarse pitch, strong trunk layer | project cards at 288x192 | big plates leave less variety per seed |
| micro-strata | long anisotropic strips, aligned fills | wide banners, blog covers | complement palettes put a yellow family in (olive on dark) |
| micro-die | rectangular macro-blocks with fractured edges | hardware / infra projects | the most "grid-like"; raise `rough` |
| micro-plates-source | k-means regions of a photo; trace angle from the photo's structure (mod 90), substrate lightness and density from its tone, hues from its dominant colours | any photo, best with colour (Toronto) | warm photos collapse to one hue; subject is abstracted, not recognisable; busy photos need fewer `plates` |
| micro-plates-data | `ctx.data` 2D float field banded into plates, 2D int label image used as is, or (n,2) points in 0..1 as Voronoi seeds (`boundary` forced to data) | euglena score matrix, label maps, point sets | noisy fields give many thin plates (smooth the data or lower `plates`) |

Small card: `micro-plates-card` (3-6 plates, `pitch` 0.0075, `hierarchy` 0.9) reads at 288x192 as broad plates with a visible coarse trace
layer; the fine grid becomes texture, as intended. Render large and downscale; native 288 px renders alias the fine layer.

## arch-* (`render_city`: ray caster over boxes, facade per face; `render_frame`: vector lattice)
City params: `scene` (canyon | aerial | street | wall), `tilt yaw fov floors bay group heights pier lit lit_shape haze`.
Facade: floor height and bay width per building, piers every K bays (windows come in groups of K-1), service floors / belts every M floors,
glass with a height gradient, lit state = group hash + per-window hash x a mask shape (`scatter floors columns stair blocks`, per building
when `mixed`), windows < 3 px fade to their mean (no moire), depth haze, roofs, street grid.
With `ctx.source` the lit probability is the photo's tone sampled at each window's own projected centre; the lit colour takes the photo's local hue.

| effect | what | best use | failure modes |
|---|---|---|---|
| arch-canyon | worm's-eye towers converging overhead | monumental heroes | dark sky slit near the centre (focal_ratio 2.6, centre_bias 1.5) |
| arch-aerial | oblique block grid, roofs, 22 rows deep | covers, OG, tall dense field | coverage 0.84-0.89 (smooth sky/street) borderline on the check |
| arch-street | two-point corridor, VP off centre | wide banners | centre_bias 1.57, focal_ratio 2.9: near the limit |
| arch-wall | frame-filling curtain wall off axis, setbacks, piers, grouped windows | project cards, backgrounds | very regular; vary `group`, `lit_shape`, `floors` |
| arch-night-photo | wall scene, windows lit by a photo | photos with strong bright/dark structure (Toronto signage reads) | low-contrast photos give random scatter; subject legible only at large size |
| arch-space-frame | rolled-camera lattice: primary/secondary members, X-braces, glazed bays | engineered / infra | glazed panels sparse; dense far field is mush near the VP |
| arch-truss-hall | heavier, braced, mostly unglazed | tactile / monumental | same |

Small card: arch-wall/aerial/space-frame read at 288x192 (thick piers and belt courses, grouped lit windows); canyon and street read as towers.

## Strongest / weakest
Strongest: micro-plates, micro-plates-card, micro-strata, arch-wall, arch-aerial (light and dark). Weakest: micro-plates-data on noisy fields,
micro-plates-source on warm monochrome photos, arch-night-photo (photo only faintly legible), arch-space-frame glazing.

## Note for a future skill
Pick by subject: hardware/infra/data -> micro-die, micro-plates (+data); a photo -> micro-plates-source (abstract) or arch-night-photo (city
mosaic); city/building/night/urban -> arch-aerial, arch-wall, arch-canyon; engineering/structure -> arch-space-frame, truss-hall.
Vary colour independently (palette, `families`, `hue_spread`) from structure (`plates`, `pitch`, `hierarchy`, `group`, `floors`) and surface
(`grain`, `wear`, `glow`). Always view at 288x192 (downscale from a large render) and in both polarities.
No shared-file changes were needed. Known engine gap: `sheet` has no `--source`/`--data`; `micro-plates-data` needs `art_render.py --data`.
