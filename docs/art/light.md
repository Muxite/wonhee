# Light and lightning effects (`family="light"`)

Owner direction 2026-11-03: light paintings and lightning are the favourites;
make many effects and see how they combine. All are deterministic numpy/scipy/
cv2 math (no generative AI), no focal point (detail spread over the frame),
readable at 288x192, <= 2 s at 1500x1000 on one CPU core (measured 0.3-1.9 s).

Code: `wonhee/art/effects/light_*.py`; shared toolkit `light_common.py`.
Tests: `tests/test_art_light.py`. Sheets: one PNG per effect (4 seeds x ember,
ultramarine, violet-dusk) in the scratchpad `art-light/`.

## How the light pipeline works (why they share a look)

Every effect accumulates scalar HDR *exposure* additively (bilinear splatting of
densely sampled curves; intensity per unit length, so it is independent of
sampling density), then `light_common.develop` applies a multi-scale glow, a
filmic curve `1 - exp(-k I)`, and a lookup in a **light ramp**
canvas -> structure -> accent -> pale tinted highlight, over a faintly wandering
tinted ground (never black). Contrast is therefore lightness only. The accent
role contributes lightness and only 30% of its hue offset, so complement-accent
palettes (ultramarine) stay one hue family with a pale warm-tinted highlight
instead of a blue+yellow rainbow. Counts (particles, strands, stars, streaks)
scale with frame size so thumbnails keep the same density and exposure.

## Hooks for real material (all effects)

* `ctx.data`: `{"nodes": [...], "edges": [...]}` or a plain list of `[x, y]`.
  Nodes may be `[x, y]`, `{"x","y","w"}` or `{"id"}` (no coordinates: a spring
  layout is computed). Edges are index pairs, id pairs or `{"source","target"}`.
  Coordinates are fitted to the whole frame (6% margin), aspect not preserved.
* `ctx.source`: image; its luminance is resampled to the frame.
Each effect says below what it does with them.

## Effects

| name | models | best palettes | data | source |
|---|---|---|---|---|
| `light-ribbons` | long-exposure light painting: bundles of strands around smooth random curves, brightness ~ 1/speed, twist pinches, chromatic fringe | ember, violet-dusk | ribbon y positions follow the point y values | luminance scales each ribbon's exposure |
| `lightning-dbm` | dielectric breakdown (Lichtenberg figures): Poisson field on a periodic grid, growth prob ~ phi^eta, many roots tile the frame, trunk brightness ~ sqrt(sub-tree) | ultramarine, violet-dusk | points are discharge roots | luminance gates growth (image shows as density) |
| `lightning-bolts` | midpoint-displacement bolts + recursive forks, 2 re-strikes, hot core and halo | ultramarine, violet-dusk | points are strike targets (bolts from the top edge) | luminance gates brightness |
| `dla-crystal` | diffusion-limited aggregation (vectorised random walkers), parent tree, young tips bright, trunks thicken | green-gold, ultramarine | points are crystal seeds | luminance sets sticking probability |
| `aurora-curtains` | stacked curtains, field-aligned rays, exponential decay above the lower rim, fold foreshortening | green-gold, teal-coral | x positions boost ray clusters | luminance modulates rays along x |
| `star-trails` | star arcs about an off-frame celestial pole, power-law magnitudes | ultramarine, ember | points are bright stars | luminance scales star flux |
| `laser-billiards` | Sinai billiard: specular beams from many emitters bouncing off walls and circular mirrors | ultramarine, violet-dusk | points are emitters | luminance attenuates beams |
| `caustic-pool` | refraction caustics: irradiance = 1/abs(det(I + d Hess h)) of a smooth height field (fold lines = bright filaments) | ember, teal-coral | unused | luminance modulates sunlight |
| `flow-streaks` | thousands of comet trails in an incompressible curl-noise flow (RK2), head bright, tail fades | green-gold, ember | points are +/- vortices | luminance steers flow along contours and sets density |
| `firework-bursts` | ballistic sparks with drag and gravity (closed form), each burst at its own age, glitter beads | ember, ultramarine | points are burst centres (size from weight) | luminance scales burst energy |
| `bokeh-field` | defocused lights through an n-blade aperture, rim brightening, 3 depth layers, blue-noise placement | ember, ultramarine | points are big bright emitters | **the image defocused**: disc brightness = luminance at its centre |
| `plasma-arcs` | fractal arcs along a Delaunay graph (or the data's edges), 5 re-strikes, current ~ 1/length, glowing electrodes | ultramarine, violet-dusk | **graph nodes -> electrodes, edges -> arcs** | luminance scales arc current |
| `neon-circuit` | neon tubes routed on a 45-degree lattice with Chaikin-rounded corners, ring terminals, some dim unpowered glass | violet-dusk, ember | **edges routed orthogonal+45 between nodes** (wiring diagram) | luminance scales tube brightness |
| `rain-shutter` | rain streaks at 3 depths seen through a rolling shutter (shear + shake + row flicker bands), beads at the heads | teal-coral, green-gold | points are lamps boosting nearby streaks | luminance scales streaks |

Common params (every effect): `exposure` (overall gain). Others are listed in
`python -m wonhee.art list --family light` with descriptions.

## Failure modes and how to avoid them

* **light-ribbons**: too few ribbons leave coverage < 0.9 on some seeds (default
  15 is fine). `twist` high makes knots that read as focal spots.
* **lightning-dbm**: `eta` > 3 gives sparse stringy figures, < 1.2 dense blobs;
  `fill` too low leaves empty tiles. The domain is periodic on purpose (a hard
  frame edge makes the discharge hug the border).
* **lightning-bolts**: more than ~12 bolts or forks > 8 turns into noise. Vertical
  by design; set `angle` for a slant.
* **dla-crystal**: isolated clusters on empty ground when `seeds` < 25 (coverage
  weakens); lower `stick` makes smoother, bushier crystals.
* **aurora-curtains**: overlapping curtains make hard-edged dark wedges;
  keep `curtains` 4-6. Blue-green hues read best (green-gold).
* **star-trails**: the lightness-range check hovers near 0.25; do not lower `exposure`.
* **laser-billiards**: large `bounces` x `rays` becomes wallpaper; mirrors are dark
  holes, keep them small.
* **caustic-pool**: `relief` > 1.6 closes into mush; < 0.6 leaves smooth blobs.
* **flow-streaks**: bright washes if `particles` is raised without lowering `exposure`.
* **firework-bursts**: densest of the set; the `lightness range` metric reports
  ~0.22 (< 0.25) because bright pixels are under 5% by design. Visually fine.
* **bokeh-field**: with `source`, a flat image gives a uniform disc field.
* **plasma-arcs**: with sparse data (< 12 nodes) empty regions appear; add
  background nodes or use the ctx.data points as a subset of the blue-noise set.
* **neon-circuit**: seed-dependent centre bias (about 1 in 12 seeds > 1.6);
  re-roll the seed. With data, node positions decide coverage.
* **rain-shutter**: `shear` > 0.5 reads as slanted hair, not rain.

## Tying effects to projects (for per-project agents)

| project shape | effect |
|---|---|
| dependency / call / import graph | `plasma-arcs` (organic), `neon-circuit` (engineered wiring) |
| time series, signal, "flow of data" | `light-ribbons`, `flow-streaks` |
| failures, errors, alerts, "spikes" | `lightning-bolts`, `lightning-dbm` |
| growth, accumulation, learning | `dla-crystal` |
| a photo / screenshot the project owns | `bokeh-field` (defocused), `caustic-pool`, `flow-streaks`, `lightning-dbm` with `source` |
| celebration, launch, scale-out | `firework-bursts` |
| optics, wave, simulation | `laser-billiards`, `caustic-pool` |
| monitoring, streaming, logs | `rain-shutter`, `star-trails` |
| ambient / atmosphere | `aurora-curtains` |

## Note for a future skill: choosing a light effect

1. Does the project have a graph or point set? Pass it as `ctx.data` first
   (`plasma-arcs` / `neon-circuit` / `lightning-dbm` roots / `firework-bursts`).
2. Does it have a real image? Pass it as `ctx.source` (`bokeh-field`, `flow-streaks`).
3. Otherwise pick by mood in the table above. One effect per card; if layering,
   screen/add two light effects with different structure (ribbons + bokeh, caustic +
   star-trails), keep one palette, add `bloom` + `grain` post ops sparingly
   (effects already bloom inside).
4. Always run `python -m wonhee.art check` (coverage >= 0.9, centre_bias <= 1.6,
   focal_ratio <= 3) and view the 288x192 downscale before accepting.
