# Physics / computation effects (`physics-*`)

Owner direction 2026-11-03: card images are made of real physics or
computation (WaveSim interference, the euglena score heatmap are the bar).
All of these are numpy/scipy/cv2 math with a seed: no generative AI. Each
fills the whole frame with evenly spread detail (the `check` metrics pass:
coverage >= 0.90, center_bias <= 1.6, focal_ratio <= 3.0), reads at 288x192,
and renders in well under 8 s at 1500x1000 (single core).

```
python -m wonhee.art render physics-marbling --out a.png --palette ember --seed 3
python -m wonhee.art sheet  physics-caustics --out s.png --seeds 0,1,2,3 --palettes ember,ultramarine,teal-coral
```

Conventions shared by every effect
- Colour: the frame is one scalar field mapped through the palette ramp
  (canvas -> structure -> accent). Body of the picture lives in canvas..structure;
  the accent only touches crests, nodal lines, hot caustic points.
  So any hue works; two-hue palettes (ultramarine = blue+yellow accent) are
  the loudest, mono-ish ones (ember, teal-coral) the calmest.
- `ctx.data` formats: a list of `(x, y, w)` with x, y in 0..1 (fractions of the
  frame, y down) = point sources / charges / vortices / drops. A 1-D list of
  numbers = a series (one tile per value) for the tiled effects.
- `ctx.source`: an image (float HxWx3, 0..1). Used as the *medium* of the
  physics (see per effect), always through luminance.
- Shared helpers live in `physics_common.py` (`lookup`, `fit`, `get_points`, ...).

## The effects

### physics-marbling  (strongest)
Paper marbling as exact area-preserving flows: ink drops and comb tines,
each pixel traced backwards through the inverse maps. Params: `drops` 48,
`combs` 2, `tines` 16, `rings` 5, `scale` (working-res factor).
data: points = drop positions (w = radius). source: the image becomes the ink
(a marbled photo). Palettes: ember, teal-coral, ultramarine. Failure: too many
drops and combs gives mush; `combs` 0 gives plain concentric drops.

### physics-caustics  (strongest)
Light refracted by a random height field; rays splatted where they land
(the pool-floor network). Params: `scale` 0.07 (feature size), `depth` 0.9
(sharpness), `octaves` 2, `gamma`. source: luminance is the surface height (a
glass relief of the image). Palettes: all three; teal-coral reads as water.
Failure: depth > 1.4 gives sparse thin threads (low coverage).

### physics-vortex
Pseudo-spectral 2D vortex dynamics stirring a striped dye (spirals, shear
roll-up). Params: `vortices` 26, `steps` 55, `stripes` 14, `strength` 1,
`radius` 0.085. data: (x, y, circulation) vortices. source: the image
luminance is the dye (stirred, recoloured). Failure: steps > 120 or
strength > 1.5 over-winds into noise; stripes < 6 loses the spiral read.

### physics-wavemedium  (strong, closest to WaveSim)
Wave equation with a smooth random speed map: wavelength shrinks in slow
patches, fronts refract. Params: `sources` 6, `contrast` 1.1, `wavelength`
0.07, `steps` 700, `patch` 0.2. data: points = sources. source: luminance =
wave speed. Failure: contrast 0 collapses to plain wave-ripple.

### physics-interference
Analytic Huygens sum of a few coherent sources (rings, hyperbolic fringes).
Params: `mode` points|lattice|slits, `sources` 4, `wavelength` 0.06, `gain`,
`spacing`. data: (x, y, amplitude) sources. `lattice` gives Talbot carpets,
`slits` two facing slit arrays. Failure: sources > 8 and short wavelength
turns into Turing-like dots; the singular dot at each source is visible.

### physics-chladni
Chladni sand figure: all rectangular-plate eigenmodes near one resonance
(`order` 17, `band` 0.7), sand on the nodal lines (`sand` 0.17). data: drive
points (mode amplitudes = response at those points). source: luminance warps
the plate. Failure: `order` < 8 reads as a blobby maze.

### physics-spiral
Complex Ginzburg-Landau spirals (BZ reaction / heart tissue). Params `alpha`
0.4, `beta` -0.7, `steps` 700, `cell` 15 (arm size), `bands`. source:
luminance detunes the local frequency. Failure: 1 + alpha*beta < 0
(alpha 0.9, beta -1.2) goes into defect chaos (noisy mush); needs ~600
steps to nucleate spirals.

### physics-grayscott
Gray-Scott reaction-diffusion, `regime` maze|spots|worms|mitosis|coral|
solitons|chaos, `steps` 7000, `cell` 4.2, `feed`/`kill` overrides. source:
luminance modulates the feed rate (regime changes across the image).
Failure: `spots` need more steps to fill; `chaos`/`solitons` regimes depend on
seeding and may leave flat areas (the tests only guard the default).

### physics-swift-hohenberg
Swift-Hohenberg pattern formation (convection): hexagons melting into
stripes with dislocations. `pattern` mixed|hex|stripes, `wavelength` 0.034,
`r` 0.35, `steps` 420. source: luminance sets hexagon vs stripe. Failure: r <
0.1 never saturates (low contrast).

### physics-fieldlines
Many charges: LIC streamlines along E plus equipotential glow. `charges` 14,
`length` 40, `equipotentials` 0.15, `soft` 0.04. data: (x, y, q) charges.
Weakest at thumbnail size: fine hair texture; raise `length` and keep
`equipotentials` low. Failure: equipotential > 0.4 makes a ladder pattern.

### physics-harmonograph
Damped coupled-pendulum curves, one per tile (`tiles` 15). Params `detune`,
`decay`, `samples`, `gap`. data: 1-D series, one tile per value (the value
seeds that tile's frequency ratio, so a project's series maps to a poster).
Reads as a lace/contact sheet; empty tile corners are deliberate.

### physics-attractor
Clifford and de Jong strange attractors as log-density, one per tile
(`tiles` 6, `kind` mixed, `points` 900k). data: 1-D series seeds each tile.
Tiles whose parameters give thin orbits are re-rolled; with `tiles` > 12 some
cells stay faint.

### physics-moire
Overlaid curved line gratings (`gratings` 2, `pitch` 0.02, `warp`, `duty`).
source: luminance displaces the lines (the moire carries the picture).
Failure: pitch < 0.012 aliases at card size.

### physics-grains
MBO curvature-flow grain growth: foam/polycrystal. `phases` 7, `steps` 55,
`sigma` 1.7, `lines`. source: luminance quantised into phases is the initial
condition (a coarsened image). Reads as terrazzo/camouflage; the quietest.

## Choosing a physics effect for a project (note for a future skill)

1. Ask what the project *is*: a wave/signal thing (WaveSim, audio, radio)
   -> `physics-wavemedium` / `physics-interference` / `physics-chladni`.
   Points (sensor sites, nodes, cities) -> use them as `data` sources/charges/
   vortices (`fieldlines`, `interference`, `wavemedium`, `vortex`).
   A series or matrix (benchmark scores, per-run metrics) -> one tile per
   value in `harmonograph` / `attractor`, or cell centres as `data` points.
   A picture (screenshot, product photo) -> `source`: marbling, vortex,
   moire, caustics, grains and gray-scott all turn the image into the medium.
   Growth/emergence (evolution, biology, ML training) -> grayscott,
   swift-hohenberg, spiral. Optics/graphics -> caustics, moire.
2. Pick the palette by mood: ember/teal-coral calm; ultramarine loud.
3. Run `sheet` with 4 seeds, `check` the pick, look at the 288x192 downscale.
4. Optional recipe post ops: a light `bloom` helps interference, spiral,
   wavemedium and chladni crests; do not bloom the line-art ones
   (harmonograph, attractor already glow).
