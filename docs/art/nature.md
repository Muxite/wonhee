# Nature, motion and contour effects (`nature_*`, `motion_*`, `contour_*`)

Round 2 of the library (owner direction 2026-11-03: site first, any media but
not fully AI-generated, many looks, 288x192 must work). 15 effects in three
families. Code: `wonhee/art/effects/{nature,motion,contour}_*.py`, helpers
`natkit.py` (fields, flow, colour ramp, surface), `motionkit.py` (frame
sequences), `contourkit.py` (field, line coverage, masks). Tests:
`tests/test_art_nature.py`. Sheets: scratchpad `art-nature/` (`fam1.png`,
`fam2.png`, `fam3.png`, `overview_288x192.png`, overlap sheets `ov1.png`,
`ov2.png`).

## Controls are separated (every effect)

| group | what | typical params |
|---|---|---|
| STRUCTURE | the scalar field / lines / flow | `scale`, `angle`, `aniso`, `layers`, `spacing`, ... |
| SOURCE | photo / data / sequence feeding it | `use_source`, `source_mode`, `crop`, `scene`, `frames`, `ctx.data` |
| COLOUR | scalar -> colour only | `ramp` (mono / dual / spectrum), `tint`, `contrast`, `invert`, `colour_mode` (photo / ramp) |
| SURFACE | spatial finish, not a LUT | `grain`, `haze`, `shade`, `min_px`, `wobble` |

Colour is one OKLCH ramp from the palette roles (lightness carries contrast,
hue drift bounded; `spectrum` sweeps ~150 degrees with monotone lightness so it
never muddies). Light palettes give ink on paper. Each effect has mood tags
(`mood:dreamy`, `mood:energetic`, `mood:cinematic`, `mood:calm`, `mood:restless`,
`mood:cartographic`, `mood:precise`, `mood:hand-drawn`, `mood:scientific`,
`mood:earthy`).

## Overlap table (what existed, what the real gap was)

| area | nearest existing (rendered, `ov1.png`/`ov2.png`) | verdict |
|---|---|---|
| pale ribbons on dark water | physics-vortex / marbling (all-over high-frequency swirl), light-ribbons (thin glowing wires) | GAP: calm, mostly empty water, few broad ribbons, one coherent direction -> `nature-plankton-ribbons` |
| granular bands, striated rock | physics-grains (voronoi blobs), feature-line-scan | GAP -> `nature-strata` |
| clouds | none (swift-hohenberg/grayscott are patterns) | GAP -> `nature-cloudfield` |
| aurora | aurora-curtains: vertical bars with black bases, flat | GAP: draped, overlapping, perspective curtains with a bright rim -> `nature-aurora-sheets` (curtains kept) |
| photo-derived flow | feature-flow-hatch (LIC hatch lines only), feature-ripple-refraction | GAP: the photo itself dragged along its own flow -> `nature-flow-wash` |
| time / motion | rain-shutter, star-trails, flow-streaks are single-image simulations of streaks; NO sequence input anywhere | GAP: real frame sequences -> five `motion-*` effects |
| contours | feature-contour-relief (one weight, 0.8 px lines, speckle on busy sources), feature-ridge-lines, feature-line-scan, feature-flow-hatch (LIC noise, random spacing) | PARTIAL: reused the idea, added hierarchy, min width, despeckle, masks, wobble, exact spacing, new line languages |

## Nature family (flow, material, natural fields)

* `nature-plankton-ribbons` (mood: dreamy, calm, cinematic). A tracer dragged
  by a stream-function current (sheared along `angle`, so ribbons share one
  direction); `boundary` adds a coast the flow must follow (psi vanishes on
  it); `use_source` lets a photo's luminance iso-lines steer the current
  (`source_mode` luminance / orientation / both); `time` slides the pattern
  (coordinate-based). Best: dark palettes (teal, ultramarine), light palette gives
  ink ribbons. Fails: very high `eddy`/`stretch` curl into soup. Card: 3-5 broad ribbons.
* `nature-strata` (earthy, calm, precise). Bedding coordinate warped by folds,
  faults, optional photo luminance; lognormal beds, per-bed tone/grain,
  ragged contacts, lamination, striations, mineral speckle. `ramp=dual|spectrum`
  adds bed-to-bed hue drift. Fails: fold > 0.8 looks like wood grain. Card: ~10 bands.
* `nature-cloudfield` (calm, dreamy, cinematic). `kind` cumulus | stratus |
  cirrus | streets; stacked decks that shade each other, lit from `sun`;
  photo luminance gathers clouds. Best kind: cumulus (dark and light variants good);
  cirrus/stratus are plainer. `time` drifts decks with parallax.
* `nature-aurora-sheets` (dreamy, cinematic). Curtains as projected surfaces
  (perspective, edge-on flare, ray noise, altitude hue via a second ramp).
  Fails: leaves the lower third dark (it is a sky). Weakest nature effect for
  the no-focal-point rule; fine for blog covers.
* `nature-flow-wash` (dreamy, painterly). Photo smeared by LIC along its own
  structure orientation, iso-contours, a drift or a curl field; `detail` keeps
  sharp structure where coherence is high; `colour_mode=ramp` repaints the smear.
  Best sources: photos with large forms; people stay legible at small `smear`.

## Motion family (sequences, stills now, video later)

Sequence source = `scene`: `frames` (directory, video via cv2 with an ffmpeg
fallback, or `ctx.data` = list/array of frames), `pan` (camera move over the
source with luminance parallax and seeded shake), `orbs`, `plankton` / `clouds`
/ `aurora` (the library's own effects rendered over their `time`),
`effect:<name>`, `auto`. Temporal coherence: every random quantity is a pure
function of (seed, x, y, t) (`natkit.tnoise`), `t0`/`span` select a window,
`n_frames` only changes sampling (test:
`test_sequence_frame_independent_of_frame_count`), so the look runs unchanged on video.

* `motion-long-exposure` (energetic, dreamy, cinematic): mean + lighten trails
  with persistence, rear-curtain `sharp`, `bloom`. Best: pan of a night photo
  (broad directional blur), orbs (light painting). 
* `motion-stroboscope` (energetic, restless): `mode=movers` repeats what
  differs from the aligned median background; `multi` is a multiple exposure
  (best on photo pans). Echoes step through the colour ramp (`temporal_hue`).
* `motion-slit-scan` (dreamy, restless, cinematic): `warp` time-displacement
  (angle, bands, seeded wobble) or classic `slit`. Needs a long move: default
  `travel` 0.9. 
* `motion-frame-diff` (energetic, restless, scientific): denoised, floored,
  auto-exposed signed differences in two hues; `widen` for thumbnails.
  Weak on photo pans (dense parallax comb texture); strong on orbs and flow sequences.
* `motion-flow-trails` (energetic, cinematic): Farneback flow drives a LIC
  smear of the mid frame; `min_blur`, `freeze`. Strong on pans of night photos.

Without a source, slit-scan / frame-diff / flow-trails default to the plankton
sequence (dense motion); others default to orbs.

## Contour family (multiscale line languages)

Shared FIELD controls: `scale` (coarse smoothing), `denoise` (median +
despeckle), `use_source`, `local`, `mask` (+`mask_amount`: tone / detail /
saliency), data arrays in `ctx.data`, procedural terrain when nothing is given;
LINE controls `min_px` (minimum width in OUTPUT pixels: thumbnails keep their
lines) and `wobble` (hand-drawn, coordinate-based).

* `contour-topo-multiscale` (cartographic, precise, scientific, calm): index +
  minor + hairline levels, packing-density suppression (`min_gap`), hypsometric
  fill, hillshade. Dark = glowing lines, light = ink on paper. Best: procedural
  terrain, data fields, smooth photos (raise `scale`).
* `contour-banded-hatch` (hand-drawn, cartographic): few flat tone bands, each
  hatched at its own angle, weight by tone, boundary lines. The most distinctive
  card look (6 bands at 288, engraved map at full size).
* `contour-streamlines` (hand-drawn, precise, scientific): evenly spaced,
  non-crossing streamlines (Jobard-Lefer) of contour / orientation / curl /
  drift / radial fields with tone-dependent spacing and width swell. ~2.5-4 s at
  1500x1000. Best overall of the family on procedural paper and on orientation of skies/water.
* `contour-hatch-engrave` (hand-drawn, precise): analytic phase-field hatching
  that bends with tone (banknote), crossing hatches in the deep tones. Pitch
  floor 3.2 px (moire guard at 288). Excellent on portraits.
* `contour-edge-ink` (hand-drawn, scientific): two-scale Canny, percentile
  thresholds, connected-component despeckle, widening. Clean on portraits and
  objects; sparse on textured scenes.

## Failure modes and notes for a future skill

* Chaotic flow effects are realization-dependent: a large-sigma `fbm` on a tiny
  grid has a mean offset (fixed: `natkit.noise` removes the mean). Coverage
  of plankton ribbons can still vary by seed (0.1-0.35 of the frame).
* Photo pans for `motion-frame-diff`/`stroboscope(movers)` are not real motion;
  use real video, orbs or the nature scenes for convincing differences.
* Light palettes are ink on paper everywhere (ramp 0 = paper).
* Budget at 1500x1000 (measured): nature 0.2-2 s, motion 0.4-5.7 s (stroboscope on orbs the slowest),
  contour 0.5-1.4 s, streamlines 4 s. All under 8 s.
* Skill recipe: pick the family by subject (water/sky/rock -> nature; any
  moving material -> motion with the same `scene` the video will use;
  portrait/map/diagram -> contour), set the structure scale for 288 first,
  then colour (`ramp`), then surface (`grain`, `haze`, `min_px`).
