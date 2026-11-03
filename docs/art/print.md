# print / swiss / dither (`print-*`, `swiss-*`, `dither-*`)

Kits: `wonhee/art/printkit.py` (separation, screens, paper, grain, registration, wear, overprint), `swisskit.py` (grid, packing, forms, type, press), `ditherkit.py` (palettes, Bayer/blue/diffusion, pixel grid). Tests: `tests/test_art_print.py`. Sheets: scratchpad `art-print/` (`ov_print|swiss|dither.png`, `small_*.png` = exactly 288x192).

## Overlap with what existed (same crops: ducks, foil portrait, wavesim)
| existing | what it does | real gap |
|---|---|---|
| feature-riso-layers | 2-4 percentile-tone layers, one hue family, noise grain, roll() offsets | no hue/channel separation, no screens, no paper, no clean variant, no dark stock, blotchy olive in dark |
| feature-halftone-screen | one faint dot screen | single ink, low contrast, no angles per ink |
| vibe screenprint / risograph | fixed 2-ink photo filters (nice weave/streaks) | no screens, no n-ink, no layout, no independent controls |
| dots-halftone-lines / dots-stipple / dots-bayer | scalar field -> mask; Bayer only in panels | photo colour, arbitrary palette, Sierra/JJN/Stucki/Burkes, scale-consistent pixel grid |
| grade-* | colour only | all spatial parts built here |

## Independent controls
Colour: `variant` (auto|light|dark), `inks`, `paper`, `colors`/`hues`. Structure: `sep`, `inkn`, `screen/kind`, `pitch`, `angle`, `cols`, `algo`, `density`. Source: `crop zoom local` (photo, 2-D `ctx.data`, or generated shapes). Surface: `clean`, `texture`, `fibre`, `grain`, `reg`, `wear`.
Scale awareness: lengths authored at 1500 px, scaled, floors (screen pitch 4.2 px, registration 1.4 px, dither pixel 2.6 px). Big shapes untouched.

## Effects
| effect | mood | what / best sources / failure | 288x192 |
|---|---|---|---|
| print-screenprint | tactile, archival, editorial | 1-4 flat ink shapes by tone/levels/hue/cmy on fibrous paper, `screen` none/dot/line/fm. Best: photos, silhouettes. Fails: text screenshots blobby | strong |
| print-riso-dots | tactile, playful | per-ink dot screens + registration, hue regions or channels (false colour). Best colourful photos. Fails: flat UI | good |
| print-newsprint | archival, editorial | one coarse dot/line screen + optional flat tint. Best readable halftone | strong |
| print-overprint-shapes | playful, structured | generated circles/rings/bars/quarters bleeding off all edges, multiply overprint; `data` sets sizes | strong |
| print-nine-ways | editorial, structured | 3x3 sheet: outline+anchors, pixels, spiral, scribble, dot rings, bands, halftone, hatch, stipple (+silhouette, contour). Subject = ink-blot or photo. Photos become abstract (use simple subjects) | readable (96x64 cells) |
| swiss-grid-poster | structured, editorial | Swiss-inspired grid: merged modules, anchor block, forms, overprint, optional photo crops, hairline grid, optional type from `ctx.data` (str / list / {title,subtitle,caption}; Liberation Sans), light/dark | strong; type dropped when < 9 px |
| swiss-form-tiling | structured, playful | every module a form; all-over pattern | strongest |
| swiss-photo-grid | editorial, structured | photo cut into merged modules, duotone/halftone, solid blocks | good |
| dither-photo | digital, archival | any palette x 14 algorithms on a logical pixel grid (`cols`), same look at all sizes. Fails: faces < 70 cols; diffusion worms on flat gradients | strong |
| dither-blocks | digital, playful | coarse blocks + light ordered dither + gaps | strongest |
| dither-gradient | digital, playful | generated warped bands dithered to a 1-2 hue ramp | strong |
| dither-stipple | archival, tactile | Lloyd colour stipple, any palette, per-dot tint | fine; dots coarsened at small size |

## Notes
- Swiss effects intentionally use negative space: `check` reports focal/centre failures for photo-grid (fr ~3.7); form-tiling borderline (fr 2.9-3.1). Poster passes.
- Render times at 1500x1000: 1.2-2.8 s print/swiss, 0.1-0.5 s dither, 4.2 s dither-gradient.
- Fonts: Liberation Sans (Helvetica metrics) from /usr/share/fonts; falls back to DejaVu; `font` param overrides. Not bundled in repo.
- Near-misses: nine-ways on busy photos; riso-dots on mono palettes only opens a second hue via the +105 rule.

## Note for a future skill
Pick a mood first: tactile/archival -> screenprint, newsprint, stipple; playful -> riso-dots, overprint-shapes, form-tiling, dither-blocks; structured/editorial -> swiss-grid-poster (pass the project name as `data` for type), nine-ways; digital -> dither-*. Always render `clean=true` and textured, light and dark, and check 288x192.
