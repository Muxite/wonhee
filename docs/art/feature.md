# Feature-edit effects (`feature-*`, family `feature`)

Owner direction (2026-11-03): a project image is a filter/edit of a FEATURE of
the real original, a visualisation of its subject, or both. Every effect here
reads `ctx.source` (float32 RGB 0..1; a synthetic stand-in is used if absent),
is deterministic in (params, seed, size, palette, source), uses palette roles
(body = canvas to structure; accent only as small highlights, so one hue
family), avoids pure black/white, and runs in <= 3 s at 1500x1000.

CLI: `python -m wonhee.art render feature-ripple-refraction --source img.png --out a.png`.
NOTE: `render --source` resizes the source to the output size (distorts aspect);
better to load it at native aspect and let `fit_cover` crop by detail. `sheet`
has no `--source` (requested change). Contact-sheet helper used here: /tmp/fe/sheet.py.

## Common params (all effects)
`crop` feature|center, `zoom` (0 auto; 0.3-0.9 forces a zoomed crop of the densest
region; use on screenshots with empty margins), `local` (0..1 CLAHE tone, the main
"no focal point" lever), `hue_from_source` (palette re-keyed to the source's dominant hue).

## Extractors (`wonhee/art/features.py`)
luminance, to_oklab/from_oklab/oklch_field; local_equalize; edges_multiscale;
detail_energy; saliency; structure_orientation (theta along structure, coherence;
flat areas filled by a smooth seeded field); ridge_valley (Hessian); dominant_palette
(OKLab k-means), dominant_hue; superpixels (SLIC-like), label_means, label_edges;
detect_grid (autocorrelation period x/y); text_mask; feature_crop (best window of an
aspect by detail energy with an evenness term, not centre), detail_crops (n disjoint
windows), tile_grid, fit_cover; synthetic_source; region_weights (detail, tone, cells,
superpixel, diagonal, rows, columns; soft or hard seam), composite_regions.

## Effects
| name | what | best sources | failure modes |
|---|---|---|---|
| feature-ripple-refraction | source seen through a physical ripple height field (refraction + caustics) | photos, posters, heatmaps | text gets wobbly; strength > 25 destroys subject |
| feature-sunset-grade | OKLCH one-hue grade (hue param / -2 = dominant), bloom, grain | any photo | stays photographic; inherits source focal point |
| feature-flow-hatch | LIC hatching along structure-tensor orientation, cross-hatch in deep tones | photos, plots, radial graphs | flat UI: invented flow |
| feature-contour-relief | luminance as terrain: banded fill, contours, index lines, hillshade | photos, glows, heatmaps | text becomes islands; noisy skies (raise `smooth`) |
| feature-ridge-lines | Hessian ridge/valley two-ink line drawing | textured photos, UI, text | empty margins give coverage < 0.9 (use `zoom`) |
| feature-line-scan | joy-division scanline relief from tone | photos, heatmaps | thin at 288 px on flat sources |
| feature-circuit-traces | tone contours re-routed as 45-degree PCB traces, pads | photos, graphs, heatmaps | chunky on busy photos |
| feature-halftone-screen | rotated dot/line/diamond screen + accent dots | photos, glows | moire on dashboards |
| feature-superpixel-mosaic | SLIC cells posterised through palette (`color=graded` keeps colour variety) | photos, heatmaps | small text lost |
| feature-quadtree-mosaic | adaptive tiles: small where detail | screenshots, dashboards, photos | busy grid at card size |
| feature-detail-tiles | even grid of the source's own most detailed crops, graded | multi-area sources (dashboards, photos) | flat source gives near copies |
| feature-thermal | palette thermal ramp, isotherms, grid of peak marks | bright-on-dark, heatmaps | NEXUS-like centred source keeps its hot centre |
| feature-riso-layers | 2-4 misregistered, grainy overprinted ink layers | photos, posters | text screenshots turn blobby; high contrast |
| feature-blueprint-notes | drafting grid, Canny linework, annotation boxes on structures + text blocks | screenshots, diagrams, photos | smooth gradients: only grid |
| feature-edge-glitch | slices shift by their edge energy, streaks, block copies, accent ghost | UI, text, photos | flat sources barely glitch |
| feature-wave-field | luminance = wave speed and initial disturbance; interference bands over the tone | any tonal source | subject fades if `show` < 0.4 |
| feature-regions | `effects="a,b"` per region from `mode` mask, soft/hard seam, `params` JSON overrides | any | pick effects that suit the source; helper `render_regions()` |

## Choosing a feature-edit for an original (note for a future skill)
- Heatmap / matrix / plot (euglena, wavesim): contour-relief, ripple-refraction, wave-field, line-scan, flow-hatch (the data becomes terrain/flow).
- Graph / diagram / workflow (moeka, nexus): ridge-lines, blueprint-notes, circuit-traces, flow-hatch, thermal.
- Dark UI / terminal / poster screenshot (ujin, wordle): `zoom` 0.4-0.6 onto text, then blueprint-notes, ridge-lines, edge-glitch, detail-tiles, quadtree-mosaic.
- Photo: sunset-grade (hue match), ripple-refraction, line-scan, riso-layers, halftone-screen, superpixel-mosaic, regions (halftone on calm + contour on busy).
- Sources with a bright centre or large empty margin will fail the focal/coverage check: crop with `zoom`, raise `local`, or use detail-tiles/regions(mode=cells).
- Always run `python -m wonhee.art check` and view at 288x192.

Sheets: scratchpad `art-feature/` (ov_ducks, ov_moeka, ov_toronto = every effect x 3 palettes; ov_small = 288x192 check).
