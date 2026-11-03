# Atmosphere and optics (`atmos-*`, `optics-*`)

Owner direction 2026-11-03 (library round 2). Real photos need *light behaving like light* (glow,
haze, reflections) and *the camera / screen as an object* (lens, display, flash, low light). A colour
grade cannot make light scatter; these are the spatial parts.

Code: `wonhee/art/atmoskit.py` (toolkit), `effects/atmos_common.py` (parameter groups, one shared
pipeline, `synth_night` generated test field), `effects/atmos_effects.py`, `effects/optics_effects.py`.
Tests: `tests/test_art_atmos.py`. Sheets: scratchpad `art-atmos/` (`atmos_overview.png`,
`optics_overview.png`, `atmos_small_288.png`, `optics_small_288.png`).

## Overlap with what exists (rendered on the Toronto night photo and the foil portrait)

| need | exists | verdict |
|---|---|---|
| bloom / halation on a photo | `Finish.bloom/halation` inside grades (cinestill-800t, crt, night-city): one fixed radius, one threshold, bound to the look, small amounts | real gap: independent, multi-scale, emitter-selected, skin- and core-aware glow |
| light effects | `light-*`, `bokeh-field`: GENERATE light; with `source` they only use luminance as a mask, `bokeh-field` replaces the photo by discs | gap: nothing keeps the photo and lights it up |
| haze / fog / depth | none (grades only lift blacks globally) | gap: depth-dependent airlight, fog banks |
| directional scatter, wet reflections, light pools | none | gap |
| lens: distortion, fringing, soft corners | `Finish.ca`, `crt barrel` (black corners, fixed) | gap: fill-cropped distortion, focus plane |
| screen capture: perspective, pixel lattice, moire, bands, glare | `crt`, `vhs`, `y2k-digicam` are colour + scanlines/jitter only | gap |
| flash falloff | `Finish.flash` (one centred blob in `cold-blue-flash`, y2k) | gap: subject-zone / depth inverse-square, skin safe |
| low-light camera | `chroma_noise`, `grain` | gap: signal-dependent noise, waxy denoise, smear |

Colour grades compose on top: run `atmos-*` first, then `grade-cinestill-800t` with `finish=0`.

## Independent controls (every effect)
colour: `haze_hue/haze_chroma`, `halation_hue`, `glow_tint`. structure: radii, `horizon`, `depth`,
`streak_angle`, `pool_count`. source: `mode` (luminance|colour|peak|both), `threshold`, `hue`, `local`,
`skin`. surface: `grain`, `anchor` (dark end kept), `core` (blown cores not stacked). Noise and
textures are functions of (pixel, seed): no frame crawl in video. Hand-authored masks via `ctx.data`
(`{"polygons": [[(x, y)...]]}` in 0..1, or a 2D array) for `depth=mask` (1 = far) and the wet ground.

## atmos effects (source = a photo; none given -> a generated night street)
| effect | what | params to touch | mood | fails on |
|---|---|---|---|---|
| `atmos-glow` | bloom (4 scales, linear light) + warm halation fringe outside highlights | bloom, bloom_radius, halation, halation_hue, threshold, mode | nocturnal, luminous, nostalgic | flat daylight (nothing emits) |
| `atmos-haze` | Koschmieder veil by depth, ambient-coloured airlight, lamps in-scatter, skin and cores kept clear | haze, depth, horizon, haze_level, haze_hue | mysterious, nocturnal | wrong `horizon`; frames with no distance cue |
| `atmos-fog` | seeded fractal fog banks, low-lying, lit by lamps | fog, fog_coverage, fog_scale, fog_height, fog_drift | mysterious, intimate | high coverage with no lamps = grey sheet |
| `atmos-streaks` | directional scatter (anamorphic, rain, diagonal, cross) | streak, streak_length, streak_angle, streak_cross | luminous, mediated | many small lights = stripes (raise threshold) |
| `atmos-light-pools` | N (default 5) big soft pools of coloured light under the strongest lamps | pools, pool_count, pool_radius, pool_squash | intimate, luminous | single dominant light |
| `atmos-wet-ground` | mirrored lights, vertical stretch, roughness smear, ripple, Fresnel gain, wet darkening inside a ground mask | wet, ground, roughness, ripple, reflect_length | nocturnal, mysterious | wrong horizon; non-street photos |
| `atmos-nocturne` | all stages, `preset` night-street / mist / neon-rain / stage / dream / none | any | nocturnal, candid | stacking everything at 1.0 blows out (presets are mild on purpose) |

Small card (288x192): scales are fractions of the frame, so glow, pools, fog, haze and wet look the
same; thin streaks and ripple detail soften. Best sources: night streets, neon, stage and window light.
Skin: emitters, haze and glow are masked by an OKLab skin wedge (`skin`); the foil hat is not skin, so
it still glows slightly.

## optics effects
| effect | what | key params | mood | fails on |
|---|---|---|---|---|
| `optics-screen-photo` | picture seen on a display: off-axis perspective (fills frame), curved glass, soft corners, pixel lattice + RGB stripes, moire beat, exposure bands, glass glare, channel displacement, bloom, vignette | tilt, barrel, softness, pitch, grid, subpixel, moire, bands, displace, glare | mediated, nostalgic | tilt > 0.12 leans faces; lattice deliberately absent below ~2 px pitch (small cards keep moire, bands, bloom) |
| `optics-lens` | distortion (cropped), lateral fringing, sharp zone or tilted focus plane, veiling bloom, vignette | distortion, fringe, focus, softness, bloom, vignette | candid, intimate | distortion > 0.2 near faces |
| `optics-flash` | 1/d^2 flash from a subject zone or depth gradient, specular shine, ambient floor, slightly warm shadows, shadow noise; skin protected | falloff, centre_y, reach, ambient, depth, specular | candid, nocturnal, nostalgic | landscapes, photos already flashed |
| `optics-lowlight-cam` | shadow push, signal-dependent shot noise, chroma blotches, waxy denoise, highlight smear, clip, sharpen halos | iso, denoise, push, smear, clip, sharpen | nocturnal, candid, mediated | bright daylight |

Render budget (Toronto 1500x1000): atmos 0.5-1.7 s, optics 0.5-2.0 s.

## Note for a future skill: atmosphere and optics
1. Night photo wants light: `atmos-nocturne` preset night-street, tune `threshold`/`mode`; add `grade-cinestill-800t`
   with `finish=0` afterwards. 2. Want depth or mystery: `atmos-haze` (set `horizon`) then `atmos-fog`.
3. Wet street: set `ground` to the real ground row; lower `wet` if the photo already shows reflections.
4. "Seen through a device": `optics-screen-photo`; "candid": `optics-flash` then `optics-lowlight-cam`;
   "old lens": `optics-lens`. 5. Always check people: skin must stay natural, then view at 288x192.
