# Looks: colour grades and finishing (`grade-*`)

Owner direction 2026-11-03: filters, adjustments and colour grades matter; a big library for the site now and video later; photos (people included) are welcome; deterministic math only.

Core: `wonhee/art/grading.py`. Looks: `wonhee/art/effects/grade_{cine,film,retro,mono,mood}.py`. Tools: `python -m wonhee.art.grade_cli`.

## How a look is built

A look = `Grade` (pointwise colour transform) + `Finish` (spatial finishing).

- `Grade` runs: exposure and white balance, highlight shoulder, channel matrix, master and per-channel pchip curves, linear-light contrast, black lift / white cap, OKLCH saturation / vibrance / selective hue ranges / hue pull, split-toning (shadows / mids / highlights), mono mix, gradient map (duotones), bleach-bypass overlay, skin-tone protection, then chroma-preserving gamut mapping. It is baked into a 33^3 LUT; rendering always goes through that LUT (trilinear), so a `.cube` export matches the render exactly (minus `Finish`).
- `Finish` (not LUT-able): soften / clarity / sharpen, lateral chromatic aberration, VHS (chroma bleed, ghost, jitter, tracking), halation, bloom, flash falloff, vignette, chroma noise, luminance-dependent per-channel film grain, CRT scanlines / mask / barrel, final triangular dither. Sizes are defined at 1500 px and scale with the frame; grain, scanlines and tape noise are attenuated at small sizes so a 288x192 tile is not static.
- Sources are 8-bit and band when a grade stretches the shadows, so the source gets a 1.5/255 triangular pre-dither before the LUT.
- Skin protection (`skin`, 0..1) keeps the source's skin chroma/hue where an OKLab orange-pink wedge is detected (idea of `wonhee.edit.filters.skin_weight`). Looks that intentionally recolour skin (cold-blue-flash, thermal, duotones, cyber-neon) use little or none.

Every look has params `intensity` (0..1, 0 = identity), `grain` (multiplier), `finish` (0 = colour only), `skin` (-1 = look default), `crop` (center|feature), and the effect `seed` drives grain / noise / tape jitter. `sunset-hue` also has `hue` and `pull`.

```
python -m wonhee.art render grade-teal-orange --source photo.jpg --size 1500x1000 --params '{"intensity":0.8}' --out a.png
python -m wonhee.art.grade_cli sheet photo.jpg --out sheet.png            # all looks, 360x240 tiles
python -m wonhee.art.grade_cli sheet photo.jpg --out small.png --small    # 288x192 rendered at that size
python -m wonhee.art.grade_cli describe photo.jpg                         # vibe vector
python -m wonhee.art.grade_cli nearest reference.jpg -k 6                 # looks closest to a reference's vibe
python -m wonhee.art.grade_cli export-web cold-blue-flash                 # CSS filter + SVG filter + overlay notes
python -m wonhee.art.grade_cli export-cube teal-orange teal.cube          # ffmpeg -i in.mp4 -vf lut3d=teal.cube out.mp4
python -m wonhee.art.grade_cli export-cube infrared-false-colour ir.cube --size 65
```

## Export fidelity (how to read the table)

- LUT: rms (0..255) between the exact `Grade` and its 33^3 trilinear LUT on random colours. Under 1.5 is invisible; infrared-false-colour (sharp hue sectors) is about 5, so use `--size 65`.
- SVG: `feComponentTransfer` per-channel table along the grey axis then a least-squares `feColorMatrix` (`color-interpolation-filters="sRGB"`). Rms vs the exact grade; close <4, good <8, rough <14, poor beyond. It cannot do hue-selective, split-tone or gradient-map behaviour well, so use it for the tonal / matte / mono / pastel looks and the real LUT or a pre-rendered image for the rest.
- CSS: best `brightness contrast sepia hue-rotate saturate` chain by Powell search. CSS cannot make a cast, so only mono, matte and gentle looks are usable. For anything marked poor, ship a rendered image.
- Finishing (grain, halation, bloom, vignette, flash, scanlines, CA) is not in either; `export-web` prints CSS approximations for the vignette, grain overlay and glow.

## The looks

| look | LUT rms | SVG | CSS |
|---|---|---|---|
| bleach-bypass | 0.1 | rough (13.23) | rough (13.12) |
| bw-airy | 0.0 | close (0.65) | good (7.99) |
| cinestill-800t | 0.7 | rough (11.93) | poor (14.63) |
| cold-blue-flash | 0.5 | rough (9.0) | poor (44.54) |
| cross-process | 0.7 | rough (10.82) | poor (19.36) |
| crt | 1.5 | good (7.83) | good (7.93) |
| cyanotype | 0.1 | rough (11.03) | poor (21.6) |
| cyber-neon | 1.4 | poor (25.97) | poor (33.0) |
| desaturated-documentary | 0.1 | rough (12.85) | rough (11.74) |
| duotone-amber | 0.2 | poor (18.47) | poor (20.74) |
| duotone-crimson | 0.1 | poor (14.32) | poor (22.05) |
| duotone-forest | 0.0 | close (2.4) | good (6.18) |
| duotone-indigo | 0.0 | close (3.08) | rough (11.19) |
| ektachrome-slide | 1.3 | poor (17.96) | poor (21.47) |
| faded-film | 0.1 | rough (8.4) | rough (9.18) |
| golden-hour | 0.8 | poor (25.12) | poor (26.06) |
| high-key-editorial | 0.2 | good (4.89) | good (7.69) |
| infrared-false-colour | 5.2 | poor (44.79) | poor (49.53) |
| kodachrome | 1.4 | poor (18.55) | poor (17.08) |
| lomo | 0.9 | rough (13.36) | poor (14.25) |
| matte-pastel | 0.0 | good (6.3) | good (7.06) |
| moody-forest | 0.4 | poor (14.94) | poor (21.54) |
| night-city | 0.9 | poor (14.18) | poor (16.85) |
| noir-bw | 0.1 | close (0.13) | poor (22.78) |
| overcast-cool | 0.1 | rough (8.72) | rough (11.57) |
| pastel-dream | 0.2 | good (6.43) | rough (10.27) |
| polaroid | 0.2 | good (7.52) | rough (10.21) |
| portra-warm | 0.3 | rough (12.96) | poor (14.46) |
| sepia-albumen | 0.0 | good (6.14) | rough (10.47) |
| seventies-warm | 0.5 | poor (24.42) | poor (24.42) |
| sunset-hue | 1.5 | poor (49.97) | poor (36.22) |
| teal-orange | 1.4 | poor (27.0) | poor (31.15) |
| thermal-map | 1.0 | poor (45.03) | poor (35.55) |
| tropical | 1.6 | poor (28.7) | poor (29.25) |
| vhs | 0.2 | rough (9.91) | rough (12.17) |
| y2k-digicam | 1.1 | rough (13.83) | rough (13.07) |

### bleach-bypass
*gritty, desaturated, high contrast*

Silver-retention film process: colour overlaid with its own luminance, half the saturation, hard contrast, crunchy grain.

- Best on: faces, architecture, war/doc moods, rain, metal and concrete.
- Fails on: pastel subjects; heavy shadows crush on dark photos.

### bw-airy
*black and white, soft, light, quiet*

High-key monochrome: lifted blacks, bright mids, soft contrast and a slightly cool silver tone, fine grain.

- Best on: portraits, flowers, snow, minimal architecture, quiet street.
- Fails on: needs enough light; dark photos go grey and flat.

### cinestill-800t
*neon night, tungsten film, red halation*

Tungsten-balanced stock shot under street light: cyan-blue shadows, warm highlights, strong red halation round every light, soft bloom, visible grain.

- Best on: night photos, signage, car lights, city streets, people lit by lamps.
- Fails on: daylight (turns everything cyan); flat low-light scenes with no bright lights have no halation to show.

### cold-blue-flash
*cold, flash-lit, candid, film*

Direct-flash party/portrait look: whole frame cast cold blue (red channel pulled down hard through the mids), deep navy blacks, bright near-white highlights, flash falloff (bright centre, dark corners), soft bloom, visible grain and a little chromatic aberration. Calibrated on the statistics of a film-flash portrait (blue 255 deg, black point L .16, white L .94); the photo itself is not used.

- Best on: people, interiors, night street portraits, anything with a close subject.
- Fails on: daylight landscapes (become blue monochrome); skin intentionally turns cool, so skin=0 by default.

### cross-process
*chemical accident, punk, poster*

E6 slide in C41 chemistry: steeper red and green curves, blue shadows lifted and highlights capped, yellow-green highlights, cyan-blue darks. Tamed so highlights keep detail.

- Best on: skies, fashion, street, anything you want hotter and weirder.
- Fails on: skin goes yellow-green (protection 0.35); text-heavy images.

### crt
*phosphor monitor, scanlines, glow*

Horizontal scanlines, RGB phosphor mask, bloom around bright pixels, slightly curved glass and corner falloff; colour a bit saturated.

- Best on: pixel art, UI screenshots, graphics, hero backgrounds, games.
- Fails on: photos of people get the 'monitor photo' feel; fine mask moire at some downsizes (period is clamped to 2 px).

### cyanotype
*Prussian blue print, botanical, sun print*

Cyanotype: tone mapped from deep Prussian blue to a pale blue-white paper, matte, grainy, slightly stained.

- Best on: plants, birds, architecture, portraits as botanical-plate cards.
- Fails on: colour contrast is lost: red and green subjects merge in tone.

### cyber-neon
*synthetic, nightclub, cyberpunk*

Shadows to deep violet-blue, highlights pushed pink/cyan, saturation high, strong contrast, heavy bloom. Colour, not realism.

- Best on: night photos, signage, lit people, glass and metal.
- Fails on: skin tones go magenta or cyan (skin=0.15 only); washed daylight scenes just look tinted.

### desaturated-documentary
*sober, honest, news-like*

Muted colour (-40%), mild contrast, neutral-cool shadows, slight clarity and light grain. Looks 'unstyled' on purpose.

- Best on: reportage, people, screenshots-with-photos, product shots that must not shout.
- Fails on: subjects that rely on colour (sunsets, neon): they go grey.

### duotone-amber
*duotone, dark brown to amber glow, warm screenprint*

Tone is mapped between two inks (dark ink to paper) in OKLab, with firm contrast so detail stays; every source collapses to the same palette, which makes mixed photos consistent.

- Best on: card art that must match a project's colour; photos of people (reads as a print); textures.
- Fails on: colour contrast is lost; very flat images give a one-tone wash.

### duotone-crimson
*duotone, crimson ink on warm paper, loud print*

Tone is mapped between two inks (dark ink to paper) in OKLab, with firm contrast so detail stays; every source collapses to the same palette, which makes mixed photos consistent.

- Best on: card art that must match a project's colour; photos of people (reads as a print); textures.
- Fails on: colour contrast is lost; very flat images give a one-tone wash.

### duotone-forest
*duotone, deep green ink on cream paper, botanical*

Tone is mapped between two inks (dark ink to paper) in OKLab, with firm contrast so detail stays; every source collapses to the same palette, which makes mixed photos consistent.

- Best on: card art that must match a project's colour; photos of people (reads as a print); textures.
- Fails on: colour contrast is lost; very flat images give a one-tone wash.

### duotone-indigo
*duotone, indigo ink on pale paper, calm and graphic*

Tone is mapped between two inks (dark ink to paper) in OKLab, with firm contrast so detail stays; every source collapses to the same palette, which makes mixed photos consistent.

- Best on: card art that must match a project's colour; photos of people (reads as a print); textures.
- Fails on: colour contrast is lost; very flat images give a one-tone wash.

### ektachrome-slide
*clean, cool, crisp slide*

Slide film: bluish-clean whites, strong blues and greens, high contrast and sharpness, little warmth. Fresh and commercial.

- Best on: daytime landscapes, water, architecture, cars, outdoors.
- Fails on: warm indoor light (turns cold); skin slightly cool, protected 0.5.

### faded-film
*worn, nostalgic, washed*

Lifted blacks, capped whites, 20% less colour, green-cyan shadows and warm highlights, visible grain: a print left on a windowsill.

- Best on: everything casual: street, friends, landscapes, interiors.
- Fails on: needs no contrast: poorly lit already-flat photos go milky.

### golden-hour
*warm, glowing, late sun*

Warm balance, amber highlights with a soft glow, lifted warm shadows, slightly richer oranges and greens-to-gold.

- Best on: landscapes, faces, water, buildings, anything that should feel like 6pm.
- Fails on: scenes with important blues (skies turn teal-grey); night.

### high-key-editorial
*bright, clean, magazine*

Exposure up, white roll-off kept, lifted shadows, soft low contrast, gentle warm skin and a clean cool-white cast; subtle diffusion.

- Best on: portraits, product, light interiors, fashion, daytime street.
- Fails on: dark or night images (washes out), already overexposed skies.

### infrared-false-colour
*Aerochrome, pink foliage, dreamy*

Infrared colour film: foliage and green water turn hot pink-red and bright, sky and blue deepen to violet, soft glow. Only greens and cyans are remapped, so skin and neutrals survive.

- Best on: trees, grass, parks, birds in grass, water edges.
- Fails on: scenes without green barely change (a cool tint only).

### kodachrome
*saturated, dense, 1970s slide*

Rich saturated reds and deep blues, warm cast, inky blacks, firm contrast, creamy yellows. Dense and graphic.

- Best on: sunny landscapes, signs, cars, buildings, people outdoors, ducks and birds.
- Fails on: dim indoor photos (blacks go dead); skin gets ruddy in strong sun (skin protection moderate).

### lomo
*toy camera, vignette, saturated*

Cross-processed colour, high saturation and contrast, a hard vignette and a touch of chromatic aberration.

- Best on: street, friends, parties, food, anything cheerful.
- Fails on: subtle subjects; the vignette hides corners.

### matte-pastel
*clean matte, calm, social-media*

Black point raised to a soft slate, muted pastel colour, slight cool-pink tint, minimal grain, no glow. Calmer and sharper than pastel-dream.

- Best on: product, interiors, portraits, landscapes, UI hero photos behind text.
- Fails on: needs no punch; deep blacks are intentionally gone.

### moody-forest
*dark, green, damp, quiet*

Greens shifted toward deep teal-moss and darkened, yellows muted, blacks lifted green, low saturation elsewhere, hard vignette.

- Best on: foliage, parks, hiking photos, wildlife, birds in grass.
- Fails on: no green to work with (becomes dull grey); skin keeps its colour but sits in a green-grey world.

### night-city
*electric, blue hour, urban*

Lifted navy blacks, magenta-orange highlights, teal mids, a bit of bloom: city lights glow without crushing the dark.

- Best on: skylines, plazas, signage, rain, people under lamps.
- Fails on: bright daytime scenes (look turns murky).

### noir-bw
*black and white, hard, graphic, grainy*

Red-filter monochrome (dark skies, bright skin), strong S-curve, deep blacks and bright whites, heavy grain and a vignette.

- Best on: faces, architecture, night scenes, textures, birds.
- Fails on: very dark photos crush; low-contrast subjects turn grey.

### overcast-cool
*calm, cool, quiet*

Slightly cool white balance, desaturated, soft contrast with blue shadows: grey-day Nordic mood.

- Best on: landscapes, buildings, forests, calm portraits.
- Fails on: already blue/grey images (little change); night.

### pastel-dream
*soft, glowing, candy, dreamy*

Lifted blacks, pastel saturation, pink-lavender shadows and mint-cream highlights, strong diffusion glow and hazy bloom.

- Best on: flowers, portraits, skies, interiors, light pictures.
- Fails on: dark/contrasty pictures wash out; fine detail softened on purpose.

### polaroid
*instant print, sun-bleached, cream*

Cream-yellow cast with cyan-teal shadows, low contrast, soft focus and edge falloff, slightly blown highlights.

- Best on: friends, food, summer, small snapshots, window light.
- Fails on: dark scenes (muddy); no frame is drawn, add one in layout if wanted.

### portra-warm
*soft, warm, natural, wedding*

Portra-like: warm-neutral balance, gentle contrast, lifted shadows, creamy pastel skin, slightly yellow-green foliage, smooth highlight shoulder, fine grain.

- Best on: portraits, weddings, everyday people photos, landscapes in soft light, food.
- Fails on: scenes that need punch (looks polite); very cool/blue-hour scenes lose their mood.

### sepia-albumen
*antique print, brown, quiet*

Albumen-print sepia: warm brown shadows to cream highlights, lowered contrast and capped whites, soft vignette.

- Best on: portraits, old-town architecture, landscapes in haze, heritage pages.
- Fails on: colour-driven subjects; modern signage looks anachronistic.

### seventies-warm
*brown-orange, sun-faded, retro*

Warm orange cast, greens turned olive-mustard, soft red/orange, lifted brown shadows, low contrast, soft halation.

- Best on: family-album feel, cars, interiors, parks, people.
- Fails on: blue skies turn grey-green; very dark pictures turn muddy brown.

### sunset-hue
*dusk, one-hue family, parametric*

Pulls every hue toward a target (default warm 40 deg OKLCH; try 20 red, 330 magenta, 195 cyan, 265 blue), then splits shadows cool-dark, highlights warm. The photographic sibling of feature-sunset-grade.

- Best on: any photo you need to match a project's hue, skies, water, plazas.
- Fails on: stays recognisable but loses real colour differences: wants a strong subject; heavy pull on people makes skin flat (lower `pull`).
- Params: `hue` (target OKLCH hue in degrees), `pull` (0..1 how far every hue is pulled toward it)

### teal-orange
*cinematic, punchy, blockbuster*

Shadows pushed to teal, highlights and warm midtones to amber, S-curve contrast, selective cyan/blue shift. Skin stays orange (protected).

- Best on: landscapes with sky or water, street scenes, people (skin protection keeps faces natural).
- Fails on: already orange-heavy scenes (sunsets turn flat orange), pure foliage.

### thermal-map
*heat camera, false colour*

Lightness mapped through a thermal ramp (black, violet, magenta, red, amber, near-white). Colour information is replaced; structure stays.

- Best on: night photos, glowing lights, faces (reads as a heat portrait), textures.
- Fails on: low-contrast flat images give a one-colour wash.

### tropical
*lush, bright, holiday*

Teal-turquoise water, vivid leafy greens leaning yellow, warm skin, lifted shadows, high vibrance and light bloom.

- Best on: beaches, plants, birds, summer, food, bright daylight.
- Fails on: dull grey scenes (nothing to saturate); night.

### vhs
*tape, home video, 80s-90s*

Chroma bleeding sideways, soft luma with tape edge halos, ghosting, row jitter and a tracking band, cool-green shadows and magenta highlights, low saturation but pushed colour.

- Best on: video stills, people, signage, anything you want to feel like a recording.
- Fails on: small text and fine detail disappear; tracking band is seed-dependent.

### y2k-digicam
*2000s compact camera, flash, early digital*

Hard on-camera flash feel: blown highlights, cyan-blue cast in shadows, crisp over-sharpened edges, chroma noise, tiny lateral aberration, saturated colour.

- Best on: friends, parties, cars, street snaps, anything candid; reads great at small size.
- Fails on: calm landscapes (looks wrong on purpose); skin gets a magenta-pink push (protected 0.45).

## Look finder

`describe(img)` returns mean_L, contrast (p95-p5), sat, warm (+ warm / - cool), shadow and highlight tint as OKLab (a, b), grain (high-frequency residual in flat areas), black / white point (p1 / p99) and the three dominant hues. `nearest_looks(img_or_vector, k)` renders every look on a neutral chart (grey ramp, hue wheel, skin patches, smooth field; computed once per process) and ranks looks by the weighted normalised distance between their chart statistics and the picture's. It is a first guess for "which grade is this vibe": on the owner's cold-blue film-flash reference it ranks overcast-cool then cold-blue-flash, then cyanotype / duotone-indigo (all cold-blue). Because the chart has fixed content, treat the top 3 as candidates, not a verdict.

## Picking a look (note for a future skill)

1. Decide the job. Card art that must match a project's hue: duotone-* or sunset-hue (hue param = project hue); background behind text: matte-pastel, overcast-cool, desaturated-documentary, or any look with `intensity` 0.5; hero / lead photo: the cinematic and film looks; LQIP / tiny: bold looks survive (duotone, thermal-map, cyanotype, noir-bw, cold-blue-flash), subtle ones (portra-warm, matte-pastel) read as the original.
2. Match the photo. People: portra-warm, high-key-editorial, golden-hour, teal-orange, bw-airy, cold-blue-flash (candid / night), y2k-digicam (party). Landscapes: kodachrome, ektachrome-slide, golden-hour, tropical, moody-forest, infrared-false-colour (foliage). Dense textures (ducks, crowds): duotones, cross-process, bleach-bypass, thermal-map. Night: cinestill-800t, night-city, cyber-neon, cold-blue-flash. Screens / UI / pixel art: crt, vhs.
3. Match the mood with `nearest` against a reference image; then lower `intensity` to 0.6-0.8 for people, keep `skin` at default.
4. Dark photos: lifting looks (pastel-dream, high-key-editorial, matte-pastel) go grey and amplify JPEG blocking; use looks that keep blacks (cinestill, night-city, cold-blue-flash) or raise exposure first.
5. Check at 288x192 (`sheet --small`): drop looks that turn to noise or mud there.
6. Video: `export-cube` and `ffmpeg -vf lut3d=file.cube`; grain/halation are added separately.

## Known limits

- A look cannot know the photo: the same look is stronger on already-contrasty sources (ducks) and flatter on dull ones; use `intensity`.
- Grading is per pixel, not semantic; skin protection is a colour wedge, so warm wood/sand is protected too, and very dark or very pale skin under coloured light may escape it.
- No look is a licensed stock emulation; names describe the character only.
