# Photo looks: shoot for the edit (different subject, different place, same vibe)

**Source:** the reference screenshots and posts saved 2026-10-18 (K-pop night MV frames, an idol fan-club flash shoot, graphic-design poster carousels, campus memes), measured with `wonhee/edit/style.py`. The reference images themselves stay in gitignored `data/refs/`. This file holds what we learned from them, in our words and numbers.

## What makes a look (and what the edit can and can't do)

A look is mostly four things, and all four are measurable:

1. **Tone:** where the darks, mids and lights sit (L* percentiles).
2. **Colour per tone band:** for example, cool shadows with warm highlights. This is a* / b* in shadows, mids and highs.
3. **Colourfulness:** chroma.
4. **Lens traits:** soft edges, dark corners, glow.

`wonhee vibe` moves a photo toward a reference set on all four. **The edit can't create light that wasn't there.** The light in the scene, the lens and the distance have to be right in camera; the grade finishes the job. Each look below says which is which.

## Look A: K-pop night (MV station / bedroom frames)

**Measured (4 frames):**
- Median L* is 15; the brightest 1% sits at about 83. That's a dark frame with lights that clip.
- b* is -10 to -19 in **every** band: blue-cyan all over, with no warm skin correction.
- Edge/centre detail is 0.59 and corner/centre brightness is 0.53: soft edges and dark corners, the fisheye signature.

**Shoot it:**
- **Lens:** phone 0.5× ultra-wide, or a clip-on fisheye. Get close (30–80 cm) so the near prop looks big.
- **Camera:** hip or desk height, tilted, not level. Handheld.
- **Light:** night, with the room light off. The lights *in* frame do the work: monitor, LED strip, vending machine, fluorescent tubes. Keep one in frame so it glows. There's no front key light; screen light on the face is enough.
- **Exposure:** lock it 1–1.5 stops under; the shadows should go black. Use 1/15–1/30 s and let one real action (Enter key, cable, sip) smear.
- **Places:** your desk at night (PC + Jetson glow), the E-building stairwell after dark, a night bus or train seat, a vending machine.
- **Edit:** tone + cool cast, `bloom` 0.4–0.7, `vignette` 0.5–0.8, a little `fisheye` / `edge_blur` if the lens was not wide enough, `mask=person` + `subject_lift` 0.15–0.3 so you stay readable.

## Look B: K-pop flash editorial (fan-club party shoot)

**Measured:**
- Median L* is 78 and the darkest 1% is 7: a bright frame with lifted (faded) blacks.
- It's warm-pink in every band (mid a* +22, b* +19), yet low chroma (0.26 saturation). Pastel, not saturated.
- Edges stay sharp and there's no vignette.

**Shoot it:**
- **Light:** direct on-camera flash (the phone flash works) in a room that is *already* pink or warm. Use props in one palette: balloons, confetti, tinsel, a lace throw.
- **Camera:** shoot slightly from above, full-body sitting on the floor, wide but not fisheye. Have the subject play with the camera: a hand over the face, a tilted head.
- **Places:** your room with a pink or peach throw, a party corner after a hackathon, any warm-lit cafe corner.
- **Edit:** tone + warm-pink cast at strength 0.7–0.9, a touch of `grain`, no vignette.

## Look C: two-colour print (screen print / halftone)

- **Shooting:** any photo with a clear silhouette (plush, laptop, gear on a desk). A plain background matters more than the light.
- **Edit:** `vibe_image print=duotone ink=#1a2fb0 paper=#f3ead8` (ultramarine on cream), or `print=halftone ink=#e8141c` (red dots). This is a joke format as much as a look.

## Formats to reuse for jokes

- **Spec-sheet poster** (`poster` node): big title, two label boxes, tilted image, a justified mono "prompt" paragraph. The design-study format, used to present something mundane as an exhibit. For example: title "Signal Shielding", labels "Hardware: 1 sheet of foil" and "Funding: GitHub Student Pack", body = an absurdly precise brief. Pair it with a duotone or halftone image.
- **Laptop screen as the second panel** ("3 signs he likes u"): you to camera, holding the laptop with the "slide" on it. Text is outline@top. It's a cheap two-layer joke, with no edit beyond the text.
- **Selfie + bold italic caption** (the Waterloo mall complaint): front camera, deadpan face, `text_image preset=outline position=upper`.
- **Talking-head meme with a long centred block** ("CS majors after building a webcrawler…"): static, headphones on, the text is the joke.

## How wonhee helps (commands)

| Step | Command | Cost |
|---|---|---|
| Learn a look + get a shooting guide | `wonhee vibe --like data/refs/<dir> [--like content:<id>] --name kpop-night` | about $0.003 for the guide (remote); `--guide-engine local` is $0, ~55 s on gemma4:26b-a4b, a little weaker |
| Edit your photos toward it | `wonhee vibe <photo...> --like ... --name kpop-night` | $0: the on-box gemma4:e4b picks the knobs (~10 s a photo) and the edit is deterministic. `--set fisheye=0.6` forces a knob |
| Which saved posts share a look | `wonhee inspire`, section "Looks (measured from the pixels)" | $0 (cached teardowns) |
| Spec-sheet poster | `wonhee node run poster --in image=<photo> --set title=... --set body=...` | $0 |
| Clean audio | `wonhee audio clip.mp4 [--preset voice\|mixed\|music] [--check]` | $0, about 1/4 real time |

The output folder `data/vibe/<date>-<name>/` holds:
- the edited photos
- `sheet.jpg` (before/after)
- `report.md`: what the references share, how to shoot it, and per photo the fit (1–5), the knobs used and **what to change next time you shoot**

## Audio defaults (why)

- **Loudness:** -14 LUFS integrated, true peak -1.5 dBTP, two-pass `loudnorm` (linear). That's what IG, TikTok and YouTube normalise to, so the app neither turns you down (squashing peaks) nor up (raising hiss).
- **Denoise:** RNNoise at full mix plus a light FFT pass for the residual hiss. Measured on a talking clip with added pink noise: speech-to-noise went from 18 dB to 42 dB. At mix 0.8 it removes only about 10 dB, because the 20% dry signal keeps the noise.
- **The reported noise metric is speech-to-noise** (RMS minus the noise floor), not the absolute floor. Loudness gain raises the floor too, so an absolute number would look worse after cleaning.
