# Project 1: "midnight in Waterloo" (raw filler reel)

Date: 2026-10-22. First wonhee project made end to end from the owner's own footage. Goal: get a raw, low-edit vlog
out fast, before the Vancouver audience goes to sleep. Media lives in gitignored `data/projects/2026-10-22-midnight-waterloo/`.

## Brief (from the owner)

- Source: one take on the attached flash drive, `20261022_000755.mp4` (filmed 00:07 on 2026-10-22).
- Raw vlog feel. No joke. **Do not cut pauses** (`tighten` off).
- On-screen caption (three lines, small, Instagram-like):
  `jiuwenswarm ai4research` / `open source btw` / `rogue agent fixed ✅`. All three happened in the 14 h of work today.
- Audio: isolate the human voice (car passing, clothing rustle), normalise so it is loud but not spiky. Intelligibility
  beats purity; some leftover noise is fine.
- Instagram-style captions for the speech, small, not big and bold.

## What the footage is

Portrait selfie (1920x1080 stored, rotation 90, so 1080x1920), 24 fps, 44.6 s, walking outdoors at night, dark, earbuds in.
Raw audio: −25.0 LUFS integrated, loudness range 11.8 LU, true peak −1.9 dBFS (quiet voice, loud car passes).
The speech was clear enough to transcribe in full (13 lines, 0.6 s to 43.8 s).

## What was done

| Step | Tool | Result |
|---|---|---|
| Copy off the drive (mounted read-only) | `cp` | `raw.mp4` |
| Voice isolation + loudness | `audio_clean isolate=true` (Demucs on the GPU, under the lock; 11 s) | −14.8 LUFS, LRA 5.2 |
| Transcript with word timings | `speech_subs` (whisper) | 13 lines, all words timed |
| Captions | hand-written info lines (`captions.json`), timed text layers | v2: each line appears about 0.3 s before the speech and stays until the next one; wording adds context and need not match the speech |
| Render | `compose_video`, inline layout (`layout5.json`), `tighten=false` | v2: one encode, 1080x1920, 44.3 s, 51 MB |
| Share copy | two-pass x264, 5.3 Mb/s video + 128 kb/s AAC | 28.8 MiB, under the 30 MiB chat upload limit |
| Archive | copied to the flash drive, `wonhee/` folder, checksum matched | master and share copy |

Final loudness: −14.7 LUFS, LRA 5.0 LU, true peak −1.1 dBFS (raw: −25.0, 11.8, −1.9).

Layout (v2 captions replace the v1 word-by-word auto-captions): full-bleed clip; the three-line caption in `caption` (Figtree) at 2.4 % of the height on a translucent black
box at y 0.14, for the whole video; seven short info lines in the `autocaption` role (Arimo, white on a dark rounded box) at y 0.70, 2.5 % of the height, moved into the 72 % safe width (`LAYOUT_ADJUSTED`, expected). All text sits inside the safe zones. No look, no
grade, no pause trimming, no music.

Files: `raw.mp4`, `clean.mp4` (cleaned audio, original picture), `subs.json`, `layout.json`, `midnight-waterloo.mp4`
(the deliverable).

## Not done, on purpose

- The picture is dark (night, phone camera). It was left as shot; a lift would lose the raw feel and noise grows.
- Pauses and filler words are kept (owner's call).
- No music: the owner adds in-app audio, or none, when posting.

## Versions

v1 to v4 were deleted (local and on the flash drive) once v5 existed. **v5 is the current cut**: speech captions follow the owner's words phrase by phrase at 3.4 % of the height (v2-v4 were looser, info-style lines), the top caption is 3.2 % and its first line is `14h internship day`. Master 48 MB and a 28.8 MiB share copy, both on the flash drive in `wonhee/`. Caption text and timings: `captions.json`.

## Sync fix (v6)

The owner reported the voice slightly out of sync. Measured by cross-correlating loudness envelopes: `audio_clean`
(isolation and denoise) delays the voice by **40 ms** against the raw audio; the raw audio's speech onset lines up
with the first mouth movement within about one frame. v6 is v5 with the audio advanced 40 ms (video stream copied, audio
re-encoded); the same measurement against the raw audio now gives 0 ms. **v6 is the current cut**; v5 was deleted.
Follow-up for the tool: `audio_clean` should compensate its own delay (or report it); logged as a drift item.
If it still looks off, the remaining suspect is the phone's variable frame rate; the owner says which direction.

## Owner corrections (2026-10-22)

- The word was **Codex**, as guessed.
- Under the car noise at about 20-24 s he said "**no overtime btw**"; whisper had dropped it. It is in the caption.
- Captions are information for the viewer, not a transcript, and should stay up longer than the speech.

- v3: top caption gains `14h internship day` as its first line (owner's pick from five options). The bottom first line reads `14 hours at work today`. The Waterloo, Ontario location is added by the owner as an Instagram location tag when posting.

## Open items

- `data/projects/` is a folder convention for now. There is no project command yet; a project is these files plus this page.
- Teardown lesson for later: the isolation step removed enough car noise to transcribe the clip, but speech under a
  passing car can still be lost. A look at the waveform (or a second whisper pass on the un-isolated audio) would have caught "no overtime btw".

## Lessons (written 2026-10-22, after v6)

The reel shipped in six versions in about an hour. Most of the rework was avoidable.

### What cost rounds, and what to do next time

| What happened | Rounds lost | Do this instead |
|---|---|---|
| Captions were my guess at style: small first, then bigger, then closer to the speech, then info-style, then closer again | 4 | Before the first render, show the owner **one still frame** with the caption layout and the first three caption lines, and ask size and wording style in one question. Still previews take seconds |
| Whisper heard "codecs" and dropped "no overtime btw" under a passing car | 2 | After transcription, list **low-confidence or suspicious spans** (rare words, long single words, gaps inside speech) and have the owner confirm them. Compare the transcript of the raw and the cleaned audio to find dropped words |
| `audio_clean` delayed the voice 40 ms and nobody checked | 1 | Run a sync check (loudness-envelope correlation against the raw audio) after any audio step. Logged as D23, so the step should compensate itself |
| The chat upload limit (30 MiB) was found after the first send | 1 | Always make a share copy under 29 MiB (two-pass at about 5.3 Mb/s video for 45 s) alongside the master |
| Inline `layout=` JSON on the command line, copied and patched by hand five times | n/a | A `raw` reel style (template with slots for the top caption lines and a caption file) so a re-render is one command |

### Rules this project settled (the owner's taste, for raw vlogs)

- Raw means: no pause trimming, no grade, no music, no big bold text. Plain white small text on a translucent dark box.
- Captions are information for the viewer, not a transcript. They can add context, should appear slightly before the
  speech and stay until the next caption, and should still follow what was said (close, not verbatim).
- Top caption is the context card (what the day was); it should be a little bigger than the speech captions' size
  in v1 but plain. The location tag is added in the Instagram editor, not burned in.
- Clear voice beats clean voice. Isolate the voice, normalise to −14 LUFS, accept leftover noise.
- Speed matters for posting windows. Deliver a postable cut first, then iterate.

### Caveats on this cut

- **No one listened.** Loudness, the voice clean-up and the sync were checked by measurement, and the captions by looking at
  six frames. Intelligibility was checked only by whisper transcribing the whole clip.
- The `no overtime btw` caption (20.2 to 24.0 s) is placed by guess; the words are masked by the car in the audio.
- Isolation costs 40 ms of delay (fixed in v6 by shifting the audio), and may still leave the voice a little off if the
  phone's variable frame rate is the real cause.
- The share copy has been compressed twice (the render, then the two-pass resize), and Instagram will compress it a third
  time. If quality matters, post the master from the flash drive (48 MB) rather than the chat copy.
- Footage is dark (night, phone camera) and was left dark on purpose. A lift adds noise.
- The caption claims (`jiuwenswarm ai4research`, `open source btw`, `rogue agent fixed ✅`, "14 hours") are the
  owner's own statements and were not checked against anything.
- The flash drive was not auto-mounted. It was mounted with `sudo mount` and unmounted after each copy, and a shell sitting inside
  the mount blocks the unmount.
