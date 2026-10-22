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
| Render | `compose_video`, inline layout (`layout2.json`), `tighten=false` | v2: one encode, 1080x1920, 44.3 s, 51 MB |
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

## Owner corrections (2026-10-22)

- The word was **Codex**, as guessed.
- Under the car noise at about 20-24 s he said "**no overtime btw**"; whisper had dropped it. It is in the caption.
- Captions are information for the viewer, not a transcript, and should stay up longer than the speech.

## Open items

- `data/projects/` is a folder convention for now. There is no project command yet; a project is these files plus this page.
- Teardown lesson for later: the isolation step removed enough car noise to transcribe the clip, but speech under a
  passing car can still be lost. A look at the waveform (or a second whisper pass on the un-isolated audio) would have caught "no overtime btw".
