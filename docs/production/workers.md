# Workers: what exists, what's on this machine, what to build

A **worker** is a long-lived container in one lane (see policies §3) that pulls jobs for one or more nodes.
Model-backed workers stay up with the model loaded. Pure-ffmpeg workers can be one container per job: start cost is about 0.15 s, against seconds of encoding.

## Already built (wonhee nodes)

| Node | Lane | Notes |
|---|---|---|
| `grab` | net | Apify post data + yt-dlp video; paid, under the cost cap |
| `speech_subs` | cuda | faster-whisper turbo; about 15 s flat for 10–100 s clips (model load dominates), so it **needs a warm worker** |
| `screen_text` | cpu-light + net | frame sample + pHash, then a remote vision LLM |
| `merge_subs`, `to_srt` | cpu-light | pure data |
| `translate` | net | 80-line batches with a glossary |
| `burn_subtitles` | cpu-encode | libass + x264. **Becomes a timeline transform** once `render` exists |
| `look`, `look_image` | nvenc (x264 fallback) / cpu-light | named looks from `wonhee/edit/looks.py`: `moody` (underexposed, desaturated, cool shadows, vignette), `grain`, `moody_grain`, `digicam` (+ optional orange date stamp), `faded`, `bw_grain`, `warm_pink`, `cool_night`; `intensity` 0–2, `grain` override. Each preset is one filter string that `render` can fuse later |
| `text_image`, `text_video` | cpu-light / nvenc | one text block in the `outline` / `band` / `script` preset at `top` / `upper` / `center` / `lower` / `bottom` (IG safe zones), colour emoji and CJK, with an optional `look` underneath, in one pass (`wonhee/edit/text.py`, bundled OFL fonts in `wonhee/assets/fonts`) |
| `carousel`, `content_images` | cpu-light | `images` port (ordered manifest). Lay photos out as 4:5 / 1:1 / 9:16 slides, with panoramas spanning slides; one look (+ date stamp) for the whole set in a single ffmpeg pass; per-photo text. `content_images` pulls a grabbed post's slides. Save with `wonhee export <ref> <dir>` (also MCP `export`). A directory or JSON list works as `--in images=` |
| `teardown` + `wonhee inspire` | net (+ local llm) | one stored post → (creator-blind since v2: the creator profile is never in the prompt, so teardowns stay a pure inspiration record; scene text such as a road sign is not an overlay) format, text/footage coupling, footage to film, text preset/position, look, capabilities (closed vocabularies). Non-English text is translated first by the on-box model (gemma4:e4b, $0). `wonhee inspire [ids] [--since] [--relation]` tears down every stored post (cached per post) and writes `data/inspire/<date>-run<id>/report.md`: what to build, best to replicate, shoot list, line bank with render hints. Measured 2026-10-17: 36 posts in 12 min for $0.085 (about $0.0024 a post); a cached rerun takes under 1 s at $0 |
| `larp` + `wonhee larp "<task>"` | net or local llm | any mundane task → professional-moment framings (LinkedIn announcement, founder update, press release, earnings call, changelog, KPI report, ...): an on-screen line + caption each, hashtags, text preset/position, look, and the shot to capture. `--image/--video` also renders the chosen line in one pass. `--engine local` is $0, about 3 s |
| `ideate` + `wonhee ideate [-n 20] [--focus ...]` | net (or local llm) | the planning step: which posts to make next and the b-roll to film for them. Two sources kept apart: **inspiration** (creator-blind teardowns of saved posts, ids `c<n>`) and **your brief** (`data/brief/*.md`, your own idea lists and persona notes, one id per bullet `b-xxxxx`). Each idea cites what it uses; its origin (blend / inspiration / own / new) is computed from the citations, never chosen by the model. The reply has one array per origin (about 40/30/30), so the mix is asked for explicitly. Unknown references are dropped (`UNKNOWN_REF`), clipped brief ids are repaired when unambiguous, and a line copying a saved post's overlay is dropped (`IDEA_COPY`). The shoot list is shared, so plates are ranked by how many ideas they serve. Writes `data/ideate/<date>-run<id>/ideas.md` + `ideas.json`. Measured 2026-10-18: 24 ideas from 37 posts + 133 brief items in about 70 s for $0.01 (remote). `--engine local` is $0 and about 23 s for 12 ideas, but weaker (no on-screen lines, over-cites), so treat it as a draft |
| `translate engine=local` | local llm | subtitle translation on the on-box model at $0 |
| GPU extras (`uv sync --group gpu`) | cuda | torch 2.13 cu130 + torchvision, transformers, demucs. **cuDNN pin:** torch pins cuDNN 9.20 for CUDA 13, so the CUDA 12 cuDNN whisper uses is pinned to 9.20 too; mismatched versions cross-load sublibraries (`CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`). Whisper and SAM verified in one process. SAM 3 is not used: gated, custom licence. DeepFilterNet is not used: abandoned, numpy<2 and old torchaudio |
| `style_profile`, `shot_guide`, `vibe_review`, `vibe_image` + `wonhee vibe` | cpu-light (+ local or remote vision) | learn a photographic look from references (folder, images, or `content:<id>`; reels give keyframes) and move your photos toward it, **no generation**. `style_profile` measures tone, colour per tone band and lens traits (`wonhee/edit/style.py`). `shot_guide` is a vision model grounded on those numbers, writing how to shoot it in your places (remote ~$0.003; local gemma4:26b-a4b $0, ~55 s, a little weaker). `vibe_review` is the on-box gemma4:e4b ($0, ~10 s) picking clamped knobs (`REVIEW_CLAMPED`) plus reshoot tips. `vibe_image` is deterministic: per-band tone/colour match, subject mask: on the GPU (`uv sync --group gpu`) a text prompt (`mask_prompt` "plush toy", picked by the reviewer) goes through Grounding-DINO tiny → SAM 2.1 base+ (both Apache-2.0, HF cache, ~0.18 s a mask after a 5 s load); without torch, or when the prompt finds nothing, the CPU ONNX models (`person` U²-Net / `object` IS-Net, ~1.3 s, downloaded to `data/models`), fisheye, edge and motion blur, bloom, vignette, grain, halftone / duotone print, crop. **Mask policy** (`mask_policy=auto`, default): masks are a last resort. It grades without one, then with a 40% smaller tone move, then blends a soft mask (guided-filter edge, 1.5% feather, opacity 0.39 then 0.65), taking the first step that keeps the subject's skin readable (not sunk below ~60% of its brightness, not newly blown); with no subject named it still checks for a person on the GPU. `always` = soft mask every time, `off` = never. Skin protection (0.6) keeps skin hue under colour grades without any mask. The chosen step goes into the report. Writes `data/vibe/<date>-<name>/`. See `photo-looks.md` |
| `poster` | cpu-light | the spec-sheet poster joke format: title, two label boxes, tilted image, justified mono paragraph (bundled Noto Sans / Mono) |
| `filter_image`, `filter_draft`, `filter_review` + `wonhee filter` | cpu-light (+ local or remote model) | filters that give a vibe without copying anyone (`wonhee/edit/filters.py`): exposure, S-curve contrast, shadows / highlights, fade, roll-off, temperature / tint, saturation / vibrance, greens / blues, mono, split toning by named hue, skin protection (a colour-range rule, not a mask), clarity, sharpen, glow, bloom, halation, light leak, chromatic aberration, vignette, grain. 13-filter pack (`wonhee filter list`), ~0.1-0.2 s a photo. `wonhee filter make "<vibe>" --name N --on <photos>`: the model picks a pack base + a few word moves ("temperature down, a lot"), code renders and measures (clipping, crushed shadows, skin drift; plain-word verdicts, because small models misread numbers), a guard fixes blown / crushed results, a vision review moves up to 4 knobs a round, best round saved to `data/filters/<name>.json` (5 previous versions kept). `--from` tweaks a saved filter (goal = original vibe + change). `filter show --cube` exports the colour part as a .cube LUT (ffmpeg `lut3d`, CapCut, VN, Resolve). Local gemma4:e4b, $0, ~20 s for 3 rounds on 2 photos |
| `audio_check`, `audio_clean` + `wonhee audio` | cpu | loudness (LUFS), true peak, range, speech-to-noise, with verdicts; clean = rumble cut, RNNoise (`arnndn`, model fetched once; `DENOISE_FALLBACK` to `afftdn` offline) + residual FFT denoise, optional gate / de-ess, compression, two-pass `loudnorm` to -14 LUFS / -1.5 dBTP + limiter; video stream copied. `isolate=true` (GPU, Demucs htdemucs, MIT) first splits the voice from music / street / room and `music_db` puts the bed back underneath. Measured: 44 s clip in ~12 s; hissy test clip 18 → 42 dB speech-to-noise; music + rumble test clip 7 → 30 dB (RNNoise) vs 51 dB (isolate), 13.6 s warm |
| `wonhee inspire` looks section | cpu-light | groups saved posts by measured look (photos or keyframes) and prints the `wonhee vibe --like content:...` command for each group |

## On this machine, ready to wrap

| Asset | Where | Good for |
|---|---|---|
| ffmpeg 6.1 with NVENC (h264/hevc/av1), CUVID decode (h264/hevc/vp9/av1), `scale_cuda`, `overlay_cuda`, libass, `loudnorm`, `rubberband`, `silencedetect`, `scdet`, `xfade`, `zscale`, libplacebo | host `/usr/bin/ffmpeg`; `linuxserver/ffmpeg:7.1.1` image | render, proxy, audio, checks |
| faster-whisper tiny / base / small / large-v3-turbo | HF cache | speech |
| SigLIP2 so400m, CLIP ViT-B/32 | HF cache | shot embeddings, "find the shot with the bus", dedupe |
| OWLv2, Grounding-DINO tiny / base | HF cache | open-vocabulary boxes: subject tracking for reframe, text placement that avoids the subject |
| SAM 2.1 base+, SAM 3 | HF cache | masks: subject cutout, text-behind-subject |
| OneFormer (COCO Swin-L) | HF cache | sky / road / building segmentation for placement and grading |
| Local vision LLMs: gemma3:12b, gemma4:e4b, gemma4:26b-a4b, ministral-3:14b, qwen3.6:35b-a3b | Ollama volume | shot description and OCR without a paid call |
| Text LLMs: qwen3:8b, qwen3.5:9b, gpt-oss:20b, phi4:14b, … | Ollama volume | captions, hashtags, planning drafts |
| Embeddings: qwen3-embedding 0.6b / 8b, nomic-embed-text-v2-moe | Ollama volume | search over past reels and captions |
| tesseract | host | cheap first-pass OCR; weak on script fonts |
| yt-dlp | host | already used by grab |
| **slopcut** (`~/projects/slopcut`) | repo | Pillow text layers (`text.py`), xfade transitions, beat cues (`audio/cues.py`), carousel slicing |
| **viv** (`~/projects/viv`) | repo | grounded vision on video: object tracking that survives leaving the frame |
| **sdown** (`~/projects/sdown`) | repo | download + silence-trimmed audio extraction |
| **gauge** (`wonhee profile`) | this repo | cost and contention measurement for any new worker |

## Build next (ordered by speed gained per effort)

| # | Worker / node | Lane | In → out | Built from | Why it speeds things up |
|---|---|---|---|---|---|
| 1 | **render** | nvenc (x264 option) | `timeline` → `video` (+ preview, thumbnail from one decode via `split`) | ffmpeg `-filter_complex_script` | Replaces chains of re-encodes: about 3× less wall time with x264, 8–11× with NVENC (60 s clip) |
| 2 | **timeline ops**: `cut`, `trim`, `speed`, `grade`, `reframe_9x16`, `text`, `audio` | none (data) | `timeline` → `timeline` | the spec's R1 ops, rewritten as transforms | Agents compose edits for free and validate them before any pixels move |
| 3 | **text_layers** | cpu-light | `timeline` → PNG layers | `slopcut/text.py` + Noto Color Emoji + bundled OFL script fonts | Emoji-correct overlays in the reference style; libass can't do colour emoji |
| 4 | **proxy** | nvenc | `video` → 360p, 2–5 fps `video` + keyframes | ffmpeg CUVID decode + `scale_cuda` | Every analysis node reads a small file; full decode happens once |
| 5 | **warm whisper** | cuda | `video` → `subtitles` | existing `analyze/speech.py` in a server loop | Removes the ~15 s per-call model load. Add a music gate (watchouts §Audio) |
| 6 | **shot_index** | cuda | proxy → per-shot embeddings + tags | SigLIP2 | Lets the planner ask "shots with a bus at sunset" across all footage |
| 7 | **local_vision** | llm | keyframes → descriptions, on-screen text | gemma3:12b or ministral-3:14b via Ollama, batched per reel | Removes the paid vision call from `screen_text`; batches all of a reel's keyframes in one request |
| 8 | **check** | cpu-light | `video` → findings | spec R3 + a safe-zone check on text boxes | Catches the expensive mistakes (black frames, loudness, letterboxing) before a human looks |
| 9 | **caption_writer** | llm | plan + style → caption, hashtags | spec R6 style memory | One call per reel; uses the voice style |
| 10 | **reframe_track** | cuda | 16:9 `video` → crop path for the timeline | OWLv2 / Grounding-DINO boxes, smoothed; viv for long clips | Turns landscape footage into 9:16 that follows the subject, without hand keyframing |
| later | stems (demucs), upscale (Real-ESRGAN), frame interpolation (RIFE) | cuda | | not installed | Slow, and seldom needed for phone footage. Measure before adopting |

## Worker contract

- One image per lane family, built from `profiling/images/…`. Models mount read-only from the host caches (`~/.cache/huggingface`, the Ollama volume). The CUDA JIT cache is a shared volume (`gauge-cuda-cache` at `/root/.nv`), or each fresh container spends about 60 s compiling.
- The worker loads its model once, then loops: claim job → run node fn → write blobs to the media store → write the result row → next.
- Jobs come from wonhee's SQLite (a `job` table: chain step, lane, node, cache key, status, attempts, timeout). Claim with an atomic `UPDATE … WHERE status='queued' … RETURNING`. No extra broker is needed on one box.
- Health: a worker reports `lane`, `node versions`, `model`, `loaded_at` and `jobs_done`. The scheduler stops sending work to a worker that misses its heartbeat.
- Profile every worker once with `wonhee profile run …` alone and next to its usual neighbour before setting its lane limit.
