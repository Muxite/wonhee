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
