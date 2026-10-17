# Watch-outs

Things that silently cost time or quality. **Seen** marks ones observed in this repo. The rest are standard ffmpeg / Instagram pitfalls worth checking on the first footage from each new source.

## Time
- **Chained re-encodes** (seen, measured). Each node that writes an mp4 costs a full decode and encode. 3 nodes = 47–76 s on a 60 s clip, against 6–8 s for one NVENC graph. Fix: lazy timeline + one `render`.
- **Parallel x264 isn't faster** (seen, measured): 4 at once 64 s vs 68 s serial. Queue CPU encodes, and put parallelism into NVENC (4 at once) or other lanes.
- **Cold model per call** (seen): whisper is about 15 s flat regardless of clip length. A fresh Ollama container without the JIT cache: +60 s. Keep workers warm and share `/root/.nv`.
- **Page cache charged to the first reader** (seen): RAM numbers for model workers depend on which container read the file first. Judge memory by the anon/file split, not the total.
- **`-ss` placement.** `-ss` before `-i` seeks fast and is frame-accurate when re-encoding. With `-c copy` it snaps to the previous keyframe, so the clip may start early. Decide per use.
- **VP9/AV1 sources** (seen: Instagram serves VP9). CPU decode is slow at 1080×1920. Use `-hwaccel cuda` (CUVID handles vp9/av1) for proxies.
- **Filter graph threading.** Custom `xfade` expressions with `st()/ld()` must run with `-filter_complex_threads 1` (slopcut learned this). Keep them out of the main graph's hot path, or render those transitions separately.

## Picture
- **Colour emoji in libass** (seen, tested): rendered as monochrome outline glyphs. Pillow with Noto Color Emoji renders them correctly. Overlay PNG text layers.
- **Script fonts aren't installed.** The reference style needs one. Bundle OFL fonts (for example Dancing Script, Pacifico) in the worker image, and never depend on host fonts.
- **Phone HDR** (iPhone HLG / Dolby Vision, 10-bit). Encoding straight to 8-bit yuv420p washes colours out. Detect `color_transfer=arib-std-b67|smpte2084` at ingest and tone-map once (`zscale` + `tonemap`, or libplacebo).
- **Rotation metadata.** Phone clips often store landscape pixels with a rotate tag. ffmpeg auto-rotates on decode, but probe-reported width/height may be swapped. Normalise at ingest.
- **VFR phone footage.** Variable frame rate drifts against audio after cuts. Set CFR at render (`fps=30`), not in a separate pass.
- **Generation loss.** Each lossy pass costs SSIM: 0.9948 after 3 x264 passes vs 0.9964 after one. Instagram re-encodes on upload as well, so give it one clean, high-quality encode.
- **Safe zones.** Instagram's UI covers roughly the top 14% and bottom 20–25% of a 9:16 reel, plus a right-side strip for buttons. Keep text inside the middle band. The reference reel centres it.
- **NVENC files are bigger** (measured: 1.8× x264 at equal quality). Fine for uploads, since Instagram re-encodes. Use x264 when archiving or when size matters.

## Audio
- **Whisper on music** (seen). The reference reel has no speech, only a Japanese pop track, yet whisper (VAD on) returned "Bye." and the language "en". Gate captions: skip when speech covers < 20% of the duration, when the `no_speech_prob` average is high, or when the detected language disagrees with the caption language. Flag the step as `weak`.
- **No audio stream.** Many edits and screen recordings have none. Concat and some filters fail on mixed inputs. Add a silent track at ingest (`anullsrc`).
- **Loudness.** Use two-pass `loudnorm` to −14 LUFS inside the render graph, using the first pass's measured values. Don't run a separate normalisation encode.
- **`atempo` range.** It accepts 0.5–2.0 per instance, so chain instances for bigger factors. Speed changes must apply to audio and video in the same graph, or they drift.
- **Licensed music.** Trending audio is attached in the Instagram app. Don't bake third-party music into uploads.

## Data and safety
- **Third-party media** (seen policy): grabbed reels are reference only, never source footage for our posts, and never committed. `data/` is gitignored, and keep it that way.
- **Vision OCR errors** (seen): `screen_text` read "OCtranspo" as "Otranspo" in the reference reel. Treat OCR as a draft. Names and places need a human glance or a second source, such as the caption.
- **Cache poisoning by version.** If a node's behaviour changes and its `version` doesn't, old cached outputs get reused silently. Bump the version.
- **Cost.** `grab` and remote LLM calls are paid. Keep `cost_cap_usd` set on agent-run chains, and prefer the local vision models for drafts.
