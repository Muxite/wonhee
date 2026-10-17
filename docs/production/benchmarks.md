# Benchmarks behind the production rules

Measured 2026-10-17 on the host ffmpeg 6.1.1: Ryzen 7 3700X (16 threads), RTX 5070 Ti.
Another session was running Docker builds during the first round, so the first-round wall times are noisier. The ranges cover both rounds.
Reproduce with `profiling/bench/ffmpeg_batching.sh <clip> <tag>`.

## Same edit, four ways

The edit is: trim 1 s off each end, apply a warm grade (`eq`), add one ASS text overlay, and output 1080×1920.
The reference for SSIM is the same graph encoded with x264 slow at CRF 14, so SSIM measures encoding loss only.

| Variant | 13 s reel: wall | 60 s clip: wall | 60 s clip: CPU-s | SSIM (60 s) | size (60 s) |
|---|---|---|---|---|---|
| A: 3 nodes, each re-encodes (x264 medium, CRF 20) | 20.4–21.0 s | 46.7–75.7 s | 558–598 | 0.9948 | 23.0 MB |
| B: one graph, x264 medium, CRF 20 | 6.2–7.0 s | 15.7–24.8 s | 191–204 | 0.9964 | 23.4 MB |
| C: one graph, NVENC p5 / hq / CQ 21 | 2.4–3.0 s | 5.6–8.2 s | 15–19 | 0.9968 | 43.2 MB |
| D: one graph, x264 veryfast | 4.1–4.2 s | 8.2–14.8 s | 75–86 | 0.9953 | 20.8 MB |
| E: trim only, `-c copy` | 0.08 s | 0.08–0.10 s | 0.1 | n/a (keyframe-snapped) | n/a |

What this shows:
- Collapsing the nodes into one graph (B vs A) gives **3× the speed and a third of the CPU**, plus a quality gain.
- NVENC (C vs B) is **2.8× faster again** with **about 12× less CPU**, at equal or better SSIM. The cost is a 1.8× larger file.
- x264 veryfast (D) is a middle ground for CPU-only machines.

## Parallel batches (4 or 8 copies of the 60 s clip, grade only)

| Batch | wall | per clip |
|---|---|---|
| 4 × x264, serial | 67.7 s | 16.9 s |
| 4 × x264, parallel | 64.2 s | 16.1 s (x264 already uses every thread) |
| 4 × NVENC, serial | 29.0 s | 7.3 s |
| 4 × NVENC, parallel | 11.4 s | 2.9 s |
| 8 × NVENC, parallel | 22.6 s | 2.8 s (saturated at about 4) |

So NVENC throughput is about **21 minutes of 1080×1920 video per wall-clock minute**, with the CPU mostly free. In parallel NVENC jobs, the limit is the CPU-side decode and filtering. Moving decode to CUVID (`-hwaccel cuda`) is the next thing to try.

## From the gauge report (2026-10-16)

- Container start: `docker run` +149 ms, `--gpus` +216 ms, a burst of 16 is 423 ms each. Negligible for encodes, but it matters for sub-second nodes.
- whisper `speech_subs`: about 15 s flat for 10–100 s clips, with the GPU only 3% busy, so model load dominates.
- Subtitle burn with x264: about 3.3 CPU-s per second of video.
- Contention: an LLM next to NVENC slowed ×1.12, an LLM next to training ×2.08, whisper ×1.0 next to anything.
- Batched LLM decode: 1,417 tok/s (batch 8) vs 258 tok/s total for 4 separate processes.

Full report: `profiling/report/shared-gpu-node.html`.
