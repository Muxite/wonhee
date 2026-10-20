# Production guides

How wonhee should turn footage into a published reel quickly. An agent plans a DAG, and long-lived worker containers run it.
Written 2026-10-17. Every number here was measured on this box (Ryzen 7 3700X, 16 threads; RTX 5070 Ti) with `wonhee profile` or `profiling/bench/ffmpeg_batching.sh`.

| Doc | Read it when |
|---|---|
| [policies.md](policies.md) | planning a chain, adding a node, or deciding what runs where |
| [workers.md](workers.md) | choosing what to build next; what tools and models are already on this machine |
| [watchouts.md](watchouts.md) | something looks wrong, or before trusting a new source of footage |
| [benchmarks.md](benchmarks.md) | you want the numbers behind the rules |
| [references.md](references.md) | what our posts and the reference reels look like, and what that means for what to build |
| [insights-2026-10-17.md](insights-2026-10-17.md) | what to record (B-roll shoot list), which formats fit, and what that means for what to build |
| [idea-bank.md](idea-bank.md) | the @frictionlessmuk niche: ideas grouped into shoot days, each with a capture, a line and a look |

## Day-to-day commands
- `wonhee grab <urls>`: store references (reels, photo posts and carousels) with analysis.
- `wonhee inspire`: turn everything stored into an actionable report (what to film, what to build, lines to use).
- `wonhee larp "<mundane task>" [--image|--video PATH]`: professional-moment framings, rendered onto your media.
- `wonhee ideate [-n 20] [--focus "cafe day"]`: what to post next and the b-roll to film, from saved posts + your brief (`data/brief/*.md`), each idea tagged with where it came from.
- `wonhee node run look|look_image|text_image|text_video|carousel ...`, then `wonhee export <ref> <dir>`: produce and save.

## The rules on one screen

Rules 1, 5 and the `render` half of 2 are the approved target; `render`, `timeline` and `proxy` aren't built yet (see [DESIGN.md](../DESIGN.md) › Stage 4).

1. **Touch pixels once.** Edit nodes produce a timeline, not a video. Only `render` encodes. A chain should have one final encode, plus at most one preview encode. Three chained re-encodes cost **3–4× the time and CPU** of one filter graph, and lose quality.
2. **Pick the encoder by lane.** NVENC for previews and most finals (**2.8× faster** than x264 medium, same SSIM, about 1.8× the file size). x264 when file size matters.
3. **Batch by lane, not by item.** x264 already uses all 16 threads, so run **one** CPU encode at a time. NVENC scales to **about 4** at once (4 × 60 s in 11 s), then flattens.
4. **Keep models warm.** A container starts in about 0.15 s, but a model loads in 5–15 s and a cold CUDA JIT takes about 60 s. Workers that hold a model stay up and take jobs from a queue.
5. **Decode once, analyse a proxy.** Make one low-res proxy (for example 360p at 2 fps), then fan out shots, OCR, vision and tagging from it.
6. **Stream copy when you can.** A trim with no other change is 0.08 s with `-c copy`, against 6 s or more re-encoded. It cuts on keyframes, so use it only where frame accuracy doesn't matter.
7. **Text with emoji is an image, not ASS.** libass draws colour emoji as monochrome outlines. Render text layers with Pillow and overlay them inside the same graph.
8. **Cache by content.** The key is input hashes + params + node version + model name, which wonhee already does. Never re-run a node whose key is cached unless asked.
9. **Gate the irreversible.** Agents plan and render freely. Publishing, paid calls above the cap, and deleting anything need a human.
10. **Measure new workers.** Run each new worker under `wonhee profile run …` before trusting its cost or putting it next to other GPU work.
