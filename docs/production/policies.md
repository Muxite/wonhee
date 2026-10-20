# Policies: planning and running video DAGs

Status: approved design. §1–2 (timeline, `render`, the one-encode validator) and the per-node timeout in §6 are **not built yet**; until they are, the encoding nodes (`look`, `text_video`, `burn_subtitles`) each encode once and chains should keep them to one per output. See [DESIGN.md](../DESIGN.md).

These apply to any agent (or person) that writes a chain for wonhee's node runner (`wonhee chain run`, MCP `chain_run`).
They extend the existing contracts: typed ports, content-hash cache, cost cap. See `docs/specs/2026-10-15-toolkit-design.md`.

## 1. Shape of a production chain

```
ingest ─┬─ proxy ─┬─ shots ──────┐
        │         ├─ ocr/vision ─┤
        │         └─ tags ───────┤
        └─ speech (if speech) ───┴─ plan ─ text_layers ─ preview ─(human ok)─ render ─ check ─ outbox
```

- **Analyse → plan → render → check.** Analysis nodes read a proxy. `plan` writes a timeline. `render` is the only node that encodes the final. `check` runs on the rendered file.
- **Branches fan out across lanes and run at the same time.** For example, speech uses CUDA while OCR uses the CPU and a caption call uses the network. Joins wait only for what they need.
- **The plan is data.** It is saved, versioned and replayable. Re-rendering an edited plan never re-runs analysis.

## 2. Lazy edits (the main speed rule)

Edit operations (trim, cut, speed, grade, reframe, text, audio mix) are **timeline transforms**. Their output port is a `timeline` (JSON), not a `video`.
`render` compiles the whole timeline into **one** `ffmpeg -filter_complex` graph and encodes it once.

The measured cost of not doing this, on a 60 s 1080×1920 clip:

| | wall | CPU-s | SSIM vs. reference |
|---|---|---|---|
| 3 separate nodes, each re-encoding (x264) | 47–76 s | 560–600 | 0.9948 |
| one graph, x264 medium | 16–25 s | 190–205 | 0.9964 |
| one graph, NVENC p5 | 5.6–8.2 s | 15–19 | 0.9968 |

Agent rules:
- A chain may contain **at most one lossy encode per output path**, plus one preview. The validator should reject a second unless the node declares `materialize: true` with a reason (for example, a model that needs real frames).
- Nodes that genuinely need pixels (vision, OCR, scene detection) read the **proxy** or decode frames themselves. They never ask for an intermediate full-quality render.
- Pure trims with nothing else applied use stream copy (`-c copy`, 0.08 s). It snaps to keyframes, so it's for rough cuts and source pulls, not the final cut points.

Timeline sketch for a reel like the reference one (3 shots, per-shot text, warm grade, music added in app):

```json
{"canvas": {"w": 1080, "h": 1920, "fps": 30},
 "clips": [{"src": "<sha>", "in": 0.0, "out": 2.83}, {"src": "<sha>", "in": 2.83, "out": 4.87}, {"src": "<sha>", "in": 4.87, "out": 12.93}],
 "grade": {"preset": "warm", "saturation": 1.1},
 "text": [{"clip": 0, "content": "🌸 Ottawa, Japan 🌸", "style": "script_outline", "pos": "center"},
          {"clip": 1, "content": "OCtranspo route 7\n>St-Laurent 🥺❤️", "style": "script_outline", "pos": "center"}],
 "audio": {"source": "mute", "note": "trending track is added in the Instagram app"}}
```

## 3. Lanes and concurrency

Each worker declares one lane. The scheduler admits jobs per lane, not globally.

| Lane | What runs there | Limit on this box | Why |
|---|---|---|---|
| `cpu-encode` | x264/x265 encodes, heavy filters | **1** at a time | x264 already uses all 16 threads; 4 in parallel took 64 s vs 68 s serial |
| `nvenc` | NVENC encodes, previews | **4** | 4 × 60 s took 11.4 s vs 29 s serial; 8 took 22.6 s, so 4 is the knee |
| `cuda` | whisper, vision models, local VLMs | 1 large model + whisper | whisper slows ×1.0 next to anything; an LLM next to training slows ×2 |
| `llm` | Ollama / llama.cpp | 1 server, **batch requests** | batched decode reached 1,417 tok/s vs 258 tok/s for 4 separate processes |
| `cpu-light` | probe, OCR, Pillow text, checks | about 8 | short, single-threaded |
| `net` | grab, Apify, remote LLMs | per-provider rate limit | money and rate caps, not hardware |

- NVENC, CUDA and the CPU are separate engines, so overlapping them is cheap. Measured: an LLM next to an NVENC encode slowed ×1.12, and whisper slowed about ×1.0 next to anything.
- Hold `gpu-lock` only for the duration of a measured run. In production the per-lane limits, not the lock, keep jobs apart.

## 4. Batching

- **Across items:** when N reels need the same node, send them to the same warm worker back to back, so the model loads once. Order the queue by `(lane, node, model)`.
- **Inside a model:** send lists, not singles. translate already sends 80 lines per call. Captions, hashtags and shot descriptions should also go in one call per reel, or per batch of reels.
- **Inside ffmpeg:** one process per output. When several outputs share an input (final + preview + thumbnail), use one decode with `split` and several outputs, not three runs.
- **Don't parallelise CPU encodes to "batch".** Run them in sequence on `cpu-encode` and put the parallelism into NVENC or other lanes.

## 5. Caching and idempotence

- The cache key is `node name + version + input refs + params + salt (model names)`, which is already in `wonhee/nodes/chain.py`. Bump a node's `version` whenever its output could change.
- Seed anything random (transition picks, sampling) through params so the key stays stable.
- Store the timeline and the rendered sha together. Re-rendering an identical timeline is a cache hit.
- Proxies and per-shot embeddings are cached per source sha. A new edit of the same footage reuses them.

## 6. Failure

- Retry only transient errors (network, 429, worker restart), at most twice with backoff. Never retry `application_error` or OOM blindly. Shrink the job instead (lower resolution, smaller model, fewer frames).
- A failed branch fails its join, not the whole chain. Steps that already succeeded stay cached for `wonhee resume`.
- Every node has a timeout of about 3× its measured p90 (`wonhee profile` gives the p90).

## 7. Human gates and provenance

- **Agents may:** plan, analyse, render previews and finals, run checks, and add to the outbox.
- **Agents may not, without a person:** publish, delete media, exceed `cost_cap_usd`, or post anything that uses another creator's footage.
- Third-party reels (from `grab`) are **reference material**: style, timing, text tone. They are never source footage for our own uploads, and never committed to git.
- Music: Instagram trending audio is added in the app, where the licence lives. A rendered upload carries either our own audio or none, with a note in the outbox.
