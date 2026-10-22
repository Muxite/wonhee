# Framing and motion: how much is going on, and where the subject sits

Status: **proposed** 2026-10-22. New node and CLI command, so large lane: the owner approves before code (PROCESS §2).
Stage 3 (capture) and Stage 2 (plan): it reads footage and ranks candidates. Lane `cpu-light`, no GPU.
Evidence: the midnight-waterloo project (`docs/production/projects/2026-10-22-midnight-waterloo.md`). `shots index`
used 2.6 GB of VRAM and 19 s for one 44 s clip, but returned one shot, and its tags describe content, not motion or
framing. Neither question can be answered from SigLIP tags.

## 0. Why

Two questions come up for every clip, and both are measurable without a model:

1. **How much is happening?** For a screen recording (tmux with six Claude panes and htop/nvtop), this is how much of
   the picture changes, and how often. For a camera clip, it is camera motion plus subject motion.
2. **Where is the subject, and how is the camera moving?** That decides which zoom or pan works, and where text can
   sit without covering the face.

The shot tags answer neither. A jump cut on a locked-off screen recording needs activity, not semantics.

## 1. The node: `motion_framing`

`video -> framing: text` (JSON, one object per sampled second).

Params (`extra="forbid"`):
- `sample_fps` (default 4): frames analysed per second of video. Decoding is `-vf fps=N,scale=320:-2` into a pipe, so
  the full-resolution frames are never stored.
- `subject: "face" | "none"` (default `face`): picks the subject detector. See §4.
- `activity_region` (default whole frame): `{x, y, w, h}` as fractions, so the screen recording's dead margins can be
  ignored.

Per sample, from consecutive frames (greyscale, 320 px wide):

| Field | Meaning | How |
|---|---|---|
| `changed` | fraction of pixels that changed | `|f_t − f_{t−1}| > 12` (8-bit), share of pixels |
| `activity` | mean absolute change, 0–1 | mean of `|f_t − f_{t−1}| / 255` |
| `edges` | busyness of the picture, 0–1 | share of Canny edge pixels (thresholds 80, 160) |
| `pan_px` | horizontal camera motion | median optical-flow x (Farneback, 160 px wide), in full-frame pixels per second |
| `tilt_px` | vertical camera motion | median optical-flow y, same |
| `zoom` | zoom rate per second | least-squares scale of the flow field about the centre, as a fraction per second |
| `still` | true if `activity < 0.01 and |pan|,|tilt| < 2 px/s` | |
| `subject` | box `{x, y, w, h}` (fractions) or `null` | detector output, `null` when none is found |

Per clip, the node also writes a summary: the means and 90th percentiles of `activity`, `edges` and `changed`; the
share of seconds that are `still`; and the time spans with activity above the 75th percentile (the "busiest" spans,
for jump-cut candidates).

Invariants:
- `null`, never 0, for unknown values (house rule). A clip with no subject gives `subject: null` throughout, and the
  summary says `subject_found: 0.0`.
- No frame is written to disk. Decoding is one pass, through a pipe.
- The node is deterministic: fixed seeds, fixed thresholds, so the output is cacheable on (node version, inputs, params).

## 2. The ranking command

`wonhee frame rank <video...> [--top N] [--sheet]`

- Runs `motion_framing` on each file (cache-aware), then prints a table sorted by the chosen score:
  `score = 0.6·activity_p90 + 0.2·edges_mean + 0.2·changed_mean`, with the weights in one place and a `--score` flag to
  sort by one column instead.
- `--sheet` writes one contact sheet per file to `data/frame/<date>/`, with the busiest span marked, so the owner
  judges the shortlist by eye. The model does not decide the winner.
- Prints the busiest spans as `start–end` seconds, ready for `ffmpeg -ss/-to` or a jump-cut list.
- MCP twin: `frame_rank(videos, top, sheet)`. Same capabilities as the CLI (DESIGN contract 4).

The ranking is a shortlist. It can't tell a good tmux moment from a busy one with a blinking cursor; that is the
owner's call on the contact sheet, and the spec says so in the help text.

## 3. Jump cuts

A jump cut is a join where `activity` is high on both sides, so the cut doesn't read as a jump. The node reports
`activity` before and after each proposed cut point. Choosing cut points and rendering them stays with `compose_video`
(`cuts`, already specified in the meme reel styles spec, 2026-10-22). This spec does not add a second cutter.

## 4. The subject detector (open)

The venv has OpenCV 5.0.0, but the Haar face cascade files are **not** in this build (`cv2.data` has none). Options:

- **A.** Bundle a small face detector model file (YuNet, ~0.3 MB ONNX, run through OpenCV's DNN module). CPU, no new
  dependency, but a model file to licence-check and bundle.
- **B.** Skip the subject box in v1 (`subject: "face"` returns `null`), and rely on `pan/tilt/zoom` and `activity`.
  Smallest change; the framing question is then answered for motion only.

Recommendation: **B** for v1, then **A** once the owner wants "where is the face" answered. The owner decides.

## 5. Tests (from this text)

- A synthetic clip with a static left half and a moving right half: `activity` is high only on the right, `changed`
  matches the moving area within 2 %.
- A clip panned by a known number of pixels per second (synthetic moving texture): `pan_px` within 10 % of the true
  speed. A zoom of a known rate: `zoom` within 10 %.
- A static clip: every sample has `still: true`, and the summary's `still` share is 1.0.
- `subject: "face"` with no detector model present returns `null` for every sample and logs a `SUBJECT_UNAVAILABLE`
  flag. It does not fail the node.
- `frame rank` sorts by the documented score, and `--score activity` sorts by that column alone.
- The node runs on the midnight-waterloo clip in under 5 s of wall time on the CPU (the reference clip is 44 s).

## 6. Open questions for the owner

1. **Approve §1–§3** (the node, the summary, the ranking command) so it can be built.
2. **Subject detector:** B (motion only, v1) or A (bundle YuNet now)?
3. **Score weights:** 0.6/0.2/0.2 for activity/edges/changed is my starting guess. Is "most activity" the right
   default for your tmux clips, or should "interesting items" (edges) weigh more?
4. **Where the footage is.** The tmux recordings and the camera clips are not on this machine, nor on the flash drive or
   the arteta/jesse drives I searched. Once they're copied somewhere (or the path is given), the spec can be checked on real footage.

## 7. Order

1. `motion_framing` node (§1) with the synthetic tests.
2. `frame rank` command and MCP twin (§2).
3. Real-footage check on the owner's clips, with an alignment check by a different agent (PROCESS §3).
4. Optional: subject detector (§4, option A).
