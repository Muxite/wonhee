"""Time for layouts: word chunking, pause trimming, and the animation state ladder (pure functions, no pixels).

Design: docs/specs/2026-10-20-reel-styles-design.md §2 (timed layers, word-driven kinds) and §3 (pause trimming).
Everything is in milliseconds on the *source* timeline unless a function says otherwise; `Timeline` re-times a source
instant onto the tightened timeline, so every layer and caption can be moved with the cuts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from wonhee.nodes.subtitles import Word

END_PUNCT = re.compile(r"[.!?;:,…。！？，]$")
SENTENCE_END = re.compile(r"[.!?…。！？]$")
MAX_STATES = 10  # per animation (spec §2)


@dataclass(frozen=True)
class Timeline:
    """Source -> output time after cutting `cuts` (sorted, non-overlapping (start_ms, end_ms) intervals)."""
    cuts: tuple[tuple[int, int], ...] = ()

    def map(self, ms: float) -> float:
        removed = 0.0
        for s, e in self.cuts:
            if ms <= s:
                break
            removed += min(ms, e) - s
        return ms - removed

    def removed_ms(self) -> int:
        return sum(e - s for s, e in self.cuts)

    def kept(self, duration_ms: int) -> list[tuple[int, int]]:
        """The (start, end) segments that stay, covering [0, duration_ms]; cuts never reach past the clip's end."""
        out, pos = [], 0
        for s, e in self.cuts:
            s, e = min(s, duration_ms), min(e, duration_ms)
            if s > pos:
                out.append((pos, s))
            pos = max(pos, e)
        if pos < duration_ms:
            out.append((pos, duration_ms))
        return out


def find_pauses(words: list[Word], min_pause_ms: int = 250, keep_ms: int = 80) -> tuple[tuple[int, int], ...]:
    """Cuts for the pauses between consecutive words: a gap of at least `min_pause_ms` is shortened to `keep_ms`, half
    of it kept on each side of the cut so speech isn't clipped. Only gaps between words are cut (not the lead-in or the
    tail of the take)."""
    cuts = []
    ws = sorted(words, key=lambda w: w.start_ms)
    reach = ws[0].end_ms if ws else 0
    for w in ws[1:]:
        gap = w.start_ms - reach
        if gap >= min_pause_ms and gap > keep_ms:
            cuts.append((reach + keep_ms // 2, w.start_ms - (keep_ms - keep_ms // 2)))
        reach = max(reach, w.end_ms)
    return tuple(cuts)


def snap_cuts(cuts: tuple[tuple[int, int], ...], fps: float | None, duration_ms: int) -> tuple[tuple[int, int], ...]:
    """Move every cut point to the video's frame grid (nearest frame boundary) so video and audio are trimmed at the
    same instants, and drop cuts that end up empty or lie past the clip."""
    if not fps:
        return tuple((s, e) for s, e in cuts if s < e and s < duration_ms)
    frame = 1000.0 / fps
    out = []
    for s, e in cuts:
        a, b = round(round(s / frame) * frame), round(round(e / frame) * frame)
        b = min(b, duration_ms)
        if b > a and a < duration_ms:
            out.append((a, b))
    return tuple(out)


def chunk_words(words: list[Word], max_words: int = 3, pause_ms: int = 350) -> list[list[Word]]:
    """Chunks of 1..max_words words, broken on pauses (>= pause_ms between words) and after punctuation."""
    out: list[list[Word]] = []
    cur: list[Word] = []
    for w in words:
        if cur and (len(cur) >= max_words or w.start_ms - cur[-1].end_ms >= pause_ms or END_PUNCT.search(cur[-1].text)):
            out.append(cur)
            cur = []
        cur.append(w)
    if cur:
        out.append(cur)
    return out


def stack_phrases(words: list[Word], max_lines: int = 4, gap_ms: int = 400) -> list[list[Word]]:
    """Phrases for the kinetic stack: a new phrase starts after a gap, at a sentence end, or when the stack is full."""
    out: list[list[Word]] = []
    cur: list[Word] = []
    for w in words:
        if cur and (len(cur) >= max_lines or w.start_ms - cur[-1].end_ms >= gap_ms or SENTENCE_END.search(cur[-1].text)):
            out.append(cur)
            cur = []
        cur.append(w)
    if cur:
        out.append(cur)
    return out


def scroll_offsets(scroll_ms: int, span_ms: int, max_states: int = 8) -> list[tuple[int, float]]:
    """The intermediate states of a list scroll that starts at an entry's time: (offset from entry start in ms, progress
    0..1 exclusive). At most `max_states`; none when scroll_ms is 0. The scroll never takes more than 80% of the entry."""
    sc = min(scroll_ms, int(span_ms * 0.8))
    if sc < 40:
        return []
    m = min(max_states, max(1, round(sc / 40)))
    return [(round(j * sc / m), (j + 1) / (m + 1)) for j in range(m)]


def states_between(times: list[int], end_ms: int) -> list[tuple[int, int]]:
    """Contiguous (start, end) windows: state k runs from times[k] to times[k+1]; the last to `end_ms`."""
    ends = times[1:] + [end_ms]
    return [(a, max(a, b)) for a, b in zip(times, ends)]


# ---- animation ladders --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class AnimState:
    """One rendered state of an animation: how long it shows, and how the layer is transformed in it."""
    ms: int  # duration of the state (the last, steady state is open-ended: ms = 0)
    alpha: float = 1.0
    scale: float = 1.0
    dy: float = 0.0  # vertical offset as a fraction of the canvas height (negative = up)
    chars: float = 1.0  # type: fraction of the characters shown


def ladder(anim: str, dur_ms: int = 250, steps: int | None = None) -> list[AnimState]:
    """The in-animation as a few states (<= MAX_STATES including the steady one); the last has ms=0 and is the layer at
    rest. `none` is just the rest state."""
    if anim == "none":
        return [AnimState(0)]
    if anim == "fade":
        n = min(MAX_STATES - 1, steps or max(2, round(dur_ms / 50)))
        return [AnimState(round(dur_ms / n), alpha=(k + 1) / (n + 1)) for k in range(n)] + [AnimState(0)]
    if anim == "pop":  # 0.85 -> 1.06 -> 1.0 over 300 ms (default) in six states
        d = dur_ms if dur_ms != 250 else 300
        scales = [0.85, 0.94, 1.02, 1.06, 1.03]
        return [AnimState(round(d / (len(scales) + 1)), scale=s, alpha=min(1.0, 0.4 + 0.2 * k)) for k, s in enumerate(scales)] \
            + [AnimState(0)]
    if anim == "rise":  # up 4% of the height while fading in
        n = min(MAX_STATES - 1, steps or 5)
        return [AnimState(round(dur_ms / n), alpha=(k + 1) / (n + 1), dy=0.04 * (1 - (k + 1) / (n + 1))) for k in range(n)] \
            + [AnimState(0)]
    if anim == "type":
        n = min(MAX_STATES - 1, steps or 9)
        return [AnimState(round(dur_ms / n), chars=(k + 1) / (n + 1)) for k in range(n)] + [AnimState(0)]
    raise ValueError(f"unknown anim '{anim}'")


def windows_for(in_ms: int, out_ms: int | None, states: list[AnimState]) -> list[tuple[int, int | None, AnimState]]:
    """(start, end, state) windows for a layer shown from in_ms to out_ms (None = to the end; the end is then None).
    The in-animation states run first; the layer then rests until out_ms. Animations are cut short by out_ms."""
    out: list[tuple[int, int | None, AnimState]] = []
    t = in_ms
    for st in states[:-1]:
        end = t + st.ms if out_ms is None else min(t + st.ms, out_ms)
        if end > t:
            out.append((t, end, st))
        t = end
    if out_ms is None or out_ms > t:
        out.append((t, out_ms, states[-1]))
    return out
