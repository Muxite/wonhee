"""Rank candidate cuts of the owner talking (post-assistant spec §2.2): pure functions on word timings, no DB, no pixels.

Why it exists: the owner records freely, several takes of one idea, and wonhee proposes which span to keep. A plan is a
start/end (or a few spans) on one take's own timeline plus the measured reasons; rendering stays with `compose_video`.

Scoring is rule-based and every weight lives in the block of constants below, so the ranking can be tuned in one
place. Order of importance (owner): delivery and fewest stumbles first. Nothing here rewards "energy", loudness, speed
or length for their own sake, because some videos are deadpan on purpose. Dead air is only penalised past an allowance
(the Tighten threshold plus a normal sentence break), so a deliberate beat survives and a long gap does not.
Unknown inputs (no audio stat) are None and the term is skipped, never scored as 0.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from wonhee.analyze.speech import SpokenWord
from wonhee.edit.timeline import SENTENCE_END

# --- Pause allowance: what the existing `tighten` keeps, plus one normal sentence break --------------------------------
# Copies of nodes.compose.Tighten's defaults (importing it would pull cv2 into a pure engine); a test pins them.
TIGHTEN_MIN_PAUSE_MS = 250   # find_pauses cuts gaps of at least this
TIGHTEN_KEEP_MS = 80         # what is left of a cut pause
SENTENCE_BREAK_MS = 400      # a normal breath between sentences; shorter than this is never dead air (guess)
DEAD_AIR_ALLOWANCE_MS = TIGHTEN_MIN_PAUSE_MS + SENTENCE_BREAK_MS

# --- Weights (points; higher score is better) ----------------------------------------------------------------------
# Stumbles dominate: one costs more than a second of dead air or ~25 dB of loudness spread.
STUMBLE_WEIGHT = {"repeat": 10.0, "false_start": 10.0, "trail_off": 10.0, "filler": 8.0, "like": 4.0}
DEAD_AIR_PER_SEC = 8.0       # per second of silence beyond DEAD_AIR_ALLOWANCE_MS
PREAMBLE_WEIGHT = 2.0        # per throat-clearing word before the first word that carries content
LEAD_IN_PER_SEC = 2.0        # per second from the plan's start to its first content word beyond LEAD_IN_GRACE_MS
LEAD_IN_GRACE_MS = 500
LOUDNESS_PER_DB = 0.3        # per dB of loudness spread beyond LOUDNESS_GRACE_DB; only when a stat is supplied
LOUDNESS_GRACE_DB = 6.0
SPLICE_WEIGHT = 12.0         # per join: dearer than a stumble, so a spliced plan only wins when the clean span is much
                             # smaller (owner: fewer cuts, one clean take over spliced fragments)
COVERAGE_WEIGHT = 20.0       # share of the take's words kept; stops "the shortest clean sentence" from always winning
MIN_SPAN_MS = 1500           # a plan shorter than this is not a usable cut (unless the whole take is shorter)

FILLERS = frozenset({"um", "uh", "uhm", "umm", "er", "erm", "ah", "hmm", "mm", "mmm"})
PREAMBLE = frozenset({"so", "okay", "ok", "alright", "anyway", "well", "hey", "hi", "right", "basically"})
_TRAIL = re.compile(r"(\.\.\.|…)$")
_CUT_OFF = re.compile(r"[-–—]$")
_TOKEN = re.compile(r"[^\w']+", re.UNICODE)


@dataclass
class Plan:
    take: int
    start_ms: int
    end_ms: int
    score: float
    why: list[str]
    dropped: list[tuple[int, int, str]]
    spans: list[tuple[int, int]]            # what is kept; one span unless the plan is spliced
    components: dict[str, float | None]     # signed points per term; None = not measured
    spliced: bool = False

    def to_dict(self) -> dict:
        return {"take": self.take, "start_ms": self.start_ms, "end_ms": self.end_ms, "score": self.score,
                "spliced": self.spliced, "spans": [list(s) for s in self.spans], "why": list(self.why),
                "dropped": [list(d) for d in self.dropped], "components": dict(self.components)}


@dataclass
class _Sentence:
    lo: int   # word index range [lo, hi)
    hi: int
    start_ms: int
    end_ms: int


def _norm(text: str) -> str:
    return _TOKEN.sub("", text.lower())


def find_stumbles(words: Sequence[SpokenWord]) -> dict[int, str]:
    """word index -> kind, for the word that is the stumble (the second of a repeat, the filler itself).

    Repeats are checked longest-first (three words, two, one) so 'the pump the pump' is one stumble, not two."""
    toks = [_norm(w.text) for w in words]
    out: dict[int, str] = {}
    covered: set[int] = set()
    for i, w in enumerate(words):
        if i in covered or not toks[i]:
            continue
        for n in (3, 2, 1):
            if i - n >= 0 and i + n <= len(words) and toks[i:i + n] == toks[i - n:i] and all(toks[i:i + n]):
                out[i] = "repeat"
                covered.update(range(i, i + n))
                break
        else:
            if toks[i] in FILLERS:
                out[i] = "filler"
            elif toks[i] == "like":
                out[i] = "like"
            elif toks[i] == "know" and i > 0 and toks[i - 1] == "you":
                out[i] = "filler"
            elif _CUT_OFF.search(w.text.strip()):
                out[i] = "false_start"
            elif _TRAIL.search(w.text.strip()):
                out[i] = "trail_off"
    return out


def _sentences(words: Sequence[SpokenWord]) -> list[_Sentence]:
    out, lo = [], 0
    for i, w in enumerate(words):
        last = i == len(words) - 1
        gap = words[i + 1].start_ms - w.end_ms if not last else 0
        if last or SENTENCE_END.search(w.text.strip()) or gap > DEAD_AIR_ALLOWANCE_MS:
            out.append(_Sentence(lo, i + 1, words[lo].start_ms, w.end_ms))
            lo = i + 1
    return out


def _first_content(words: Sequence[SpokenWord], lo: int, hi: int) -> tuple[int, int]:
    """(index of the first word that carries content, number of throat-clearing words before it)."""
    n = 0
    for i in range(lo, hi):
        t = _norm(words[i].text)
        if t in FILLERS or t in PREAMBLE:
            n += 1
            continue
        return i, n
    return hi, n


def _score(words, stumbles, sents: list[_Sentence], keep: list[int], total_words: int,
           loudness_db: float | None) -> tuple[float, dict[str, float | None], list[str]]:
    """Score the sentences `keep` (indices into sents, ascending; more than one run = spliced)."""
    comp: dict[str, float | None] = {}
    why: list[str] = []
    kept_words = sum(sents[k].hi - sents[k].lo for k in keep)
    cover = kept_words / total_words
    comp["coverage"] = round(COVERAGE_WEIGHT * cover, 2) + 0.0
    why.append(f"keeps {kept_words}/{total_words} words ({cover:.0%})")
    kinds: dict[str, int] = {}
    for k in keep:
        for i in range(sents[k].lo, sents[k].hi):
            if i in stumbles:
                kinds[stumbles[i]] = kinds.get(stumbles[i], 0) + 1
    st = -sum(STUMBLE_WEIGHT[kd] * n for kd, n in kinds.items())
    comp["stumbles"] = round(st, 2) + 0.0
    why.append("stumbles: " + (", ".join(f"{n} {kd}" for kd, n in sorted(kinds.items())) if kinds else "none"))
    # Dead air: only gaps inside a kept run are heard; a gap between two runs is the splice's business.
    air_ms, longest = 0, 0
    for k in keep:
        s = sents[k]
        for i in range(s.lo, s.hi - 1):
            gap = words[i + 1].start_ms - words[i].end_ms
            longest = max(longest, gap)
            air_ms += max(0, gap - DEAD_AIR_ALLOWANCE_MS)
    for a, b in zip(keep, keep[1:]):   # adjacent sentences with no splice between them: the break counts too
        if sents[b].lo == sents[a].hi:
            gap = words[sents[b].lo].start_ms - words[sents[a].hi - 1].end_ms
            longest = max(longest, gap)
            air_ms += max(0, gap - DEAD_AIR_ALLOWANCE_MS)
    comp["dead_air"] = round(-DEAD_AIR_PER_SEC * air_ms / 1000, 2) + 0.0
    why.append(f"dead air: {air_ms} ms past the {DEAD_AIR_ALLOWANCE_MS} ms allowance (longest gap {longest} ms)")
    first = keep[0]
    fi, pre = _first_content(words, sents[first].lo, sents[keep[-1]].hi)
    lead = max(0, (words[fi].start_ms if fi < len(words) else sents[first].start_ms) - sents[first].start_ms)
    pt = -(PREAMBLE_WEIGHT * pre + LEAD_IN_PER_SEC * max(0, lead - LEAD_IN_GRACE_MS) / 1000)
    comp["point_early"] = round(pt, 2) + 0.0
    why.append(f"point: {pre} lead-in word(s), content starts {lead} ms in")
    joins = sum(1 for a, b in zip(keep, keep[1:]) if sents[b].lo != sents[a].hi)
    comp["splice"] = round(-SPLICE_WEIGHT * joins, 2) + 0.0
    if joins:
        why.append(f"{joins} join(s)")
    if loudness_db is None:
        comp["loudness"] = None
    else:
        comp["loudness"] = round(-LOUDNESS_PER_DB * max(0.0, loudness_db - LOUDNESS_GRACE_DB), 2) + 0.0
        why.append(f"loudness spread {loudness_db:g} dB (grace {LOUDNESS_GRACE_DB:g})")
    total = round(sum(v for v in comp.values() if v is not None), 2) + 0.0
    return total, comp, why


def _pad_bounds(words, lo: int, hi: int) -> tuple[int, int]:
    """Span times for words [lo, hi): half of Tighten's kept pause on each side, never into a neighbouring word."""
    pad = TIGHTEN_KEEP_MS // 2
    s, e = words[lo].start_ms, words[hi - 1].end_ms
    ps = max(0, s - pad) if lo == 0 else max(s - pad, (words[lo - 1].end_ms + s) // 2)
    pe = e + pad if hi == len(words) else min(e + pad, (e + words[hi].start_ms) // 2)
    return ps, pe


def _join_ok(words, a_hi: int, b_lo: int) -> bool:
    """A splice lands cleanly when both cut points have the room Tighten would leave (keep_ms / 2) in silence. The
    join then sounds like a tightened pause (keep_ms), the longest pause the spec allows a join to leave."""
    pad = TIGHTEN_KEEP_MS // 2
    return (words[a_hi].start_ms - words[a_hi - 1].end_ms >= pad
            and words[b_lo].start_ms - words[b_lo - 1].end_ms >= pad)


def _stumbly(stumbles, sent: _Sentence) -> bool:
    return any(i in stumbles for i in range(sent.lo, sent.hi))


def _describe(stumbles, words, lo, hi) -> str:
    kinds: dict[str, int] = {}
    for i in range(lo, hi):
        if i in stumbles:
            kinds[stumbles[i]] = kinds.get(stumbles[i], 0) + 1
    return ", ".join(f"{n} {k}" for k, n in sorted(kinds.items())) or "outside the best span"


def _plan(take: int, words, stumbles, sents, keep, total_words, loudness_db) -> Plan:
    score, comp, why = _score(words, stumbles, sents, keep, total_words, loudness_db)
    runs: list[list[int]] = []
    for k in keep:
        if runs and sents[k].lo == sents[runs[-1][-1]].hi:
            runs[-1].append(k)
        else:
            runs.append([k])
    spans = [_pad_bounds(words, sents[r[0]].lo, sents[r[-1]].hi) for r in runs]
    dropped: list[tuple[int, int, str]] = []
    cursor = 0   # word index: everything before it is accounted for
    for r in runs:
        lo = sents[r[0]].lo
        if lo > cursor:
            dropped.append((_pad_bounds(words, cursor, lo)[0] if cursor else 0, spans[runs.index(r)][0],
                            _describe(stumbles, words, cursor, lo)))
        cursor = sents[r[-1]].hi
    if cursor < len(words):
        dropped.append((spans[-1][1], words[-1].end_ms, _describe(stumbles, words, cursor, len(words))))
    return Plan(take=take, start_ms=spans[0][0], end_ms=spans[-1][1], score=score, why=why, dropped=dropped,
                spans=spans, components=comp, spliced=len(spans) > 1)


def rank_takes(takes: Sequence[Sequence[SpokenWord]], loudness_db: Sequence[float | None] | None = None,
               top_n: int = 3) -> list[Plan]:
    """Ranked candidate cuts across all takes, best first.

    Per take: the best contiguous span (any run of whole sentences long enough to use) and, when a stumbly sentence
    sits inside it and the cut points have room, one spliced variant with those sentences removed. `loudness_db` is
    one spread per take (dB between the loud and quiet parts of the take, measured by the caller), or None / a None
    entry when unmeasured. Output is deterministic: ties break on take, then start, then end."""
    plans: list[Plan] = []
    for t, raw in enumerate(takes):
        words = sorted((w for w in raw if w.text.strip()), key=lambda w: (w.start_ms, w.end_ms))
        if not words:
            continue
        spread = loudness_db[t] if loudness_db is not None and t < len(loudness_db) else None
        stumbles = find_stumbles(words)
        sents = _sentences(words)
        total = len(words)
        whole = words[-1].end_ms - words[0].start_ms
        best: tuple[float, list[int]] | None = None
        base: tuple[float, list[int]] | None = None   # best window with a stumbly interior sentence to cut out
        for a in range(len(sents)):
            for b in range(a, len(sents)):
                span_ms = sents[b].end_ms - sents[a].start_ms
                if span_ms < MIN_SPAN_MS and whole >= MIN_SPAN_MS:
                    continue
                keep = list(range(a, b + 1))
                score = _score(words, stumbles, sents, keep, total, spread)[0]
                if best is None or score > best[0]:   # first wins ties: earliest start, then shortest
                    best = (score, keep)
                if any(_stumbly(stumbles, sents[k]) for k in keep[1:-1]) and (base is None or score > base[0]):
                    base = (score, keep)
        if best is None:   # every window under the minimum and the take itself is long: cannot happen, but be safe
            best = (0.0, list(range(len(sents))))
        plans.append(_plan(t, words, stumbles, sents, best[1], total, spread))
        # One spliced variant: drop the interior sentences that hold stumbles, if every join lands cleanly. It is
        # built from the best window that has such a sentence, which need not be the best contiguous window.
        if base is not None:
            keep = base[1]
            kept = [k for k in keep if k in (keep[0], keep[-1]) or not _stumbly(stumbles, sents[k])]
            joins = [(a, b) for a, b in zip(kept, kept[1:]) if sents[b].lo != sents[a].hi]
            if all(_join_ok(words, sents[a].hi, sents[b].lo) for a, b in joins):
                plans.append(_plan(t, words, stumbles, sents, kept, total, spread))
    plans.sort(key=lambda p: (-p.score, p.take, p.start_ms, p.end_ms))
    return plans[:top_n] if top_n else plans

