"""Filling a reel style's slots from the words and the options (the `wonhee reel` command's brain, no DB, no ffmpeg).

Design: docs/specs/2026-10-20-reel-styles-design.md §6 (the styles and the one command) and §7 (the lecture slots).
A style is a `compose_video` template; its text slots come from the options (--title, --cta, --items), and its *times*
(chapter tiles, insets, theses, footnotes, the closing card) come from the spoken words: they land on phrase starts,
because cuts and overlays follow words, not a beat (study rule 2). Everything is in source-timeline milliseconds;
`compose_video` maps it through the pause cuts.

`fill_slots` returns (slots, notes): the notes say what an option could not do for this style, so the report can
show it instead of dropping it silently.
"""
from __future__ import annotations

import re

from wonhee.nodes.subtitles import Word

CARD_MS = 2500  # the closing card / lines at the end of the take
CTA_MS = 3000
PHRASE_GAP_MS = 300  # a pause this long, or a sentence end, makes the next word a phrase start
MIN_APART_MS = 1200  # chapters, theses and insets never start closer than this
NEVER_MS = 9999000  # a template's default for a timed slot that has not been set: never shown
MARKERS = "¹²³"
_STOP = {"which", "their", "there", "these", "those", "about", "would", "could", "should", "where", "being", "because",
         "while", "other", "after", "before", "every", "thing", "things"}


def phrase_starts(words: list[Word]) -> list[int]:
    """The start of each phrase: the first word, and any word after a pause of PHRASE_GAP_MS or a sentence end."""
    out: list[int] = []
    for i, w in enumerate(words):
        if i == 0 or w.start_ms - words[i - 1].end_ms >= PHRASE_GAP_MS or re.search(r"[.!?…]$", words[i - 1].text):
            out.append(w.start_ms)
    return out


def spread_times(words: list[Word], n: int, lo_ms: int, hi_ms: int) -> list[int]:
    """`n` increasing times in [lo_ms, hi_ms], evenly spread and snapped to phrase starts (else to word starts, else
    the even point), at least MIN_APART_MS apart when the range allows it."""
    if n <= 0 or hi_ms <= lo_ms:
        return []
    starts = [s for s in phrase_starts(words) if lo_ms <= s <= hi_ms] or [w.start_ms for w in words if lo_ms <= w.start_ms <= hi_ms]
    out: list[int] = []
    for k in range(n):
        target = lo_ms + (k + 0.5) * (hi_ms - lo_ms) / n
        pool = [s for s in starts if not out or s >= out[-1] + MIN_APART_MS]
        out.append(min(pool, key=lambda s: abs(s - target)) if pool else round(max(target, (out[-1] + MIN_APART_MS) if out else 0)))
    return sorted(set(out)) if len(set(out)) == n else [round(lo_ms + (k + 0.5) * (hi_ms - lo_ms) / n) for k in range(n)]


def windows(times: list[int], end_ms: int, max_ms: int | None = None) -> list[dict]:
    """{in_ms, out_ms} windows: each starts at its time and runs to the next (or `end_ms`), at most `max_ms` long."""
    out = []
    for k, t in enumerate(times):
        nxt = times[k + 1] if k + 1 < len(times) else end_ms
        out.append({"in_ms": t, "out_ms": min(nxt, t + max_ms) if max_ms else nxt})
    return out


def _tail(duration_ms: int, ms: int) -> dict:
    return {"in_ms": max(0, duration_ms - ms)}


def _sentences(text: str) -> list[str]:
    parts = re.findall(r"[^.!?]+[.!?]?", text)
    return [p.strip() for p in parts if p.strip()]


def footnote_words(words: list[Word], n: int, lo_ms: int, hi_ms: int) -> list[Word]:
    """`n` caption words to hang footnote markers on: long enough to be a noun-ish word, spread over the take, each
    spoken once at its first use (the marker goes on the first use) and distinct."""
    first_use: dict[str, Word] = {}  # the marker goes on a word's first use in the captions, so the footnote anchors there
    for w in words:
        first_use.setdefault(re.sub(r"\W", "", w.text).lower(), w)
    uniq = [w for key, w in first_use.items() if lo_ms <= w.start_ms <= hi_ms and len(key) >= 5 and key not in _STOP]
    out: list[Word] = []
    for k in range(n):
        target = lo_ms + (k + 0.5) * (hi_ms - lo_ms) / n
        pool = [w for w in uniq if w not in out and (not out or w.start_ms >= out[-1].start_ms + MIN_APART_MS)]
        if not pool:
            break
        out.append(min(pool, key=lambda w: abs(w.start_ms - target)))
    return sorted(out, key=lambda w: w.start_ms)


def lecture_slots(words: list[Word], duration_ms: int, lecture: dict, notes: list[str]) -> dict:
    """The `lecture` style's slots from larp's `lecture` dict (title, subtitle, theses, footnotes, citation) and the words:
    theses fade in at phrase gaps and are replaced by the next; footnotes hang on caption words and show from that word;
    the closing card takes the last CARD_MS."""
    slots: dict = {}
    for k in ("title", "subtitle", "citation"):
        if lecture.get(k):
            slots[k] = str(lecture[k]).replace("*", "").strip("_ ")  # a model may mark the italic subtitle up
    card_in = max(0, duration_ms - CARD_MS)
    first = words[0].start_ms if words else 0
    theses = list(lecture.get("theses") or [])[:4]
    if len(lecture.get("theses") or []) > 4:
        notes.append("more than 4 theses given; the first 4 are used")
    times = spread_times(words, len(theses), first + 800, card_in - 800) if theses else []
    if len(times) < len(theses):
        notes.append("the take is too short for that many theses; the last ones were dropped")
        theses = theses[:len(times)]
    for k, (text, win) in enumerate(zip(theses, windows(times, card_in)), 1):
        slots[f"thesis_{k}"], slots[f"thesis_{k}_t"] = text, win
    feet = list(lecture.get("footnotes") or [])[:3]
    anchors = footnote_words(words, len(feet), first + 1500, card_in - 1500) if feet else []
    if len(anchors) < len(feet):
        notes.append("not enough long words to hang every footnote on; the extra footnotes were dropped")
    marks, ft = {}, windows([w.start_ms for w in anchors], card_in)
    for k, (w, text, win) in enumerate(zip(anchors, feet, ft), 1):
        marks[re.sub(r"\W", "", w.text).lower()] = MARKERS[k - 1]
        slots[f"footnote_{k}"], slots[f"footnote_{k}_t"] = f"{MARKERS[k - 1]} {text}", win
    if marks:
        slots["marks"] = marks
    return slots


def fill_slots(style: str, declared: dict, words: list[Word], duration_ms: int, *, title: str | None = None,
               cta: str | None = None, items: list[str] | None = None, n_insets: int = 0,
               lecture: dict | None = None) -> tuple[dict, list[str]]:
    """Slot values for `style` (declared: the template's slots). Options the style has no slot for are reported."""
    slots: dict = {}
    notes: list[str] = []
    first = words[0].start_ms if words else 0
    tail_in = max(0, duration_ms - CTA_MS)
    if title and "title" in declared:
        slots["title"] = title
    if cta and style != "list":  # a list's closing lines are entries of the list itself
        if "cta" in declared:
            slots["cta"] = cta
            if "cta_t" in declared:  # the lecture's closing card has one time for both lines (card_t)
                slots["cta_t"] = _tail(duration_ms, CARD_MS if style == "series" else CTA_MS)
        else:
            notes.append(f"--cta ignored: '{style}' has no closing line")
    n_in = min(n_insets, 4)
    if n_insets and "inset1" not in declared:
        notes.append(f"--inset ignored: '{style}' has no inset slots")
        n_in = 0
    elif n_insets > 4:
        notes.append(f"{n_insets} insets given; the first 4 are used")
    if n_in:
        lo = {"series": 3200, "list": first}.get(style, max(first, 4200))
        times = spread_times(words, n_in, lo, max(lo + MIN_APART_MS, tail_in))
        wins = windows(times, tail_in, 2400 if style == "scrapbook" else None)
        for k, w in enumerate(wins, 1):
            slots[f"inset{k}"], slots[f"inset{k}_t"] = k - 1, w
    if style == "scrapbook":
        labels = list(items or [])[:4]
        if items and len(items) > 4:
            notes.append(f"{len(items)} items given; chapters take the first 4")
        times = spread_times(words, len(labels), max(first, 4200), tail_in - 500)
        for k, (lab, w) in enumerate(zip(labels, windows(times, tail_in, 5000)), 1):
            slots[f"ch{k}"], slots[f"ch{k}_t"] = lab, w
    elif style == "list":
        entries = list(items or [])
        if cta:
            entries += _sentences(cta)
        if entries:
            slots["items"] = entries
            lo = first
            slots["at_ms"] = spread_times(words, len(entries), lo, max(lo + MIN_APART_MS, duration_ms - 800)) or None
            if slots["at_ms"] is not None and len(slots["at_ms"]) < len(entries):
                slots["at_ms"] = None  # too short to place each entry on a phrase: the layer spreads them evenly
    elif style == "lecture":
        if items:
            lecture = {**(lecture or {}), "theses": items}
        slots = {**lecture_slots(words, duration_ms, lecture or {}, notes), **slots}  # explicit options win
        slots["card_t"] = _tail(duration_ms, CARD_MS)
    elif items:
        notes.append(f"--items ignored: '{style}' has no list slots")
    return slots, notes
