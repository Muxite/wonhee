"""Subtitles -> ASS text: plain white lower-third with a black outline (basic, not noticeable)."""
from __future__ import annotations

import re

from wonhee.nodes.subtitles import Subtitles

LATIN_FONT = "DejaVu Sans"
CJK_FONT = "Noto Sans CJK SC"
_CJK_LANGS = {"zh", "ja", "ko"}
_CJK_CHARS = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿＀-￯]")


def pick_font(subs: Subtitles, font: str | None = None) -> str:
    if font:
        return font
    lang = (subs.language or "").lower().split("-")[0]
    if lang in _CJK_LANGS or any(_CJK_CHARS.search(ln.text) for ln in subs.lines):
        return CJK_FONT
    return LATIN_FONT


def ass_time(ms: int) -> str:
    cs = round(max(ms, 0) / 10)
    h, rem = divmod(cs, 360_000)
    m, rem = divmod(rem, 6000)
    s, c = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def escape(text: str) -> str:
    # backslashes become a neutral look-alike so source text can never form \N, \n or \h
    return text.replace("\\", "\u29f5").replace("{", "\\{").replace("}", "\\}")


def wrap_lines(text: str, max_chars: int) -> list[str]:
    """At most 2 lines of max_chars; words when there are spaces, character chunks otherwise."""
    n = max(max_chars, 1)
    text = " ".join(text.split())
    if not text:
        return []
    if " " in text:
        lines: list[str] = []
        cur = ""
        for w in text.split(" "):
            if cur and len(cur) + 1 + len(w) > n:
                lines.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}" if cur else w
        lines.append(cur)
    else:
        lines = [text[i:i + n] for i in range(0, len(text), n)]
    if len(lines) > 2:
        # keep 2 lines; libass (WrapStyle 0) wraps whatever is left over in line 2
        lines = [lines[0], " ".join(lines[1:]) if " " in text else "".join(lines[1:])]
    return lines


def render_ass(subs: Subtitles, width: int, height: int, font: str | None = None, size_frac: float = 0.035,
               margin_frac: float = 0.22, max_chars: int = 28) -> str:
    fontname = pick_font(subs, font)
    size = max(round(height * size_frac), 1)
    margin_v = round(height * margin_frac)
    head = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {width}\nPlayResY: {height}\nWrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
        "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{fontname},{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,3,0,2,"
        f"20,20,{margin_v},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    events = []
    for ln in subs.lines:
        body = "\\N".join(escape(x) for x in wrap_lines(ln.text, max_chars))
        if not body:
            continue
        events.append(f"Dialogue: 0,{ass_time(ln.start_ms)},{ass_time(max(ln.end_ms, ln.start_ms))},"
                      f"Default,,0,0,0,,{body}")
    return head + "\n".join(events) + "\n"
