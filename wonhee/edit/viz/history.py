"""Two PIL charts that need no browser: a commit heatmap (weeks x weekday, like a GitHub contribution grid) and a
language-mix bar (lines of code per language, from a depgraph scan).

Both are honest small multiples rather than decoration: every cell or segment is a real count, the caption line
states the totals, and colours come from the same theme tokens and language hashing as the HTML graphs, so a
heatmap and a dependency graph of one repo sit together on a page.
"""
from __future__ import annotations

import subprocess
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from wonhee.edit.viz.colors import language_rgb
from wonhee.edit.viz.graph import Graph
from wonhee.edit.viz.themes import theme as get_theme

FONTS = Path(__file__).resolve().parents[2] / "assets" / "fonts"
DISPLAY = {"python": "Python", "javascript": "JavaScript", "typescript": "TypeScript", "c": "C", "cpp": "C++",
           "cuda": "CUDA", "rust": "Rust"}
HEAT_HUE = {"paper": (31, 122, 61), "dark": (64, 196, 99)}  # one green ramp per theme, darkest = busiest


def _rgb(hex_: str) -> tuple[int, int, int]:
    h = hex_.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _mix(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))  # type: ignore[return-value]


def _font(px: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONTS / ("NotoSans-Bold.ttf" if bold else "NotoSans-Regular.ttf")), px)


def commit_days(path: Path, rev: str = "HEAD") -> list[date]:
    """One date per commit reachable from `rev` (UTC), oldest first."""
    r = subprocess.run(["git", "-C", str(path), "log", "--format=%ct", rev], capture_output=True, text=True)
    if r.returncode != 0:
        raise ValueError(f"git log {rev}: {r.stderr.strip()}")
    return sorted(datetime.fromtimestamp(int(t), tz=timezone.utc).date() for t in r.stdout.split())


def commit_heatmap(days: list[date], theme: str, size: tuple[int, int], title: str = "") -> Image.Image:
    t = get_theme(theme)
    W, H = size
    bg, ink, muted = _rgb(t["bg"]), _rgb(t["ink"]), _rgb(t["hint"])
    im = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(im)
    if not days:
        d.text((W // 2, H // 2), "no commits", fill=muted, font=_font(18), anchor="mm")
        return im
    start = days[0] - timedelta(days=days[0].weekday())  # Monday of the first week
    weeks = (days[-1] - start).days // 7 + 1
    counts: dict[date, int] = {}
    for day in days:
        counts[day] = counts.get(day, 0) + 1
    peak = max(counts.values())

    # title, month row, grid and caption are laid out as one block and centred, so a short (year-wide) grid does
    # not float in empty space
    pad = round(W * 0.05)
    label = _font(max(11, round(H * 0.035)))
    title_font = _font(max(14, round(H * 0.05)), bold=True)
    lh = label.size * 1.6
    title_h = title_font.size * 1.8 if title else 0
    cell = min((W - 2 * pad) / weeks, (H - 2 * pad - title_h - 2 * lh) / 7)
    gap = max(1.0, cell * 0.14)
    block = title_h + lh + 7 * cell + lh
    x0 = (W - cell * weeks) / 2
    top = (H - block) / 2
    y0 = top + title_h + lh
    empty = _mix(bg, ink, 0.08)
    hot = HEAT_HUE.get(theme, HEAT_HUE["paper"])
    for w in range(weeks):
        for wd in range(7):
            day = start + timedelta(days=7 * w + wd)
            if day > days[-1]:
                break
            n = counts.get(day, 0)
            # sqrt scale: one busy day should not wash every other active day out to the empty colour
            fill = empty if n == 0 else _mix(_mix(bg, hot, 0.35), hot, (n / peak) ** 0.5)
            x, y = x0 + w * cell, y0 + wd * cell
            d.rectangle([x, y, x + cell - gap, y + cell - gap], fill=fill)

    last_month, last_x = None, -1e9
    for w in range(weeks):
        day = start + timedelta(days=7 * w)
        x = x0 + w * cell
        if day.month != last_month:
            last_month = day.month
            text = day.strftime("%b")
            if x - last_x >= d.textlength(text, font=label) + 8:  # skip a label that would overlap the last one
                d.text((x, y0 - lh * 0.35), text, fill=muted, font=label, anchor="ls")
                last_x = x
    caption = (f"{len(days):,} commits on {len(counts):,} days, "
               f"{days[0].strftime('%b %Y')} to {days[-1].strftime('%b %Y')}")
    d.text((x0, y0 + 7 * cell + lh), caption, fill=muted, font=label, anchor="ls")
    if title:
        d.text((x0, top + title_font.size), title, fill=ink, font=title_font, anchor="ls")
    return im


def language_mix(graph: Graph, theme: str, size: tuple[int, int], title: str = "") -> Image.Image:
    t = get_theme(theme)
    W, H = size
    bg, ink, muted = _rgb(t["bg"]), _rgb(t["ink"]), _rgb(t["hint"])
    lightness = float(t["node_lightness"].rstrip("%")) / 100
    im = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(im)
    loc: dict[str, int] = {}
    for n in graph.nodes:
        loc[n.language] = loc.get(n.language, 0) + n.loc
    total = sum(loc.values())
    if not total:
        d.text((W // 2, H // 2), "no code files", fill=muted, font=_font(18), anchor="mm")
        return im
    parts = sorted(loc.items(), key=lambda kv: -kv[1])
    pad = round(W * 0.06)
    bar_y, bar_h = round(H * 0.38), round(H * 0.18)
    x = float(pad)
    span = W - 2 * pad
    label = _font(max(12, round(H * 0.045)))
    for i, (lang, n) in enumerate(parts):
        w = span * n / total
        d.rectangle([x, bar_y, x + w, bar_y + bar_h], fill=language_rgb(lang, lightness))
        x += w
    # legend under the bar, wrapping; tiny languages are still named so nothing is hidden
    lx, ly = pad, bar_y + bar_h + round(H * 0.08)
    for lang, n in parts:
        name = DISPLAY.get(lang, lang)
        text = f"{name} {100 * n / total:.0f}%" if n / total >= 0.005 else f"{name} <1%"
        tw = d.textlength(text, font=label) + round(H * 0.06) + 24
        if lx + tw > W - pad:
            lx, ly = pad, ly + round(H * 0.08)
        sw = round(H * 0.04)
        d.rectangle([lx, ly - sw, lx + sw, ly], fill=language_rgb(lang, lightness))
        d.text((lx + sw + 8, ly), text, fill=ink, font=label, anchor="ls")
        lx += tw
    d.text((pad, round(H * 0.26)), f"{total:,} lines of code in {len(graph.nodes):,} files",
           fill=muted, font=label, anchor="ls")
    if title:
        d.text((pad, round(H * 0.14)), title, fill=ink, font=_font(max(14, round(H * 0.06)), bold=True), anchor="ls")
    return im
