from __future__ import annotations

from datetime import date, timedelta

from wonhee.intent.schemas import Window

_MONTHS = {"6mo": 6, "1y": 12, "2y": 24}
_LABEL = {"6mo": "6 months ago", "1y": "1 year ago", "2y": "2 years ago"}


def _months_ago(today: date, n: int) -> date:
    y, m = divmod(today.year * 12 + today.month - 1 - n, 12)
    day = min(today.day, 28)
    return date(y, m + 1, day)


def resolve_window(phrase: str | None, start: date | None, end: date | None, today: date) -> Window | None:
    if start and end:
        return Window(start=start, end=end, label=f"{start}..{end}")
    if phrase == "now":
        return Window(start=today - timedelta(days=30), end=today, label="now")
    if phrase in _MONTHS:
        center = _months_ago(today, _MONTHS[phrase])
        return Window(start=center - timedelta(days=30), end=center + timedelta(days=30), label=_LABEL[phrase])
    return None
