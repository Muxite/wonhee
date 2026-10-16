from __future__ import annotations

import re

PORT_TYPES = ("content", "video", "image", "subtitles", "text", "file")
# A ref is a str: "content:<id>" for content; a 64-hex blob sha for the rest.
SHA = re.compile(r"^[0-9a-f]{64}$")
CONTENT = re.compile(r"^content:(\d+)$")


def parse_type(t: str) -> tuple[str, bool]:
    """'subtitles?' -> ('subtitles', True). Unknown type -> ValueError."""
    optional = t.endswith("?")
    base = t[:-1] if optional else t
    if base not in PORT_TYPES:
        raise ValueError(f"unknown port type '{t}'")
    return base, optional


def ref_ok(type_: str, ref: str) -> bool:
    pat = CONTENT if type_ == "content" else SHA
    return bool(pat.match(ref))


def is_literal(value: str) -> bool:
    return bool(SHA.match(value) or CONTENT.match(value))
