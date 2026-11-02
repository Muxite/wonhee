"""text_style node: reads the on-photo text style of a reference image post slide (colours, relative size, lines,
alignment, box or outline, dimming, where the block sits) and returns it as a `TextSpec`-shaped profile (JSON), so the
renderer needs no new vocabulary. Measured locally by `edit/textstyle`; the font role is never measured here, it comes
from `role_hint` (see docs/specs/2026-10-26-post-assistant-design.md, 2.4)."""
from __future__ import annotations

import json

import cv2
from pydantic import BaseModel, ConfigDict

from wonhee.db import repo
from wonhee.edit import text as T
from wonhee.edit import textstyle as ts
from wonhee.media import path_for
from wonhee.nodes import registry
from wonhee.nodes.base import Node
from wonhee.nodes.subtitles import save_text

ANALYZER_VERSION = 1  # the text_style table's key; bump with the measurements, like the node version


class TextStyleParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role_hint: str | None = None  # a key of edit.text.ROLES, read by eye or by the vision model; None = unknown


def text_style_fn(ctx, inputs, p: TextStyleParams) -> dict[str, str]:
    if p.role_hint is not None and p.role_hint not in T.ROLES:
        raise ValueError(f"unknown font role '{p.role_hint}'; known: {', '.join(T.ROLES)}")
    sha = inputs["image"]
    im = cv2.imread(str(path_for(ctx.settings.media_dir, sha)))
    if im is None:
        raise RuntimeError("cannot read the image")
    row = ctx.conn.execute("SELECT on_screen_text FROM image_analysis WHERE media_sha256=? AND on_screen_text IS NOT NULL "
                           "ORDER BY analyzer_version DESC LIMIT 1", (sha,)).fetchone()
    style = ts.measure(im, row[0] if row else None)
    if not style.found:
        ctx.flag("NO_TEXT_FOUND", "no text block found on the image; every field of the profile is null")
    elif style.notes:
        ctx.flag("TEXT_STYLE_UNSURE", "; ".join(style.notes)[:300])
    profile = json.dumps(ts.to_profile(style, p.role_hint), indent=1)
    repo.save_text_style(ctx.conn, sha=sha, analyzer_version=ANALYZER_VERSION, profile_json=profile)
    return {"profile": save_text(ctx.conn, ctx.settings.media_dir, profile)}


registry.register(Node("text_style", 1, {"image": "image"}, {"profile": "text"}, TextStyleParams, text_style_fn,
                       "On-photo text style of a reference image: colour pair, relative size, lines, alignment, box or "
                       "outline, dimming and block position, as a TextSpec-shaped profile (JSON). Font role only from "
                       "role_hint."))
