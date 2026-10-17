"""The teardown of one inspiration post. Closed vocabularies, so teardowns aggregate and map onto wonhee's tools:
text style/position are the text_image/text_video presets, `look` is a look preset, capabilities are a fixed list."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from wonhee.edit.looks import Preset as LookPreset

Format = Literal["text-on-broll", "pov-text-meme", "talking-head", "skit", "music-edit", "dance-or-trend-audio",
                 "vlog", "tutorial-or-explainer", "news-or-info-card", "greenscreen", "photo-dump", "carousel-info",
                 "meme-image", "ad", "other"]
Coupling = Literal["decoupled", "loose", "coupled"]
Camera = Literal["static", "handheld", "pan", "pov", "gimbal", "screen-recording", "photo"]
Capability = Literal[
    "text_overlay", "colour_emoji", "band_text", "script_text", "per_shot_text", "look_grade", "film_grain",
    "datestamp", "carousel_layout", "panorama_span", "hard_cuts", "beat_sync_cuts", "speed_ramp", "zoom_or_pan",
    "transitions", "green_screen_key", "screen_recording", "subtitles_from_speech", "translation",
    "voiceover", "two_person_shoot", "on_camera_performance", "props_or_costume", "location_shoot",
    "music_in_app", "reframe_9x16", "stock_or_archive_footage", "illustration_or_drawing"]


class Coupled(BaseModel):
    level: Coupling
    why: str


class Hook(BaseModel):
    first_2s: str
    mechanism: str


class Overlay(BaseModel):
    present: bool
    english: str | None = None  # the overlay text, translated to English if needed
    style: Literal["outline", "band", "script", "plain", "other", "none"] = "none"
    position: Literal["top", "upper", "center", "lower", "bottom", "none"] = "none"


class Audio(BaseModel):
    kind: Literal["trending-song", "original-voice", "voiceover", "ambient", "none", "unknown"]
    role: str


class Footage(BaseModel):
    subject: str
    setting: str
    light: str
    camera: Camera
    framing: str
    duration_s: float = Field(ge=0, le=180)
    person_on_camera: bool
    difficulty_1to5: int = Field(ge=1, le=5)


class Claim(BaseModel):
    claim: str
    evidence: str


class Replicate(BaseModel):
    score_1to5: int = Field(ge=1, le=5)
    effort_minutes: int = Field(ge=0, le=6000)
    blockers: list[str] = []
    fit_for_creator: str


class Teardown(BaseModel):
    summary: str
    format: Format
    coupling: Coupled
    hook: Hook
    text_overlay: Overlay
    look: LookPreset | None = None  # the closest wonhee look preset to the post's grade, or null
    audio: Audio
    footage_to_record: list[Footage] = []
    edit_recipe: list[str] = []
    why_it_works: list[Claim] = []
    replicate: Replicate
    text_template: str | None = None  # reusable line with [slots]
    learn: list[str] = []
    capabilities: list[Capability] = []
    confidence: Literal["high", "medium", "low"]
    gaps: list[str] = []
