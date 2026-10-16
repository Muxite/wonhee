from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_NAME = "scout"
SCHEMA_VERSION = 1


class Window(BaseModel):
    start: date
    end: date
    label: str


class Constraints(BaseModel):
    on_camera: bool | None = None
    max_length_s: int | None = None
    tone: str | None = None


class ScoutRequest(BaseModel):
    mode: Literal["discover", "from_reference"] | None = None
    niche: str | None = None
    window: Window | None = None
    reference_urls: list[str] = Field(default_factory=list)
    trend_name: str | None = None
    platform: Literal["instagram"] = "instagram"
    constraints: Constraints = Field(default_factory=Constraints)
    acknowledged: list[str] = Field(default_factory=list)  # fields the user chose to keep as-is

    def describe(self) -> str:
        subject = self.niche or self.trend_name or ", ".join(self.reference_urls) or "?"
        how = "from your reference" if self.mode == "from_reference" else "I pick the trend"
        win = f"{self.window.label} ({self.window.start}..{self.window.end})" if self.window else "no window"
        cam = {True: "on-camera", False: "off-camera", None: "camera: creator default"}[self.constraints.on_camera]
        length = f"≤{self.constraints.max_length_s}s" if self.constraints.max_length_s else "length: creator default"
        return f"Scout: {subject}, {self.platform}, {win}, {how}, {cam}, {length}."


class Extracted(BaseModel):
    """What the model may pull out of one user message. Unknown → null. Never invent."""
    niche: str | None = None
    window_phrase: Literal["now", "6mo", "1y", "2y"] | None = None
    window_start: date | None = None
    window_end: date | None = None
    reference_urls: list[str] = Field(default_factory=list)
    trend_name: str | None = None
    on_camera: bool | None = None
    max_length_s: int | None = None
    tone: str | None = None
    keep_as_is: list[Literal["on_camera", "max_length_s"]] = Field(default_factory=list)
    drop_bad_urls: bool = False
