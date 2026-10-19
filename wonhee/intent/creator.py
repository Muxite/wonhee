from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class Skin(BaseModel):
    """Skin undertone the creator likes; filters and vibe edits nudge skin-coloured pixels toward it."""
    undertone: Literal["none", "cool", "neutral", "warm", "rose", "gold"] = "none"
    strength: float = Field(default=0.5, ge=0, le=1)


class Creator(BaseModel):
    on_camera: bool = True
    niches: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    props: list[str] = Field(default_factory=list)
    language: str = "en"
    max_length_s: int = 30
    skin: Skin = Field(default_factory=Skin)


def load_creator(path: Path) -> Creator:
    if not Path(path).is_file():
        return Creator()
    return Creator.model_validate(yaml.safe_load(Path(path).read_text()) or {})
