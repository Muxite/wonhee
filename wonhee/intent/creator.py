from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class Creator(BaseModel):
    on_camera: bool = True
    niches: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    props: list[str] = Field(default_factory=list)
    language: str = "en"
    max_length_s: int = 30


def load_creator(path: Path) -> Creator:
    if not Path(path).is_file():
        return Creator()
    return Creator.model_validate(yaml.safe_load(Path(path).read_text()) or {})
