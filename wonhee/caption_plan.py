"""Deterministic source-time caption plans; no hidden writing or rendering calls."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CaptionCue(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    start_seconds: float = Field(ge=0, le=45)
    end_seconds: float = Field(gt=0, le=45)
    text: str = Field(min_length=1, max_length=500)

    @model_validator(mode='after')
    def valid(self):
        if self.end_seconds <= self.start_seconds or not self.text.strip():
            raise ValueError('caption cue needs nonempty text and end after start')
        for value in (self.start_seconds, self.end_seconds):
            if abs(value*1000 - round(value*1000)) > 1e-6:
                raise ValueError('caption cue times must use whole milliseconds')
        return self


def validate_cues(cues: list[dict], duration_ms: int) -> list[CaptionCue]:
    if not 1 <= len(cues) <= 12:
        raise ValueError('caption_cues needs 1..12 contiguous cues covering the full source')
    parsed = [CaptionCue.model_validate(cue) for cue in cues]
    end = 0
    for cue in parsed:
        if round(cue.start_seconds*1000) != end:
            raise ValueError('caption cues must be ordered, contiguous and non-overlapping from zero')
        end = round(cue.end_seconds*1000)
    if end != duration_ms:
        raise ValueError(f'caption cues must cover the source exactly ({duration_ms/1000:g} seconds)')
    return parsed


def render_layout(slots: dict, cues: list[dict], duration_ms: int) -> dict:
    from wonhee.edit import compose as C
    parsed = validate_cues(cues, duration_ms)
    layout = C.load_template('desktop_reel', {'bg':slots.get('bg','#080d12')})
    layout.layers = [ly for ly in layout.layers if ly.type != 'text']
    for cue in parsed:
        checked = C.load_template('desktop_reel', {'title':cue.text})
        title = next(ly for ly in checked.layers if ly.type == 'text')
        title.t = C.TimeSpan(in_ms=round(cue.start_seconds*1000), out_ms=round(cue.end_seconds*1000))
        layout.layers.append(title)
    return layout.model_dump(mode='json', exclude_none=True)
