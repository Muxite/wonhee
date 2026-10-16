from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Deps:
    apify: Any
    search: Any
    downloader: Any
    transcriber: Any

    @classmethod
    def live(cls, settings) -> "Deps":
        from wonhee.analyze.speech import WhisperTranscriber
        from wonhee.config import secret
        from wonhee.sources.apify import LiveApify
        from wonhee.sources.fetch import LiveDownloader
        from wonhee.sources.websearch import BraveSearch

        apify_token, brave_key = secret("APIFY_API_KEY", settings), secret("BRAVE_API_KEY", settings)
        return cls(
            apify=LiveApify(apify_token, settings.apify_price_per_1k) if apify_token else Missing("APIFY_API_KEY"),
            search=BraveSearch(brave_key, settings.brave_endpoint) if brave_key else Missing("BRAVE_API_KEY"),
            downloader=LiveDownloader(),
            transcriber=WhisperTranscriber(settings.whisper_model, settings.whisper_device, settings.whisper_compute_type),
        )


class Missing:
    def __init__(self, key: str):
        self.key = key

    def __getattr__(self, name):
        from wonhee.pipeline.context import StageFailed

        def _raise(*a, **k):
            raise StageFailed("MISSING_KEY", f"{self.key} is not set; put it in .env or export it (see `wonhee doctor`)")
        return _raise
