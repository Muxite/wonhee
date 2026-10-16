from pathlib import Path

from scenedetect import ContentDetector, detect

from wonhee.media import probe


def detect_shots(path: Path) -> list[tuple[int, int]]:
    scenes = detect(str(path), ContentDetector())
    if not scenes:
        dur = probe(path).duration_ms or 0
        return [(0, max(dur, 1))]
    return [(int(s.seconds * 1000), int(e.seconds * 1000)) for s, e in scenes]
