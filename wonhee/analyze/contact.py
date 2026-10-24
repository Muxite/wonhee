"""Timestamped contact sheets let an agent inspect the actual frames of an uploaded take."""
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from wonhee.analyze.frames import extract_frame
from wonhee.media import probe


def contact_sheet(src: Path, dest: Path, *, frames: int = 6) -> None:
    duration = probe(src).duration_ms
    if duration is None:
        raise ValueError('the video duration is unknown')
    columns = min(3, frames)
    rows = (frames + columns - 1) // columns
    sheet = Image.new('RGB', (columns * 320, rows * 264), '#202020')
    draw = ImageDraw.Draw(sheet)
    with tempfile.TemporaryDirectory(prefix='wonhee-contact-') as td:
        for i in range(frames):
            at = min(max(0, duration - 100), int((i + .5) * duration / frames))
            path = extract_frame(src, at, Path(td) / f'{i}.jpg')
            with Image.open(path) as frame:
                frame = ImageOps.exif_transpose(frame).convert('RGB')
                thumb = ImageOps.contain(frame, (316, 238))
                x, y = (i % columns) * 320, (i // columns) * 264
                sheet.paste(thumb, (x + (320 - thumb.width) // 2, y))
                draw.text((x + 8, y + 244), f'{at / 1000:.2f}s', fill='white')
    sheet.save(dest, format='JPEG', quality=85)
