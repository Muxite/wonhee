from pathlib import Path

import imagehash
from PIL import Image


def phash_hex(image_path: Path) -> str:
    with Image.open(image_path) as im:
        return str(imagehash.phash(im))
