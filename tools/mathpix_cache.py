"""
Gets Mathpix readings for a corpus image, reading a saved copy when there is one
and calling the live service otherwise.

Saved readings live in corpus/mathpix_cache/, named after the image, and hold
both the line reading and the word reading together. They let the tools run
without hitting Mathpix every time (and without a key at all, once saved), and
avoid paying to read the same image twice.
"""

import json
from pathlib import Path

from ocr.mathpix import read_all

CACHE_DIR = Path("corpus/mathpix_cache")


def get_readings(image_path: Path) -> dict:
    """Return {"line": <line response>, "word": <word entries>} for an image."""

    cache_file = CACHE_DIR / f"{image_path.stem}.json"
    if cache_file.exists():
        cached = json.loads(cache_file.read_text())
        if "line" in cached and "word" in cached:
            return cached

    readings = read_all(image_path.read_bytes())
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(readings, indent=2))
    return readings


def get_response(image_path: Path) -> dict:
    """Return just the line reading, for tools that only need page structure."""

    return get_readings(image_path)["line"]
