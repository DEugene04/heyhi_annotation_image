"""
Reads a cleaned image with Mathpix and returns its raw response.

This layer only talks to Mathpix and hands back what it says. Turning that raw
response into our own structure is the IR builder's job, so if we ever change
OCR provider, this file and the IR builder are the only things that change.

Mathpix is asked for line-by-line and word-by-word detail, because the IR needs
to know where each line sits on the page, not just the text.
"""

import base64
import io

import httpx
from PIL import Image

from config import settings


class MathpixError(Exception):
    """Raised when Mathpix cannot read the image or the request fails."""


def _compress(image_bytes: bytes) -> bytes:
    """
    Turn any image into a JPEG below the configured size, so requests come back
    faster and the format is one Mathpix reads.

    A JPEG that is already small enough is returned unchanged. Anything else
    (a larger photo, or another format like PNG or WebP) is re-saved as a JPEG,
    shrinking the quality step by step until it fits.
    """

    image = Image.open(io.BytesIO(image_bytes))
    if image.format == "JPEG" and len(image_bytes) <= settings.max_image_bytes:
        return image_bytes

    image = image.convert("RGB")
    for quality in (85, 70, 55, 40):
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        data = buffer.getvalue()
        if len(data) <= settings.max_image_bytes:
            return data

    # Still too big: scale the image down and try once more.
    image.thumbnail((1600, 1600))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=60)
    return buffer.getvalue()


def _request(encoded: str, options: dict) -> dict:
    """Send one request to Mathpix for an already-encoded image and return JSON."""

    if not settings.mathpix_app_id or not settings.mathpix_app_key:
        raise MathpixError("Mathpix credentials are not set (see .env).")

    payload = {"src": f"data:image/jpeg;base64,{encoded}", **options}
    headers = {
        "app_id": settings.mathpix_app_id,
        "app_key": settings.mathpix_app_key,
        "Content-type": "application/json",
    }

    try:
        response = httpx.post(
            settings.mathpix_endpoint, json=payload, headers=headers, timeout=60.0
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise MathpixError(f"Mathpix request failed: {exc}") from exc

    result = response.json()
    if result.get("error") or result.get("error_info"):
        detail = result.get("error_info") or result.get("error")
        raise MathpixError(f"Mathpix could not read the image: {detail}")
    return result


def read_image(image_bytes: bytes) -> dict:
    """Read an image and return the line-by-line response (line_data)."""

    encoded = base64.b64encode(_compress(image_bytes)).decode("ascii")
    return _request(encoded, {"include_line_data": True})


def read_all(image_bytes: bytes) -> dict:
    """
    Read an image both ways and return both results together.

    Mathpix will not return line data and word data in one call, so this makes
    two: the line reading gives the page's structure, the word reading gives the
    precise shape of each word. The image is only compressed and encoded once.

    Returns {"line": <line response>, "word": <list of word entries>}.
    (The two calls run one after another for now; they could be run at the same
    time later to save time — see the latency work in WS5.)
    """

    encoded = base64.b64encode(_compress(image_bytes)).decode("ascii")
    line = _request(encoded, {"include_line_data": True})
    word = _request(encoded, {"include_word_data": True})
    return {"line": line, "word": word.get("word_data", []) or []}
