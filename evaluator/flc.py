"""
Thin client for the FLC (first-level-checking) spelling/grammar service.

FLC is a deployed HTTP service in another repo -- we never import its code, we
call it over the network (like ocr/vlm.py calls the VLM). It reads a student
composition and returns per-issue findings, each an offset-anchored span with a
category ("spelling"/"grammar") and a suggestion.

We depend ONLY on the `checking[]` field of the response (the stable contract);
`diagnostics` and `checker_samples` are theirs / test-only and ignored. The
`X-Is-Unit-Testing` header is never sent -- it bypasses auth and bills the
unit-testing account.

The service is slow (~40s) and remote, so `check()` degrades gracefully: any
failure (network, timeout, unexpected shape) returns [] rather than raising, so
a marking request is never brought down by the spell-checker being unavailable.
"""

import logging
from typing import List, Optional, TypedDict

import httpx

from config import settings

logger = logging.getLogger(__name__)

# Appended to the configured base host to form the full endpoint.
_FLC_PATH = "/global/first-level-checking/v2"


class FlcFinding(TypedDict):
    """One issue FLC reports. `start`/`end` are character offsets into the exact
    text we send (so they map straight onto flat_text)."""

    target: str
    suggestion: str
    category: str  # "spelling" or "grammar"
    start: int
    end: int


async def check(
    student_composition: str,
    question_statement: str,
    *,
    language: str = "English",
    base_url: Optional[str] = None,
    timeout: Optional[float] = None,
) -> List[FlcFinding]:
    """Return FLC's findings for one composition, or [] if the service can't be
    reached or answers unexpectedly (never raises)."""

    base = (base_url or settings.flc_base_url).rstrip("/")
    if not base:
        logger.warning("FLC base URL is not configured; skipping spelling/grammar check.")
        return []
    url = base if base.endswith(_FLC_PATH) else base + _FLC_PATH

    payload = {
        "student_composition": student_composition,
        "question_statement": question_statement,
        "language": language,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout or settings.flc_timeout_seconds) as client:
            response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        # Network error, timeout, non-2xx, or non-JSON body: degrade to no findings.
        # Log the exception TYPE and repr -- some (e.g. timeouts) have an empty str,
        # which would otherwise hide the real reason.
        logger.warning(
            "FLC call to %s failed (%s: %r); marking without spelling/grammar findings.",
            url,
            type(exc).__name__,
            exc,
        )
        return []

    checking = data.get("checking")
    if not isinstance(checking, list):
        logger.warning("FLC response had no 'checking' list; got keys %s.", list(data))
        return []
    return checking
