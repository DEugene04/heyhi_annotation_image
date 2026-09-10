"""
Loads the project's settings from the environment.

The actual secret values live in a .env file that is never committed. This
module only reads them and gives the rest of the code a single, typed place to
ask for them.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Every setting the project needs, read from the environment / .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Mathpix credentials, used by the OCR layer to read handwriting.
    mathpix_app_id: str = ""
    mathpix_app_key: str = ""
    mathpix_endpoint: str = "https://api.mathpix.com/v3/text"

    # Images are re-compressed below this size before sending, following
    # Mathpix's own guidance that smaller images come back faster.
    max_image_bytes: int = 100_000

    # If the reading is less sure than this (area-weighted across the page), we
    # tell the student to retake the photo instead of annotating a bad reading.
    confidence_threshold: float = 0.6

    # VLM used to re-read the text (experiment: VLM text + Mathpix geometry).
    # Override the model per experiment via .env: OPENAI_MODEL=<model-id>.
    openai_api_key: str = ""
    openai_model: str = "gpt-5.1"

    # EXPERIMENT toggle: feed Mathpix's literal reading to the VLM as a spelling
    # hint (see ocr/vlm.py). Set VLM_USE_MATHPIX_HINT=false in .env to disable, so
    # the model / hint can be A/B'd without code changes.
    vlm_use_mathpix_hint: bool = True

    # FLC (first-level-checking) spelling/grammar service. Base host only; the
    # client appends the /global/first-level-checking/v2 path. Overridable via
    # .env (FLC_BASE_URL) since the deploy host may differ (staging vs prod).
    flc_base_url: str = (
        "https://n47xec5kukaax54bihdh4bc5pq0zgoso.lambda-url.ap-southeast-1.on.aws"
    )
    # Above this many seconds we give up on FLC and mark without its findings.
    flc_timeout_seconds: float = 90.0

    # EXPERIMENT toggle: draw each feedback box tight to the character span it
    # points at (e.g. FLC's single wrong word/phrase) instead of boxing the whole
    # line. Set TIGHT_FEEDBACK_BOXES=false in .env to revert to whole-line boxes.
    tight_feedback_boxes: bool = True


settings = Settings()

import os

_key = os.environ.get("OPENAI_API_KEY")

OPENAI_API_KEY_DICT = {
    "AI_AUTOMARKING": os.environ.get("OPENAI_API_KEY_AI_AUTOMARKING", _key),
    "GLOBAL": _key,
}

