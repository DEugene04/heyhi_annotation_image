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
    openai_api_key: str = ""
    openai_model: str = "gpt-5.6"


settings = Settings()
