"""
Re-reads the page text with a vision language model (GPT-5).

Mathpix is very good at telling us *where* each line sits, but on messy or unusual
handwriting it misreads letters. A VLM reads far more accurately because it uses
language understanding — but it does not give reliable positions. So we use the
VLM only for the text, and keep Mathpix's shapes (see ocr/fusion.py).

It transcribes everything on the page — printed and handwritten — one line per
physical line, so the reading lines up with Mathpix's per-line shapes. Deciding
what to ignore (for example a printed question) is the evaluator's job, not this
layer's. This returns the transcription as a list of lines, in reading order.
"""

import base64

from config import settings


class VLMError(Exception):
    """Raised when the VLM cannot be called (e.g. no API key)."""


_PROMPT = (
    "You are transcribing a student's handwriting for a SPELLING AND GRAMMAR "
    "assessment. A teacher needs to see the student's exact spelling, including "
    "every mistake. Your job is to copy the letters that are actually written, "
    "NOT to read for meaning. "
    "Transcribe every piece of text in this image exactly as it appears — "
    "printed and handwritten alike, including any question, instructions, or "
    "numbering already on the page. Do not decide what is relevant: transcribe "
    "all of it. "
    "Copy the writing letter for letter. If a word is misspelled, or is not a "
    "real word, keep it exactly as written — do NOT fix it. For example, if the "
    "page says 'laazy' write 'laazy' (not 'lazy'); if it says 'outsde' write "
    "'outsde' (not 'outside'). Never correct spelling, grammar, or punctuation, "
    "and never translate, summarise, or add anything of your own. When you are "
    "tempted to write the correct word, write the wrong one that is actually on "
    "the page instead. "
    "Output one line for each physical line on the page, breaking where the text "
    "breaks on the page — even when a sentence continues onto the next line. "
    "Use LaTeX for any mathematics. "
    "Output only the transcription — no commentary, no code fences."
)


def transcribe_lines(image_bytes: bytes) -> list[str]:
    """Read the handwriting with GPT-5 and return one string per written line."""

    if not settings.openai_api_key:
        raise VLMError("OPENAI_API_KEY is not set (see .env).")

    from openai import OpenAI  # imported here so the module loads without the SDK

    client = OpenAI(api_key=settings.openai_api_key)
    encoded = base64.b64encode(image_bytes).decode("ascii")

    response = client.responses.create(
        model=settings.openai_model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": _PROMPT},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{encoded}",
                    },
                ],
            }
        ],
    )

    return [line for line in response.output_text.splitlines() if line.strip()]
