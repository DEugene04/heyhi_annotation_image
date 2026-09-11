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


# The instruction is enforced twice: the full rules and worked examples live in
# the system prompt (_SYSTEM_PROMPT), and a condensed restatement rides with the
# image in the user prompt (_USER_PROMPT). The autocorrect prior is strong, so the
# examples below SHOW the required behaviour (keep the wrong spelling) rather than
# only describing it -- few-shot demonstration is more effective than instruction.
_SYSTEM_PROMPT = (
    "You are a transcription engine for a SPELLING AND GRAMMAR assessment. Your "
    "ONLY job is to copy the exact letters a student wrote, mistakes and all. A "
    "teacher needs to see every error, so you must NEVER correct, complete, "
    "translate, summarise, or improve anything.\n\n"
    "Rules:\n"
    "- Copy the writing letter for letter. Keep every misspelling, wrong tense, "
    "missing or extra letter, and bad punctuation exactly as written.\n"
    "- You are copying letters, NOT reading for meaning. When a word is unclear, "
    "transcribe the letters that are actually on the page, not the word you expect.\n"
    "- Transcribe every piece of text on the page — printed and handwritten alike, "
    "including any question, instructions, or numbering. Do not decide what is "
    "relevant; that is the marker's job, not yours.\n"
    "- Output one line for each physical line on the page, breaking where the text "
    "breaks — even when a sentence continues onto the next line.\n"
    "- Use LaTeX for any mathematics.\n"
    "- Output only the transcription: no commentary, no code fences, no fixes.\n\n"
    "Examples of the exact behaviour required — keep the wrong spelling, never fix it:\n\n"
    "  Written on the page:  The cat is verry laazy and sat outsde.\n"
    "  CORRECT output:       The cat is verry laazy and sat outsde.\n"
    "  WRONG output:         The cat is very lazy and sat outside.\n\n"
    "  Written on the page:  She dont no the anser to the qeustion.\n"
    "  CORRECT output:       She dont no the anser to the qeustion.\n"
    "  WRONG output:         She doesn't know the answer to the question.\n\n"
    "  Written on the page:  I recieved my award yesterday.\n"
    "  CORRECT output:       I recieved my award yesterday.\n"
    "  WRONG output:         I received my award yesterday.\n\n"
    "Whenever you feel tempted to write the correct word, write the wrong one that "
    "is actually on the page instead."
)

# Rides with the image. Restates the one rule that matters most, so the constraint
# is present both before and alongside the task (the 2x enforcement).
_USER_PROMPT = (
    "Transcribe this page exactly as written, for a spelling assessment. Copy "
    "every letter, including all misspellings and mistakes — do NOT correct, "
    "complete, or translate anything. One line per physical line on the page. "
    "Output only the transcription."
)


# EXPERIMENT (Case 1: Mathpix-as-hint). Appended to the user prompt when a literal
# OCR reading is supplied, to counter the model's autocorrect prior with evidence
# of the exact letters written -- without letting the (often misread) OCR override
# what the image actually shows.
_HINT_TEMPLATE = (
    "\n\nTo help with difficult handwriting, here is a literal character-level OCR "
    "of the same page. It preserves the exact spelling that was written (including "
    "misspellings) but it sometimes misreads letters:\n---\n{hint}\n---\n"
    "Use it ONLY as a hint about spelling: where the image supports the OCR's exact "
    "spelling, keep that spelling (including any misspelling). But trust the image "
    "over the OCR whenever they conflict on which letters are actually written."
)


def transcribe_lines(image_bytes: bytes, mathpix_hint: str = "") -> list[str]:
    """Read the handwriting with GPT-5 and return one string per written line.

    `mathpix_hint` is the literal OCR reading of the same page; when given, it is
    appended to the user prompt as a spelling hint (see _HINT_TEMPLATE).
    """

    if not settings.openai_api_key:
        raise VLMError("OPENAI_API_KEY is not set (see .env).")

    from openai import OpenAI  # imported here so the module loads without the SDK

    client = OpenAI(api_key=settings.openai_api_key)
    encoded = base64.b64encode(image_bytes).decode("ascii")

    user_prompt = _USER_PROMPT
    if mathpix_hint.strip():
        user_prompt += _HINT_TEMPLATE.format(hint=mathpix_hint)

    # The full rules + few-shot examples go in the system prompt; the user message
    # carries the image and a condensed restatement (enforced twice on purpose).
    response = client.responses.create(
        model=settings.openai_model,
        instructions=_SYSTEM_PROMPT,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{encoded}",
                    },
                ],
            }
        ],
    )

    return [line for line in response.output_text.splitlines() if line.strip()]
