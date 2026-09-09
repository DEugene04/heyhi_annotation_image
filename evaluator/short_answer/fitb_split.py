"""
Split one flat OCR blob into per-blank student answers for multi-blank FITB.

Context
-------
A student uploads a photo of a fill-in-the-blank answer. The reading pipeline
OCRs the whole page into a single flat string (`fused.flat_text`) -- every blank
mashed together, with no labels telling us where blank 1's answer ends and blank
2's begins. The marking chain, however, wants one answer *per blank* (a
`List[str]`, in blank order) so it can grade each blank against its own model
answer. This module is the missing step between the two.

It is the flat-text analogue of the smartjen PDF flow's `get_pdf_answers`: that
one reads a scanned worksheet region-by-region using the `<ans>` tags to keep
blanks separate; we don't have separate regions -- just one blob -- so we ask an
LLM to segment it, guided by the `(blank N)` position markers recovered from the
question's answer spans.

Design notes
------------
- **Verbatim.** Each returned answer is an exact slice of the OCR text, never
  reworded or re-spelled. The image highlighter positions a highlight by matching
  its text back into `flat_text`, so a paraphrased answer would fail to anchor.
- **No silent truncation.** If the model returns anything other than exactly
  `num_blanks` answers we raise `BlankSplitError` rather than zipping a short
  list and dropping blanks (the root bug in the reference implementation). The
  caller should refuse the page -- ask for a retake -- instead of marking a
  misaligned set of answers.
- **Single-blank is a no-op.** With one blank the whole blob *is* the answer, so
  we return it directly without an LLM call. This keeps the existing, working
  single-blank path byte-for-byte unchanged.

This module is purely additive: it imports nothing from the marking chain and
mutates no shared state, so it can be removed by deleting this file.
"""

import json
from typing import Awaitable, Callable, List, Optional

from bs4 import BeautifulSoup

import config
from evaluator.dynamic_llm_call import DynamicAsyncOpenAI

# A small, cheap text model is plenty for segmentation; this mirrors the marking
# chain's choice for simple tasks. Override per call via `model=`.
DEFAULT_SPLIT_MODEL = "gpt-4.1-mini"

# The class the question's answer spans carry. Mirrors
# QuestionExtractor._replace_ans_tags in logic.py -- kept here so this module
# stays self-contained (single source of truth for the two should be unified if
# the markup convention ever changes).
_ANS_SPAN_CLASS = "ans-div"


class BlankSplitError(ValueError):
    """The flat OCR text could not be split into exactly the expected number of
    blanks. Signals the caller to refuse the page (retake), never to mark a
    partial/misaligned set of answers."""


def extract_blanks(question_html: str) -> tuple[str, list[str]]:
    """Parse the answer spans once, returning both the marked question and answers.

    A single source of truth for the `ans-div` convention: it returns

      - the question with each answer span replaced by an ordered ``(blank N)``
        marker (document order, 1-based) and the span's own content discarded, so
        the correct answer never leaks into the text shown to the model; and
      - the text that was inside each span, in blank order -- which, on a payload
        that stores the answer inside the span, IS that blank's correct answer.

    `count_blanks` and `mark_blanks` are thin views over this.
    """
    soup = BeautifulSoup(question_html, "html.parser")
    spans = soup.find_all("span", class_=_ANS_SPAN_CLASS)
    answers = [span.get_text(strip=True) for span in spans]
    for idx, span in enumerate(spans, start=1):
        span.replace_with(f"(blank {idx})")
    return str(soup), answers


def count_blanks(question_html: str) -> int:
    """Number of answer blanks in the question = count of answer spans.

    This is the most reliable blank count: it counts the blanks the student
    actually sees on the page. The caller should cross-check it against the
    `blank N :` lines in the correct answer and flag a mismatch.
    """
    return len(extract_blanks(question_html)[1])


def mark_blanks(question_html: str) -> str:
    """Replace each answer span with an ordered ``(blank N)`` marker.

    Numbering follows document order (1-based), matching how the correct answer's
    `blank N :` lines are numbered, so the model sees exactly where each blank
    sits in the sentence. The span's own content (which is the *answer* on a
    worksheet) is discarded -- we only want the position.
    """
    return extract_blanks(question_html)[0]


_SYSTEM_PROMPT = """You are given a fill-in-the-blank question and a student's answer.

The question contains numbered blanks written as (blank 1), (blank 2), and so on, \
in order. The student's answer has been transcribed from a photo into ONE block of \
text, with every blank's answer run together and no labels separating them.

Your only job is to separate that text into what the student wrote for each blank.

Rules:
1. Return each blank's answer EXACTLY as it appears in the transcription. Do not \
fix spelling, capitalisation, punctuation, or spacing, and do not reword or \
translate anything. Copy the student's characters verbatim.
2. Return exactly one entry per blank, in blank-number order (blank 1 first).
3. If the student clearly left a blank unanswered, return an empty string "" for \
that blank. Do not invent text for it.
4. Do not add commentary. Output only the JSON object described below.

Output JSON in exactly this shape:
{"answers": ["<blank 1 text>", "<blank 2 text>", ...]}"""


async def _default_complete(messages: List[dict], api_key: str, model: str) -> str:
    """Real LLM call. Returns the raw model output (a JSON string)."""
    client = DynamicAsyncOpenAI(api_key=api_key)
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )
    return response.output_text


async def split_student_blanks(
    question_html: str,
    flat_ocr_text: str,
    num_blanks: int,
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    _complete: Optional[Callable[[List[dict], str, str], Awaitable[str]]] = None,
) -> List[str]:
    """Split `flat_ocr_text` into the student's answer for each blank.

    Args:
        question_html: The question as delivered on the request -- HTML (or the
            worksheet JSON-segments string) carrying the `<span class="ans-div">`
            answer spans. Used to place the `(blank N)` markers in the prompt.
        flat_ocr_text: The student's writing, OCR'd to a single string
            (`fused.flat_text`).
        num_blanks: How many blanks to split into. The caller derives this from
            the correct answer / `count_blanks` and is responsible for the
            cross-check; here it is the exact length the result must have.
        api_key: OpenAI key. Defaults to the environment, like `evaluate()`.
        model: Split model. Defaults to `DEFAULT_SPLIT_MODEL`.
        _complete: Test seam -- an async `(messages, api_key, model) -> json_str`
            used instead of the real client. Not for production callers.

    Returns:
        A list of exactly `num_blanks` strings, in blank order, each a verbatim
        slice of `flat_ocr_text` (empty string for an unanswered blank).

    Raises:
        ValueError: `num_blanks` is not a positive integer (a caller bug).
        BlankSplitError: the model's output is unparseable or not exactly
            `num_blanks` answers -- the page should be refused, not marked.
    """
    if num_blanks < 1:
        raise ValueError(f"num_blanks must be >= 1, got {num_blanks}")

    # Single blank: the whole blob is the one answer. No LLM call, and the
    # existing single-blank path is left exactly as it was.
    if num_blanks == 1:
        return [flat_ocr_text]

    question_with_markers = mark_blanks(question_html)

    user_prompt = (
        f"Question (blanks marked (blank 1), (blank 2), ...):\n{question_with_markers}\n\n"
        f"Student's transcribed answer (one block of text):\n{flat_ocr_text}\n\n"
        f"There are {num_blanks} blanks. Split the answer into exactly "
        f"{num_blanks} entries, in blank order."
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": _SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
    ]

    complete = _complete or _default_complete
    resolved_key = (
        api_key
        or config.settings.openai_api_key
        or config.OPENAI_API_KEY_DICT["AI_AUTOMARKING"]
    )
    resolved_model = model or DEFAULT_SPLIT_MODEL

    raw = await complete(messages, resolved_key, resolved_model)

    try:
        parsed = json.loads(raw)
        answers = parsed["answers"]
    except (json.JSONDecodeError, TypeError, KeyError) as exc:
        raise BlankSplitError(
            f"Could not parse an 'answers' list from the split model output: {raw!r}"
        ) from exc

    if not isinstance(answers, list) or not all(isinstance(a, str) for a in answers):
        raise BlankSplitError(
            f"Split model returned a non-list-of-strings 'answers': {answers!r}"
        )

    if len(answers) != num_blanks:
        raise BlankSplitError(
            f"Split produced {len(answers)} answers but the question has "
            f"{num_blanks} blanks; refusing rather than dropping blanks."
        )

    return answers
