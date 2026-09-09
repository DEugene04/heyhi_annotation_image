"""
Multi-blank FITB orchestration for the live evaluator.

`evaluate()` marks one student answer against one answer key. A fill-in-the-blank
question with several blanks is just that operation run once per blank, then the
results stitched together -- which is all this module does:

    detect it's multi-blank FITB  ->  split the OCR blob into per-blank answers
    ->  mark each blank with the ordinary evaluate() path  ->  aggregate.

The pieces it leans on already exist and are trusted:
  - `extract_blanks` (fitb_split.py) parses the answer spans: it rewrites each
    `<ans>` span to a `(blank N)` marker (dropping the answer out of the question
    text so nothing leaks into the split prompt) AND returns the text that was
    inside each span -- which, at this project's entry point, IS that blank's
    correct answer. The blanks are the single source of truth for count, order,
    and (when no `correct answer:` segment is present) the answer key itself.
  - `split_student_blanks` turns the one OCR blob into one verbatim answer per
    blank (this project's new piece; see fitb_split.py).
  - `evaluate()` itself marks each blank -- the same single-answer path that
    already works for one blank today, reused unchanged.

The correct answer can arrive three ways, all inside the `question` params; they
are tried in order of richness: an explicit `answer_key` with `blank N :` markers,
then a `correct answer: blank N : ...` segment (which can carry `/`-alternatives),
then the text inside the `<ans>` spans. This deliberately does NOT go through the
smartjen `extract_question_obj`, which requires `correct answer:` and `Solution:`
segments that this project's entry-point payload need not have.

Scoring is `per_blank`: each blank is marked out of its own per-blank mark and the
marks are summed (the user's model: `mark` is per-blank, the question total is
their sum). The senior's two-stage holistic combine (a second LLM pass that
re-marks across blanks with whole-question rubric awareness) is a deliberate
future swap -- it would replace `_aggregate` only, which is why aggregation is
isolated in one function.

Purely additive: `evaluate()` delegates here only for genuine multi-blank FITB
(>= 2 blanks); single-blank and non-FITB take their existing path untouched.
"""

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Awaitable, Callable, List, Optional

from evaluator.short_answer.fitb_split import (
    _ANS_SPAN_CLASS,
    extract_blanks,
    split_student_blanks,
)

# Recognises a "blank N :" marker anywhere in a string.
_BLANK_MARKER = re.compile(r"blank\s*(\d+)\s*:", re.IGNORECASE)

# One blank's answer: "blank N : <value up to the next blank marker or end>".
# DOTALL so a value spanning newlines (blanks separated by <br/> -> \n) is kept
# whole -- the fragility that breaks logic.py's _convert_fitb_answer.
_BLANK_SEGMENT = re.compile(
    r"blank\s*(\d+)\s*:\s*(.*?)(?=blank\s*\d+\s*:|$)", re.IGNORECASE | re.DOTALL
)

# Segment labels that mark the end of the correct-answer block in a payload.
_ANSWER_END_LABELS = ("mark:", "mark :", "solution:", "solution :", "difficulty")


@dataclass
class FitbSpec:
    """Everything the orchestrator needs, parsed from the request once.

    - question: question-facing HTML with `(blank N)` markers and NO answers.
    - per_blank_answers: the answer key for each blank, in blank order, with the
      `blank N :` prefixes stripped and alternatives kept `/`-separated.
    - num_blanks: how many blanks (the length the split must have).
    - blank_full_mark: the per-blank maximum mark.
    """

    question: str
    per_blank_answers: List[str]
    num_blanks: int
    blank_full_mark: float


def _blank_answers(raw_correct: str) -> List[str]:
    """Parse a raw correct answer into one answer-key string per blank.

    Handles both shapes the payload can carry: repeated-prefix alternatives
    ("blank 1 : Water / blank 1 : H2O") and single-prefix alternatives
    ("blank 1 : Water / H2O"), and blanks separated by spaces OR newlines. All of
    a blank's alternatives are collected and re-joined with " / " so the ordinary
    marking prompt still reads them as alternatives. Returns the answers ordered
    by blank number.
    """

    by_number: dict = {}
    for match in _BLANK_SEGMENT.finditer(raw_correct):
        number = int(match.group(1))
        value = match.group(2).strip().strip("/").strip()
        if not value:
            continue
        by_number.setdefault(number, []).append(value)

    return [" / ".join(by_number[n]) for n in sorted(by_number)]


def _isolated_question(question: str, per_blank_answers: List[str], target_index: int) -> str:
    """Build a single-gap version of the question for marking one blank.

    Every blank EXCEPT the target is filled with its own (first) correct answer,
    and only the target blank is left as its `(blank N)` gap. This is what stops
    the single-answer marker from seeing several gaps, one student answer, and
    docking the blank as an "incomplete" whole-question answer: with one gap it
    judges just that blank against just that blank's key.
    """

    result = question
    for index, answer in enumerate(per_blank_answers):
        if index == target_index:
            continue  # keep this blank's gap -- it's the one being marked
        blank_no = index + 1
        first_alt = answer.split(" / ")[0].strip() if answer else ""
        if first_alt:
            result = re.sub(
                rf"\(blank\s*{blank_no}\)", first_alt, result, flags=re.IGNORECASE
            )
    return result


def _raw_correct_from_segments(segments: List[dict]) -> str:
    """Pull the raw correct-answer text out of the request's segments.

    Collects the segment(s) after a "correct answer:" label up to the next block
    label ("mark:", "Solution:", ...). Used instead of the payload's
    already-canonicalised correct answer because that canonicalisation
    (_convert_fitb_answer) drops blanks separated by newlines.
    """

    collecting = False
    parts: List[str] = []
    for seg in segments:
        text = seg.get("text", "") if isinstance(seg, dict) else ""
        stripped = text.strip().lower()
        if stripped.startswith("correct answer"):
            collecting = True
            after = text.split(":", 1)[1] if ":" in text else ""
            if after.strip():
                parts.append(after)
            continue
        if collecting:
            if any(stripped.startswith(label) for label in _ANSWER_END_LABELS):
                break
            parts.append(text)
    return "\n".join(parts)


def _parse_blank_mark(texts: List[str], default_full_mark: float, num_blanks: int) -> float:
    """Read the per-blank mark from a "mark: N" segment.

    The user's model: `mark` is the PER-BLANK maximum (the question total is their
    sum). When no usable mark segment is present, fall back to an even split of the
    caller's full_mark across the blanks.
    """

    for text in texts:
        stripped = text.strip().lower()
        if stripped.startswith("mark:") or stripped.startswith("mark :"):
            after = text.split(":", 1)[1].strip() if ":" in text else ""
            try:
                mark = float(after)
                if mark > 0:
                    return mark
            except ValueError:
                pass
    return default_full_mark / num_blanks if num_blanks else default_full_mark


async def maybe_parse_fitb(
    question: str,
    answer_key: Optional[str],
    default_full_mark: float,
) -> Optional[FitbSpec]:
    """Parse `question` into a `FitbSpec`, or return None if it isn't FITB.

    Returns None (rather than raising) for anything that isn't a FITB payload --
    plain-text questions, essays, malformed JSON -- so non-FITB callers fall
    straight through to their existing path.

    Everything is derived from the `<ans>` answer spans (count, `(blank N)`
    markers, and their order). The correct answer is taken, in order of richness,
    from: an `answer_key` carrying `blank N :` markers, else a `correct answer:`
    segment, else the text inside the spans themselves. No dependency on the
    smartjen `extract_question_obj` (which needs `correct answer:` and `Solution:`
    segments this project's entry-point payload need not have).
    """

    try:
        segments = json.loads(question)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(segments, list) or not segments:
        return None

    texts = [seg.get("text", "") for seg in segments if isinstance(seg, dict)]

    # The question-facing HTML is the segment(s) that carry the answer spans.
    # Restricting to these keeps the correct-answer / solution / audio-script
    # segments out of the text shown to the splitter and marker: no answer leak.
    span_html = "\n".join(text for text in texts if _ANS_SPAN_CLASS in text)
    marked_question, span_answers = extract_blanks(span_html)
    num_spans = len(span_answers)

    # FITB either by an explicit "question type: ... FITB ..." label or by the
    # mere presence of answer spans -- so a leaner entry-point payload (spans but
    # no type label) is still recognised.
    typed_fitb = any(
        text.strip().lower().startswith("question type") and "fitb" in text.lower()
        for text in texts
    )
    if num_spans == 0 and not typed_fitb:
        return None

    # Correct answer, in order of richness (all live in the question params).
    if answer_key and _BLANK_MARKER.search(answer_key):
        per_blank_answers = _blank_answers(answer_key)
    else:
        segment_correct = _raw_correct_from_segments(segments)
        if _BLANK_MARKER.search(segment_correct):
            per_blank_answers = _blank_answers(segment_correct)
        else:
            per_blank_answers = list(span_answers)

    # The visible spans are authoritative for the count; fall back to the answer
    # count only when a typed payload somehow carries no spans.
    num_blanks = num_spans if num_spans else len(per_blank_answers)
    if num_blanks < 1:
        return None

    # Keep the answer key aligned to the blanks. If the chosen source disagrees
    # with the visible span count, prefer the spans when they carry the answers;
    # otherwise the data is inconsistent -- refuse rather than mark misaligned.
    if len(per_blank_answers) != num_blanks:
        if len(span_answers) == num_blanks and any(span_answers):
            per_blank_answers = list(span_answers)
        else:
            return None

    return FitbSpec(
        question=marked_question,
        per_blank_answers=per_blank_answers,
        num_blanks=num_blanks,
        blank_full_mark=_parse_blank_mark(texts, default_full_mark, num_blanks),
    )


def _aggregate(results: List[dict]) -> dict:
    """Combine per-blank evaluate() results into one evaluate()-shaped result.

    per_blank scoring: sum the marks, concatenate the highlights in blank order
    (each target is a verbatim slice of the reading, so the adapter positions them
    by walking a cursor forward), and label each blank's remarks/reason. Swapping
    in the holistic two-stage combine would mean replacing this function only.
    """

    total_mark = sum(r["mark"] for r in results)
    highlights = [h for r in results for h in r.get("mark_highlights", [])]

    def labelled(key: str) -> str:
        return "\n\n".join(
            f"Blank {i + 1}: {r.get(key, '')}" for i, r in enumerate(results)
        )

    models = sorted({m for r in results for m in r.get("models_used", [])})

    aggregated = {
        "mark": total_mark,
        "reason": labelled("reason"),
        "remarks": labelled("remarks"),
        "remarks_detailed": labelled("remarks_detailed"),
        "mark_highlights": highlights,
        "total_tokens": sum(r.get("total_tokens", 0) for r in results),
        "total_cost": sum(r.get("total_cost", 0) for r in results),
        "models_used": models,
    }

    # Surface per-blank debug prompts as lists when they are present.
    if any("messages_automarking" in r for r in results):
        aggregated["messages_automarking"] = [
            r.get("messages_automarking") for r in results
        ]
        aggregated["messages_breakdown"] = [
            r.get("messages_breakdown") for r in results
        ]

    return aggregated


async def evaluate_fitb(
    spec: FitbSpec,
    student_answer: str,
    *,
    subject: str = "",
    language: str = "English",
    marking_note: str = "",
    rubric: str = "",
    api_key: Optional[str] = None,
    debug_mode: bool = False,
    _split: Optional[Callable[..., Awaitable[List[str]]]] = None,
    _mark_blank: Optional[Callable[..., Awaitable[dict]]] = None,
) -> dict:
    """Mark a multi-blank FITB answer: split the blob, mark each blank, aggregate.

    Args:
        spec: The parsed question (from `maybe_parse_fitb`).
        student_answer: The student's writing as one OCR blob (`fused.flat_text`).
        subject / language / marking_note / rubric / api_key / debug_mode: passed
            straight through to each per-blank marking call.
        _split / _mark_blank: test seams. `_split` defaults to
            `split_student_blanks`; `_mark_blank` defaults to the real `evaluate()`
            (imported lazily to avoid an import cycle).

    Returns:
        An `evaluate()`-shaped dict (mark, reason, remarks, remarks_detailed,
        mark_highlights, usage totals), with the marks summed across blanks.

    Raises:
        BlankSplitError: the blob could not be split into exactly `num_blanks`
            answers (propagated from the splitter) -- the page should be refused.
    """

    split = _split or split_student_blanks
    mark_blank = _mark_blank
    if mark_blank is None:
        # Lazy import: evaluate.py imports this module, so importing it at module
        # load time would be circular.
        from evaluator.short_answer.evaluate import evaluate as mark_blank

    pieces = await split(
        spec.question, student_answer, spec.num_blanks, api_key=api_key
    )

    async def mark_one(index: int) -> dict:
        return await mark_blank(
            _isolated_question(spec.question, spec.per_blank_answers, index),
            pieces[index],
            spec.per_blank_answers[index],
            full_mark=spec.blank_full_mark,
            subject=subject,
            language=language,
            marking_note=marking_note,
            rubric=rubric,
            api_key=api_key,
            debug_mode=debug_mode,
        )

    results = await asyncio.gather(*[mark_one(i) for i in range(spec.num_blanks)])
    return _aggregate(list(results))
