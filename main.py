"""
The backend service entrypoint.

It exposes two routes:

  - /health   : a simple check that the service is up.
  - /annotate : the pipeline route. It takes an uploaded photo and returns the
                annotation payload the frontend draws.

The pipeline is strictly serial: read the page (Mathpix shapes + GPT-5 text,
fused), package it into the IR, judge it, turn feedback into shapes, hand it to
the frontend. The judging step (the evaluator) marks the reading against the
question and an optional answer key, and its verbatim-text spans are located
back into the reading before the resolver draws them. A page the two readers
disagree on, or one read too unclearly to trust, is not annotated: the student
is asked for a clearer photo instead.
"""

import asyncio
import json
import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Form, HTTPException, UploadFile

from contracts.schema import AnnotationPayload
from evaluator import flc
from evaluator.essay.adapter import score_detail_to_scorecard
from evaluator.essay.evaluate import evaluate_essay
from evaluator.flc_adapter import to_feedback as flc_to_feedback
from evaluator.short_answer.adapter import to_feedback
from evaluator.short_answer.evaluate import evaluate
from geometry.resolver import resolve_payload
from ocr.fusion import RETAKE_WARNING, AlignmentError, fuse
from ocr.ir_builder import build_ir, is_readable
from ocr.vlm import transcribe_lines

app = FastAPI(title="Handwritten Answer Annotation")

# Uvicorn configures its own named loggers but leaves the root logger at WARNING
# with no handler, so a plain INFO log here would be dropped. Send INFO logs to
# the console (stderr) here; each /annotate call additionally attaches a fresh
# timestamped file under logs/ (see _annotate_log_file) so every run gets its own
# log instead of all runs piling into a single evaluator.log.
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@contextmanager
def _annotate_log_file(question_type: str):
    """Attach a fresh timestamped log file for the duration of one /annotate run.

    Runs are separated by evaluator so they are easy to find later: each writes to
    logs/<essay|short_answer>/<timestamp>.log. The handler is added to the root
    logger so every INFO log emitted during the request is captured, and removed
    afterwards.
    """
    subdir = "essay" if question_type == "essay" else "short_answer"
    run_dir = LOG_DIR / subdir
    run_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    handler = logging.FileHandler(run_dir / f"{timestamp}.log")
    handler.setLevel(logging.INFO)
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield
    finally:
        root.removeHandler(handler)
        handler.close()


def _log_stage(label: str, value) -> None:
    """Log one pipeline layer's input or output as pretty JSON under a banner.

    Pydantic models (IR, AnnotationPayload) are dumped via model_dump; everything
    else goes through json.dumps with a str fallback so nothing can break the
    request. Each entry is fenced by a banner so the layers are easy to scan in
    logs/evaluator.log.
    """

    if hasattr(value, "model_dump"):
        value = value.model_dump()
    try:
        body = json.dumps(value, indent=2, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        body = str(value)
    logger.info("----- %s -----\n%s", label, body)

# Returned when the photo cannot be annotated reliably and should be retaken.
RETAKE_STATUS = 422


@app.get("/health")
def health() -> dict:
    """Report that the service is running."""

    return {"status": "ok"}


@app.post("/annotate", response_model=AnnotationPayload)
async def annotate(
    photo: UploadFile,
    question: str = Form(...),
    question_type: str = Form("short_answer"),
    answer_key: Optional[str] = Form(None),
    rubric: Optional[str] = Form(None),
    marking_note: Optional[str] = Form(None),
) -> AnnotationPayload:
    """
    Turn an uploaded photo into an annotated result.

    Reads the page with both readers and fuses them, checks the reading is
    trustworthy, then judges it with the evaluator chosen by `question_type` and
    runs the resolver to produce the payload the frontend draws. When the readers
    disagree on the line count, or the reading is too unclear, the page is refused
    with a retake-the-photo message instead of being annotated wrongly.

    `question_type` selects the evaluator: "short_answer" (the default) marks
    against an optional answer key / rubric / marking note and highlights each
    sentence; "essay" marks against a rubric and highlights strength excerpts,
    with the per-criterion scores shown in the panel scorecard.

    Everything but the question and type is optional: the answer key, rubric, and
    marking note only guide short-answer scoring when supplied.
    """

    image_bytes = await photo.read()

    # Give this run its own timestamped file under logs/<question_type>/ so every
    # layer's trace lands in a fresh, per-evaluator log that is easy to find later.
    with _annotate_log_file(question_type):
        return await _run_annotate(
            image_bytes,
            photo.filename,
            question,
            question_type,
            answer_key,
            rubric,
            marking_note,
        )


async def _run_annotate(
    image_bytes: bytes,
    photo_filename: Optional[str],
    question: str,
    question_type: str,
    answer_key: Optional[str],
    rubric: Optional[str],
    marking_note: Optional[str],
) -> AnnotationPayload:
    # Trace every layer's input and output into this run's log file so the weak
    # link in the pipeline can be pinpointed (e.g. where spelling is lost).
    _log_stage(
        "REQUEST input",
        {
            "photo_filename": photo_filename,
            "photo_bytes": len(image_bytes),
            "question": question,
            "question_type": question_type,
            "answer_key": answer_key,
            "rubric": rubric,
            "marking_note": marking_note,
        },
    )

    # LAYER 1 -- Mathpix reader: shapes + its own (often misread) text.
    ir = build_ir(image_bytes)
    _log_stage("LAYER 1 build_ir output (Mathpix IR)", ir)

    # EXPERIMENT (Case 1): pass Mathpix's literal reading to the VLM as a spelling
    # hint, to counter the VLM's autocorrect prior without trusting the OCR's
    # (often wrong) word choices. Revert this commit to disable.
    mathpix_hint = "\n".join(seg.text for seg in ir.segments)
    _log_stage("LAYER 2 VLM Mathpix hint (input)", mathpix_hint)

    # LAYER 2 -- VLM reader: accurate text, no reliable positions. This is the
    # prime suspect for silent spelling correction; log it verbatim.
    try:
        vlm_lines = transcribe_lines(image_bytes, mathpix_hint)
        _log_stage("LAYER 2 transcribe_lines output (raw VLM lines)", vlm_lines)

        # LAYER 3 -- Fusion: VLM words laid onto Mathpix shapes.
        fused = fuse(ir, vlm_lines)
    except AlignmentError as exc:
        _log_stage("LAYER 3 fuse REFUSED (AlignmentError)", str(exc))
        raise HTTPException(status_code=RETAKE_STATUS, detail=exc.warning)
    # flat_text is the exact string the evaluator judges -- compare it against the
    # raw VLM lines above to see whether fusion changed any words.
    _log_stage("LAYER 3 fuse output (fused IR + flat_text)", fused)

    # LAYER 4 -- Readability guard.
    readable = is_readable(fused)
    _log_stage("LAYER 4 is_readable decision", {"readable": readable})
    if not readable:
        raise HTTPException(status_code=RETAKE_STATUS, detail=RETAKE_WARNING)

    # LAYERS 5-7 -- Evaluate, adapt, and resolve. The reading stages above are
    # shared; only the evaluator differs, chosen here by question_type.
    if question_type == "essay":
        return await _run_essay(fused, question, rubric)
    return await _run_short_answer(fused, question, answer_key, rubric, marking_note)


def _parse_essay_rubric(rubric: Optional[str]) -> Optional[list]:
    """Parse the essay rubric form value into a rubric table, or None to fall back.

    The essay rubric is a STRUCTURED table (a JSON list of criteria), unlike the
    short-answer free-text rubric. It is optional: an empty, non-JSON, or non-list
    value returns None so `evaluate_essay` uses its hardcoded default rubric.
    """

    if not rubric or not rubric.strip():
        return None
    try:
        parsed = json.loads(rubric)
    except (ValueError, TypeError):
        logger.warning("Essay rubric is not valid JSON; falling back to default rubric.")
        return None
    if isinstance(parsed, list) and parsed:
        return parsed
    logger.warning("Essay rubric JSON is not a non-empty list; falling back to default rubric.")
    return None


async def _run_short_answer(
    fused,
    question: str,
    answer_key: Optional[str],
    rubric: Optional[str],
    marking_note: Optional[str],
) -> AnnotationPayload:
    """Short-answer strategy: mark against the answer key / rubric, highlight
    each sentence."""

    # LAYER 5 -- Evaluator (idea marking) + FLC (spelling/grammar), run
    # concurrently so the total wait is ~max(evaluator, FLC), not the sum. FLC
    # degrades to [] on failure, so a down spell-checker never fails the request.
    _log_stage(
        "LAYER 5 evaluate input",
        {
            "question": question,
            "student_answer": fused.flat_text,
            "answer_key": answer_key,
            "rubric": rubric or "",
            "marking_note": marking_note or "",
        },
    )
    _log_stage(
        "LAYER 5 FLC input",
        {"student_composition": fused.flat_text, "question_statement": question},
    )
    evaluation, flc_findings = await asyncio.gather(
        evaluate(
            question,
            fused.flat_text,
            answer_key,
            rubric=rubric or "",
            marking_note=marking_note or "",
            debug_mode=True,
        ),
        flc.check(fused.flat_text, question),
    )
    _log_stage("LAYER 5 evaluate output (full evaluator result + prompts)", evaluation)
    _log_stage(
        "USAGE / COST",
        {
            "total_tokens": evaluation.get("total_tokens"),
            "total_cost": evaluation.get("total_cost"),
            "models_used": evaluation.get("models_used"),
            # FLC's dollar cost is spent in its own process and not returned, so
            # it cannot be logged here -- only its finding count.
            "flc_findings": len(flc_findings),
        },
    )
    _log_stage("LAYER 5 FLC output (checking[])", flc_findings)

    # LAYER 6 -- Adapters: evaluator highlights + FLC findings -> positioned
    # feedback. FLC findings are additive (extra red spelling/grammar highlights).
    feedback = to_feedback(fused.flat_text, evaluation)
    flc_feedback = flc_to_feedback(fused.flat_text, flc_findings)
    _log_stage("LAYER 6 to_feedback output (evaluator)", [f.model_dump() for f in feedback])
    _log_stage("LAYER 6 FLC adapter output", [f.model_dump() for f in flc_feedback])

    # LAYER 7 -- Resolver: char spans -> polygons -> the payload the frontend draws.
    payload = resolve_payload(fused, feedback + flc_feedback)
    _log_stage("LAYER 7 resolve_payload output (final payload)", payload)

    return payload


async def _run_essay(fused, question: str, rubric: Optional[str]) -> AnnotationPayload:
    """Essay strategy: mark against a rubric, highlight FLC spelling/grammar
    findings on the page, and show the per-criterion scores in the panel.

    The on-page highlights come from FLC (red spelling/grammar), not the compo
    marker's strengths; the rubric feedback lives in the scorecard. The rubric is
    optional -- an empty or unparseable rubric falls back to the hardcoded default.
    """

    rubric_table = _parse_essay_rubric(rubric)

    # LAYER 5 -- Essay evaluator (idea/rubric marking) + FLC (spelling/grammar),
    # run concurrently so the wait is ~max(evaluator, FLC). FLC degrades to [].
    _log_stage(
        "LAYER 5 evaluate_essay input",
        {
            "question": question,
            "student_answer": fused.flat_text,
            "rubric_provided": rubric_table is not None,
        },
    )
    _log_stage(
        "LAYER 5 FLC input",
        {"student_composition": fused.flat_text, "question_statement": question},
    )
    evaluation, flc_findings = await asyncio.gather(
        evaluate_essay(question, fused.flat_text, rubric_table),
        flc.check(fused.flat_text, question),
    )
    _log_stage("LAYER 5 evaluate_essay output (full essay result)", evaluation)
    _log_stage(
        "USAGE / COST",
        {
            "total_tokens": evaluation.get("total_tokens"),
            "total_cost": evaluation.get("total_cost"),
            "models_used": evaluation.get("models_used"),
            "flc_findings": len(flc_findings),
        },
    )
    _log_stage("LAYER 5 FLC output (checking[])", flc_findings)

    # LAYER 6 -- Adapters: FLC findings -> red ERROR feedback (the on-page
    # highlights); per-criterion results -> panel scorecard.
    flc_feedback = flc_to_feedback(fused.flat_text, flc_findings)
    scorecard = score_detail_to_scorecard(
        evaluation["score_detail"], evaluation["rubric_table"]
    )
    _log_stage("LAYER 6 FLC adapter output", [f.model_dump() for f in flc_feedback])
    _log_stage("LAYER 6 essay adapter output (scorecard)", [s.model_dump() for s in scorecard])

    # LAYER 7 -- Resolver: char spans -> polygons -> the payload the frontend draws.
    payload = resolve_payload(fused, flc_feedback, scorecard)
    _log_stage("LAYER 7 resolve_payload output (final payload)", payload)

    return payload
