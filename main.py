"""
The backend service entrypoint.

It exposes two routes:

  - /health   : a simple check that the service is up.
  - /annotate : the pipeline route. It takes an uploaded photo and returns the
                annotation payload the frontend draws.

The pipeline is strictly serial: read the page (Mathpix shapes + GPT-5 text,
fused), package it into the IR, judge it, turn feedback into shapes, hand it to
the frontend. The judging step (the evaluator) is on hold, so the route stands in
fixture feedback anchored to the real reading. A page the two readers disagree
on, or one read too unclearly to trust, is not annotated: the student is asked
for a clearer photo instead.
"""

from fastapi import FastAPI, HTTPException, UploadFile

from contracts.schema import AnnotationPayload
from fixtures.sample_payload import stand_in_feedback
from geometry.resolver import resolve_payload
from ocr.fusion import RETAKE_WARNING, AlignmentError, fuse
from ocr.ir_builder import build_ir, is_readable
from ocr.vlm import transcribe_lines

app = FastAPI(title="Handwritten Answer Annotation")

# Returned when the photo cannot be annotated reliably and should be retaken.
RETAKE_STATUS = 422


@app.get("/health")
def health() -> dict:
    """Report that the service is running."""

    return {"status": "ok"}


@app.post("/annotate", response_model=AnnotationPayload)
async def annotate(photo: UploadFile) -> AnnotationPayload:
    """
    Turn an uploaded photo into an annotated result.

    Reads the page with both readers and fuses them, checks the reading is
    trustworthy, then (while the evaluator is on hold) stands in fixture feedback
    and runs the real span-to-geometry resolver. When the readers disagree on the
    line count, or the reading is too unclear, the page is refused with a
    retake-the-photo message instead of being annotated wrongly.
    """

    image_bytes = await photo.read()

    # Read the page two ways: Mathpix for the shapes, GPT-5 for the text.
    ir = build_ir(image_bytes)
    try:
        fused = fuse(ir, transcribe_lines(image_bytes))
    except AlignmentError as exc:
        raise HTTPException(status_code=RETAKE_STATUS, detail=exc.warning)

    # Too unclear to place annotations on reliably: ask for a better photo.
    if not is_readable(fused):
        raise HTTPException(status_code=RETAKE_STATUS, detail=RETAKE_WARNING)

    # Evaluator ON HOLD: stand in feedback anchored to the real reading, then run
    # the real resolver to turn it into the payload the frontend draws.
    return resolve_payload(fused, stand_in_feedback(fused))
