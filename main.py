"""
The backend service entrypoint.

It exposes two routes:

  - /health   : a simple check that the service is up.
  - /annotate : the pipeline route. It takes an uploaded photo and returns the
                annotation payload the frontend draws.

The pipeline is strictly serial: clean the image, read it, package it, judge it,
turn feedback into shapes, hand it to the frontend. The judging step (the
evaluator) is on hold, so for now the route closes the loop with fixture feedback
and the earlier layers are not yet wired in. This lets the frontend and the
overall shape be exercised end-to-end before the evaluator returns.
"""

from fastapi import FastAPI, UploadFile

from contracts.schema import AnnotationPayload
from fixtures.sample_payload import SAMPLE_PAYLOAD

app = FastAPI(title="Handwritten Answer Annotation")


@app.get("/health")
def health() -> dict:
    """Report that the service is running."""

    return {"status": "ok"}


@app.post("/annotate", response_model=AnnotationPayload)
async def annotate(photo: UploadFile) -> AnnotationPayload:
    """
    Turn an uploaded photo into an annotated result.

    While the evaluator is on hold, this returns fixture feedback so the chain is
    testable end-to-end. The real serial pipeline (clean -> read -> package ->
    evaluate -> shapes) is wired in here as each layer is built.
    """

    # The upload is read so the route accepts a real file, even though the
    # earlier pipeline layers are not yet wired in.
    await photo.read()

    return SAMPLE_PAYLOAD
